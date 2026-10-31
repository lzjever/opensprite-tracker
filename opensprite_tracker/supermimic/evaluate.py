"""Evaluate a frozen policy in real mjlab/MuJoCo, never in the world model.

python -m supermimic_x2.evaluate --checkpoint RUN/model_015000.pt \
    --output outputs/supermimic/eval-15000-seed42 --num-envs 8 --seed 42

The output directory must be new. Default: attempt every transition of --clip 0;
stop early only if every instance has already failed (reported success 0).
Motion paths and full-body collision default to the checkpoint configuration.
Override with --motion FIRST.npz SECOND.npz --clip 1 --full-body-collision.
--max-steps is for an explicitly incomplete diagnostic, never a success result.
rollout.npz contains T+1 states and T transition metrics. live[t] is survival
after arriving at state t; score_mask[t] includes the transition causing the
first failure and excludes ALL later transitions. Quaternions are wxyz.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

from opensprite_tracker.runtime import prepare_runtime

prepare_runtime()

import numpy as np
import torch

from . import networks
from .networks import Policy


def resolve_motion_files(override, cfg):
    """Accept old single-path checkpoints and current lists, with CLI priority."""
    from .robot import MOTION_PATH

    files = override if override is not None else cfg.get("motion", MOTION_PATH)
    files = [files] if isinstance(files, (str, Path)) else list(files)
    if not files:
        raise ValueError("at least one motion path is required")
    return [Path(path).resolve(strict=True) for path in files]


def clip_bounds(motion, clip):
    """Inclusive global reference bounds for a selected clip."""
    if not 0 <= clip < motion.num_clips:
        raise ValueError(f"clip must be in [0,{motion.num_clips}), got {clip}")
    return int(motion.clip_start[clip]), int(motion.clip_end[clip])


def new_output(path, kind):
    if path is not None:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    parent = Path("outputs/supermimic").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=kind + "-" + time.strftime("%Y%m%d-%H%M%S") + "-", dir=parent
        )
    )


def load_policy(checkpoint, device="cpu", motion_override=None):
    """Own checkpoints only; infer exact architecture from weights and train args."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    from .robot import physical_contract

    if (
        ck.get("method") != "SuperMimic-OpenSprite-mjlab"
        or ck.get("robot_contract") != physical_contract()
    ):
        raise ValueError(
            "Checkpoint is not compatible with current native OpenSprite physics"
        )
    from opensprite_tracker.motion_contract import validate_motion_contract

    for path in resolve_motion_files(motion_override, cfg):
        validate_motion_contract(
            path,
            cfg.get("dataset_revision", ""),
            allow_diagnostic_motion=cfg.get("allow_diagnostic_motion") == "True",
        )
    if "reward_config" in ck:
        cfg["reward_config"] = ck["reward_config"]
    weights = ck["pi"]

    def layers(prefix):
        keys = sorted(
            (
                k
                for k, v in weights.items()
                if k.startswith(prefix + ".") and k.endswith(".weight") and v.ndim == 2
            ),
            key=lambda k: int(k.split(".")[1]),
        )
        return [weights[k].shape[0] for k in keys]

    enc, dec = layers("enc"), layers("dec")
    obs_dim = weights["obs_norm.mean"].numel()
    goal_dim = weights["goal_norm.mean"].numel()
    hist = int(cfg.get("hist", 10))
    goal_frames = int(cfg.get("goal_frames", goal_dim // 77))
    if goal_dim != 77 * goal_frames:
        raise ValueError(
            "checkpoint goal dimension is not the OpenSprite reference layout"
        )
    pi = Policy(
        obs_dim,
        goal_dim,
        hist=hist,
        z_dim=enc[-1],
        action_dim=dec[-1],
        encoder_hidden=enc[:-1],
        decoder_hidden=dec[:-1],
    ).to(device)
    pi.load_state_dict(weights, strict=True)
    pi.eval().requires_grad_(False)
    return pi, cfg, ck.get("iteration"), goal_frames


def scalar_stats(x, mask):
    """Statistics over valid transitions; missing coverage is null, not zero."""
    values = np.asarray(x)[mask].reshape(-1)
    finite = values[np.isfinite(values)]
    result = {
        "count": int(values.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    for name, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        result[name] = float(fn(finite)) if finite.size else None
    result["p95"] = float(np.quantile(finite, 0.95)) if finite.size else None
    return result


def require_finite_live(values, live, frame, quantity):
    """Abort numerical failures while scored; NaN comparisons cannot mean success."""
    finite = torch.ones_like(live)
    for value in values:
        finite &= torch.isfinite(value).reshape(len(live), -1).all(-1)
    bad = live & ~finite
    if bad.any():
        ids = bad.nonzero().flatten().tolist()
        raise RuntimeError(
            f"Nonfinite {quantity} at frame {frame}, envs {ids}; "
            "evaluation aborted without a success result"
        )


def summarize(metrics, mask, joint_names):
    result = {
        name: scalar_stats(value, mask)
        for name, value in metrics.items()
        if name not in ("joint_error_rad",)
    }
    result["joint_error_abs_rad"] = scalar_stats(
        np.abs(metrics["joint_error_rad"]), mask
    )
    result["per_joint"] = {
        name: {
            key: scalar_stats(metrics[key][..., j], mask)
            for key in (
                "speed_peak_rad_s",
                "speed_ratio",
                "raw_torque_peak_nm",
                "applied_torque_peak_nm",
            )
        }
        for j, name in enumerate(joint_names)
    }
    return result


@torch.no_grad()
def evaluate(args):
    from .env import TrackingSim
    from .mdp import OBS_DIM, make_mdp, reward_config
    from .motion import MotionLib
    from .robot import (
        CONTROL_DT,
        JOINT_NAMES,
        MJCF_PATH,
        default_joint_pos,
        velocity_limits,
        make_kinematics,
    )
    from .rotations import mat_to_quat, rot_angle_sq

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint.resolve(strict=True)
    pi, cfg, iteration, goal_frames = load_policy(checkpoint, args.device, args.motion)
    if pi.obs_dim != OBS_DIM or pi.action_dim != len(JOINT_NAMES):
        raise ValueError(
            "checkpoint actor interface does not match current OpenSprite MDP"
        )
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "none") == "bf16" else None
    motion_paths = resolve_motion_files(args.motion, cfg)
    requested_collision = getattr(args, "full_body_collision", None)
    full_body_collision = (
        bool(cfg.get("full_body_collision", False))
        if requested_collision is None
        else requested_collision
    )
    kin = make_kinematics(args.device, full_body_collision)
    motion = MotionLib(motion_paths, kin, args.device)
    clip = getattr(args, "clip", 0)
    ref_start, ref_end = clip_bounds(motion, clip)
    expected_steps = ref_end - ref_start
    steps = (
        min(expected_steps, args.max_steps)
        if args.max_steps is not None
        else expected_steps
    )
    requested_steps = steps
    if steps < 1 or args.num_envs < 1:
        raise ValueError("num-envs/max-steps must be positive")
    reward_cfg = reward_config(
        cfg.get("reward_profile", "paper"), cfg.get("reward_config")
    )
    interface = checkpoint.parent / "interface.json"
    reward_source = (
        "embedded checkpoint"
        if "reward_config" in cfg
        else "recorded profile / paper defaults"
    )
    if interface.exists() and "reward_config" not in cfg:
        recorded = json.loads(interface.read_text())
        if "reward" in recorded:
            reward_cfg = reward_config(recorded=recorded["reward"])
            reward_source = str(interface)
    mdp = make_mdp(
        kin,
        motion,
        default_joint_pos(args.device),
        velocity_limits(args.device),
        reward_cfg,
    )
    output = new_output(args.output, "eval")
    env = TrackingSim(
        num_envs=args.num_envs,
        motion=motion,
        mdp=mdp,
        device=args.device,
        seed=args.seed,
        eval_mode=True,
        randomize=args.randomize,
        reset_noise=False,
        auto_reset=False,
        gamma=float(cfg.get("gamma", 0.99)),
        full_body_collision=full_body_collision,
        eval_clip=clip,
    )
    cpu = lambda x: x.detach().cpu().numpy().copy()
    records = {
        k: []
        for k in (
            "root_pos",
            "root_quat_wxyz",
            "joint_pos",
            "joint_vel",
            "ref_frame",
            "live",
        )
    }
    metrics, masks, terminated = {}, [], []
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    first_failure = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=args.device
    )
    feet = mdp.feet
    limits = velocity_limits(args.device)
    initial_context = cpu(env.xi)
    force_limits = env.actuator.force_limit[:, env.act_ids].clone()

    def record(S, frame, live):
        values = (S[0], mat_to_quat(S[1]), S[4], S[5], frame, live)
        for key, value in zip(records, values):
            records[key].append(cpu(value))

    started = time.monotonic()
    try:
        obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        if args.sensor_noise:
            obs = mdp.add_noise(obs)
        history = obs[:, None].repeat(1, pi.hist, 1)
        record(env.state(), env.f, alive)
        for t in range(steps):
            # history already ends at s_t / frame t, including the previous command.
            action = pi(history, motion.goal(env.f, goal_frames))
            require_finite_live((action,), alive, t, "policy action")
            action = action.clamp(-20, 20)
            action = torch.where(alive[:, None], action, torch.zeros_like(action))
            before = alive.clone()
            out = env.step(action)  # advances physics AND reference exactly once
            S, frame = out["S1"], out["f1"]
            require_finite_live((*S, out["r"]), before, t + 1, "state/reward")
            newly_failed = before & out["term"]
            first_failure[newly_failed] = frame[newly_failed]
            alive = before & ~out["term"]
            masks.append(cpu(before))
            terminated.append(cpu(out["term"]))
            record(S, frame, alive)
            bp, bR, bv, bw, tp, tR = mdp._bodies_and_target(S, frame)
            foot_pos = bp[:, feet]
            reference_foot_pos = motion.body_p[frame][:, feet]
            clearance = kin.sole_bottom_z(foot_pos, bR[:, feet])
            ref_clearance = kin.sole_bottom_z(
                reference_foot_pos, motion.body_R[frame][:, feet]
            )
            contact = out["contact_peak_n"]
            speed = out["speed_peak"]
            speed_ratio = speed / limits
            terms = dict(out["terms"])
            terms.update(
                {
                    "reward": out["r"],
                    "body_pos_world_rmse_m": (bp - motion.body_p[frame])
                    .square()
                    .sum(-1)
                    .mean(-1)
                    .sqrt(),
                    "body_angle_rmse_rad": rot_angle_sq(tR, bR).mean(-1).sqrt(),
                    "body_linvel_rmse_m_s": (bv - motion.body_v[frame])
                    .square()
                    .sum(-1)
                    .mean(-1)
                    .sqrt(),
                    "body_angvel_rmse_rad_s": (bw - motion.body_w[frame])
                    .square()
                    .sum(-1)
                    .mean(-1)
                    .sqrt(),
                    "joint_error_rad": S[4] - motion.q[frame],
                    "foot_pos_error_m": (foot_pos - tp[:, feet]).norm(dim=-1),
                    "foot_world_error_m": (foot_pos - reference_foot_pos).norm(dim=-1),
                    "foot_angle_error_rad": rot_angle_sq(
                        tR[:, feet], bR[:, feet]
                    ).sqrt(),
                    "foot_clearance_m": clearance,
                    "reference_foot_clearance_m": ref_clearance,
                    "foot_clearance_error_m": clearance - ref_clearance,
                    "speed_peak_rad_s": speed,
                    "speed_ratio": speed_ratio,
                    "speed_excess_rad_s": (speed - limits).clamp_min(0),
                    "overspeed_joint_fraction": (speed_ratio > 1).float().mean(-1),
                    "raw_torque_peak_nm": out["raw_torque_peak"],
                    "applied_torque_peak_nm": out["applied_torque_peak"],
                    "raw_torque_ratio": out["raw_torque_peak"] / force_limits,
                    "applied_torque_ratio": out["applied_torque_peak"] / force_limits,
                    "contact_peak_n": contact,
                    "contact_resultant_peak_n": out["contact_force_history"]
                    .norm(dim=-1)
                    .amax(-1),
                    "both_feet_airborne": (
                        (clearance > 0.002).all(-1)
                        & (contact < args.contact_threshold).all(-1)
                    ).float(),
                }
            )
            if "body_contact_peak_n" in out:
                terms["body_contact_peak_n"] = out["body_contact_peak_n"]
            require_finite_live(terms.values(), before, t + 1, "diagnostic metrics")
            for key, value in terms.items():
                metrics.setdefault(key, []).append(cpu(value))
            # Noise-free observations, frozen normalization, no extra frame increment.
            obs = mdp.obs(S, action, env.xi, env.f)
            if args.sensor_noise:
                obs = mdp.add_noise(obs)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            if (t + 1) % 250 == 0 or t + 1 == steps:
                print(
                    f"EVAL {t + 1}/{steps} alive={int(alive.sum())}/{args.num_envs}",
                    flush=True,
                )
            if not alive.any():
                print(
                    f"ALL_FAILED at transition {t + 1}; no post-failure simulation needed",
                    flush=True,
                )
                break
    finally:
        env.close()

    records = {key: np.stack(value) for key, value in records.items()}
    metrics = {key: np.stack(value) for key, value in metrics.items()}
    score_mask = np.stack(masks)
    steps = len(masks)
    local_frames = records["ref_frame"] - ref_start
    arrival_time = local_frames[1:] * CONTROL_DT
    jump_mask = score_mask & (arrival_time >= 32.0) & (arrival_time <= 34.0)
    full_times = np.arange(1, expected_steps + 1) * CONTROL_DT
    window_expected = int(((full_times >= 32.0) & (full_times <= 34.0)).sum())
    complete = steps == expected_steps
    failures = cpu(first_failure)
    per_env = []
    for i in range(args.num_envs):
        failure = int(failures[i])
        mask = score_mask[:, i]
        per_env.append(
            {
                "env": i,
                "success": bool(failure < 0)
                if complete
                else (False if failure >= 0 else None),
                "first_failure_frame": failure if failure >= 0 else None,
                "first_failure_local_frame": failure - ref_start
                if failure >= 0
                else None,
                "first_failure_seconds": (failure - ref_start) * CONTROL_DT
                if failure >= 0
                else None,
                "scored_transitions": int(mask.sum()),
                "valid_return": float(metrics["reward"][:, i][mask].sum()),
                "jump_window_scored_transitions": int(jump_mask[:, i].sum()),
            }
        )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    metadata = {
        "schema": 1,
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "motion": str(motion_paths[0])
        if len(motion_paths) == 1
        else [str(path) for path in motion_paths],
        "source_files": motion.source_files,
        "clip": clip,
        "clip_name": motion.clip_names[clip],
        "ref_start": ref_start,
        "ref_end": ref_end,
        "full_body_collision": full_body_collision,
        "model_path": str(MJCF_PATH.resolve()),
        "model_path_role": "render mesh asset; full_body_spec supplies whole-body physics when enabled",
        "git_commit": git_commit,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "policy_hist": pi.hist,
        "goal_frames": goal_frames,
        "train_args": cfg,
        "reward_config": asdict(reward_cfg),
        "reward_config_source": reward_source,
        "reward_profile": reward_cfg.profile,
        "relative_body_anchor": "waist_yaw_link"
        if reward_cfg.profile == "legacy_full"
        else "pelvis_link",
        "control_dt": CONTROL_DT,
        "expected_transitions": expected_steps,
        "requested_transitions": requested_steps,
        "executed_transitions": steps,
        "complete_clip": complete,
        "stopped_after_all_failed": bool((failures >= 0).all()),
        "success_rate": (
            float((failures < 0).mean())
            if complete
            else (
                0.0
                if requested_steps == expected_steps and (failures >= 0).all()
                else None
            )
        ),
        "action_noise": False,
        "sensor_noise": args.sensor_noise,
        "domain_randomization": args.randomize,
        "actuator_delay": "baseline actuator/delay dynamics retained; seed supplied",
        "score_mask": "alive before transition; first failure included, all later samples excluded",
        "live": "alive after arriving at stored state; final clip completion is not failure",
        "torque_speed_sampling": "maximum absolute value over four physics substeps (plus endpoint speed)",
        "contact_sampling": "four physics substeps; normal positive=-sensor force_z",
        "clearance": "signed official box-sole minimum z above the flat z=0 plane; left,right",
        "airborne_thresholds": {
            "clearance_m": 0.002,
            "contact_peak_n": args.contact_threshold,
        },
        "wall_seconds": time.monotonic() - started,
        "per_env": per_env,
        "overall": summarize(metrics, score_mask, JOINT_NAMES),
        "jump_32_34s": {
            "interpretation": "fixed clip-local time window; no jump annotation implied for other motions",
            "expected_transitions_per_env": window_expected,
            "scored_transitions": int(jump_mask.sum()),
            "coverage": float(jump_mask.sum() / (window_expected * args.num_envs))
            if window_expected
            else None,
            "metrics": summarize(metrics, jump_mask, JOINT_NAMES),
        },
    }
    ref = torch.arange(ref_start, ref_start + steps + 1, device=args.device)
    arrays = {
        **records,
        "score_mask": score_mask,
        "terminated": np.stack(terminated),
        "ref_local_frame": local_frames,
        "time_seconds": local_frames * CONTROL_DT,
        "first_failure_frame": failures,
        "joint_names": np.asarray(JOINT_NAMES),
        "first_failure_local_frame": np.where(failures >= 0, failures - ref_start, -1),
        "body_names": np.asarray(kin.tracked_names),
        "fps": np.asarray(1 / CONTROL_DT),
        "ref_root_pos": cpu(motion.root_p[ref]),
        "ref_root_quat_wxyz": cpu(mat_to_quat(motion.root_R[ref])),
        "ref_joint_pos": cpu(motion.q[ref]),
        "initial_context": initial_context,
        "velocity_limits": cpu(limits),
        "force_limits": cpu(force_limits),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    k: v
                    for k, v in metadata.items()
                    if k not in ("overall", "jump_32_34s")
                },
                default=str,
            )
        ),
        **{"metric_" + key: value for key, value in metrics.items()},
    }
    np.savez_compressed(output / "rollout.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "complete_clip": complete,
                "success_rate": metadata["success_rate"],
                "per_env": per_env,
            }
        ),
        flush=True,
    )
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--motion", type=Path, nargs="+", help="default: checkpoint motion path(s)"
    )
    p.add_argument(
        "--clip",
        type=int,
        default=0,
        help="clip index within the packed motion library",
    )
    p.add_argument(
        "--full-body-collision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: checkpoint collision model",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-steps",
        type=int,
        help="diagnostic only; success_rate=null unless full clip",
    )
    p.add_argument("--contact-threshold", type=float, default=10.0)
    p.add_argument(
        "--randomize",
        action="store_true",
        help="use the training physics DR, default is nominal",
    )
    p.add_argument(
        "--sensor-noise",
        action="store_true",
        help="use training observation noise, default is noiseless",
    )
    return p


def main():
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
