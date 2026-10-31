"""Fresh-simulation, open-loop 1/8-step OpenSprite world-model diagnostic (no training).

Example:
  python -m supermimic_x2.validate_model --checkpoint RUN/model_001000.pt \
      --output outputs/supermimic/model-check-1000

Default budget: 64 independent environments, four random-reference resets,
eight real control steps per segment (32 vectorized steps total). The frozen
policy acts on REAL observations. The WM gets only the segment's initial real
state/context, then rolls autonomously through the same recorded actions.
The transition causing termination/timeout is scored; all later transitions
in that segment are masked, even though the real environment can auto-reset.
Motion paths and full-body collision default to the checkpoint configuration.
--clip selects random start frames only within that clip; omitted, environments
are assigned clips by env_id % num_clips, including every subsequent reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from . import networks
from .networks import Policy, WorldModel
from .robot import CONTEXT_DIM, MEM_DIM
from .evaluate import clip_bounds, resolve_motion_files
from .rotations import rot_angle_sq

HORIZON = 8
SCORED_STEPS = (1, 8)


def load_models(path, device, motion_override=None):
    """Load this project's trusted checkpoint, including frozen normalization."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("method") != "SuperMimic-OpenSprite-mjlab":
        raise ValueError("Expected a SuperMimic-OpenSprite-mjlab checkpoint")
    cfg = dict(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    from .robot import physical_contract

    if ck.get("robot_contract") != physical_contract():
        raise ValueError("Checkpoint physical robot contract changed")
    from opensprite_tracker.motion_contract import validate_motion_contract
    from .evaluate import resolve_motion_files

    for source in resolve_motion_files(motion_override, cfg):
        validate_motion_contract(
            source,
            cfg.get("dataset_revision", ""),
            allow_diagnostic_motion=cfg.get("allow_diagnostic_motion") == "True",
        )
    if "reward_config" in ck:
        cfg["reward_config"] = ck["reward_config"]

    def widths(weights, prefix):
        keys = sorted(
            (
                k
                for k, value in weights.items()
                if k.startswith(prefix + ".")
                and k.endswith(".weight")
                and value.ndim == 2
            ),
            key=lambda k: int(k.split(".")[1]),
        )
        return [weights[k].shape[0] for k in keys]

    weights = ck["pi"]
    enc, dec = widths(weights, "enc"), widths(weights, "dec")
    obs_dim, goal_dim = (
        weights["obs_norm.mean"].numel(),
        weights["goal_norm.mean"].numel(),
    )
    if obs_dim != 105 or dec[-1] != 31 or goal_dim % 77:
        raise ValueError(
            "Checkpoint actor must use OpenSprite 105-observation / 31-action interface"
        )
    pi = Policy(
        obs_dim,
        goal_dim,
        hist=int(cfg["hist"]),
        z_dim=enc[-1],
        encoder_hidden=enc[:-1],
        decoder_hidden=dec[:-1],
    ).to(device)
    pi.load_state_dict(weights, strict=True)
    wm = WorldModel(
        widths(ck["wm"], "net")[:-1], context_dim=CONTEXT_DIM, memory_dim=MEM_DIM
    ).to(device)
    wm.load_state_dict(ck["wm"], strict=True)
    for model in (pi, wm):
        model.eval().requires_grad_(False)
    return pi, wm, cfg, ck.get("iteration"), goal_dim // 77


def squared_errors(prediction, actual):
    """Per-environment mean-square error; vectors average their components.

    Rotation is the SO(3) geodesic angle, not component-wise quaternion error.
    Memory is already normalized by the adapter; no WM loss scaling is applied.
    """
    return {
        "root_position_rmse_m": (prediction[0] - actual[0]).square().mean(-1),
        "root_rotation_rmse_rad": rot_angle_sq(prediction[1], actual[1]),
        "joint_position_rmse_rad": (prediction[4] - actual[4]).square().mean(-1),
        "root_linear_velocity_rmse_m_s": (prediction[2] - actual[2]).square().mean(-1),
        "joint_velocity_rmse_rad_s": (prediction[5] - actual[5]).square().mean(-1),
        "memory_normalized_rmse": (prediction[6] - actual[6]).square().mean(-1),
    }


class ErrorTotals:
    def __init__(self):
        self.count = 0
        self.terminal_count = 0
        self.sums = {}
        self.nonfinite = {}

    def add(self, errors, valid, done):
        self.count += int(valid.sum())
        self.terminal_count += int((valid & done).sum())
        for name, values in errors.items():
            selected = values[valid].double()
            finite = torch.isfinite(selected)
            self.sums[name] = self.sums.get(name, 0.0) + float(selected[finite].sum())
            self.nonfinite[name] = self.nonfinite.get(name, 0) + int((~finite).sum())

    def report(self, candidates):
        result = {
            "valid_count": self.count,
            "candidate_count": candidates,
            "valid_fraction": self.count / candidates,
            "terminal_transitions_included": self.terminal_count,
            "nonfinite_count": self.nonfinite,
        }
        result.update(
            {
                name: (total / self.count) ** 0.5
                if self.count and not self.nonfinite[name]
                else None
                for name, total in self.sums.items()
            }
        )
        return result


@torch.no_grad()
def validate(args):
    from .env import make_env
    from .mdp import make_mdp, reward_config
    from .motion import MotionLib
    from .robot import CONTROL_DT, default_joint_pos, velocity_limits, make_kinematics
    from .sampling import ReturnSampler

    if args.num_envs < 1 or args.segments < 1:
        raise ValueError("num-envs and segments must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pi, wm, cfg, iteration, goal_frames = load_models(
        checkpoint, args.device, args.motion
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
    selected_clip = getattr(args, "clip", None)
    if selected_clip is not None:
        clip_bounds(motion, selected_clip)
    reward_cfg = reward_config(
        cfg.get("reward_profile", "paper"), cfg.get("reward_config")
    )
    interface = checkpoint.parent / "interface.json"
    if interface.is_file() and "reward_config" not in cfg:
        settings = json.loads(interface.read_text())
        if "reward" in settings:
            reward_cfg = reward_config(recorded=settings["reward"])
    mdp = make_mdp(
        kin,
        motion,
        default_joint_pos(args.device),
        velocity_limits(args.device),
        reward_cfg,
    )
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(
        args.num_envs,
        motion,
        mdp,
        device=args.device,
        seed=args.seed,
        reset_noise=True,
        randomize=True,
        auto_reset=True,
        gamma=float(cfg.get("gamma", 0.99)),
        full_body_collision=full_body_collision,
    )
    sampler = ReturnSampler(motion)
    env.clip_sampler = (
        sampler.for_clips
        if selected_clip is None
        else lambda ids: sampler.for_clips(torch.full_like(ids, selected_clip))
    )
    totals = {step: ErrorTotals() for step in SCORED_STEPS}
    segment_reports, initial_frames, initial_local_frames, initial_clips = (
        [],
        [],
        [],
        [],
    )
    start = time.monotonic()
    try:
        for segment in range(args.segments):
            env.reset()  # Apply selection to segment zero as well as later resets.
            initial_frames.append(env.f.cpu().tolist())
            clip_ids = motion.clip_id[env.f]
            initial_clips.append(clip_ids.cpu().tolist())
            initial_local_frames.append(
                (env.f - motion.clip_start[clip_ids]).cpu().tolist()
            )
            prediction = env.state()  # the ONLY real-state input to this WM segment
            context = env.normalize_xi(env.xi)
            history = torch.zeros(args.num_envs, pi.hist, 99, device=args.device)
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
            segment_counts = {}
            for step in range(1, HORIZON + 1):
                obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                if args.observation_noise:
                    obs = mdp.add_noise(obs)
                fresh = env.ep_step == 0
                history = torch.where(
                    fresh[:, None, None],
                    obs[:, None].expand_as(history),
                    torch.cat((history[:, 1:], obs[:, None]), 1),
                )
                action = pi(history, motion.goal(env.f, goal_frames)).clamp(-20, 20)
                out = env.step(action)
                prediction = wm.step(prediction, out["a"], context)
                # s1 is the cloned terminal-before-reset state. Score before
                # clearing alive so the terminal transition itself is retained.
                if step in totals:
                    totals[step].add(
                        squared_errors(prediction, out["S1"]), alive, out["done"]
                    )
                    segment_counts[str(step)] = int(alive.sum())
                alive &= ~out["done"]
            segment_reports.append({"segment": segment, "valid_counts": segment_counts})
        report = {
            "kind": "fresh_simulation_heldout_world_model_diagnostic",
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": iteration,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip(),
            "motion": str(motion_paths[0])
            if len(motion_paths) == 1
            else [str(path) for path in motion_paths],
            "source_files": motion.source_files,
            "clip_names": motion.clip_names,
            "clip_start": motion.clip_start.cpu().tolist(),
            "clip_end": motion.clip_end.cpu().tolist(),
            "clip": selected_clip,
            "full_body_collision": full_body_collision,
            "seed": args.seed,
            "training_seed": cfg.get("seed"),
            "num_envs": args.num_envs,
            "segments": args.segments,
            "horizon": HORIZON,
            "real_vector_steps": args.segments * HORIZON,
            "real_transitions_including_masked": args.num_envs
            * args.segments
            * HORIZON,
            "control_dt_s": CONTROL_DT,
            "physics_dt_s": 0.005,
            "randomize_physics": True,
            "reset_noise": True,
            "policy_observation_noise": args.observation_noise,
            "action_exploration_noise": 0.0,
            "action_clip": [-20, 20],
            "amp": cfg.get("amp", "none"),
            "wm_teacher_forcing": False,
            "normalizers_updated": False,
            "mask": "include first terminal/timeout transition; exclude every later transition in segment",
            "rmse_convention": "sqrt(mean over valid environments and vector components); rotation uses geodesic angle",
            "empty_or_nonfinite_rmse": "null; inspect valid_count and nonfinite_count",
            "memory_interpretation": "normalized conditional-mean prediction, not deterministic RNG replay",
            "initial_reference_frames": initial_frames,
            "segment_coverage": segment_reports,
            "initial_reference_clip_ids": initial_clips,
            "initial_reference_local_frames": initial_local_frames,
            "initial_reference_seconds": [
                [frame * CONTROL_DT for frame in row] for row in initial_local_frames
            ],
            "source_motion_fk_error": motion.source_fk_error,
            "steps": {
                str(step): total.report(args.segments * args.num_envs)
                for step, total in totals.items()
            },
            "elapsed_seconds": time.monotonic() - start,
        }
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"report": str(path), "steps": report["steps"]},
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return report
    finally:
        env.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new report directory")
    p.add_argument(
        "--motion", type=Path, nargs="+", help="default: checkpoint motion path(s)"
    )
    p.add_argument(
        "--clip", type=int, help="random-reference starts restricted to this clip"
    )
    p.add_argument(
        "--full-body-collision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: checkpoint collision model",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument(
        "--observation-noise", action=argparse.BooleanOptionalAction, default=True
    )
    return p


if __name__ == "__main__":
    validate(parser().parse_args())
