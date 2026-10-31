"""Train the X2 single-motion SuperMimic reproduction on mjlab.

Example: python -m supermimic_x2.train --output outputs/supermimic/run-01
All updates are model-based; the old PPO task/runner is not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from opensprite_tracker.runtime import prepare_runtime

prepare_runtime()

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import networks
from .buffer import ReplayBuffer
from .engine import Learner
from .mdp import OBS_DIM, make_mdp, reward_config
from .motion import MotionLib
from .networks import Critic, Policy, WorldModel
from .robot import (
    CONTEXT_DIM,
    MEM_DIM,
    MOTION_PATH,
    default_joint_pos,
    velocity_limits,
    make_kinematics,
)
from .state import unflatten_state
from .sampling import ReturnSampler


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-revision", default="")
    p.add_argument(
        "--allow-diagnostic-motion", choices=("True", "False"), default="False"
    )
    p.add_argument("--resume-runtime-seconds-floor", type=float, default=0.0)
    p.add_argument("--motion", type=Path, nargs="+", default=[MOTION_PATH])
    p.add_argument(
        "--full-body-collision", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="current attempt wall budget including setup/compile; zero disables",
    )
    p.add_argument("--save-every-seconds", type=float, default=600.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument(
        "--iterations",
        type=int,
        default=320000,
        help="absolute end iteration for this runner",
    )
    p.add_argument("--buffer-steps", type=int, default=1000)
    p.add_argument("--utd", type=float, default=0.4)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--goal-frames", type=int, default=10)
    p.add_argument("--hist", type=int, default=10)
    p.add_argument("--wm-window", type=int, default=8)
    p.add_argument("--wm-batches", type=int, default=10)
    p.add_argument("--wm-batch", type=int, default=2048)
    p.add_argument("--pi-batch", type=int, default=4096)
    p.add_argument("--wm-width", type=int, default=2048)
    p.add_argument("--wm-microbatch", type=int, default=256, help="gradient accumulation within the full logical WM batch; 0 disables")
    p.add_argument("--pi-microbatch", type=int, default=128, help="gradient accumulation within the full logical imagined rollout batch; 0 disables")
    p.add_argument("--critic-microbatch", type=int, default=8192, help="gradient accumulation within each logical critic minibatch; 0 disables")
    p.add_argument("--critic-storage", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--replay-device", choices=["cpu", "cuda"], default="cpu", help="storage only; sampled transitions are returned to --device")
    p.add_argument("--policy-hidden", type=int, nargs="+", default=[2048, 1024, 512, 256])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--action-noise", type=float, default=0.1)
    p.add_argument("--critic-epochs", type=int, default=5)
    p.add_argument("--critic-minibatches", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=64)
    p.add_argument("--wm-only-iters", type=int, default=0)
    p.add_argument("--lr-decay-iters", type=int, default=250000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--amp", choices=["none", "bf16"], default="none")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--reward-profile", choices=["paper", "legacy_full"], default="legacy_full"
    )
    p.add_argument(
        "--initialize-from",
        type=Path,
        help="copy only world model and actor; reset value/optimizers/reward sampling and iteration",
    )
    p.add_argument("--impact-ramp-updates", type=int, default=1000)
    return p


def checkpoint(learner, args, iteration, sim_steps, wall_seconds, sampler=None):
    result = {
        "wm": learner.wm.state_dict(),
        "pi": learner.pi.state_dict(),
        "critic": learner.critic.state_dict(),
        "opt_wm": learner.opt_wm.state_dict(),
        "opt_pi": learner.opt_pi.state_dict(),
        "opt_v": learner.opt_v.state_dict(),
        "iteration": iteration,
        "sim_steps": sim_steps,
        "wall_seconds": wall_seconds,
        "runtime_seconds": wall_seconds,
        "robot_contract": __import__(
            "opensprite_tracker.supermimic.robot", fromlist=["physical_contract"]
        ).physical_contract(),
        "dataset_revision": args.dataset_revision,
        "args": vars(args),
        "method": "SuperMimic-OpenSprite-mjlab",
        "schema": 1,
        "sampling": {"returns": sampler.returns, "past": sampler.past}
        if sampler
        else None,
        "reward_config": vars(learner.mdp.cfg),
    }
    if learner.transition_model is not None:
        result.update(
            transition_model=learner.transition_model.state_dict(),
            opt_transition=learner.opt_transition.state_dict(),
        )
    return result


def main():
    attempt_started = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True  # Save only at a coherent update boundary.

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parser().parse_args()
    torch.set_num_threads(4)
    if args.resume and args.initialize_from:
        raise ValueError("Use resume OR initialize-from, not both")
    if not args.full_body_collision:
        raise ValueError("OpenSprite task requires its full physical collision model")
    args.output = args.output.resolve()
    args.motion = [path.resolve(strict=True) for path in args.motion]
    # One immutable launch per directory; resuming uses a new output directory.
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("model_*.pt")) and not args.resume:
        raise ValueError("Existing output checkpoints require an explicit resume")
    from opensprite_tracker.motion_contract import validate_motion_contract

    for path in args.motion:
        validate_motion_contract(
            path,
            args.dataset_revision,
            allow_diagnostic_motion=args.allow_diagnostic_motion == "True",
        )
    args.context_dim, args.memory_dim = CONTEXT_DIM, MEM_DIM
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    launch_path = args.output / (
        "launch.json"
        if not (args.output / "launch.json").exists()
        else f"launch-resume-{time.time_ns()}.json"
    )
    launch_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "git_commit": git_commit,
                "source": str(Path(__file__).resolve().parents[1]),
                "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "python": os.sys.executable,
                "initialization": "resume"
                if args.resume
                else "wm_actor_only"
                if args.initialize_from
                else "all_networks_random",
                "resume_semantics": "iterations is absolute end; new output directory for every attempt",
            },
            indent=2,
            default=str,
        )
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    # The simulation adapter owns actual physics and per-env DR/memory.
    from .env import TrackingSim

    kin = make_kinematics(args.device, args.full_body_collision)
    motion = MotionLib(args.motion, kin, args.device)
    rcfg = reward_config(args.reward_profile)
    if args.full_body_collision and rcfg.profile == "legacy_full":
        rcfg.include_body_impact = True
    mdp = make_mdp(
        kin, motion, default_joint_pos(args.device), velocity_limits(args.device), rcfg
    )
    env = TrackingSim(
        num_envs=args.num_envs,
        motion=motion,
        mdp=mdp,
        device=args.device,
        seed=args.seed,
        gamma=args.gamma,
        full_body_collision=args.full_body_collision,
    )
    sampler = ReturnSampler(motion, args.gamma, reward_scale=mdp.reward_scale)
    env.frame_sampler = sampler
    if motion.num_clips > 1:
        env.clip_sampler = sampler.for_clips
    observe = torch.compile(mdp.obs) if args.compile else mdp.obs
    if args.compile:
        env.reward_head = torch.compile(mdp.reward)
    wm = WorldModel(
        hidden=(args.wm_width,) * 3, context_dim=CONTEXT_DIM, memory_dim=MEM_DIM
    ).to(args.device)
    pi = Policy(
        OBS_DIM, 77 * args.goal_frames, hist=args.hist, hidden=args.policy_hidden
    ).to(args.device)
    goal = motion.goal(env.f, args.goal_frames)
    with torch.no_grad():
        critic_dim = (
            mdp.critic_features(env.state(), env.f, env.normalize_xi(env.xi)).shape[-1]
            + goal.shape[-1]
        )
    critic = Critic(critic_dim).to(args.device)
    transition_model = None
    extras = None
    if mdp.uses_transition_metrics:
        from .transition_metrics import TransitionMetricsModel

        transition_model = TransitionMetricsModel(
            include_body=args.full_body_collision
        ).to(args.device)
        extras = {"transition_metrics": (transition_model.metric_dim, torch.float32)}
    buffer = ReplayBuffer(
        args.buffer_steps,
        args.num_envs,
        args.device,
        state_dim=75 + MEM_DIM,
        context_dim=CONTEXT_DIM,
        extra_fields=extras,
        storage_device=args.replay_device,
    )
    learner = Learner(
        wm,
        pi,
        critic,
        mdp,
        motion,
        buffer,
        env.normalize_xi,
        args,
        transition_model=transition_model,
    )
    learner.impact_ramp = env.impact_ramp
    start, sim_steps, wall_offset = 0, 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        from .robot import physical_contract

        if (
            ck.get("robot_contract") != physical_contract()
            or ck.get("dataset_revision") != args.dataset_revision
        ):
            raise ValueError("Checkpoint robot or UMR dataset revision differs")
        if ck.get("method") != "SuperMimic-OpenSprite-mjlab":
            raise ValueError("Only native OpenSprite training state may resume")
        if (
            bool(ck["args"].get("full_body_collision", False))
            != args.full_body_collision
        ):
            raise ValueError(
                "Collision model changed; resume must use the same physical model"
            )
        if ck["args"].get("max_wall_seconds", 0.0) != args.max_wall_seconds:
            raise ValueError(
                "Resume must preserve the campaign cumulative max-wall-seconds budget"
            )
        old_profile = ck.get("reward_config", {}).get("profile", "paper")
        if old_profile != args.reward_profile:
            raise ValueError(
                "Reward changed: use --initialize-from to reset the critic and reward optimizer state"
            )
        for name, module in (
            ("wm", wm),
            ("pi", pi),
            ("critic", critic),
            ("opt_wm", learner.opt_wm),
            ("opt_pi", learner.opt_pi),
            ("opt_v", learner.opt_v),
        ):
            module.load_state_dict(ck[name])
        if transition_model is not None:
            transition_model.load_state_dict(ck["transition_model"])
            learner.opt_transition.load_state_dict(ck["opt_transition"])
        start, sim_steps, wall_offset = (
            ck["iteration"],
            ck["sim_steps"],
            max(ck["wall_seconds"], args.resume_runtime_seconds_floor),
        )
        if ck.get("sampling"):
            sampler.returns.copy_(ck["sampling"]["returns"])
            sampler.past.copy_(ck["sampling"]["past"])
        print(
            f"RESUME absolute iteration {start} -> {args.iterations}; replay refilled from real simulation",
            flush=True,
        )
    elif args.initialize_from:
        ck = torch.load(
            args.initialize_from, map_location=args.device, weights_only=False
        )
        wm.load_state_dict(ck["wm"])
        pi.load_state_dict(ck["pi"])
        print(
            f"INITIALIZE physics WM + actor from iteration {ck['iteration']}; "
            "new reward/value/transition heads/optimizers/sampling; new iteration 0",
            flush=True,
        )
    writer = SummaryWriter(str(args.output / "tensorboard"))
    history = torch.zeros(args.num_envs, args.hist, OBS_DIM, device=args.device)
    stats = {}
    stats_count = {}

    def metric(key, value):
        # Keep reductions on GPU until a sparse log point, not sync every term.
        stats[key] = stats.get(key, 0.0) + value.detach()
        stats_count[key] = stats_count.get(key, 0) + 1

    @torch.no_grad()
    def collect(count):
        nonlocal history, sim_steps
        for _ in range(count):
            state = env.state()
            obs = mdp.add_noise(observe(state, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            history = torch.where(
                fresh[:, None, None],
                obs[:, None].expand_as(history),
                torch.cat((history[:, 1:], obs[:, None]), 1),
            )
            goal = motion.goal(env.f, args.goal_frames)
            pi.obs_norm.update(obs)
            pi.goal_norm.update(goal)
            action = (
                pi(history, goal) + args.action_noise * torch.randn_like(env.a_prev)
            ).clamp(-20, 20)
            out = env.step(action)
            if transition_model is not None:
                from .transition_metrics import pack_targets

                out["transition_metrics"] = pack_targets(
                    out, out["xi"], mdp.velocity_limits, kin.total_mass_kg * 9.81
                )
                transition_model.update_normalization(
                    out["S"], out["S1"], out["a"], env.normalize_xi(out["xi"])
                )
            sampler.update(out)
            buffer.add(out)
            before, after = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(before, out["a"], env.normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(before, after))
            sim_steps += args.num_envs
            metric("real/reward", out["r"].mean() / mdp.reward_scale)
            for key, value in out["terms"].items():
                if key.startswith("per_reward/"):
                    metric("real/" + key, value.mean())
            metric("real/terminated_per_step", out["term"].float().mean())
            # Per-clip averages, not length-weighted packed-array metrics.
            clip_id = motion.clip_id[out["f"]]
            for clip in range(motion.num_clips):
                selected = (clip_id == clip).float()
                count = selected.sum().clamp_min(1)
                metric(f"clip{clip}/active_fraction", selected.mean())
                metric(
                    f"clip{clip}/reward",
                    (out["r"] * selected).sum() / count / mdp.reward_scale,
                )
                metric(
                    f"clip{clip}/terminated_per_step",
                    (out["term"] * selected).sum() / count,
                )
                metric(f"clip{clip}/reset_frequency", (fresh * selected).sum() / count)
                metric(
                    f"clip{clip}/joint_pos_rmse_rad",
                    (out["terms"]["joint_error_rad"] * selected).sum() / count,
                )
            if "ep_len" in out and out["ep_len"].numel():
                metric("real/episode_length_steps", out["ep_len"].float().mean())
            if "ep_ret" in out and out["ep_ret"].numel():
                metric(
                    "real/episode_return_normalized",
                    (out["ep_ret"] / (mdp.reward_scale * out["ep_len"])).mean(),
                )

    started = attempt_started
    env.impact_ramp.fill_(0.0 if mdp.uses_transition_metrics else 1.0)
    collect(args.warmup_steps)
    print(
        f"READY real_steps={sim_steps} actor={OBS_DIM}x{args.hist}+goal={goal.shape[-1]} "
        f"action=31 state={75 + MEM_DIM} context={CONTEXT_DIM} critic={critic_dim}",
        flush=True,
    )
    (args.output / "interface.json").write_text(
        json.dumps(
            {
                "obs_dim": OBS_DIM,
                "history": args.hist,
                "goal_dim": goal.shape[-1],
                "action_dim": 31,
                "state_dim": 75 + MEM_DIM,
                "memory_dim": MEM_DIM,
                "context_dim": CONTEXT_DIM,
                "critic_dim": critic_dim,
                "physics_dt": 0.005,
                "policy_dt": 0.02,
                "reward": vars(mdp.cfg),
                "actor_p_v_inputs": False,
                "reward_scale": mdp.reward_scale,
                "full_body_collision": args.full_body_collision,
                "self_collision": False,
                "motion_files": motion.source_files,
                "clip_names": motion.clip_names,
                "clip_start": motion.clip_start.tolist(),
                "clip_end": motion.clip_end.tolist(),
                "reset_clip_probabilities": [1 / motion.num_clips] * motion.num_clips,
                "clip_allocation": "env_id modulo num_clips; adaptive phase within assigned clip",
                "imagined_transition_metrics": "two supervised heads, per-cost conservative minimum"
                if transition_model
                else None,
                "reference_cache_fk_difference": motion.source_fk_error,
                "context_fields": getattr(env, "context_names", None),
            },
            indent=2,
            default=str,
        )
    )
    csv_path = args.output / "metrics.csv"
    existing_fields = None
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open() as old_csv:
            existing_fields = next(csv.reader(old_csv))
    log = csv_path.open("a", newline="")
    csv_writer = None
    last_log, last_iteration, step_acc = time.monotonic(), start, 0.0
    last_save = time.monotonic()
    finished_iteration, stop_reason = start, "iteration_limit"
    for it in range(start, args.iterations):
        lr = args.lr * (
            0.03
            + 0.97
            * 0.5
            * (
                1
                + math.cos(math.pi * min(it, args.lr_decay_iters) / args.lr_decay_iters)
            )
        )
        for optimizer in (
            learner.opt_wm,
            learner.opt_pi,
            learner.opt_v,
            learner.opt_transition,
        ):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr
        env.impact_ramp.fill_(min(1.0, (it + 1) / max(1, args.impact_ramp_updates)))
        step_acc += 1 / args.utd
        count = int(step_acc)
        step_acc -= count
        collect(count)
        losses = []
        for _ in range(args.wm_batches):
            loss, err8 = learner.wm_update()
            losses.append(loss)
        metric("model/loss", torch.stack(losses).mean())
        metric("model/error_last_step", err8)
        for key, value in learner.last_model_metrics.items():
            metric("model/" + key, value)
        if it >= args.wm_only_iters:
            for key, value in learner.policy_update().items():
                metric("imagination/" + key, value)
        finished_iteration = it + 1
        budget_stop = (
            args.max_wall_seconds > 0
            and time.monotonic() - attempt_started + wall_offset
            >= args.max_wall_seconds
        )
        stopping = stop_requested or budget_stop
        if it == start or (it + 1) % args.log_every == 0 or it + 1 == args.iterations or stopping:
            now = time.monotonic()
            values = {
                key: float(value / stats_count[key]) for key, value in stats.items()
            }
            values.update(
                {
                    "time/iterations_per_second": (it + 1 - last_iteration)
                    / (now - last_log),
                    "time/wall_hours": (now - started + wall_offset) / 3600,
                    "time/attempt_wall_hours": (now - attempt_started) / 3600,
                    "time/sim_steps": sim_steps,
                    "optimizer/lr": lr,
                    "memory/peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            if csv_writer is None:
                # Actor metrics exist only after world-model warmup, so keep a
                # stable superset of columns from the first row.
                fields = sorted(
                    set(values)
                    | {"real/episode_length_steps", "real/episode_return_normalized"}
                    | {
                        f"clip{c}/{key}"
                        for c in range(motion.num_clips)
                        for key in (
                            "active_fraction",
                            "reward",
                            "terminated_per_step",
                            "reset_frequency",
                            "joint_pos_rmse_rad",
                        )
                    }
                    | {
                        "imagination/" + k
                        for k in (
                            "actor_loss",
                            "critic_loss",
                            "imagined_reward",
                            "actor_grad_norm",
                            "imagined_survival",
                            "action_peak",
                        )
                    }
                )
                csv_writer = csv.DictWriter(
                    log,
                    fieldnames=existing_fields or ["iteration"] + fields,
                    extrasaction="ignore",
                )
                if not existing_fields:
                    csv_writer.writeheader()
            csv_writer.writerow({"iteration": it + 1, **values})
            log.flush()
            for key, value in values.items():
                writer.add_scalar(key, value, it + 1)
            writer.flush()
            print("UPDATE", it + 1, json.dumps(values, sort_keys=True), flush=True)
            stats.clear()
            stats_count.clear()
            last_log, last_iteration = now, it + 1
        timed_save = (
            args.save_every_seconds > 0
            and time.monotonic() - last_save >= args.save_every_seconds
        )
        if (
            it == start
            or (it + 1) % args.save_every == 0
            or it + 1 == args.iterations
            or timed_save
            or stopping
        ):
            ck = checkpoint(
                learner,
                args,
                it + 1,
                sim_steps,
                time.monotonic() - started + wall_offset,
                sampler,
            )
            destination = args.output / f"model_{it + 1:06d}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(ck, temporary)
            os.replace(temporary, destination)
            print("SAVED", destination, flush=True)
            last_save = time.monotonic()
        if stopping:
            stop_reason = "wall_budget" if budget_stop else "signal"
            break
    log.close()
    writer.close()
    env.close()
    (args.output / "completion.json").write_text(
        json.dumps(
            {
                "iteration": finished_iteration,
                "stop_reason": stop_reason,
                "attempt_wall_seconds": time.monotonic() - attempt_started,
                "sim_steps": sim_steps,
            },
            indent=2,
        )
    )
    print(
        "TRAINING_COMPLETE",
        finished_iteration,
        "reason=" + stop_reason,
        f"attempt_wall_seconds={time.monotonic() - attempt_started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
