#!/usr/bin/env python3
"""Repository-local demo commands. --dry-run prints argv without GPU work or writes."""
from __future__ import annotations

import argparse
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_install import DEFAULT_CHECKPOINT, DEFAULT_MOTION, file_problem, python_path, repo_path


def owned(value):
    path = repo_path(value, root=ROOT)
    if not path.is_relative_to(ROOT) or (path.relative_to(ROOT).parts and path.relative_to(ROOT).parts[0] == ".reference"):
        raise ValueError(f"Use a repository-owned path, not .reference or another checkout: {value}")
    return path


def fresh_output(value, kind):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = owned(value or f"outputs/demo/{kind}-{stamp}-{uuid.uuid4().hex[:6]}")
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Choose a new directory; this command never reuses an old run's budget.")
    return path


def build_plan(args):
    python = python_path(args.python, ROOT/".venv/bin/python")
    umr_override = getattr(args, "umr_python", None)
    # The two environments intentionally have different NumPy/MuJoCo versions;
    # never silently run native UMR in the training interpreter.
    umr = python_path(umr_override, ROOT/".venv-umr/bin/python") if umr_override else str(ROOT/".venv-umr/bin/python")
    plan = {"cwd": str(ROOT), "commands": [], "required_files": []}
    motion = owned(getattr(args, "motion", DEFAULT_MOTION))
    if args.command == "check":
        command = [sys.executable, str(ROOT/"scripts/check_install.py"), "--profile", args.profile,
                   "--python", python, "--umr-python", umr, "--motion", str(motion), "--checkpoint", str(owned(args.checkpoint))]
        if args.json: command.append("--json")
        plan["commands"].append(command)
    elif args.command in ("evaluate", "record"):
        checkpoint = owned(args.checkpoint)
        output = fresh_output(args.output, args.command)
        evaluation = output if args.command == "evaluate" else output/"evaluation"
        plan["required_files"] = [str(checkpoint), str(motion), str(ROOT/"assets/opensprite/opensprite.xml")]
        command = [python, str(ROOT/"scripts/evaluate_opensprite.py"), "--checkpoint", str(checkpoint),
                   "--motion", str(motion), "--clip", "0", "--num-envs", str(args.num_envs),
                   "--seed", str(args.seed), "--device", args.device, "--output", str(evaluation)]
        if args.command == "record":
            command.append("--continue-after-failure")
        elif args.max_steps is not None:
            command += ["--max-steps", str(args.max_steps)]
        plan["commands"].append(command)
        if args.command == "record":
            plan["create_parent"] = str(output)
            plan["commands"].append([python, "-m", "opensprite_tracker.supermimic.render",
                "--rollout", str(evaluation/"rollout.npz"), "--output", str(output/"render"),
                "--model", str(ROOT/"assets/opensprite/opensprite.xml"), "--env-index", "0",
                "--width", "1920", "--height", "1080", "--fps", "50", "--include-post-failure"])
            plan["note"] = "Full real actor/physics recording after first failure, no reset/freeze. Standard success still ends at first failure; a full video is not proof of success."
    elif args.command == "train":
        hard = round(args.hours*3600)
        if hard <= 180: raise ValueError("hours must leave more than 180 seconds for shutdown reserve")
        output = fresh_output(args.output, f"train-dance-{args.preset}")
        template = "dance-8h.json" if args.preset == "goal-sim" else "dance-goal-sim2real-8h.json"
        cfg = json.loads((ROOT/"configs/demo"/template).read_text())
        relative = str(output.relative_to(ROOT))
        replacements = {"{run_dir}": relative, "{python}": python, "{motion}": str(motion.relative_to(ROOT)), "{soft_budget_seconds}": str(hard-180)}
        source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       check=True, capture_output=True, text=True).stdout.strip()
        cfg.update(run_dir=relative, budget_seconds=hard-180, hard_budget_seconds=hard,
                   prior_training_seconds=0, source_commit=source_commit,
                   delivery_dir=relative + "/delivery", evaluation_clips=[0],
                   evaluation_seeds=[42, 43, 44], delivery_video_count=0)
        cfg["command"] = [replacements.get(value, value) for value in cfg["command"]]
        if any(flag in cfg["command"] for flag in ("--resume", "--initialize-from", "--world-model-from", "--resume-budget-record")):
            raise ValueError("Fresh demo training template must not inherit checkpoint state")
        config_path = output/"demo-training.json"
        plan.update(create_parent=str(output), config_path=str(config_path), config=cfg,
                    required_files=[str(motion), str(ROOT/"assets/opensprite/opensprite.xml")])
        plan["commands"].append([python, str(ROOT/"scripts/monitor_training.py"), "--config", str(config_path)])
        plan["note"] = f"NEW fresh {args.hours:g}h ledger, no old budget/state inherited. This does not change configs/training.json or manage any other trainer."
    else:
        stage = "convert" if args.command in ("build-motion", "buildmotion") else args.stage
        command = [umr, str(ROOT/"scripts/retarget_lafan.py"), "--stage", stage,
                   "--motions", *args.motions, "--output-dir", str(owned(args.output_dir)),
                   "--cache-dir", str(owned(args.cache_dir)), "--motion-output-dir", str(owned(args.motion_output_dir))]
        if args.command == "retarget":
            command += ["--max-frames", str(args.max_frames), "--correspondence-device", args.correspondence_device]
            command += ["--correspondence-dir", str(owned(args.correspondence_dir))]
        plan["commands"].append(command)
        plan["required_files"] = [str(ROOT/"data/sources/lafan1-resolved"/f"{name}.bvh") for name in args.motions]
        plan["required_files"] += [str(ROOT/"data/sources/lafan1-resolved/Geno_stance.bvh"), str(ROOT/"assets/geno/Geno.fbx")]
        if stage == "retarget":
            plan["required_files"].append(str(owned(args.correspondence_dir)/"training/correspondence_slots_final.npz"))
        if stage == "convert":
            plan["required_files"] += [str(owned(args.output_dir)/name/"native.npz") for name in args.motions]
        plan["note"] = "Native Geno/OpenSprite UMR; retarget uses CPU. Compatible bundled correspondence is reused. For fresh correspondence use retarget --stage all --correspondence-dir outputs/rebuild-dance/new-correspondence. Correspondence training defaults to CPU; CUDA requires an explicit flag."
    return plan


def execute(plan, command):
    if command == "train":
        from scripts.monitor_training import validate_source
        validate_source(plan["config"], ROOT)
    for value in plan["required_files"]:
        problem = file_problem(value)
        if problem: raise ValueError(problem)
    for argv in plan["commands"]:
        executable = argv[0]
        if not (Path(executable).is_file() or shutil.which(executable)):
            raise ValueError(f"Interpreter missing: {executable}. Install the appropriate requirements into .venv or .venv-umr first.")
    if command == "record":
        if not shutil.which("ffmpeg") or not ctypes.util.find_library("EGL"):
            raise ValueError("Recording requires ffmpeg and an EGL runtime. Run scripts/demo.py check first.")
    if command == "retarget" and not shutil.which("node"):
        raise ValueError("Native UMR requires Node.js >=18 and the bundled Three.js parser. Run check --profile retarget.")
    if plan.get("create_parent"):
        Path(plan["create_parent"]).mkdir(parents=True, exist_ok=False)
    if plan.get("config_path"):
        with Path(plan["config_path"]).open("x") as stream:
            json.dump(plan["config"], stream, indent=2)
            stream.write("\n")
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
                   "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    for argv in plan["commands"]:
        print(json.dumps(argv), flush=True)
        subprocess.run(argv, cwd=ROOT, env=environment, check=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "evaluate", "record", "train", "retarget", "build-motion"):
        s = sub.add_parser(name, aliases=["buildmotion"] if name == "build-motion" else [])
        s.add_argument("--python", help="training/render Python; default .venv/bin/python")
        s.add_argument("--dry-run", action="store_true")
        if name in ("check", "retarget", "build-motion"):
            s.add_argument("--umr-python", help="default .venv-umr/bin/python")
        if name in ("check", "evaluate", "record", "train"):
            s.add_argument("--motion", default=str(DEFAULT_MOTION))
        if name in ("check", "evaluate", "record"):
            s.add_argument("--checkpoint", default="artifacts/checkpoints/model_018000.pt" if name == "record" else str(DEFAULT_CHECKPOINT))
        if name == "check":
            s.add_argument("--profile", choices=("demo", "retarget", "all"), default="demo")
            s.add_argument("--json", action="store_true")
        elif name in ("evaluate", "record"):
            s.add_argument("--output")
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--num-envs", type=int, default=1 if name == "record" else 3)
            s.add_argument("--device", default="cuda:0")
            if name == "evaluate": s.add_argument("--max-steps", type=int, help="incomplete diagnostic only")
        elif name == "train":
            s.add_argument("--output")
            s.add_argument("--hours", type=float, default=8.)
            s.add_argument("--preset", choices=("goal-sim", "goal-sim2real"), default="goal-sim")
        else:
            s.add_argument("--motions", nargs="+", choices=("dance2_subject1",), default=["dance2_subject1"])
            s.add_argument("--output-dir", default="outputs/rebuild-dance/native")
            s.add_argument("--cache-dir", default="outputs/rebuild-dance/cache")
            s.add_argument("--motion-output-dir", default="outputs/rebuild-dance/motions")
            if name == "retarget":
                s.add_argument("--stage", choices=("retarget", "build", "train", "all"), default="retarget")
                s.add_argument("--max-frames", type=int, default=0)
                s.add_argument("--correspondence-device", choices=("cpu", "cuda"), default="cpu")
                s.add_argument("--correspondence-dir", default="assets/umr_correspondence/geno_opensprite0825")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan, args.command)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        p.exit(1, f"demo: {error}\n")


if __name__ == "__main__":
    main()
