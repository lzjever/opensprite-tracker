#!/usr/bin/env python3
"""Run immediate, then hourly, isolated Codex audits without a system scheduler.

The launcher never signals a trainer. Only its own audit subprocess group may
be terminated on shutdown/timeout. Training corrections belong to the explicitly
authorized audit task, which must preserve the existing campaign budget.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path("outputs/delivery/opensprite_lafan3_tracker")


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, content):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves repository: {value}")
    return path


def trainer_alive(root, state):
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = Path("/proc") / str(pid)
        argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return argv == state.get("command") and proc.joinpath("cwd").resolve() == root
    except (OSError, ValueError):
        return False


def snapshot(root, config, run):
    supervisor = read_json(run / "supervisor.json")
    prior = float(config.get("prior_training_seconds", 0))
    cap = float(config["budget_seconds"])
    if not (math.isfinite(prior) and math.isfinite(cap) and 0 <= prior <= cap <= 43200):
        raise ValueError("Invalid training ledger: require 0 <= prior <= cap <= 43200")
    # active_seconds ALREADY includes prior; checkpoint runtime excludes prior.
    campaign = float(supervisor.get("active_seconds", prior))
    if not math.isfinite(campaign) or campaign < 0:
        raise ValueError("Invalid supervisor active_seconds")
    recent = []
    metrics = run / "metrics.csv"
    if metrics.is_file():
        import csv

        with metrics.open() as stream:
            recent = list(deque(csv.DictReader(stream), maxlen=12))
    checkpoints = [p for p in run.glob("model_*.pt") if p.stem[6:].isdigit()]
    checkpoints.sort(key=lambda p: int(p.stem[6:]))
    return {
        "captured_at": utc(),
        "root": str(root),
        "run_dir": str(run),
        "source_branch": config.get("source_branch"),
        "supervisor": supervisor,
        "verified_trainer_alive": trainer_alive(root, supervisor),
        "campaign_seconds": campaign,
        "prior_training_seconds": prior,
        "formal_elapsed_seconds": max(0.0, campaign - prior),
        "supervisor_soft_campaign_budget_seconds": cap,
        "hard_total_training_budget_seconds": 43200,
        "remaining_to_supervisor_soft_limit_seconds": max(0.0, cap - campaign),
        "accounting": "campaign=supervisor.active_seconds ALREADY includes prior; checkpoint.runtime_seconds excludes prior; resume floor=campaign-prior. Never reset or enlarge the budget.",
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "recent_logged_metrics": recent,
    }


def delivery_complete(root, run):
    marker = run / "hourly/delivery-complete.json"
    if not marker.is_file():
        return False
    value = read_json(marker)
    if value.get("status") != "complete" or type(value.get("quality_pass")) is not bool:
        raise ValueError("Delivery needs status=complete and honest boolean quality_pass")
    if not value.get("quality_notes") or not value.get("completed_at"):
        raise ValueError("Delivery needs quality evidence/limitations and completed_at")
    delivery = root / DELIVERY
    if local_path(root, value.get("delivery_dir", "")) != delivery:
        raise ValueError("Unexpected delivery directory")

    def artifact(name):
        # Check lexical ancestors before resolve: resolving first hides symlinks.
        lexical = Path(os.path.abspath(root / name))
        if not lexical.is_relative_to(delivery):
            raise ValueError(f"Artifact outside delivery: {name}")
        if any(p.is_symlink() for p in (lexical, *lexical.parents) if p != root):
            raise ValueError(f"Delivery cannot depend on symlinks: {name}")
        path = local_path(root, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty delivery artifact: {name}")
        return path

    artifact(value["checkpoint"])
    artifact(value["readme"])
    reports, videos = value.get("evaluation_reports", []), value.get("videos", [])
    if len(reports) != 9 or len(set(reports)) != 9 or len(videos) != 3 or len(set(videos)) != 3:
        raise ValueError("Delivery needs nine distinct reports and three distinct videos")
    pairs, shared = set(), set()
    for name in reports:
        result = read_json(artifact(name))
        pairs.add((result.get("clip"), result.get("seed")))
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("Summary must identify its evaluated checkpoint")
        shared.add(str(local_path(root, checkpoint)))
        expected = result.get("expected_transitions", 0)
        if expected <= 0 or result.get("requested_transitions") != expected:
            raise ValueError("Diagnostic prefix cannot count as final full-motion request")
        if result.get("num_envs") != 3:
            raise ValueError("Final evaluations must each use three environments")
    if len(shared) != 1:
        raise ValueError("All nine summaries must evaluate the same shared checkpoint")
    if pairs != {(clip, seed) for clip in range(3) for seed in (42, 43, 44)}:
        raise ValueError("Final reports must cover clip 0/1/2 x seed 42/43/44")
    for name in videos:
        if artifact(name).suffix.lower() != ".mp4":
            raise ValueError("Expected three ghost MP4 videos")
    supervisor = read_json(run / "supervisor.json")
    if trainer_alive(root, supervisor) or supervisor.get("status") in ("running", "starting"):
        raise ValueError("Cannot finish auditing while training is still running")
    return True


class Launcher:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.config_path = local_path(self.root, args.config)
        self.config = read_json(self.config_path)
        self.run = local_path(self.root, self.config["run_dir"])
        self.folder = self.run / "hourly"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()
        self.child = None

    def signal_child(self, sig):
        if self.child is not None:
            try:
                os.killpg(self.child.pid, sig)
            except ProcessLookupError:
                pass

    def request_stop(self, *_):
        self.stopping.set()
        self.signal_child(signal.SIGTERM)

    def save(self, state, attempt=None):
        state.update(updated_at=utc(), runner_pid=os.getpid())
        content = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        atomic_write(self.folder / "runner-state.json", content)
        if attempt is not None:
            atomic_write(attempt, content)

    def delivered(self):
        return delivery_complete(self.root, self.run)

    def audit(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        report = self.folder / f"{stamp}.md"
        attempt = self.folder / f"{stamp}.state.json"
        state = {"status": "starting", "started_at": utc(), "report": str(report)}
        started = time.monotonic()
        try:
            # Re-read mutable orchestration configuration for authorized repairs.
            config = read_json(self.config_path)
            if local_path(self.root, config["run_dir"]) != self.run:
                raise ValueError("Campaign run_dir changed; refusing to monitor another run")
            context = snapshot(self.root, config, self.run)
            try:
                if self.delivered():
                    state["status"] = "delivery_complete"
                    self.save(state, attempt)
                    return True, True
            except (ValueError, KeyError, TypeError, OSError) as error:
                context["delivery_marker_error"] = str(error)
            task = local_path(self.root, self.args.task).read_text()
            prompt = task + "\n\n本次只读启动快照；必须重新核实动态状态：\n```json\n"
            prompt += json.dumps(context, ensure_ascii=False, indent=2) + "\n```\n"
            prompt += f"\nUTC审计标识：{stamp}。上一报告：{self.folder / 'latest.md'}。本次最终分析将由CLI保存；必须实际分析逐clip趋势、WM有效样本和GPU/NaN，并对照上次报告。健康时不改训练源码/配置、不重复smoke。总GPU训练上限43200秒，active_seconds已含prior，不能再次相加。任务结束后独立训练继续，不要停止健康trainer。\n"
            prompt_path = self.folder / f"{stamp}.prompt.md"
            atomic_write(prompt_path, prompt)
            atomic_write(self.folder / f"{stamp}.snapshot.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
            requested = self.args.codex or os.environ.get("CODEX_BIN") or config.get("codex_binary") or "codex"
            executable = shutil.which(str(requested))
            if not executable:
                raise FileNotFoundError(f"Codex executable not found: {requested}")
            # Keep nvm's bin path, not the resolved JS symlink's parent, for node.
            executable = str(Path(executable).absolute())
            last_message = self.folder / f"{stamp}.last-message.md"
            command = [executable, "-a", "never", "-c", 'approval_policy="never"',
                       "-c", 'sandbox_mode="danger-full-access"', "exec", "--ephemeral",
                       "-C", str(self.root), "--sandbox", "danger-full-access",
                       "--output-last-message", str(last_message), "-"]
            environment = dict(os.environ)
            environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")
            stdout_path, stderr_path = self.folder / f"{stamp}.stdout.log", self.folder / f"{stamp}.stderr.log"
            state.update(status="running", command=command, prompt=str(prompt_path),
                         stdout=str(stdout_path), stderr=str(stderr_path), last_message=str(last_message))
            self.save(state, attempt)
            timed_out = False
            with prompt_path.open("rb") as stdin, stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                self.child = subprocess.Popen(command, cwd=self.root, env=environment,
                                              stdin=stdin, stdout=stdout, stderr=stderr,
                                              start_new_session=True)
                state["agent_pid"] = self.child.pid
                self.save(state, attempt)
                deadline = time.monotonic() + self.args.timeout_seconds
                while self.child.poll() is None:
                    if self.stopping.is_set() or time.monotonic() >= deadline:
                        timed_out = not self.stopping.is_set()
                        self.signal_child(signal.SIGTERM)
                        try:
                            self.child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            self.signal_child(signal.SIGKILL)
                            self.child.wait()
                        # Also clean audit helpers remaining in our own group.
                        self.signal_child(signal.SIGKILL)
                        break
                    try:
                        self.child.wait(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            state["agent_returncode"] = self.child.returncode
            if self.stopping.is_set():
                state["status"] = "interrupted"
                return False, False
            if timed_out:
                state["status"] = "timeout"
                state["error"] = f"Audit exceeded {self.args.timeout_seconds} seconds"
                return False, False
            if self.child.returncode:
                raise RuntimeError(f"Codex audit exited {self.child.returncode}; inspect {stderr_path}")
            if not last_message.is_file() or not last_message.read_text().strip():
                raise RuntimeError("Codex returned without a nonempty final analysis")
            atomic_write(report, last_message.read_text())
            atomic_write(self.folder / "latest.md", report.read_text())
            delivered = self.delivered()
            state.update(status="delivery_complete" if delivered else "completed", finished_at=utc())
            return True, delivered
        except Exception as error:
            state.update(status="error", error=f"{type(error).__name__}: {error}")
            print(state["error"], flush=True)
            return False, False
        finally:
            # An unexpected launcher error after Popen must not leave an audit
            # agent running after the lifetime lock is released.
            if self.child is not None and self.child.poll() is None:
                self.signal_child(signal.SIGTERM)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.signal_child(signal.SIGKILL)
                    self.child.wait()
            self.child = None
            state["elapsed_seconds"] = time.monotonic() - started
            self.save(state, attempt)

    def run_forever(self):
        # Held for the entire launcher lifetime, including the hourly wait.
        with (self.folder / "audit.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another audit launcher owns this campaign; exiting without changes", flush=True)
                return 0
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            while not self.stopping.is_set():
                started = time.monotonic()
                success, delivered = self.audit()
                if delivered:
                    return 0
                if self.args.once:
                    return 0 if success else 1
                if self.stopping.is_set():
                    break
                # Skip missed ticks after an unusually long audit, never overlap
                # or launch a burst of catch-up agents. Failures wait too.
                elapsed = time.monotonic() - started
                slots = max(1, math.floor(elapsed / self.args.interval_seconds) + 1)
                delay = started + slots * self.args.interval_seconds - time.monotonic()
                state = read_json(self.folder / "runner-state.json")
                state["next_run_at"] = datetime.fromtimestamp(time.time() + max(0, delay), timezone.utc).isoformat()
                state["launcher_status"] = "waiting"
                self.save(state)
                self.stopping.wait(max(0, delay))
            state = read_json(self.folder / "runner-state.json")
            state.update(launcher_status="stopped", stopped_at=utc())
            state.pop("next_run_at", None)
            self.save(state)
            return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--task", default="docs/plan/hourly-audit-task.md")
    parser.add_argument("--codex", help="Codex executable, else CODEX_BIN/config/PATH")
    parser.add_argument("--timeout-seconds", type=float, default=3300)
    parser.add_argument("--interval-seconds", type=float, default=3600)
    parser.add_argument("--once", action="store_true", help="One immediate audit, for explicit validation")
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout_seconds, args.interval_seconds)):
        parser.error("timeout and interval must be finite and positive")
    return Launcher(args).run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
