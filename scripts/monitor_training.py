#!/usr/bin/env python3
"""Supervise one authorized training run with a persistent total-time budget.

The JSON configuration contains command (an argv list), run_dir, budget_seconds,
and optionally prior_training_seconds, review_seconds and max_restarts. Run this
script in tmux. It only signals its own child process group. Source corrections
require an explicit new committed run; automatic retries preserve configuration.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def latest_checkpoint(run):
    candidates = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return candidates[-1] if candidates else None


def with_resume(command, checkpoint, elapsed):
    command = list(command)
    for key in ("--resume", "--resume-runtime-seconds-floor"):
        while key in command:
            index = command.index(key)
            del command[index:index+2]
    return command + ["--resume", str(checkpoint), "--resume-runtime-seconds-floor", str(elapsed)]


def previous_child_alive(previous):
    """Identify an orphan by its original argv and cwd, avoiding PID reuse."""
    pid = previous.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        proc = Path("/proc") / str(pid)
        argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return argv == previous.get("command") and proc.joinpath("cwd").resolve() == ROOT
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not isinstance(cfg["command"], list) or not all(isinstance(x, str) for x in cfg["command"]):
        raise ValueError("command must be a list of argv strings")
    run = (ROOT / cfg["run_dir"]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another supervisor already owns this run directory")
    if float(cfg["budget_seconds"]) <= 0 or float(cfg.get("review_seconds", 3600)) <= 0:
        raise ValueError("Budget and review interval must be positive")
    state_path = run / "supervisor.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous_child_alive(previous):
        raise SystemExit(f"Previous owned trainer PID {previous['pid']} is still alive; "
                         "inspect its log and stop it safely before restarting supervision")
    prior_training = float(cfg.get("prior_training_seconds", 0))
    if previous and float(previous.get("prior_training_seconds", 0)) != prior_training:
        raise ValueError("Resume must preserve the campaign's prior training charge")
    elapsed = max(float(previous.get("active_seconds", 0)), prior_training)
    restarts = int(previous.get("restarts", 0))
    stopping = False
    child = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = list(cfg["command"])
    checkpoint = latest_checkpoint(run)
    if checkpoint:
        command = with_resume(command, checkpoint, max(0., elapsed - prior_training))
    review_interval = float(cfg.get("review_seconds", 3600))
    next_review = (int(elapsed // review_interval) + 1) * review_interval
    state = {"status": "starting", "active_seconds": elapsed, "restarts": restarts,
             "prior_training_seconds": prior_training}

    def review():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = run / "reviews"
        folder.mkdir(exist_ok=True)
        output = folder / f"{stamp}.json"
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/training_snapshot.py"),
                                    "--run-dir", str(run), "--output", str(output)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=90)
        state["last_review"] = str(output)
        state["last_review_returncode"] = completed.returncode
        print(f"[REVIEW] {stamp} active_hours={state['active_seconds']/3600:.4f} "
              f"snapshot={output} returncode={completed.returncode}", flush=True)
        if completed.returncode:
            (folder / f"{stamp}.error.txt").write_text(completed.stderr)

    with (run / "supervisor.log").open("a", buffering=1) as supervisor_log:
        def announce(message):
            print(message, flush=True)
            supervisor_log.write(message + "\n")

        while not stopping and elapsed < float(cfg["budget_seconds"]):
            segment = restarts
            logfile = run / f"train-segment-{segment:02d}.log"
            started = time.monotonic()
            before = elapsed
            announce(f"[LAUNCH] segment={segment} prior_hours={elapsed/3600:.6f} command={json.dumps(command)}")
            with logfile.open("a", buffering=1) as training_log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=training_log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
                state.update(status="running", pid=child.pid, command=command, log=str(logfile))
                while child.poll() is None:
                    elapsed = before + time.monotonic() - started
                    state.update(active_seconds=elapsed, restarts=restarts,
                                 utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(state_path, state)
                    if elapsed >= next_review:
                        try:
                            review()
                        except subprocess.TimeoutExpired:
                            announce("[REVIEW] telemetry timed out; training continues")
                        next_review += review_interval
                    if elapsed >= float(cfg["budget_seconds"]) or stopping:
                        announce("[STOP] supervisor budget or stop request; requesting checkpoint at outer boundary")
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=180)
                        except subprocess.TimeoutExpired:
                            announce("[STOP] child did not stop within 180s; terminating owned process group")
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    time.sleep(5)
                elapsed = before + time.monotonic() - started
            state.update(active_seconds=elapsed, returncode=child.returncode)
            if child.returncode == 0 or stopping or elapsed >= float(cfg["budget_seconds"]):
                state["status"] = "stopped" if stopping else "finished"
                break
            checkpoint = latest_checkpoint(run)
            if restarts >= int(cfg.get("max_restarts", 1)) or not checkpoint:
                state["status"] = "needs_review"
                announce(f"[FAIL] rc={child.returncode}; checkpoint={checkpoint}; review required")
                break
            restarts += 1
            state["restarts"] = restarts
            command = with_resume(cfg["command"], checkpoint, max(0., elapsed - prior_training))
            announce(f"[RECOVER] rc={child.returncode}; resuming {checkpoint}; runtime preserved")
        state.update(active_seconds=elapsed, utc=datetime.now(timezone.utc).isoformat())
        if state["status"] == "starting":
            state["status"] = "budget_exhausted"
        atomic_json(state_path, state)
        try:
            review()
        except subprocess.TimeoutExpired:
            announce("[REVIEW] final telemetry timed out")
        atomic_json(state_path, state)
        announce(f"[DONE] status={state['status']} total_hours={elapsed/3600:.6f}")
    return 1 if state["status"] == "needs_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
