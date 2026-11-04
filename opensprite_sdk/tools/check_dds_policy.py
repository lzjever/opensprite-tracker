#!/usr/bin/env python3
"""Validate the real CPU actor through a separate MuJoCo Fast DDS runtime."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    bundle = args.bundle.resolve()
    contract = json.loads((bundle/"contract.json").read_text())
    frames = contract["frames"]-1
    if args.duration is not None:
        if not math.isfinite(args.duration) or args.duration < .02:
            raise ValueError("duration must include at least one policy step")
        frames = min(frames, int(args.duration*50))
    seconds = frames/50.
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent/(output.stem+"-work-"+uuid.uuid4().hex[:8])
    work.mkdir()
    report = {"schema":"opensprite-dds-policy-acceptance-v1", "passed":False,
        "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_directory":str(work), "expected_commands":frames,
        "reference_seconds":seconds, "video_recorded":False,
        "hardware_accessed":False, "source_sha256":{}}
    for name in ("src/opensprite_sdk/runtime.py", "src/opensprite_sdk/messages.py",
                 "src/opensprite_sdk/core/channel.py", "src/opensprite_sdk/core/dds.py",
                 "examples/robot_runtime.py", "examples/remote_policy.py",
                 "native-dds/generated/manifest.json"):
        report["source_sha256"][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    server = client = None
    try:
        common = ["--transport", "dds", "--domain", str(args.domain),
                  "--interface", args.interface, "--bundle", str(bundle)]
        runtime_command = [sys.executable, str(ROOT/"examples/robot_runtime.py"),
            "--backend", "mujoco", "--profile", contract["profile_id"], *common,
            "--enable-sim", "--duration", str(seconds+30), "--output", str(work/"runtime.json")]
        policy_command = [sys.executable, str(ROOT/"examples/remote_policy.py"), *common,
            "--output", str(work/"policy.json")]
        if args.duration is not None:
            policy_command += ["--duration", str(args.duration)]
        with (work/"runtime.log").open("w") as rlog, (work/"policy.log").open("w") as plog:
            server = subprocess.Popen(runtime_command, cwd=ROOT, stdout=rlog,
                                      stderr=subprocess.STDOUT, env=dict(os.environ))
            client = subprocess.Popen(policy_command, cwd=ROOT, stdout=plog,
                                      stderr=subprocess.STDOUT, env=dict(os.environ))
            report["policy_exit_code"] = client.wait(timeout=seconds+25)
            if report["policy_exit_code"]:
                raise RuntimeError(f"policy failed; see {work/'policy.log'}")
            report["runtime_exit_code"] = server.wait(timeout=5)
            if report["runtime_exit_code"]:
                raise RuntimeError(f"runtime failed; see {work/'runtime.log'}")
        runtime = json.loads((work/"runtime.json").read_text())
        policy = json.loads((work/"policy.json").read_text())
        report.update(runtime=runtime, policy=policy)
        if (runtime["fault"] is not None or runtime["accepted_commands"] != frames
                or runtime["accepted_command_seq"] != frames
                or not seconds-1e-9 <= runtime["simulated_seconds"] <= seconds+.02+1e-9
                or policy["transitions_sent"] != frames
                or policy["last_backend_accepted_command_seq"] != frames
                or not policy["disable_execution_verified"]):
            raise AssertionError("runtime/actor command, timing or shutdown acceptance mismatch")
        report["passed"] = True
        return report
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_error = None
        try:
            for process in (client, server):
                try:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                except ProcessLookupError:
                    # A process may exit between poll and terminate/kill.
                    pass
                except Exception as error:
                    cleanup_error = error
                    report.setdefault("cleanup_errors", []).append(str(error))
                    report["passed"] = False
        finally:
            report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            output.write_text(json.dumps(report, indent=2)+"\n")
        if cleanup_error is not None and "error" not in report:
            raise RuntimeError("policy acceptance process cleanup failed") from cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--domain", type=int, default=42)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--duration", type=float, help="omit to run the complete reference")
    print(json.dumps(check(parser.parse_args()), indent=2))
