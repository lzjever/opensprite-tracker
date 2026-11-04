#!/usr/bin/env python3
"""Run real DDS integration and optional actual ROS 2 interoperability checks."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(args):
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "opensprite-sdk-dds-acceptance-v1", "passed": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hardware_accessed": False, "checks": {}}
    try:
        checks = [("dds_integration", [sys.executable, "-m", "pytest",
                    str(ROOT/"tests/test_dds_integration.py"),
                    str(ROOT/"tests/test_runtime_dds_hardware.py"), "-q"])]
        if args.ros2:
            ros_report = args.output.parent / "ros2-interop.json"
            checks.append(("ros2_interoperability", [sys.executable,
                str(ROOT/"tools/check_ros2_interop.py"), "--output", str(ros_report),
                "--image", args.ros_image]))
        for name, command in checks:
            log = args.output.parent / (name + ".log")
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=240)
            report["checks"][name] = {"exit_code": completed.returncode, "log": str(log)}
            if completed.returncode:
                raise RuntimeError(f"{name} failed; see {log}")
            if name == "dds_integration" and "skipped" in log.read_text():
                raise RuntimeError("DDS integration acceptance cannot pass with skipped tests")
            if name == "ros2_interoperability":
                details = json.loads(ros_report.read_text())
                if not details.get("passed"):
                    raise RuntimeError("ROS2 interoperability report did not pass")
        report["passed"] = True
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--ros-image", default="ros:humble-ros-base")
    print(json.dumps(main(parser.parse_args()), indent=2))
