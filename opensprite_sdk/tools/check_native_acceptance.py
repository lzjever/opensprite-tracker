#!/usr/bin/env python3
"""Build and independently validate native SDK/runtime without real devices."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(command, work, name, *, env=None, timeout=180):
    log = work/(name+".log")
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"{name} failed ({completed.returncode}); see {log}")
    return log


def check(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent/("native-acceptance-"+uuid.uuid4().hex[:10])
    work.mkdir()
    report = {"schema":"opensprite-native-acceptance-v1", "passed":False,
        "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_directory":str(work), "architecture":platform.machine(),
        "hardware_accessed":False, "video_recorded":False,
        "ros2_requested":args.ros2, "build_skipped":args.skip_build}
    try:
        environment = dict(os.environ, OPENSPRITE_PYTHON=sys.executable)
        if not args.skip_build:
            run(["bash", str(ROOT/"tools/build_dds.sh")], work, "build-dds", env=environment)
            run(["bash", str(ROOT/"tools/build_robot.sh")], work, "build-robot", env=environment)
            run(["bash", str(ROOT/"tools/build_robot.sh"), "--hardware-only"],
                work, "build-hardware-robot", env=environment)
        robot = ROOT/"native-runtime/build/opensprite_robot"
        library = ROOT/"native-dds/build/libopensprite_dds.so"
        if not robot.is_file() or not library.is_file():
            raise RuntimeError("native robot or DDS library is absent; build is required")
        linkage = subprocess.run(["ldd", str(robot)], check=True, capture_output=True, text=True).stdout
        if "not found" in linkage or any(name in linkage.lower() for name in ("libpython", "onnxruntime")):
            raise AssertionError("native robot dependencies are missing or include Python/ONNX")
        report["native_robot_sha256"] = hashlib.sha256(robot.read_bytes()).hexdigest()
        report["native_robot_linkage"] = linkage
        hardware_robot = ROOT/"native-runtime/build-hardware/opensprite_robot"
        if not hardware_robot.is_file():
            raise RuntimeError("hardware-only native robot is absent; run tools/build_robot.sh --hardware-only")
        hardware_linkage = subprocess.run(["ldd", str(hardware_robot)], check=True,
            capture_output=True, text=True).stdout
        if "not found" in hardware_linkage or any(name in hardware_linkage.lower()
                for name in ("libpython", "onnxruntime", "libmujoco")):
            raise AssertionError("hardware robot unexpectedly depends on Python/ONNX/MuJoCo")
        report["hardware_robot_linkage"] = hardware_linkage
        report["hardware_robot_sha256"] = hashlib.sha256(hardware_robot.read_bytes()).hexdigest()
        tests = run([sys.executable, "-m", "pytest", str(ROOT/"tests/test_native_sdk.py"),
                     str(ROOT/"tests/test_native_robot.py"), str(ROOT/"tests/test_native_history.py"),
                     str(ROOT/"tests/test_native_build.py"), "-q"],
                    work, "native-tests", env=environment)
        text = tests.read_text()
        if re.search(r"\b\d+ skipped\b", text):
            raise AssertionError("native acceptance cannot pass with skipped tests")
        count = re.search(r"\b(\d+) passed\b", text)
        if not count:
            raise AssertionError("native test suite did not report successful tests")
        report["native_tests_passed"] = int(count.group(1))
        prefix = work/"installed"
        run(["cmake", "--install", str(ROOT/"native-dds/build"), "--prefix", str(prefix)],
            work, "install-sdk")
        consumer = work/"consumer"
        shutil.copytree(ROOT/"tests/native_sdk", consumer)
        consumer_build = work/"consumer-build"
        run(["cmake", "-S", str(consumer), "-B", str(consumer_build),
             f"-DCMAKE_PREFIX_PATH={prefix}", "-DCMAKE_BUILD_TYPE=Release"], work, "consumer-configure")
        run(["cmake", "--build", str(consumer_build), "--parallel", "2"], work, "consumer-build")
        peer = consumer_build/"typed_peer"
        report["installed_cpp_peer"] = str(peer)
        report["cpp_peer_sha256"] = hashlib.sha256(peer.read_bytes()).hexdigest()
        report["sdk_install_prefix"] = str(prefix)
        if args.ros2:
            ros_report = work/"cpp-ros2.json"
            run([sys.executable, str(ROOT/"tools/check_ros2_interop.py"), "--cpp-peer", str(peer),
                 "--domain", "174", "--output", str(ros_report), "--image", args.ros_image],
                work, "cpp-ros2", env=environment)
            report["cpp_ros2"] = json.loads(ros_report.read_text())
            if not report["cpp_ros2"].get("passed"):
                raise AssertionError("installed typed C++ SDK/ROS 2 interoperability failed")
        report["passed"] = True
        return report
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--ros-image", default="ros:humble-ros-base")
    args = parser.parse_args()
    result = check(args)
    print(json.dumps({"passed":result["passed"], "native_tests_passed":result["native_tests_passed"],
        "ros2_requested":args.ros2, "report":str(args.output.resolve())}, indent=2))
