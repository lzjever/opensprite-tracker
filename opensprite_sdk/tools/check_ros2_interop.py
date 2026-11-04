#!/usr/bin/env python3
"""Build canonical ROS messages and prove bidirectional native DDS/ROS 2 I/O.

Requires Docker and a ROS image containing colcon, rclpy and Fast DDS RMW.
The host Python environment needs the installed SDK and its native DDS library.
No hardware devices, simulator truth or video are used.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from opensprite_sdk import messages
from opensprite_sdk.core.dds import DdsEndpoint


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("LowCmd", "LowState", "ControlRequest", "ControlReply")


def fixtures():
    result = {}
    for name in NAMES:
        message = getattr(messages, name + "_")()
        message.profile_id = "legacy_v4"
        message.seq = 2**53 + 7
        message.stamp_ns = 1234567890123456
        message.session_id = "interop-" + name
        message.clock_domain = "sim"
        if name == "LowCmd":
            message.source_state_seq = 2**53 + 3
            message.source_state_session_id = "interop-robot"
            for index, joint in enumerate(message.joint_cmd):
                joint.mode = 1
                joint.q, joint.dq = index * .25, -.5
                joint.kp, joint.kd, joint.tau = 2., .125, .375
        elif name == "LowState":
            for index, joint in enumerate(message.joint_state):
                joint.q, joint.dq, joint.tau_est = index * .25, -.5, .375
                joint.valid = joint.tau_est_valid = True
                joint.sample_ns = message.stamp_ns - 1000 - index
                joint.sample_time_kind = "simulation"
            message.imu_state.valid = True
            message.imu_state.gyroscope = (.25, -.5, .75)
            message.imu_state.accelerometer = (1., 2., 3.)
            message.imu_state.sample_ns = message.stamp_ns - 2500
            message.imu_state.sample_time_kind = "simulation"
            message.status = "enabled"
            message.faults = ["interop diagnostic", "中文诊断"]
            message.accepted_command_seq = 2**53 + 5
            message.accepted_controller_session = "interop-controller"
        else:
            message.request_id = "interop-request"
            message.controller_session = "interop-controller"
            message.robot_session = "interop-robot"
            if name == "ControlRequest":
                message.operation = "claim"
            else:
                message.accepted = True
                message.status = "claimed"
                message.detail = "接口测试 / interface test"
                message.execution_verified = False
        result[name] = messages.message_to_dict(message)
    # Canonical JSON representation removes tuple/list differences.
    return json.loads(json.dumps(result))


def docker_arguments(args, work, *, network="none"):
    return ["docker", "run", "--rm", "--network", network,
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-v", f"{work}:/ws", "-v", f"{ROOT}/ros2/opensprite_msgs:/ws/src/opensprite_msgs:ro",
            "-v", f"{ROOT}/tools/ros2:/scripts:ro", "-w", "/ws"]


def check(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / ("ros2-workspace-" + uuid.uuid4().hex[:8])
    (work / "src").mkdir(parents=True)
    expected = fixtures()
    (work / "fixtures.json").write_text(json.dumps(expected, indent=2) + "\n")
    report = {"schema": "opensprite-sdk-ros2-interop-v1", "passed": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "image": args.image, "domain": args.domain, "work_directory": str(work),
              "native_received": {}, "matched_peak": {}, "hardware_accessed": False}
    instance = uuid.uuid4().hex[:12]
    containers = ("opensprite-sdk-msgs-" + instance, "opensprite-sdk-peer-" + instance)
    endpoints, peer, cpp_peer = [], None, None
    try:
        image = subprocess.run(["docker", "image", "inspect", args.image,
            "--format", "{{.Id}}"], check=True, capture_output=True, text=True)
        report["image_id"] = image.stdout.strip()
        package_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/"ros2/opensprite_msgs").rglob("*")) if p.is_file()}
        report["canonical_package_sha256"] = package_hashes
        build = docker_arguments(args, work) + ["--name", containers[0], args.image, "bash", "-lc",
            "source /opt/ros/$ROS_DISTRO/setup.bash && "
            "colcon build --packages-select opensprite_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release"]
        with (work / "build.log").open("w") as log:
            subprocess.run(build, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        prefix = "opensprite_acceptance_" + uuid.uuid4().hex[:10]
        writers, readers = {}, {}
        if getattr(args, "cpp_peer", None):
            if args.interface not in ("lo", "localhost", "127.0.0.1"):
                raise ValueError("installed C++ acceptance peer uses loopback")
            report["native_mode"] = "installed_typed_cpp_roundtrip"
            report["cpp_peer_binary"] = str(args.cpp_peer.resolve())
        else:
            report["native_mode"] = "python_ctypes_native_dds"
            for name in NAMES:
                writers[name] = DdsEndpoint(args.domain, args.interface,
                    f"rt/{prefix}/native/{name}", name + "_", "publisher")
                endpoints.append(writers[name])
                readers[name] = DdsEndpoint(args.domain, args.interface,
                    f"rt/{prefix}/ros/{name}", name + "_", "subscriber")
                endpoints.append(readers[name])
        peer_command = docker_arguments(args, work, network="host") + [
            "--name", containers[1],
            "-e", f"ROS_DOMAIN_ID={args.domain}", "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
            "-e", f"ROS_LOCALHOST_ONLY={1 if args.interface in ('lo', 'localhost', '127.0.0.1') else 0}",
            args.image, "bash", "-lc",
            "source /opt/ros/$ROS_DISTRO/setup.bash && source /ws/install/setup.bash && "
            f"python3 /scripts/interop_peer.py --fixtures /ws/fixtures.json "
            f"--output /ws/peer-result.json --prefix {prefix} --duration 15"]
        with (work / "peer.log").open("w") as log:
            peer = subprocess.Popen(peer_command, stdout=log, stderr=subprocess.STDOUT)
            if getattr(args, "cpp_peer", None):
                # This separately compiled executable imports only the public
                # installed C++ SDK. Its data path does not use this script's
                # Python DDS adapter or a JSON ABI.
                cpp_peer = subprocess.Popen([str(args.cpp_peer.resolve()), str(args.domain),
                    prefix, "ros-echo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                for name in writers:
                    peak = report["matched_peak"].setdefault(name, {"writer": 0, "reader": 0})
                    peak["writer"] = max(peak["writer"], writers[name].matches())
                    peak["reader"] = max(peak["reader"], readers[name].matches())
                    writers[name].write(expected[name], timeout=.1)
                    incoming = readers[name].take(timeout=0.)
                    if incoming is not None:
                        report["native_received"][name] = incoming
                        if incoming != expected[name]:
                            raise AssertionError(f"ROS2→native field mismatch: {name}")
                if cpp_peer is not None and cpp_peer.poll() not in (None, 0):
                    raise RuntimeError(f"typed C++ peer failed: {cpp_peer.stderr.read()}")
                if peer.poll() is not None:
                    break
                time.sleep(.025)
            peer.wait(timeout=3)
        report["peer_exit_code"] = peer.returncode
        report["ros2_received"] = json.loads((work / "peer-result.json").read_text())
        report["matched_at_end"] = {name: {"writer": writers[name].matches(),
                                    "reader": readers[name].matches()} for name in writers}
        if cpp_peer is not None:
            stdout, stderr = cpp_peer.communicate(timeout=3)
            report["cpp_peer_exit_code"] = cpp_peer.returncode
            report["cpp_peer_stdout"] = stdout
            report["cpp_peer_stderr"] = stderr
            lines = [line for line in stdout.splitlines() if line.startswith("{")]
            result = json.loads(lines[-1]) if lines else {}
            native_passed = cpp_peer.returncode == 0 and result.get("passed") is True and result.get("message_types") == 4
        else:
            native_passed = set(report["native_received"]) == set(NAMES)
        report["passed"] = (peer.returncode == 0 and report["ros2_received"]["passed"]
            and native_passed)
        if not report["passed"]:
            raise AssertionError("bidirectional ROS2/native DDS acceptance failed")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if cpp_peer is not None:
            if cpp_peer.poll() is None:
                try:
                    cpp_peer.terminate()
                    cpp_peer.wait(timeout=3)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    cpp_peer.kill()
                    cpp_peer.wait(timeout=3)
            for stream in (cpp_peer.stdout, cpp_peer.stderr):
                if stream is not None:
                    stream.close()
        if peer is not None and peer.poll() is None:
            peer.terminate()
            try:
                peer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                peer.kill()
                peer.wait(timeout=3)
        for endpoint in reversed(endpoints):
            endpoint.close()
        # Terminating the docker client alone does not guarantee container exit.
        # Remove only these uniquely named test containers on every error path.
        for name in containers:
            try:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="ros:humble-ros-base")
    parser.add_argument("--domain", type=int, default=173)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--cpp-peer", type=Path,
        help="independently compiled tests/native_sdk/typed_peer executable")
    args = parser.parse_args()
    if not 0 <= args.domain <= 232:
        parser.error("domain must be in 0..232")
    result = check(args)
    print(json.dumps({"passed": result["passed"], "report": str(args.output.resolve()),
                      "native_types": (list(NAMES) if args.cpp_peer else list(result["native_received"])),
                      "ros2_counts": result["ros2_received"]["counts"],
                      "matched_peak": result["matched_peak"]}, indent=2))
