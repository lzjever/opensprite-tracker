#!/usr/bin/env python3
"""Build, relocate, install and exercise only this SDK's distributable files.

Uses installed third-party dependencies but removes the source checkout from
Python's import path. No hardware interface is opened. All generated files stay
under outputs/extraction and the requested JSON report.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".reference", ".venv", "outputs", "build", "dist", "__pycache__", ".pytest_cache", ".git", ".cache"}


def excluded(path):
    return (any(part in EXCLUDED or part.endswith(".egg-info") for part in path.parts)
            or path.suffix in (".pyc", ".pyo") or path.name == "MUJOCO_LOG.TXT"
            or path.match("config/*.local.json"))


def run(command, *, cwd, log, env=None, timeout=180):
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}: {log.read_text()[-3000:]}")
    return result.returncode


def direct_native_build(source, work, logs):
    """Compile the same targets without CMake when only the C/C++ tools exist."""
    compiler, c_compiler = shutil.which("c++"), shutil.which("cc")
    if not compiler or not c_compiler:
        raise RuntimeError("native acceptance needs cmake or both cc and c++")
    native = source/"native"
    build = work/"native-build"
    build.mkdir()
    third = native/"third_party/tdk_icm45686"
    objects = []
    for name in ("inv_imu_driver", "inv_imu_transport"):
        obj = build/(name+".o")
        run([c_compiler, "-O2", "-I", str(third), "-c", str(third/"imu"/(name+".c")), "-o", str(obj)],
            cwd=work, log=logs/(name+".log"))
        objects.append(str(obj))
    common = [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-pthread", "-I", str(native/"src"), "-I", str(third)]
    targets = [
        ("sprite_can_shadow", native/"vendor/sprite_can_shadow.cpp", []),
        ("sprite_icm45686", native/"icm45686/main.cpp", objects),
        ("test_native_contract", native/"tests/contract_test.cpp", []),
        ("icm_offline_test", native/"icm45686/test.cpp", objects),
        ("icm_packet_test", native/"icm45686/packet_test.cpp", []),
    ]
    for name, code, linked in targets:
        run(common+[str(code), *linked, "-o", str(build/name)], cwd=work, log=logs/(name+"-build.log"))
    for name in ("test_native_contract", "icm_offline_test", "icm_packet_test"):
        run([str(build/name)], cwd=work, log=logs/(name+"-run.log"))
    return {"compiled":True, "build_system":"direct_cc_cxx_fallback", "compiled_targets":[x[0] for x in targets],
            "offline_tests_passed":["test_native_contract", "icm_offline_test", "icm_packet_test"],
            "architecture":platform.machine(), "cross_compiled_for_jetson":False,
            "hardware_accessed":False, "cmake_available":False}


ISOLATE = '''import importlib.util, pathlib, runpy, sys
install=pathlib.Path(sys.argv.pop(1)).resolve()
sys.path[:]=[str(install)]+[p for p in sys.path if p and
    ("site-packages" in pathlib.Path(p).parts or str(pathlib.Path(p)).startswith(sys.base_prefix+"/"))]
assert importlib.util.find_spec("opensprite_tracker") is None,"parent tracker unexpectedly importable"
import opensprite_sdk
assert pathlib.Path(opensprite_sdk.__file__).resolve().is_relative_to(install),"SDK imported from original checkout"
script=sys.argv.pop(1)
sys.argv=[script]+sys.argv[1:]
runpy.run_path(script,run_name="__main__")
'''

SELFTEST = '''import hashlib,json,pathlib,sys
import numpy as np
from opensprite_sdk.backends.mujoco import MujocoBackend,PACKAGE_ROOT
from opensprite_sdk.client import joint_command
manifest=json.loads((PACKAGE_ROOT/"assets/manifest.json").read_text())
for name,row in manifest["files"].items():
 data=(PACKAGE_ROOT/name).read_bytes()
 assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"],name
with MujocoBackend() as backend:
 assert (backend.model.nq,backend.model.nv,backend.model.nu)==(38,37,0)
 backend.enable()
 minimum=float("inf")
 for i in range(100):
  state=backend.read()
  target=joint_command(state,backend.profile["default_q"],backend.profile["kp"],backend.profile["kd"])
  target.seq=i+1;target.session_id="standalone-installed-wheel"
  backend.write(target)
  backend.step(backend.policy_steps)
  truth=backend.debug_truth()
  minimum=min(minimum,float(truth["root_position"][2]))
  assert np.all(np.abs(truth["applied_torque"])<=np.asarray(backend.profile["effort_limit"])+1e-12)
 assert minimum>.4 and backend.read().status=="enabled"
 assert abs(backend.data.time-2)<1e-9
 result={"imported_sdk":str(PACKAGE_ROOT),"parent_tracker_import_blocked":True,
         "asset_sha256_count":len(manifest["files"]),"stand_simulated_seconds":2,
         "minimum_root_height_m":minimum,"enabled_at_end":True,"passed":True}
pathlib.Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\\n")
print(json.dumps(result))
'''


DDS_SELFTEST = '''import json,pathlib,sys,time
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.backends.mujoco import load_profile
from opensprite_sdk.runtime import ManagementClient
ChannelFactoryInitialize(218, "lo", transport="dds")
profile=load_profile()
with ManagementClient() as robot:
 state=robot.Read(10.)
 assert state.clock_domain=="sim"
 assert robot.request("claim").accepted
 assert robot.request("enable").execution_verified
 for i in range(20):
  target=joint_command(state,profile["default_q"],profile["kp"],profile["kd"])
  target.seq=i+1
  robot.Write(target)
  deadline=time.monotonic()+2.
  previous=state
  while True:
   state=robot.Read(max(.001,deadline-time.monotonic()))
   assert state.session_id==previous.session_id
   if (state.stamp_ns>previous.stamp_ns and state.accepted_command_seq>=i+1
       and state.accepted_controller_session==robot.controller_session): break
   assert time.monotonic()<deadline,"command acceptance timed out"
 reply=robot.request("disable")
 assert reply.accepted and reply.execution_verified
result={"passed":True,"commands_accepted":state.accepted_command_seq,
        "simulated_seconds":state.stamp_ns/1e9,"disable_verified":True,
        "loaded_from_installed_wheel":True,"parent_tracker_import_blocked":True}
pathlib.Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\\n")
print(json.dumps(result))
'''


def check(output, *, native=True, dds=False):
    report = {"schema": "opensprite-sdk-standalone-acceptance-v1",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host_architecture": platform.machine(), "jetson_hardware_tested": False,
              "hardware_interfaces_opened": False, "native_skipped": not native,
              "dds_requested":dds, "passed": False}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = ROOT / "outputs/extraction"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="standalone-", dir=base))
    source, wheels, install, logs = (work/name for name in ("source", "wheels", "install", "logs"))
    for directory in (source, wheels, logs):
        directory.mkdir()
    report["work_directory"] = str(work)
    try:
        copied = []
        inventory = {}
        for original in sorted(ROOT.rglob("*")):
            relative = original.relative_to(ROOT)
            if excluded(relative) or not original.is_file():
                continue
            if original.is_symlink():
                raise RuntimeError(f"distributable symlink requires an explicit portability review: {relative}")
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            copied.append(str(relative))
            inventory[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
        assert copied and all(not excluded(Path(name)) for name in copied)
        inventory_bytes = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        (work/"source-inventory.json").write_bytes(inventory_bytes)
        report.update(copied_source_files=len(copied), source_ignores_verified=True,
                      source_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest())
        run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--no-index",
             str(source), "-w", str(wheels)], cwd=work, log=logs/"wheel.log")
        wheel_files = list(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise RuntimeError("expected exactly one SDK wheel")
        wheel = wheel_files[0]
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
        assert all(not excluded(Path(name)) for name in members)
        assert "opensprite_sdk/backends/mujoco.py" in members
        assert "opensprite_sdk/profiles/legacy_v4.json" in members
        assert any(name.endswith("/meshes/pelvis_link.STL") for name in members)
        report.update(wheel_name=wheel.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      wheel_bytes=wheel.stat().st_size, wheel_members=len(members),
                      reference_and_output_files_absent_from_wheel=True)
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(install),
             str(wheel)], cwd=work, log=logs/"install.log")
        isolated = work/"isolated.py"
        isolated.write_text(ISOLATE)
        selftest = work/"selftest.py"
        selftest.write_text(SELFTEST)
        state_report = work/"installed-physics.json"
        prefix = [sys.executable, "-I", str(isolated), str(install)]
        run(prefix+[str(selftest), str(state_report)], cwd=work, log=logs/"physics.log")
        report["installed_physics"] = json.loads(state_report.read_text())
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="sprite-sdk-standalone-") as temporary:
            endpoint = str(Path(temporary)/"robot.sock")
            server_log = logs/"sim-server.log"
            with server_log.open("w") as stream:
                server = subprocess.Popen(prefix+[str(source/"examples/sim_robot.py"), "--endpoint", endpoint,
                    "--duration", "2", "--enable-sim"], cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+15
                    while not Path(endpoint).exists():
                        if server.poll() is not None or time.monotonic()>deadline:
                            raise RuntimeError(f"standalone server failed startup: {server_log.read_text()}")
                        time.sleep(.02)
                    run(prefix+[str(source/"examples/low_level.py"), "--endpoint", endpoint,
                        "--duration", ".6", "--gains-profile", "legacy_v4"], cwd=work,
                        env=env, log=logs/"low-level-client.log", timeout=15)
                    rc = server.wait(timeout=10)
                    if rc:
                        raise RuntimeError(f"standalone server failed: {server_log.read_text()}")
                finally:
                    if server.poll() is None:
                        server.terminate()
                        server.wait(timeout=5)
            client = json.loads((logs/"low-level-client.log").read_text())
            assert client["commands"]>10 and client["status"]=="disabled"
            report["two_process_example"] = {**client, "server_returncode":rc, "client_returncode":0,
                "loaded_from_installed_wheel":True, "passed":True}
        if dds:
            dds_build, dds_prefix = work/"dds-build", work/"dds-prefix"
            run([sys.executable, str(source/"tools/generate_dds_schema.py"), "--check"],
                cwd=work, log=logs/"dds-schema.log")
            run(["cmake", "-S", str(source/"native-dds"), "-B", str(dds_build),
                 "-DCMAKE_BUILD_TYPE=Release"], cwd=work, log=logs/"dds-configure.log")
            run(["cmake", "--build", str(dds_build), "--parallel", "2"],
                cwd=work, log=logs/"dds-build.log")
            run(["cmake", "--install", str(dds_build), "--prefix", str(dds_prefix)],
                cwd=work, log=logs/"dds-install.log")
            dds_env = dict(env, OPENSPRITE_DDS_LIBRARY=str(dds_prefix/"lib/libopensprite_dds.so"))
            dds_selftest = work/"dds-selftest.py"
            dds_selftest.write_text(DDS_SELFTEST)
            dds_result = work/"installed-dds.json"
            dds_runtime_result = work/"installed-dds-runtime.json"
            with (logs/"dds-runtime.log").open("w") as stream:
                server = subprocess.Popen(prefix+[str(source/"examples/robot_runtime.py"),
                    "--transport", "dds", "--domain", "218", "--interface", "lo",
                    "--enable-sim", "--duration", "15", "--output", str(dds_runtime_result)],
                    cwd=work, env=dds_env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    run(prefix+[str(dds_selftest), str(dds_result)], cwd=work,
                        env=dds_env, log=logs/"dds-client.log", timeout=20)
                    rc = server.wait(timeout=5)
                    if rc:
                        raise RuntimeError(f"installed DDS runtime failed; see {logs/'dds-runtime.log'}")
                finally:
                    if server.poll() is None:
                        server.terminate()
                        server.wait(timeout=5)
            report["relocated_dds"] = {**json.loads(dds_result.read_text()),
                "runtime":json.loads(dds_runtime_result.read_text()),
                "library":str(dds_prefix/"lib/libopensprite_dds.so"),
                "built_from_relocated_source":True}
        if native:
            if shutil.which("cmake"):
                build = work/"native-build"
                run(["cmake", "-S", str(source/"native"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                    cwd=work, log=logs/"native-configure.log")
                run(["cmake", "--build", str(build), "--parallel", "2"], cwd=work, log=logs/"native-build.log")
                run(["ctest", "--test-dir", str(build), "--output-on-failure"], cwd=work, log=logs/"native-tests.log")
                report["relocated_native"] = {"compiled":True, "build_system":"cmake",
                    "offline_ctest_passed":True, "architecture":platform.machine(),
                    "cross_compiled_for_jetson":False, "hardware_accessed":False}
            else:
                report["relocated_native"] = direct_native_build(source, work, logs)
        report.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/acceptance/standalone.json")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--dds", action="store_true", help="also build/install and test real DDS")
    args = parser.parse_args()
    print(json.dumps(check(args.output, native=not args.skip_native, dds=args.dds), indent=2))
