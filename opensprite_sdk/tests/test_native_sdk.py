"""Compile an external C++ consumer of installed headers, then use real DDS."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import pytest

from opensprite_sdk.core.dds import DdsEndpoint
from opensprite_sdk.messages import message_from_dict
from test_dds_integration import native_dds_available

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed_cpp_peer(tmp_path_factory):
    library = ROOT/"native-dds/build/libopensprite_dds.so"
    header = ROOT/"native-dds/include/opensprite_sdk/core/channel.hpp"
    if not library.is_file() or not header.is_file():
        pytest.skip("optional native C++ SDK is not built; run tools/build_dds.sh")
    if not shutil.which("cmake"):
        pytest.skip("CMake is required for installed C++ SDK acceptance")
    work = tmp_path_factory.mktemp("installed-cpp-sdk")
    prefix = work/"prefix"
    subprocess.run(["cmake", "--install", str(ROOT/"native-dds/build"), "--prefix", str(prefix)],
                   check=True, capture_output=True, text=True)
    source = work/"consumer"
    shutil.copytree(ROOT/"tests/native_sdk", source)
    build = work/"build"
    # This consumer contains only its own files; all SDK includes and linkage
    # must resolve through the installed exported target.
    subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                    f"-DCMAKE_PREFIX_PATH={prefix}", "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"],
                   check=True, capture_output=True, text=True)
    return build/"typed_peer"


def stop(process):
    if process.poll() is None and process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.write("\n")
            process.stdin.close()
            process.wait(timeout=1)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=3)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def fixtures():
    # Reuse independent ROS acceptance values, including >2**53 integers,
    # fixed31 nested arrays, UTF-8, IMU and command acceptance correlation.
    import importlib.util
    spec = importlib.util.spec_from_file_location("native_sdk_fixtures", ROOT/"tools/check_ros2_interop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixtures()


def test_installed_cpp_sdk_four_types_roundtrip_python(native_dds_available, installed_cpp_peer):
    prefix, domain = "typed_"+uuid.uuid4().hex[:12], 224
    expected = fixtures()
    writers, readers, process = {}, {}, None
    try:
        for name in expected:
            writers[name] = DdsEndpoint(domain, "lo", f"rt/{prefix}/to_cpp/{name}", name+"_", "publisher")
            readers[name] = DdsEndpoint(domain, "lo", f"rt/{prefix}/from_cpp/{name}", name+"_", "subscriber")
        process = subprocess.Popen([str(installed_cpp_peer), str(domain), prefix, "echo"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=installed_cpp_peer.parent)
        received = {}
        deadline = time.monotonic()+10
        while time.monotonic() < deadline and len(received) < 4:
            for name, envelope in expected.items():
                writers[name].write(envelope)
                result = readers[name].take(timeout=0)
                if result is not None:
                    message_from_dict(result).validate()
                    assert result == envelope
                    received[name] = result
            if process.poll() is not None and process.returncode:
                raise AssertionError(process.stderr.read())
            time.sleep(.01)
        assert set(received) == set(expected)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stderr
        assert '"passed":true' in stdout
    finally:
        for endpoint in (*writers.values(), *readers.values()):
            endpoint.close()
        if process is not None:
            stop(process)


@pytest.mark.parametrize("mode", ["callback-close", "callback-throw", "callback-destroy"])
def test_installed_cpp_callback_lifecycle(installed_cpp_peer, mode):
    # The original failure occurred after the callback had printed success,
    # during process teardown; repeat in fresh processes to exercise that race.
    for _ in range(3):
        completed = subprocess.run([str(installed_cpp_peer), "225", "callback_"+uuid.uuid4().hex[:12], mode],
            capture_output=True, text=True, timeout=8, cwd=installed_cpp_peer.parent)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["passed"] is True
