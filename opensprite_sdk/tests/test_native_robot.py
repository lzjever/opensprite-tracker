"""Real C++ robot gateway with binary synthetic devices and typed DDS clients."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.backends.native_ipc import StatePacket, TARGET1
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.core.dds import DdsEndpoint
from opensprite_sdk.messages import ControlRequest_, message_to_dict
from opensprite_sdk.runtime import ManagementClient
from opensprite_sdk.policy import sensor_observation
from test_dds_integration import native_dds_available
from test_hardware_config import measured_fixture
from test_native_ipc import imu_packet

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_robot(native_dds_available):
    binary = ROOT/"native-runtime/build/opensprite_robot"
    if not binary.is_file():
        pytest.skip("optional native robot is not built; run tools/build_robot.sh")
    assert binary.read_bytes()[:4] == b"\x7fELF"
    return binary


def compile_config(source, output):
    spec = importlib.util.spec_from_file_location("native_robot_config_compiler", ROOT/"tools/compile_native_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compile_hardware(source, output)


class NativePeers:
    """Actual filesystem IPC, distinct raw sensor times and500Hz native state."""
    def __init__(self, directory):
        self.directory = Path(directory)
        data = measured_fixture()
        data["native"]["socket_path"] = str(self.directory/"can.sock")
        data["imu"]["socket_path"] = str(self.directory/"imu.sock")
        self.source = self.directory/"hardware.json"
        self.source.write_text(json.dumps(data))
        self.config = HardwareConfig.from_dict(data)
        self.compiled = self.directory/"robot.json"
        compile_config(self.source, self.compiled)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server.bind(data["native"]["socket_path"])
        self.server.listen(1)
        self.server.settimeout(.05)
        self.imu = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.imu.bind(data["imu"]["socket_path"])
        self.imu.settimeout(.05)
        self.stop = threading.Event()
        self.disconnected = threading.Event()
        self.freeze_motor = threading.Event()
        self.freeze_imu = threading.Event()
        self.suppress_feedback = threading.Event()
        self.targets = []
        self.errors = []
        self.threads = [threading.Thread(target=self.native, daemon=True, name="native-cpp-fake-can"),
                        threading.Thread(target=self.inertial, daemon=True, name="native-cpp-fake-imu")]

    def start(self):
        for thread in self.threads:
            thread.start()

    def native(self):
        peer = None
        try:
            while not self.stop.is_set():
                try:
                    peer, _ = self.server.accept()
                    break
                except socket.timeout:
                    continue
            if peer is None:
                return
            peer.setblocking(False)
            sequence = 0
            deadline = time.monotonic()
            frozen = None
            while not self.stop.is_set():
                now = time.monotonic_ns()
                if time.monotonic() >= deadline:
                    sequence += 1
                    # Every q encodes the raw native sequence. This lets the
                    # client verify source mapping without exposing a new
                    # privileged field in LowState.
                    q = np.full(31, sequence*1e-6)
                    position, velocity = self.config.motor_state(q, np.zeros(31))
                    samples = [now-500_000]*31
                    if self.freeze_motor.is_set():
                        frozen = frozen or now-500_000
                        samples[0] = frozen
                    if not self.suppress_feedback.is_set():
                        peer.send(StatePacket(sequence, now, self.config.motor_order_hash,
                            tuple(position), tuple(velocity), tuple(samples), False).pack())
                    deadline += .002
                    if deadline < time.monotonic()-.01:
                        deadline = time.monotonic()+.002
                if select.select([peer], [], [], .0005)[0]:
                    packet = peer.recv(1281)
                    if not packet:
                        self.disconnected.set()
                        return
                    self.targets.append(TARGET1.unpack(packet))
        except (BrokenPipeError, ConnectionResetError):
            self.disconnected.set()
        except (OSError, ValueError) as error:
            if not self.stop.is_set():
                self.errors.append(error)
        finally:
            if peer is not None:
                peer.close()

    def inertial(self):
        sequence, frozen = 0, None
        try:
            while not self.stop.is_set():
                try:
                    request, address = self.imu.recvfrom(9)
                except socket.timeout:
                    continue
                assert request == b"GETIMU01"
                sequence += 1
                now = time.monotonic_ns()-100_000
                if self.freeze_imu.is_set():
                    frozen = frozen or now
                    now = frozen
                self.imu.sendto(imu_packet(now, sequence=sequence), address)
        except (FileNotFoundError, ConnectionRefusedError):
            # A correctly stopped native client removes its temporary socket.
            pass
        except (OSError, AssertionError) as error:
            if not self.stop.is_set():
                self.errors.append(error)

    def close(self):
        self.stop.set()
        self.server.close()
        self.imu.close()
        for thread in self.threads:
            thread.join(timeout=1.)


@contextmanager
def hardware_runtime(binary, *, allow_enable=True, duration=8., send_feedback=True):
    # A short filesystem path avoids the108-byte AF_UNIX path limit.
    with tempfile.TemporaryDirectory(prefix="opensprite-native-") as directory:
        peers = NativePeers(directory)
        report, log = Path(directory)/"report.json", Path(directory)/"robot.log"
        args = [str(binary), "--backend", "hardware", "--config", str(peers.compiled),
                "--domain", "226", "--interface", "lo", "--duration", str(duration), "--report", str(report)]
        if allow_enable:
            args.append("--allow-enable")
        environment = dict(os.environ, PYTHONPATH="/no-python-imports-needed", PATH="/usr/bin:/bin")
        if not send_feedback:
            peers.suppress_feedback.set()
        peers.start()
        with log.open("w") as stream:
            process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=directory, env=environment)
            try:
                ChannelFactoryInitialize(226, "lo", transport="dds")
                yield peers, process, report, log
            finally:
                if process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=3)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                peers.close()
        assert all(not thread.is_alive() for thread in peers.threads)
        assert not peers.errors, peers.errors


def wait_acceptance(client, sequence, *, timeout=2.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        state = client.Read(timeout=.5)
        if state.accepted_command_seq == sequence:
            return state
    raise AssertionError("native robot did not report command acceptance")


def test_native_gateway_shadow_source_mapping_and_no_python_dependency(native_robot):
    with hardware_runtime(native_robot) as (peers, process, report, log):
        with ManagementClient(controller_session="native-shadow-owner") as client:
            state = client.Read(timeout=5.)
            assert state.profile_id == "physical_v5" and state.status == "shadow"
            assert state.clock_domain == "monotonic"
            assert sensor_observation(state, np.zeros(31), np.zeros(31),
                imu_link=state.imu_state.attached_link, now_ns=state.stamp_ns).shape == (99,)
            maps = Path(f"/proc/{process.pid}/maps").read_text().lower()
            assert "libpython" not in maps and "onnxruntime" not in maps
            assert client.request("claim").accepted
            enabled = client.request("enable")
            assert enabled.accepted and not enabled.execution_verified
            expected_sources = []
            for sequence in (42, 43, 44):
                expected_sources.append(round(state.joint_state[0].q/1e-6))
                target = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
                target.seq = sequence
                client.Write(target)
                state = wait_acceptance(client, sequence)
                assert state.accepted_controller_session == client.controller_session
            deadline = time.monotonic()+1
            while len(peers.targets) < 3 and time.monotonic() < deadline:
                time.sleep(.001)
            assert [packet[6] for packet in peers.targets] == expected_sources
            disabled = client.request("disable")
            assert disabled.accepted and not disabled.execution_verified
        process.wait(timeout=3)
        assert process.returncode == 0, log.read_text()
        assert peers.disconnected.wait(timeout=1)
        result = json.loads(report.read_text())
        assert result["accepted_commands"] == 3
        assert result["fault"] is None


@pytest.mark.parametrize("kind", ["motor", "imu"])
def test_native_gateway_stale_individual_sensor_closes_native_ipc(native_robot, kind):
    with hardware_runtime(native_robot) as (peers, process, report, log):
        with ManagementClient(controller_session="native-fault-owner") as client:
            client.Read(timeout=5.)
            client.request("claim")
            client.request("enable")
            state = client.Read(timeout=1.)
            client.Write(joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31))
            if kind == "motor":
                peers.freeze_motor.set()
            else:
                peers.freeze_imu.set()
            process.wait(timeout=3)
        assert process.returncode != 0, log.read_text()
        assert peers.disconnected.wait(timeout=1)
        result = json.loads(report.read_text())
        assert result["fault"], result
        assert ("motor" if kind == "motor" else "imu") in result["fault"].lower(), result


def test_native_gateway_requires_local_enable_permission(native_robot):
    with hardware_runtime(native_robot, allow_enable=False) as (_, process, _, _):
        with ManagementClient(controller_session="native-permission-owner") as client:
            client.Read(timeout=5.)
            assert client.request("claim").accepted
            with pytest.raises(RuntimeError, match="enable|permission|local"):
                client.request("enable")
            assert process.poll() is None


def test_native_gateway_missing_initial_feedback_is_not_success(native_robot):
    with hardware_runtime(native_robot, duration=.15, send_feedback=False) as (peers, process, report, log):
        process.wait(timeout=3.)
        assert process.returncode != 0, log.read_text()
        result = json.loads(report.read_text())
        assert result["device_ready"] is False and "feedback" in result["fault"]
        assert not peers.targets and peers.disconnected.wait(timeout=1.)


def test_native_gateway_foreign_controller_cannot_claim_enable_or_disable(native_robot):
    with hardware_runtime(native_robot) as (peers, process, report, log):
        with ManagementClient(controller_session="native-legitimate") as owner:
            owner.Read(timeout=5.)
            owner.request("claim")
            with ManagementClient(controller_session="native-foreign") as foreign:
                state = foreign.Read(timeout=3.)
                for operation in ("claim", "enable", "disable", "release"):
                    with pytest.raises(RuntimeError, match="own|controller"):
                        foreign.request(operation)
                for _ in range(5):
                    foreign.Write(joint_command(state, [0.]*31, [0.]*31, [0.]*31))
                    time.sleep(.002)
            assert not peers.targets
            owner.request("disable")
        process.wait(timeout=3)
        assert process.returncode == 0, log.read_text()
        result = json.loads(report.read_text())
        assert result["rejected_commands"] > 0 and result["accepted_commands"] == 0


def test_native_gateway_malformed_management_cannot_poison_shared_replies(native_robot):
    with hardware_runtime(native_robot) as (_, process, _, log):
        raw = DdsEndpoint(226, "lo", "rt/control_request", "ControlRequest_", "publisher")
        try:
            with ManagementClient(controller_session="native-management-owner") as client:
                state = client.Read(timeout=5.)
                client.request("claim")
                deadline = time.monotonic()+2
                while raw.matches() == 0 and time.monotonic() < deadline:
                    time.sleep(.005)
                assert raw.matches() > 0
                request = ControlRequest_(profile_id=state.profile_id, clock_domain=state.clock_domain,
                    stamp_ns=state.stamp_ns, seq=1, session_id="malformed-publisher",
                    request_id="will-be-empty", controller_session="malformed-publisher",
                    robot_session=state.session_id, operation="claim")
                envelope = message_to_dict(request)
                envelope["data"]["request_id"] = ""
                raw.write(envelope)
                time.sleep(.03)
                # All clients share the reply topic. A request without an ID
                # cannot receive a correlated reply and must not emit a reply
                # that violates other clients' required-field contract.
                assert client.request("release").accepted
                client.request("claim")
                client.request("disable")
            process.wait(timeout=3)
            assert process.returncode == 0, log.read_text()
        finally:
            raw.close()


@pytest.mark.parametrize("corruption,expected", [
    ("source_epoch", "epoch"), ("unknown_source", "source"),
    ("sequence_replay", "sequence"), ("profile", "profile"),
])
def test_native_gateway_rejects_owner_source_and_replay(native_robot, corruption, expected):
    with hardware_runtime(native_robot) as (peers, process, report, log):
        raw = DdsEndpoint(226, "lo", "rt/lowcmd", "LowCmd_", "publisher")
        try:
            deadline = time.monotonic()+3
            while raw.matches() == 0 and time.monotonic() < deadline:
                time.sleep(.005)
            assert raw.matches() > 0
            with ManagementClient(controller_session="native-source-owner") as client:
                client.Read(timeout=5.)
                client.request("claim")
                client.request("enable")
                state = client.Read(timeout=1.)
                good = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
                good.seq = 10
                client.Write(good)
                state = wait_acceptance(client, 10)
                bad = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
                bad.seq, bad.session_id = 11, client.controller_session
                if corruption == "source_epoch":
                    bad.source_state_session_id = "a-previous-robot-session"
                elif corruption == "unknown_source":
                    bad.source_state_seq = 2**64-1
                elif corruption == "sequence_replay":
                    bad.seq = 10
                else:
                    bad.profile_id = "a-different-robot-profile"
                raw.write(message_to_dict(bad))
                process.wait(timeout=3)
            assert process.returncode != 0, log.read_text()
            assert peers.disconnected.wait(timeout=1)
            result = json.loads(report.read_text())
            assert expected in result["fault"].lower(), result
            assert result["accepted_commands"] == 1
        finally:
            raw.close()


STREAMING_OWNER = r'''
import time
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.runtime import ManagementClient
ChannelFactoryInitialize(226,"lo",transport="dds")
with ManagementClient(controller_session="crashed-native-owner") as robot:
    state=robot.Read(5.)
    robot.request("claim")
    robot.request("enable")
    print("streaming",flush=True)
    while True:
        robot.Write(joint_command(state,[j.q for j in state.joint_state],[2.]*31,[.1]*31))
        time.sleep(.02)
        state=robot.Read(1.)
'''


def test_native_gateway_owner_process_crash_triggers_independent_watchdog(native_robot):
    with hardware_runtime(native_robot) as (peers, process, report, log):
        owner = subprocess.Popen([sys.executable, "-u", "-c", STREAMING_OWNER],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert select.select([owner.stdout], [], [], 6)[0], "owner did not initialize"
            line = owner.stdout.readline().strip()
            assert line == "streaming", line
            time.sleep(.06)
            owner.terminate()  # bypasses the SDK's normal request/cleanup path
            owner.wait(timeout=2)
            process.wait(timeout=3)
            assert process.returncode != 0, log.read_text()
            assert peers.disconnected.wait(timeout=1)
            result = json.loads(report.read_text())
            assert result["fault"] == "controller_stream_watchdog", result
            assert result["accepted_commands"] >= 1
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=2)
            owner.stdout.close()
            owner.stderr.close()


@pytest.mark.parametrize("paused_diagnostic", [False, True])
def test_native_sim_dispatch_500hz_preserves_200hz_physics(native_robot, tmp_path, paused_diagnostic):
    spec = importlib.util.spec_from_file_location("native_scene_preparer", ROOT/"tools/prepare_sim_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = module.prepare(SimpleNamespace(output=tmp_path/"scene", profile="legacy_v4", seed=42, motion=None))
    scene_file = Path(prepared["scene"])
    scene = json.loads(scene_file.read_text())
    assert scene["joint_execution_hz"] == 500 and scene["physics_hz"] == 200
    profile = scene["profile"]
    report, log, truth = tmp_path/"report.json", tmp_path/"robot.log", tmp_path/"truth.jsonl"
    command = [str(native_robot), "--backend", "mujoco", "--config", str(scene_file),
        "--domain", "227", "--interface", "lo", "--allow-enable", "--duration", "6", "--report", str(report),
        "--truth-output", str(truth)]
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH="/no-python-or-policy-needed"))
        diagnostic = None
        try:
            if paused_diagnostic:
                code = """import time
from opensprite_sdk.core.dds import DdsEndpoint
reader=DdsEndpoint(227,'lo','rt/lowstate','LowState_','subscriber')
assert reader.take(5.) is not None
print('matched-and-received',flush=True)
time.sleep(30.)
"""
                diagnostic = subprocess.Popen([sys.executable, "-u", "-c", code],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                assert select.select([diagnostic.stdout], [], [], 6.)[0]
                # A received sample proves discovery was complete before STOP;
                # stopping a reader before discovery cannot test writer coupling.
                assert diagnostic.stdout.readline().strip() == "matched-and-received"
                os.kill(diagnostic.pid, signal.SIGSTOP)
            ChannelFactoryInitialize(227, "lo", transport="dds")
            with ManagementClient(controller_session="native-sim-rate-owner") as client:
                state = client.Read(timeout=5.)
                assert state.stamp_ns == 0 and state.clock_domain == "sim"
                client.request("claim")
                client.request("enable")
                for frame in range(50):
                    target = joint_command(state, profile["default_q"], profile["kp"], profile["kd"])
                    target.seq = frame+1
                    client.Write(target)
                    deadline = time.monotonic()+1.
                    target_stamp = (frame+1)*20_000_000
                    while True:
                        state = client.Read(max(.001, deadline-time.monotonic()))
                        if state.stamp_ns >= target_stamp and state.accepted_command_seq >= frame+1:
                            break
                        assert time.monotonic() < deadline
                disabled = client.request("disable")
                assert disabled.accepted and disabled.execution_verified
            process.wait(timeout=3.)
            assert process.returncode == 0, log.read_text()
            result = json.loads(report.read_text())
            assert result["fault"] is None and result["accepted_commands"] == 50
            assert result["execution_hz"] == 500
            assert 1. <= result["simulated_seconds"] <= 1.02
            assert 200 <= result["physics_steps"] <= 204
            assert abs(result["execution_ticks"]-result["physics_steps"]*2.5) <= 4
            assert result["physics_hz"] == 200
            # The external diagnostic snapshot at exactly one second avoids
            # counting the small shutdown-message transit window as motion.
            rows = [json.loads(line) for line in truth.read_text().splitlines()]
            exact = next(row for row in rows if row["time"] == 1.)
            assert exact["physics_steps"] == 200
            assert exact["joint_command_dispatches"] == 500
        finally:
            if diagnostic is not None:
                if diagnostic.poll() is None:
                    os.kill(diagnostic.pid, signal.SIGCONT)
                    diagnostic.terminate()
                    diagnostic.wait(timeout=3.)
                diagnostic.stdout.close()
                diagnostic.stderr.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.)
