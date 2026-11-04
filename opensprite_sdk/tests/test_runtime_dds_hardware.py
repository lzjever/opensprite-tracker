"""Real DDS runtime over synthetic native/IMU socket peers; no device access."""
import select
import socket
import struct
import threading
import time

import numpy as np

from opensprite_sdk.backends.hardware import HardwareBackend
from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.backends.native_ipc import StatePacket, TARGET1
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.runtime import ManagementClient, RobotRuntime
from opensprite_sdk.policy import sensor_observation
from test_dds_integration import native_dds_available  # shared optional-dependency gate/lock
from test_hardware_config import measured_fixture
from test_native_ipc import NOW, imu_packet


class SyntheticNativePeers:
    """Use the actual binary protocols with deliberately distinct sample times."""
    def __init__(self):
        self.config = HardwareConfig.from_dict(measured_fixture())
        self.local, self.peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.imu_local, self.imu_peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.backend = HardwareBackend(self.config, state_socket=self.local,
            imu_socket=self.imu_local, clock=lambda: NOW, allow_actuation=False)
        self.motor_stamps = tuple(NOW - 2_000_000 - 10_000*i for i in range(31))
        self.imu_sample_ns = NOW-500_000
        self.targets = []
        self.errors = []
        self.stop = threading.Event()
        self.disconnected = threading.Event()
        self.threads = [threading.Thread(target=self.native, name="synthetic-native", daemon=True),
                        threading.Thread(target=self.imu, name="synthetic-imu", daemon=True)]

    def start(self):
        self.peer.setblocking(False)
        self.imu_peer.settimeout(.05)
        for thread in self.threads:
            thread.start()

    def native(self):
        sequence, next_send = 0, 0.
        position, velocity = self.config.motor_state(np.zeros(31), np.zeros(31))
        writable = True
        try:
            while not self.stop.is_set():
                if writable and time.monotonic() >= next_send:
                    sequence += 1
                    packet = StatePacket(sequence, NOW-1_000_000, self.config.motor_order_hash,
                        tuple(position), tuple(velocity), self.motor_stamps, False)
                    try:
                        self.peer.send(packet.pack())
                    except (BrokenPipeError, ConnectionResetError):
                        writable = False
                    next_send = time.monotonic()+.02
                if select.select([self.peer], [], [], .005)[0]:
                    payload = self.peer.recv(1281)
                    if not payload:
                        self.disconnected.set()
                        return
                    self.targets.append(TARGET1.unpack(payload))
        except (BrokenPipeError, ConnectionResetError):
            # Closing SOCK_SEQPACKET with unread feedback may reset the peer
            # rather than returning EOF; both prove an actual IPC disconnect.
            self.disconnected.set()
        except (OSError, ValueError) as error:
            if not self.stop.is_set():
                self.errors.append(error)

    def imu(self):
        sequence = 0
        try:
            while not self.stop.is_set():
                try:
                    request = self.imu_peer.recv(9)
                except socket.timeout:
                    continue
                if not request:
                    return
                assert request == b"GETIMU01"
                sequence += 1
                try:
                    payload = bytearray(imu_packet(NOW, sequence=sequence))
                    # IMU acquisition follows the older native aggregate, as
                    # it does in real collection. The public header must anchor
                    # the fused snapshot without changing either sensor time.
                    struct.pack_into("<QQ", payload, 16, self.imu_sample_ns, NOW-100_000)
                    self.imu_peer.send(payload)
                except ConnectionRefusedError:
                    return
        except (OSError, AssertionError) as error:
            if not self.stop.is_set():
                self.errors.append(error)

    def close(self):
        self.stop.set()
        for sock in (self.peer, self.imu_peer):
            sock.close()
        self.backend.close()
        for thread in self.threads:
            thread.join(timeout=1.)


def test_fake_hardware_and_imu_through_real_dds_runtime(native_dds_available):
    # Fixed synthetic device clock intentionally differs from host monotonic.
    # Every observation and command must keep that robot clock unchanged.
    ChannelFactoryInitialize(222, interface="lo", transport="dds")
    peers = SyntheticNativePeers()
    runtime = RobotRuntime(peers.backend, allow_enable=True, shadow=True)
    stop = threading.Event()
    errors, summaries = [], []

    def run():
        try:
            summaries.append(runtime.run(duration=8., stop_event=stop))
        except BaseException as error:
            errors.append(error)

    peers.start()
    worker = threading.Thread(target=run, name="synthetic-hardware-runtime")
    worker.start()
    try:
        with ManagementClient(controller_session="dds-synthetic-shadow") as client:
            state = client.Read(timeout=5.)
            assert state.profile_id == "physical_v5" and state.clock_domain == "monotonic"
            assert state.status == "shadow" and state.stamp_ns == NOW
            assert state.imu_state.sample_ns == peers.imu_sample_ns
            assert state.imu_state.sample_time_kind == "host_read"
            observation = sensor_observation(state, np.zeros(31), np.zeros(31),
                imu_link=state.imu_state.attached_link, now_ns=state.stamp_ns)
            assert observation.shape == (99,) and np.isfinite(observation).all()
            assert peers.backend._packet.stamp_ns == NOW-1_000_000
            for index, joint in enumerate(state.joint_state):
                dependencies = np.flatnonzero(np.abs(peers.config.inverse[index]) > 1e-12)
                expected = min(peers.motor_stamps[i] for i in dependencies)
                assert joint.sample_ns == expected and joint.sample_time_kind == "host_receive"
                assert joint.valid and not joint.tau_est_valid
            assert client.request("claim", state=state).accepted
            enabled = client.request("enable")
            assert enabled.accepted and enabled.status == "shadow_ready"
            assert not enabled.execution_verified and not peers.backend._enabled
            state = client.Read(timeout=2.)
            sources = []
            for sequence in (42, 43, 44):
                command = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
                command.seq = sequence
                sources.append(command.source_state_seq)
                client.Write(command)
                deadline = time.monotonic()+2.
                while time.monotonic() < deadline:
                    accepted = client.Read(timeout=.5)
                    if accepted.accepted_command_seq == sequence:
                        break
                else:
                    raise AssertionError("no matching DDS backend acceptance")
                assert accepted.accepted_controller_session == "dds-synthetic-shadow"
                assert accepted.imu_state.sample_ns == peers.imu_sample_ns
                assert sensor_observation(accepted, np.zeros(31), np.zeros(31),
                    imu_link=accepted.imu_state.attached_link,
                    now_ns=accepted.stamp_ns).shape == (99,)
                state = accepted
            assert runtime.accepted_commands == 3 and len(peers.targets) == 3
            assert [packet[6] for packet in peers.targets] == sources
            # The outgoing native request retains the actual referenced source.
            assert peers.targets[-1][4] == NOW
            disabled = client.request("disable", timeout=3.)
            assert disabled.accepted and disabled.status == "disable_requested"
            assert not disabled.execution_verified
        worker.join(timeout=3.)
        assert not worker.is_alive() and not errors
        assert peers.disconnected.wait(timeout=1.)
        assert not peers.backend._enabled and peers.backend._closed
        assert summaries[0]["native_motor_disable_verified"] is False
        assert runtime._reader is not None and not runtime._reader.is_alive()
        assert runtime._reply_sender is not None and not runtime._reply_sender.is_alive()
    finally:
        stop.set()
        worker.join(timeout=3.)
        peers.close()
    assert not worker.is_alive() and not peers.errors
    assert all(not thread.is_alive() for thread in peers.threads)
