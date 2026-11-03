"""Real SDK channels with fake native/IMU sockets; never opens CAN devices."""
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from opensprite_sdk.backends.hardware import HardwareBackend
from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.backends.native_ipc import StatePacket, TARGET1
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelServer
from opensprite_sdk.policy import sensor_observation
from opensprite_sdk.service import BackendService
from test_hardware_config import measured_fixture
from test_native_ipc import NOW, imu_packet


def environment(*, actuate=False):
    config = HardwareConfig.from_dict(measured_fixture())
    local, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    imu_local, imu_peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    current = [NOW]
    backend = HardwareBackend(config, allow_actuation=actuate, clock=lambda: current[0],
                              state_socket=local, imu_socket=imu_local)
    backend.start()
    peer.settimeout(1)
    imu_peer.settimeout(1)
    return backend, peer, imu_peer, current


def enqueue(backend, peer, imu_peer, current, sequence, *, actuate=False,
            aggregate_age=1_000_000, motor_age=2_000_000, imu_age=0):
    q, dq = backend.config.motor_state(np.zeros(31), np.zeros(31))
    peer.send(StatePacket(sequence, current[0]-aggregate_age,
                         backend.config.motor_order_hash, tuple(q), tuple(dq),
                         (current[0]-motor_age,)*31, actuate).pack())
    errors = []

    def answer():
        try:
            assert imu_peer.recv(8) == b'GETIMU01'
            imu_peer.send(imu_packet(current[0]-imu_age, sequence=sequence))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=answer)
    thread.start()
    return thread, errors


def snapshot(backend, peer, imu_peer, current, sequence, **kwargs):
    thread, errors = enqueue(backend, peer, imu_peer, current, sequence, **kwargs)
    try:
        return backend.read()
    finally:
        thread.join(timeout=1)
        assert not thread.is_alive() and not errors


def command(state, *, sequence=1, disabled=False):
    target = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.2]*31)
    target.seq = sequence
    target.session_id = 'asynchronous-controller'
    if disabled:
        for joint in target.joint_cmd:
            joint.mode = 0
    return target


CLIENT = '''
import sys
from opensprite_sdk.client import LowLevelClient, joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(43, transport="unix", endpoint=sys.argv[1])
with LowLevelClient() as client:
    print("ready", flush=True)
    state = client.Read()
    print(state.seq, flush=True)
    sys.stdin.readline()
    cmd = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.2]*31)
    if sys.argv[2] == "disable":
        for joint in cmd.joint_cmd:
            joint.mode = 0
    client.Write(cmd)
    print("sent", flush=True)
    sys.stdin.readline()
'''


@pytest.mark.parametrize('mode', ['write', 'preview', 'disable'])
def test_unix_client_can_use_a_fresh_previous_50hz_state(tmp_path, mode):
    """A new feedback tick must not invalidate an already-computed command."""
    actuate = mode != 'preview'
    backend, peer, imu_peer, current = environment(actuate=actuate)
    client = None
    try:
        with ChannelServer(tmp_path/'sdk.sock', domain_id=43):
            ChannelFactoryInitialize(43, transport='unix', endpoint=tmp_path/'sdk.sock')
            with BackendService(backend, shadow=not actuate) as service:
                client = subprocess.Popen([sys.executable, '-c', CLIENT,
                                           str(tmp_path/'sdk.sock'), mode],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)
                assert client.stdout.readline().strip() == 'ready'
                thread, errors = enqueue(backend, peer, imu_peer, current, 1, actuate=actuate)
                first = service.publish_state()
                thread.join(timeout=1)
                assert not thread.is_alive() and not errors
                assert client.stdout.readline().strip() == '1'
                if actuate:
                    backend.enable()
                    assert TARGET1.unpack(peer.recv(1281))[6] == 1
                current[0] += 20_000_000
                thread, errors = enqueue(backend, peer, imu_peer, current, 2, actuate=actuate)
                newest = service.publish_state()
                thread.join(timeout=1)
                assert not thread.is_alive() and not errors
                # Source: aggregate21ms, motor22ms, IMU20ms; execution:2ms.
                sensor_observation(first, np.zeros(31), np.zeros(31),
                                   imu_link='pelvis_link', now_ns=current[0])
                assert newest.seq == 2
                client.stdin.write('send\n')
                client.stdin.flush()
                assert client.stdout.readline().strip() == 'sent'
                if mode != 'disable':
                    thread, errors = enqueue(backend, peer, imu_peer, current, 3, actuate=actuate)
                deadline = time.monotonic()+1
                # Broker acknowledgement means accepted into a bounded slot;
                # wait for actual subscriber delivery before one service tick.
                while not service.command._queue and time.monotonic() < deadline:
                    time.sleep(.001)
                assert service.command._queue and service.tick()
                if mode == 'disable':
                    assert service.stop_requested and backend._closed
                    assert peer.recv(1281) == b''
                else:
                    packet = TARGET1.unpack(peer.recv(1281))
                    assert packet[6] == first.seq == 1  # Never re-sign as2/3.
                    thread.join(timeout=1)
                    assert not thread.is_alive() and not errors
                client.stdin.write('exit\n')
                client.stdin.flush()
                assert client.wait(timeout=2) == 0, client.stderr.read()
    finally:
        if client is not None and client.poll() is None:
            client.kill()
            client.wait(timeout=2)
        backend.close()
        peer.close()
        imu_peer.close()


@pytest.mark.parametrize('mode', ['write', 'preview', 'disable'])
@pytest.mark.parametrize('invalid', ['unknown', 'future', 'epoch', 'stale_imu', 'stale_joint'])
def test_history_does_not_accept_unknown_or_stale_sources(mode, invalid):
    actuate = mode != 'preview'
    backend, peer, imu_peer, current = environment(actuate=actuate)
    try:
        first = snapshot(backend, peer, imu_peer, current, 1, actuate=actuate)
        if actuate:
            backend.enable()
            peer.recv(1281)
        target = command(first, disabled=mode == 'disable')
        if invalid == 'unknown':
            target.source_state_seq = 0
        elif invalid == 'future':
            target.source_state_seq = 3
        elif invalid == 'epoch':
            target.source_state_session_id = 'other-robot-session'
        elif invalid == 'stale_imu':
            current[0] += 31_000_000
        elif invalid == 'stale_joint':
            # 60ms-old aggregate remains in range, but original motor sample
            # is now61ms old. Refresh the latest execution state separately.
            current[0] += 59_000_000
        snapshot(backend, peer, imu_peer, current, 2, actuate=actuate)
        method = backend.write if actuate else backend.preview
        with pytest.raises((ValueError, RuntimeError), match='source|session'):
            method(target)
        assert not backend._closed
        peer.settimeout(.01)
        with pytest.raises(socket.timeout):
            peer.recv(1281)
    finally:
        backend.close()
        peer.close()
        imu_peer.close()


def test_mutating_public_source_does_not_refresh_cached_sensor_ages():
    backend, peer, imu_peer, current = environment()
    try:
        first = snapshot(backend, peer, imu_peer, current, 1)
        target = command(first)
        current[0] += 31_000_000
        snapshot(backend, peer, imu_peer, current, 2)
        first.stamp_ns = current[0]
        first.imu_state.sample_ns = current[0]
        for joint in first.joint_state:
            joint.sample_ns = current[0]
        target.stamp_ns = current[0]
        with pytest.raises(RuntimeError, match='source IMU'):
            backend.preview(target)
    finally:
        backend.close()
        peer.close()
        imu_peer.close()


def test_history_is_bounded_and_evicted_sources_are_not_accepted():
    backend, peer, imu_peer, current = environment()
    try:
        first = None
        for sequence in range(1, backend.source_cache_size+2):
            state = snapshot(backend, peer, imu_peer, current, sequence)
            first = state if first is None else first
        assert len(backend._source_states) == backend.source_cache_size
        with pytest.raises(ValueError, match='unknown, expired'):
            backend.preview(command(first))
    finally:
        backend.close()
        peer.close()
        imu_peer.close()


@pytest.mark.parametrize('mode', ['write', 'preview', 'disable'])
def test_valid_actor_history_does_not_relax_execution_ankle_freshness(mode):
    actuate = mode != 'preview'
    backend, peer, imu_peer, current = environment(actuate=actuate)
    try:
        first = snapshot(backend, peer, imu_peer, current, 1, actuate=actuate)
        if actuate:
            backend.enable()
            peer.recv(1281)
        target = command(first, disabled=mode == 'disable')
        current[0] += 11_000_000
        # Actor measurements are all younger than30ms, but the latest physical
        # ankle feedback is13ms old and must still stop execution.
        with pytest.raises(RuntimeError, match='stale'):
            (backend.write if actuate else backend.preview)(target)
    finally:
        backend.close()
        peer.close()
        imu_peer.close()
