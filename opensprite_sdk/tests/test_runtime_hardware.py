"""Blocking feedback acquisition must not hold the command delivery lock."""
import threading
import time
import numpy as np
import pytest

from test_hardware_async import environment, snapshot, command, enqueue
from test_native_ipc import imu_packet
from opensprite_sdk.backends.native_ipc import TARGET1, StatePacket
from opensprite_sdk.policy import sensor_observation


def test_waiting_for_native_feedback_does_not_block_command_delivery():
    backend, peer, imu_peer, current = environment(actuate=False)
    reader = None
    try:
        first = snapshot(backend, peer, imu_peer, current, 1)
        result, errors = [], []
        waiting = threading.Event()
        native_socket = backend._state_socket

        class ObservedSocket:
            def recv(self, *args, **kwargs):
                waiting.set()
                return native_socket.recv(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(native_socket, name)

        backend._state_socket = ObservedSocket()

        def receive():
            try:
                result.append(backend.read())
            except BaseException as error:
                errors.append(error)

        reader = threading.Thread(target=receive)
        reader.start()
        assert waiting.wait(timeout=1.)
        # The native packet is intentionally withheld. A writer must still
        # validate and send against the independently cached fresh snapshot.
        began = time.monotonic()
        backend.preview(command(first))
        assert time.monotonic()-began < .1
        packet = TARGET1.unpack(peer.recv(1281))
        assert packet[6] == first.seq
        assert reader.is_alive()
        answer, answer_errors = enqueue(backend, peer, imu_peer, current, 2)
        reader.join(timeout=1.)
        answer.join(timeout=1.)
        assert not reader.is_alive() and not errors and not answer_errors
        assert result[0].seq == 2
    finally:
        backend.close()
        peer.close()
        imu_peer.close()
        if reader is not None:
            reader.join(timeout=1.)


def test_fused_state_robot_clock_includes_newer_imu_without_restamping_samples():
    backend, peer, imu_peer, current = environment(actuate=False)
    try:
        state = snapshot(backend, peer, imu_peer, current, 1,
                         aggregate_age=4_000_000, motor_age=5_000_000)
        assert state.stamp_ns == current[0]
        assert backend._packet.stamp_ns == current[0]-4_000_000
        assert state.imu_state.sample_ns > backend._packet.stamp_ns
        assert state.joint_state[0].sample_ns == current[0]-5_000_000
        assert state.imu_state.sample_ns < state.stamp_ns
        obs = sensor_observation(state, np.zeros(31), np.zeros(31),
                                 imu_link="pelvis_link", now_ns=state.stamp_ns)
        assert obs.shape == (99,) and np.isfinite(obs).all()
    finally:
        backend.close(); peer.close(); imu_peer.close()


def test_slow_imu_collection_cannot_refresh_an_expired_native_packet():
    backend, peer, imu_peer, current = environment(actuate=False)
    responder = None
    try:
        q, dq = backend.config.motor_state(np.zeros(31), np.zeros(31))
        peer.send(StatePacket(1, current[0]-4_000_000, backend.config.motor_order_hash,
            tuple(q), tuple(dq), (current[0]-5_000_000,)*31, False).pack())

        def respond():
            assert imu_peer.recv(8) == b"GETIMU01"
            current[0] += 58_000_000
            imu_peer.send(imu_packet(current[0], sequence=1))

        responder = threading.Thread(target=respond)
        responder.start()
        with pytest.raises(RuntimeError, match="native aggregate state"):
            backend.read()
        assert peer.recv(1) == b""
    finally:
        backend.close(); peer.close(); imu_peer.close()
        if responder is not None:
            responder.join(timeout=1.)
