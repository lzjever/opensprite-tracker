import socket
import threading
import numpy as np
import pytest
from opensprite_sdk.backends.hardware import HardwareBackend
from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.backends.native_ipc import StatePacket,TARGET1
from opensprite_sdk.messages import LowCmd_
from test_hardware_config import measured_fixture
from test_native_ipc import imu_packet,NOW


def fake_backend(allow_actuation=True):
    config=HardwareConfig.from_dict(measured_fixture())
    local,peer=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
    ilocal,ipeer=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
    backend=HardwareBackend(config,allow_actuation=allow_actuation,clock=lambda:NOW,state_socket=local,imu_socket=ilocal)
    backend.start()
    return backend,peer,ipeer


def send_state(backend,peer,imu_peer,*,seq=1,imu_stamp=NOW,actuation_allowed=True):
    q,dq=backend.config.motor_state(np.zeros(31),np.zeros(31))
    peer.send(StatePacket(seq,NOW-1_000_000,backend.config.motor_order_hash,tuple(q),tuple(dq),(NOW-2_000_000,)*31,actuation_allowed).pack())
    def respond():
        assert imu_peer.recv(8)==b'GETIMU01';imu_peer.send(imu_packet(imu_stamp,sequence=seq))
    t=threading.Thread(target=respond);t.start()
    try:return backend.read()
    finally:t.join(timeout=1)


def test_fake_native_full_joint_target_exchange():
    backend,peer,imu_peer=fake_backend()
    try:
        state=send_state(backend,peer,imu_peer)
        np.testing.assert_allclose([x.q for x in state.joint_state],0.,atol=1e-14)
        assert state.imu_state.attached_link=='pelvis_link'
        assert state.joint_state[0].sample_time_kind=='host_receive'
        assert state.imu_state.sample_time_kind=='host_read'
        backend.enable();startup=TARGET1.unpack(peer.recv(1281))
        assert startup[:3]==(b'SPRT',1,2)
        assert max(abs(v) for v in startup[7+62:])==0
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,
                       source_state_seq=state.seq,source_state_session_id=state.session_id,session_id='controller-1')
        for target in command.joint_cmd:target.mode=1;target.kp=2.;target.kd=.2
        command.joint_cmd[4].q=.1;command.joint_cmd[4].tau=.3
        backend.write(command);packet=TARGET1.unpack(peer.recv(1281))
        assert packet[3]==2 and packet[6]==state.seq
        assert packet[7+4]==.1 and packet[7+124+4]==.3
        assert state.status=='ready' # Physical enable status is not fabricated.
        with pytest.raises(ValueError,match='sequence'):backend.write(command)
    finally:backend.close();peer.close();imu_peer.close()


def test_enable_required_and_source_session():
    backend,peer,imu_peer=fake_backend()
    try:
        state=send_state(backend,peer,imu_peer)
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,source_state_seq=1,
                       source_state_session_id=state.session_id,session_id='controller-1')
        for target in command.joint_cmd:target.mode=1
        with pytest.raises(RuntimeError,match='enable'):backend.write(command)
        backend.enable();peer.recv(1281)
        command.source_state_session_id='old-process-session'
        with pytest.raises(ValueError,match='session'):backend.write(command)
        command.joint_cmd[0].mode=0
        with pytest.raises(ValueError,match='mixed'):backend.write(command)
    finally:backend.close();peer.close();imu_peer.close()


def test_frozen_imu_disconnects_native_peer():
    backend,peer,imu_peer=fake_backend()
    try:
        with pytest.raises(RuntimeError,match='stale'):
            send_state(backend,peer,imu_peer,imu_stamp=NOW-40_000_000)
        assert peer.recv(1)==b''
        with pytest.raises(RuntimeError,match='unavailable'):backend.read()
    finally:backend.close();peer.close();imu_peer.close()


def test_controller_owner_applies_to_disable_too():
    backend,peer,imu_peer=fake_backend()
    try:
        state=send_state(backend,peer,imu_peer);backend.enable();peer.recv(1281)
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,session_id='owner',
                       source_state_seq=state.seq,source_state_session_id=state.session_id)
        for target in command.joint_cmd:target.mode=1
        backend.write(command);peer.recv(1281)
        command.seq=2;command.session_id='intruder'
        for target in command.joint_cmd:target.mode=0
        with pytest.raises(ValueError,match='owner'):backend.write(command)
        assert not backend._closed
    finally:backend.close();peer.close();imu_peer.close()


def test_cached_motor_imu_freshness_rechecked_at_enable_and_write():
    backend,peer,imu_peer=fake_backend()
    try:
        state=send_state(backend,peer,imu_peer)
        backend.clock=lambda:NOW+11_000_000
        with pytest.raises(RuntimeError,match='stale'):backend.enable()
        backend.clock=lambda:NOW;backend.enable();peer.recv(1281)
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,session_id='owner',
                       source_state_seq=state.seq,source_state_session_id=state.session_id)
        for target in command.joint_cmd:target.mode=1
        backend.clock=lambda:NOW+31_000_000
        with pytest.raises(RuntimeError,match='IMU'):backend.write(command)
    finally:backend.close();peer.close();imu_peer.close()


def test_policy_shadow_sends_full_target_only_to_declared_shadow_daemon():
    backend,peer,imu_peer=fake_backend(allow_actuation=False)
    try:
        state=send_state(backend,peer,imu_peer,actuation_allowed=False)
        assert state.status=='shadow'
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,session_id='shadow-owner',
                       source_state_seq=state.seq,source_state_session_id=state.session_id)
        for target in command.joint_cmd:target.mode=1;target.kp=2;target.kd=.2
        command.joint_cmd[0].q=.2
        with pytest.raises(RuntimeError,match='actuation permission'):backend.enable()
        with pytest.raises(RuntimeError,match='preview'):backend.write(command)
        backend.preview(command)
        target=TARGET1.unpack(peer.recv(1281))
        assert target[7]==.2 and target[7+62]==2
        assert not backend._enabled
    finally:backend.close();peer.close();imu_peer.close()


def test_actuating_daemon_rejects_preview_and_shadow_daemon_rejects_arm_client():
    backend,peer,imu_peer=fake_backend()
    try:
        state=send_state(backend,peer,imu_peer)
        command=LowCmd_(profile_id='physical_v5',seq=1,stamp_ns=NOW,session_id='owner',
                       source_state_seq=state.seq,source_state_session_id=state.session_id)
        with pytest.raises(RuntimeError,match='zero-gain-only'):backend.preview(command)
    finally:backend.close();peer.close();imu_peer.close()
    backend,peer,imu_peer=fake_backend()
    try:
        q,dq=backend.config.motor_state(np.zeros(31),np.zeros(31))
        peer.send(StatePacket(1,NOW-1_000_000,backend.config.motor_order_hash,tuple(q),tuple(dq),(NOW-2_000_000,)*31,False).pack())
        with pytest.raises(RuntimeError,match='shadow daemon'):backend.read()
    finally:backend.close();peer.close();imu_peer.close()


def test_real_socket_wait_covers_native_250ms_disable_verification():
    import time
    backend,peer,imu_peer=fake_backend()
    def native_finishes():
        time.sleep(.25)
        peer.close()
    thread=threading.Thread(target=native_finishes)
    thread.start();began=time.monotonic()
    try:
        with pytest.raises(EOFError,match='native controller closed'):
            backend.read()
        assert time.monotonic()-began>=.23
    finally:thread.join(timeout=1);backend.close();imu_peer.close()
