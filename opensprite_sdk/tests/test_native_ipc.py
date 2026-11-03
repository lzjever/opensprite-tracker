import math
import struct
import numpy as np
import pytest
from opensprite_sdk.backends.native_ipc import StatePacket,STATE2,TARGET1,pack_target,decode_icm
from opensprite_sdk.messages import LowCmd_

NOW=1_000_000_000


def state_packet(**kw):
    d=dict(sequence=1,stamp_ns=NOW-1_000_000,motor_order_hash=123,position=(.1,)*31,velocity=(.2,)*31,host_receive_ns=(NOW-2_000_000,)*31);d.update(kw)
    return StatePacket(**d)


def imu_packet(now=NOW,sequence=1):
    p=bytearray(192);p[:8]=b'SPIMU003';struct.pack_into('<QQQ',p,8,sequence,now-2_000_000,now-1_000_000)
    struct.pack_into('<6h',p,32,0,0,8192,0,0,0)
    struct.pack_into('<3d',p,56,0.,0.,0.)
    struct.pack_into('<13d',p,80,0.,0.,9.80665,0.,0.,0.,1.,0.,0.,0.,0.,0.,9.80665)
    return bytes(p)


def test_packet_layout_and_roundtrip():
    assert STATE2.size==776 and TARGET1.size==1280
    a=state_packet();b=StatePacket.unpack(a.pack(),expected_hash=123,previous_sequence=0,now=NOW)
    assert a==b
    command=LowCmd_();command.joint_cmd[2].q=.25;command.joint_cmd[2].tau=.3
    p=pack_target(command,NOW,456,8,1);header=TARGET1.unpack(p)
    assert header[:8]==(b'SPRT',1,2,8,NOW,456,1,.0)
    assert header[7+2]==.25 and header[7+124+2]==.3


@pytest.mark.parametrize('kw,expected',[
    ({'sequence':0},ValueError),({'motor_order_hash':124},ValueError),
    ({'stamp_ns':NOW+1},RuntimeError),({'host_receive_ns':(NOW-61_000_000,)*31},RuntimeError),
    ({'host_receive_ns':(NOW,)*31},ValueError),({'position':(float('nan'),)*31},ValueError)
])
def test_native_rejects_stale_replay_order_nonfinite(kw,expected):
    with pytest.raises(expected):StatePacket.unpack(state_packet(**kw).pack(),expected_hash=123,previous_sequence=0,now=NOW)


def test_native_v1_not_silently_accepted():
    with pytest.raises(ValueError,match='legacy ABI1'):
        StatePacket.unpack(bytes(528),expected_hash=123,previous_sequence=0,now=NOW)


def test_icm_raw_bias_rotation_and_gravity():
    r=np.array([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])
    imu=decode_icm(imu_packet(),now=NOW,rotation=r)
    np.testing.assert_allclose(imu.projected_gravity,[1,0,0]);assert imu.sample_ns==NOW-2_000_000
    corrupt=bytearray(imu_packet());struct.pack_into('<d',corrupt,80,.01)
    with pytest.raises(ValueError,match='raw/bias'):decode_icm(corrupt,now=NOW,rotation=r)
    with pytest.raises(RuntimeError,match='stale'):decode_icm(imu_packet(NOW-40_000_000),now=NOW,rotation=r)
