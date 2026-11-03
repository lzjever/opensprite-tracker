"""Pinned native CAN IPC and ICM45686 packet contracts; no hardware side effects."""
from dataclasses import dataclass
import math
import struct
import numpy as np

STATE2 = struct.Struct('<4sHHQqQ62d31q')
TARGET1 = struct.Struct('<4sHHQqQQ155d')


def fresh(stamp,now,maximum_age_ns,label):
    if type(stamp) is not int or stamp<=0 or stamp>now or now-stamp>maximum_age_ns:
        raise RuntimeError(f'{label}: stale/future timestamp')


@dataclass(frozen=True)
class StatePacket:
    sequence: int
    stamp_ns: int
    motor_order_hash: int
    position: tuple
    velocity: tuple
    host_receive_ns: tuple
    actuation_allowed: bool = True

    def pack(self):
        return STATE2.pack(b'SPRT',2,1 if self.actuation_allowed else 3,self.sequence,self.stamp_ns,self.motor_order_hash,*self.position,*self.velocity,*self.host_receive_ns)

    @classmethod
    def unpack(cls,payload,*,expected_hash,previous_sequence,now,max_age_ns=60_000_000):
        if len(payload)!=STATE2.size:
            raise ValueError('native SDK state ABI2 requires 776 bytes; legacy ABI1 is not deployable')
        magic,version,kind,seq,stamp,order,*values=STATE2.unpack(payload)
        if magic!=b'SPRT' or version!=2 or kind not in (1,3) or order!=expected_hash:
            raise ValueError('native state version/order mismatch')
        if seq<=previous_sequence:
            raise ValueError('native state sequence regression/replay')
        fresh(stamp,now,max_age_ns,'native aggregate state')
        if not all(math.isfinite(x) for x in values[:62]):
            raise ValueError('native state nonfinite')
        for rx in values[62:]:
            fresh(rx,now,max_age_ns,'motor host receive')
            if rx>stamp:
                raise ValueError('motor receive timestamp follows aggregate publication')
        return cls(seq,stamp,order,tuple(values[:31]),tuple(values[31:62]),tuple(values[62:]),kind==1)


def pack_target(command,stamp_ns,joint_hash,sequence,source_sequence):
    values=[]
    for field in ('q','dq','kp','kd','tau'):
        values.extend(getattr(m,field) for m in command.joint_cmd)
    if len(values)!=155 or not all(math.isfinite(x) for x in values):
        raise ValueError('target requires 155 finite values')
    if any(x<0 for x in values[62:124]):
        raise ValueError('target gains must be nonnegative')
    return TARGET1.pack(b'SPRT',1,2,sequence,stamp_ns,joint_hash,source_sequence,*values)


@dataclass(frozen=True)
class ImuPacket:
    sequence: int
    sample_ns: int
    read_end_ns: int
    accelerometer: tuple
    gyro: tuple
    projected_gravity: tuple


def decode_icm(payload,*,now,rotation,previous_sequence=0,max_age_ns=30_000_000):
    if len(payload)!=192 or payload[:8]!=b'SPIMU003':
        raise ValueError('ICM packet requires SPIMU003 and 192 bytes')
    if payload[44:56]!=bytes(12) or payload[184:]!=bytes(8):
        raise ValueError('ICM packet reserved bytes nonzero')
    seq,start,end=struct.unpack_from('<QQQ',payload,8)
    if not seq or seq<previous_sequence:
        raise ValueError('ICM sequence regression')
    fresh(start,now,max_age_ns,'ICM sample')
    if not start<=end<=now:
        raise ValueError('ICM read timestamp ordering')
    bias=np.asarray(struct.unpack_from('<3d',payload,56));v=np.asarray(struct.unpack_from('<13d',payload,80))
    raw=np.asarray(struct.unpack_from('<6h',payload,32))
    if not np.isfinite(v).all() or not np.isfinite(bias).all() or bias@bias>.01:
        raise ValueError('ICM nonfinite values/excess gyro bias')
    expected_accel=raw[:3]*(4*9.80665/32768)
    expected_gyro=raw[3:]*(2000*math.pi/180/32768)-bias
    if not np.allclose(v[:3],expected_accel,rtol=0,atol=1e-12) or not np.allclose(v[3:6],expected_gyro,rtol=0,atol=1e-12):
        raise ValueError('ICM raw/bias disagreement')
    w,x,y,z=v[6:10]
    if abs(w*w+x*x+y*y+z*z-1)>1e-6:
        raise ValueError('ICM quaternion norm')
    up=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
    if not np.allclose(v[10:],9.80665*up,rtol=0,atol=1e-9):
        raise ValueError('ICM gravity/quaternion disagreement')
    rotation=np.asarray(rotation,dtype=float)
    if rotation.shape!=(3,3):
        raise ValueError('ICM sensor-to-body rotation requires 3x3')
    return ImuPacket(seq,start,end,tuple(rotation@v[:3]),tuple(rotation@v[3:6]),tuple(-rotation@up))
