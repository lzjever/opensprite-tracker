import copy
import json
from pathlib import Path
import numpy as np
import pytest
from opensprite_sdk.constants import JOINT_NAMES,JOINT_ORDER_HASH
from opensprite_sdk.backends.hardware_config import HardwareConfig

ROOT=Path(__file__).resolve().parents[1]


def measured_fixture():
    """Synthetic CPU fixture; never an individual robot calibration."""
    d=json.loads((ROOT/'config/hardware.jetson.example.json').read_text())
    d.update(configured=True,hardware_identity='SYNTHETIC-CPU-ONLY',synthetic=True)
    d['estop'].update(configured=True,hardware_chain='synthetic_offline')
    d['can_adapter']['interfaces']=['vcan0','vcan1','vcan2','vcan3']
    d['imu'].update(mounting_verified=True,socket_path='/tmp/fake-imu.sock',sensor_to_body_rotation=np.eye(3).tolist())
    for record in d['motor_map'].values():
        record.update(firmware='synthetic',motor_zero_rad=.01,encoder_sign=1,reduction_ratio=1.,linkage_ratio=1.,soft_limit_rad=[-8.,8.],hard_limit_rad=[-10.,10.],rated_torque_nm=3.,peak_torque_nm=40.,max_speed_rad_s=20.,rated_current_a=1.,peak_current_a=5.,temperature_limit_c=80.,commissioning_torque_cap_nm=5.,feedback_torque_cap_nm=10.)
        record['mit_ranges']={'position_rad':[-12.5,12.5],'velocity_rad_s':[-20.,20.],'kp':[0.,500.],'kd':[0.,5.],'torque_nm':[-40.,40.],'source':'motor_register_readback'}
    for label,p in d['differentials'].items():
        p.update(calibrated=True,joint_to_motor_matrix=[[1.,1.],[1.,-1.]],motor_zero_rad=[.02,-.02],source='synthetic CPU fixture only')
    d['joint_limits']={name:{'position_rad':[-3.,3.],'velocity_rad_s':10.,'kp':100.,'kd':3.,'effort_nm':40.} for name in JOINT_NAMES}
    return d


def test_placeholder_cannot_arm():
    with pytest.raises(ValueError,match='completed configuration'):
        HardwareConfig.load(ROOT/'config/hardware.jetson.example.json')


def test_coupled_head_and_ankles_roundtrip_and_power():
    c=HardwareConfig.from_dict(measured_fixture());rng=np.random.default_rng(2)
    for _ in range(30):
        q,dq,tau=rng.normal(size=(3,31))
        motor_q,motor_dq=c.motor_state(q,dq);qq,dd=c.joint_state(motor_q,motor_dq)
        np.testing.assert_allclose(qq,q,atol=1e-12);np.testing.assert_allclose(dd,dq,atol=1e-12)
        np.testing.assert_allclose(c.joint_to_motor_torque(tau)@motor_dq,tau@dq,atol=1e-12)
    assert c.motor_order_hash!=JOINT_ORDER_HASH


@pytest.mark.parametrize('mutation',[
    lambda d:d['motor_map'][next(iter(d['motor_map']))].update(encoder_sign=None),
    lambda d:d['differentials']['head'].update(calibrated=False),
    lambda d:d['differentials']['right_ankle'].update(joint_to_motor_matrix=[[1.,1.],[1.,1.]]),
    lambda d:d['imu'].update(sensor_to_body_rotation=np.ones((3,3)).tolist()),
    lambda d:d['motor_map'][next(iter(d['motor_map']))]['mit_ranges'].update(source='urdf_guess'),
    lambda d:d['motor_map'][next(iter(d['motor_map']))].update(temperature_limit_c=None),
    lambda d:d['imu'].update(max_age_ms=31),
    lambda d:d['controller'].update(native_state_max_age_ms=61),
    lambda d:d['controller'].update(native_target_timeout_ms=101),
])
def test_missing_calibration_rejected(mutation):
    d=measured_fixture();mutation(d)
    with pytest.raises((ValueError,TypeError)):HardwareConfig.from_dict(d)


def test_native_tsv_and_explicit_launch(tmp_path):
    c=HardwareConfig.from_dict(measured_fixture());files=c.generate(tmp_path)
    assert all(Path(p).exists() for p in files.values())
    rows=(tmp_path/'kinematics.tsv').read_text().splitlines()
    assert len(rows)==63
    assert '--policy-actuation' not in c.launch_command(tmp_path)
    assert '--ipc-socket' in c.launch_command(tmp_path)
    with pytest.raises(ValueError,match='synthetic CPU'):
        c.launch_command(tmp_path,actuate=True)
