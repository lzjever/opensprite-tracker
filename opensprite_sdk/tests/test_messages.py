import json
import math
import pytest
from opensprite_sdk.constants import JOINT_NAMES, JOINT_COUNT, JOINT_ORDER_HASH, ordered_name_hash
from opensprite_sdk.messages import (LowCmd_, LowState_, JointCmd, JointState, IMUState,
                                     message_to_dict, message_from_dict)


def test_joint_order_and_aliases_are_joint_coordinates():
    assert len(JOINT_NAMES) == JOINT_COUNT == 31
    assert JOINT_ORDER_HASH == 17325335851961314422
    assert ordered_name_hash(reversed(JOINT_NAMES)) != JOINT_ORDER_HASH
    command = LowCmd_()
    command.motor_cmd[0].q = .3
    assert command.joint_cmd[0].q == .3
    assert LowCmd_().joint_cmd[0].q == 0
    state = LowState_()
    assert state.motor_state is state.joint_state
    assert not hasattr(state, "global_position")
    assert not hasattr(state.imu_state, "quaternion")


@pytest.mark.parametrize("cls", [LowCmd_, LowState_])
def test_json_roundtrip(cls):
    original = cls(seq=21, session_id="test", stamp_ns=123, clock_domain="sim")
    original.validate()
    restored = message_from_dict(json.loads(json.dumps(message_to_dict(original))))
    assert message_to_dict(restored) == message_to_dict(original)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1", 10**1000])
def test_nonfinite_or_non_numeric_command_rejected(value):
    command = LowCmd_()
    command.joint_cmd[4].tau = value
    with pytest.raises(ValueError):
        command.validate()


@pytest.mark.parametrize("field,value", [("seq", True), ("seq", -1), ("stamp_ns", 1.5),
    ("joint_order_hash", 3), ("clock_domain", "wall"), ("profile_id", "")])
def test_header_validation(field, value):
    command = LowCmd_()
    setattr(command, field, value)
    with pytest.raises(ValueError):
        command.validate()


def test_shape_gains_mode_and_nested_types():
    for mutate in (lambda c: c.joint_cmd.pop(), lambda c: setattr(c.joint_cmd[0], "kp", -1),
                   lambda c: setattr(c.joint_cmd[0], "mode", 2),
                   lambda c: c.joint_cmd.__setitem__(0, {})):
        command = LowCmd_()
        mutate(command)
        with pytest.raises(ValueError):
            command.validate()


def test_sensor_timestamp_kind_and_frame_validation():
    state = LowState_()
    state.imu_state.frame_id = "waist_yaw_link"
    assert state.imu_state.attached_link == "waist_yaw_link"
    for kind in ("simulation", "host_receive", "unknown"):
        JointState(sample_time_kind=kind).validate()
    for kind in ("simulation", "host_read", "unknown"):
        IMUState(sample_time_kind=kind).validate()
    with pytest.raises(ValueError):
        JointState(sample_time_kind="host_read").validate()
    with pytest.raises(ValueError):
        IMUState(sample_time_kind="host_receive").validate()
    with pytest.raises(ValueError):
        IMUState(frame="world").validate()
    assert JointState(valid=True).tau_est_valid is False
    with pytest.raises(ValueError):
        JointState(tau_est_valid=1).validate()


def test_malformed_wire_envelope_rejected():
    for value in ({}, [], {"schema": 2, "type": "LowCmd_", "data": {}},
                  {"schema": 1, "type": [], "data": {}},
                  {"schema": 1, "type": "LowCmd_", "data": {}}):
        with pytest.raises(ValueError):
            message_from_dict(value)
    encoded = message_to_dict(LowCmd_())
    encoded["data"]["unexpected"] = 1
    with pytest.raises(ValueError):
        message_from_dict(encoded)


def test_source_session_is_preserved_and_bounded():
    command = LowCmd_(source_state_seq=4, source_state_session_id="state-epoch")
    restored = message_from_dict(message_to_dict(command))
    assert restored.source_state_session_id == "state-epoch"
    assert restored.source_state_seq == 4
    for value in (False, [], "x" * 129):
        command.source_state_session_id = value
        with pytest.raises(ValueError):
            command.validate()
