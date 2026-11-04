"""Management wire semantics and bounded strings across all transports."""
import copy

import pytest

from opensprite_sdk.messages import (
    ControlRequest_, ControlReply_, LowState_, message_to_dict, message_from_dict,
)


@pytest.mark.parametrize("message", [
    ControlRequest_(request_id="request", controller_session="controller",
                    robot_session="robot", operation="claim"),
    ControlReply_(request_id="request", controller_session="controller",
                  robot_session="robot", accepted=True, execution_verified=False),
])
def test_management_roundtrip_preserves_execution_confirmation(message):
    envelope = message_to_dict(message)
    assert message_to_dict(message_from_dict(envelope)) == envelope
    assert "joint_state" not in envelope["data"]


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("controller_session", ""), ("robot_session", ""),
    ("operation", "reset"), ("request_id", "a\0b"),
    ("controller_session", "舞"*43), ("robot_session", "\ud800"),
])
def test_management_identity_and_operation_rejected(field, value):
    request = ControlRequest_(request_id="request", controller_session="controller",
                              robot_session="robot")
    setattr(request, field, value)
    with pytest.raises(ValueError):
        message_to_dict(request)


def test_reply_does_not_claim_execution_for_rejected_request():
    reply = ControlReply_(request_id="request", controller_session="controller",
                          robot_session="robot", execution_verified=True)
    with pytest.raises(ValueError):
        reply.validate()
    reply.execution_verified = False
    reply.validate()


def test_utf8_wire_bounds_not_character_counts():
    state = LowState_()
    for mutate in (
        lambda: setattr(state, "profile_id", "舞"*43),
        lambda: setattr(state.imu_state, "attached_link", "舞"*43),
        lambda: setattr(state, "faults", ["舞"*342]),
    ):
        old = copy.deepcopy(state)
        mutate()
        with pytest.raises(ValueError):
            state.validate()
        state = old
    state.profile_id = "舞"*42
    state.validate()


def test_management_unknown_fields_rejected():
    request = ControlRequest_(request_id="request", controller_session="controller",
                              robot_session="robot")
    envelope = message_to_dict(request)
    envelope["data"]["enable_now"] = True
    with pytest.raises(ValueError):
        message_from_dict(envelope)
