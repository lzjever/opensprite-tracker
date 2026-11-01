"""Focused checks for the copied SuperMimic core and Sprite-specific adapter."""

import numpy as np
import pytest
import torch
from opensprite_tracker.robot import BODY_JOINT_NAMES, JOINT_CONTROL, TRACKED_BODY_NAMES
from opensprite_tracker.task import ACTUATORS, FilteredIdealPdActuator
from opensprite_tracker.supermimic.robot import CONTEXT_DIM, MEM_DIM
from opensprite_tracker.supermimic.networks import Policy, WorldModel
from opensprite_tracker.supermimic.state import flatten_state, unflatten_state


def test_paper_policy_architecture_uses_sprite_dimensions():
    actor = Policy(obs_dim=105, goal_dim=770, action_dim=31)
    assert [m.out_features for m in actor.enc if isinstance(m, torch.nn.Linear)] == [
        2048,
        1024,
        512,
        256,
        32,
    ]
    assert [m.out_features for m in actor.dec if isinstance(m, torch.nn.Linear)] == [
        2048,
        1024,
        512,
        256,
        31,
    ]
    out = actor(torch.zeros(2, 10, 105), torch.zeros(2, 770))
    assert out.shape == (2, 31) and torch.isfinite(out).all()
    assert len(BODY_JOINT_NAMES) == 31 and len(TRACKED_BODY_NAMES) == 32
    assert CONTEXT_DIM == 221 and MEM_DIM == 66


def test_real_one_physics_tick_delay_and_lowpass_torque():
    from mjlab.actuator.actuator import ActuatorCmd
    from mjlab.utils.buffers.delay_buffer import DelayBuffer

    cfg = ACTUATORS[0]
    assert (
        cfg.delay_min_lag,
        cfg.delay_max_lag,
        cfg.delay_update_period,
        cfg.delay_hold_prob,
    ) == (0, 1, 4, 0.9)
    assert cfg.torque_time_constant_range == (0.005, 0.010)
    values = [JOINT_CONTROL[n] for n in BODY_JOINT_NAMES]
    actuator = FilteredIdealPdActuator.__new__(FilteredIdealPdActuator)
    actuator.cfg = cfg
    actuator.stiffness = torch.tensor([[c.kp for c in values]])
    actuator.damping = torch.tensor([[c.kd for c in values]])
    actuator.force_limit = torch.tensor([[c.command_effort_nm for c in values]])
    for name in (
        "filtered_effort",
        "sent_target",
        "speed_peak",
        "raw_peak",
        "applied_peak",
    ):
        setattr(actuator, name, torch.zeros(1, 31))
    alpha = 1 - np.exp(-0.005 / 0.010)
    actuator.filter_alpha = torch.full((1, 31), alpha)
    actuator.filter_initialized = torch.zeros(1, dtype=torch.bool)
    actuator._delay_buffer = DelayBuffer(
        min_lag=1,
        max_lag=1,
        batch_size=1,
        hold_prob=1.0,
        update_period=4,
        per_env_phase=False,
    )
    actuator._delay_buffer.set_lags(torch.ones(1, dtype=torch.long))
    previous = torch.zeros(1, 31)
    for tick, target in enumerate((0.0, 0.1, 0.1, 0.1)):
        command = ActuatorCmd(
            position_target=torch.full((1, 31), target),
            velocity_target=torch.zeros(1, 31),
            effort_target=torch.zeros(1, 31),
            pos=torch.zeros(1, 31),
            vel=torch.zeros(1, 31),
        )
        delayed = actuator.apply_delay(command)
        expected_target = 0.0 if tick < 2 else 0.1
        assert torch.allclose(
            delayed.position_target, torch.full((1, 31), expected_target)
        )
        requested = torch.clamp(
            actuator.stiffness * expected_target,
            -actuator.force_limit,
            actuator.force_limit,
        )
        expected = previous + alpha * (requested - previous)
        actual = actuator.compute(delayed)
        torch.testing.assert_close(actual, expected)
        previous = actual.clone()
    assert actuator._delay_buffer._step_count.item() == 4
    assert (actuator.applied_peak <= actuator.force_limit).all()


def test_world_model_preserves_and_supervises_delay_memory():
    model = WorldModel(
        hidden=(64, 64, 64), num_joints=31, context_dim=221, memory_dim=66
    )
    zero = torch.zeros(2, 3)
    q = torch.zeros(2, 31)
    h = torch.randn(2, 66)
    h[:, 62:] = torch.tensor([1.0, 1 / 3, 1.0, 1.0])
    state = (zero, torch.eye(3).expand(2, 3, 3), zero, zero, q, q, h)
    restored = unflatten_state(flatten_state(state))
    torch.testing.assert_close(restored[6], h)
    features = model.inputs(state, torch.zeros(2, 31), torch.zeros(2, 221))
    torch.testing.assert_close(features[:, 75:141], h)
    new = (*state[:6], h + 0.01)
    labels = model.targets(state, new)
    torch.testing.assert_close(labels[:, -66:], torch.full((2, 66), 0.01))


def test_formal_motion_identity_rejects_prototype_before_gpu(tmp_path):
    from opensprite_tracker.motion_contract import validate_motion_contract

    path = tmp_path / "motions.npz"
    np.savez(path, fps=50.0, joint_names=np.array(["joint"] * 31))
    with pytest.raises(ValueError, match="retarget_method"):
        validate_motion_contract(path, "umr-v1")
    assert validate_motion_contract(path, "", allow_diagnostic_motion=True)[
        "diagnostic"
    ]
    np.savez(
        path,
        retarget_method="umr_native_nr_surface_vector_v1",
        robot_model="opensprite0825",
        dataset_revision="umr-v1",
        correspondence_names=np.array(["geno", "opensprite0825"]),
    )
    assert not validate_motion_contract(path, "umr-v1")["diagnostic"]
    with pytest.raises(ValueError, match="dataset_revision"):
        validate_motion_contract(path, "wrong")
