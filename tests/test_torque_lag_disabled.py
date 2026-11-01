"""CPU checks for zero-lag PD while retaining actual command-delay memory."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCmd
from mjlab.utils.buffers.delay_buffer import DelayBuffer

from opensprite_tracker.robot import BODY_JOINT_NAMES
from opensprite_tracker.task import ACTUATORS, FilteredIdealPdActuator, robot_cfg
from opensprite_tracker.supermimic.env import OpenSpriteTrackingEnv
from opensprite_tracker.supermimic.evaluate import checkpoint_torque_lag_enabled, parser
from opensprite_tracker.supermimic.robot import (
    CONTEXT_DIM, CONTEXT_SLICES, MEM_DIM, context_normalization,
    make_env_cfg, physical_contract,
)


def initialized_cpu_actuator(monkeypatch, enabled):
    # No simulator, Warp allocation, or CUDA initialization is needed to test
    # the actuator's real initialization parameters and control law.
    monkeypatch.setattr(Actuator, "initialize", lambda *a, **kw: None)
    cfg = robot_cfg(torque_lag_enabled=enabled).articulation.actuators[0]
    actuator = FilteredIdealPdActuator(cfg, None, list(range(31)), list(BODY_JOINT_NAMES))
    actuator.initialize(SimpleNamespace(opt=SimpleNamespace(timestep=.005)),
                        None, SimpleNamespace(nworld=1), "cpu")
    actuator._delay_buffer = DelayBuffer(min_lag=1, max_lag=1, batch_size=1,
        hold_prob=1., update_period=4, per_env_phase=False)
    actuator._delay_buffer.set_lags(torch.ones(1, dtype=torch.long))
    return actuator


def test_no_lag_uses_clipped_pd_and_retains_physics_tick_delay_and_memory(monkeypatch):
    actuator = initialized_cpu_actuator(monkeypatch, False)
    assert torch.equal(actuator.torque_time_constant, torch.zeros(1, 31))
    assert torch.equal(actuator.filter_alpha, torch.ones(1, 31))
    fake_env = SimpleNamespace(actuator=actuator, act_ids=list(range(31)),
        default_q=torch.zeros(31), a_prev=torch.zeros(1, 31))
    before = OpenSpriteTrackingEnv.memory(fake_env)
    assert before.shape == (1, MEM_DIM) and MEM_DIM == 66
    assert not before[:, :62].any() and not before[:, 64:].any()
    for tick, target in enumerate((0., .1, .1, -.5, -.5)):
        command = ActuatorCmd(position_target=torch.full((1, 31), target),
            velocity_target=torch.zeros(1, 31), effort_target=torch.zeros(1, 31),
            pos=torch.zeros(1, 31), vel=torch.zeros(1, 31))
        delayed = actuator.apply_delay(command)
        expected_target = (0., 0., .1, .1, -.5)[tick]
        torch.testing.assert_close(delayed.position_target, torch.full((1, 31), expected_target))
        expected = (actuator.stiffness * expected_target).clamp(-actuator.force_limit, actuator.force_limit)
        actual = actuator.compute(delayed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        memory = OpenSpriteTrackingEnv.memory(fake_env)
        torch.testing.assert_close(memory[:, :31], actual/actuator.force_limit)
        assert memory.shape == (1, 66) and memory[:, 64:].eq(1).all()
        assert torch.isfinite(memory).all()
    assert actuator._delay_buffer._step_count.item() == 5
    assert (actuator.applied_peak <= actuator.force_limit).all()


def test_zero_lag_config_context_and_contract_preserve_delay_and_old_default():
    original = ACTUATORS[0]
    disabled = make_env_cfg(1, randomize=False, torque_lag_enabled=False).scene.entities["robot"].articulation.actuators[0]
    assert original.torque_lag_enabled and original.torque_time_constant_range == (.005, .010)
    assert not disabled.torque_lag_enabled and disabled.torque_time_constant_range == (0., 0.)
    for name in ("delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period"):
        assert getattr(disabled, name) == getattr(original, name)
    assert (disabled.delay_min_lag, disabled.delay_max_lag, disabled.delay_hold_prob, disabled.delay_update_period) == (0, 1, .9, 4)
    old, new = physical_contract(), physical_contract(torque_lag_enabled=False)
    assert "torque_lag_enabled" not in old  # Exact old enabled contract remains resumable.
    assert new["torque_lag_enabled"] is False and new["torque_tau_range"] == (0., 0.)
    for key in old.keys()-{"torque_tau_range", "temporal_model"}:
        assert old[key] == new[key]
    off, scale = context_normalization(torque_lag_enabled=False)
    tau = CONTEXT_SLICES["torque_time_constant"]
    assert len(off) == CONTEXT_DIM == 221 and (off[tau] == 0).all() and (scale[tau] == 1).all()
    assert torch.isfinite((torch.zeros(31)-off[tau])/scale[tau]).all()


def test_checkpoint_lag_metadata_and_explicit_eval_assertion():
    assert checkpoint_torque_lag_enabled({"args": {}})
    assert not checkpoint_torque_lag_enabled({"args": {"disable_torque_lag": True}, "torque_lag_enabled": False})
    with pytest.raises(ValueError, match="Contradictory"):
        checkpoint_torque_lag_enabled({"args": {"disable_torque_lag": True}, "torque_lag_enabled": True})
    with pytest.raises(ValueError, match="boolean"):
        checkpoint_torque_lag_enabled({"args": {"disable_torque_lag": "False"}})
    assert parser().parse_args(["--checkpoint", "unused.pt"]).torque_lag is None
    assert parser().parse_args(["--checkpoint", "unused.pt", "--no-torque-lag"]).torque_lag is False
