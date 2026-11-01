"""CPU-only preflight checks for the new no-torque-lag cold-start task."""
from types import SimpleNamespace

import pytest

from opensprite_tracker.supermimic.train import (
    parser, validate_checkpoint_actuator, warmup_boundary_checkpoint,
)


def test_cli_keeps_cold_start_and_five_hundred_wm_only_iterations():
    args = parser().parse_args(["--output", "unused", "--disable-torque-lag",
                               "--policy-hidden", "1024", "512", "256"])
    assert args.disable_torque_lag is True
    assert args.resume is args.initialize_from is args.world_model_from is None
    assert args.wm_only_iters == 500
    assert args.policy_hidden == [1024, 512, 256]
    assert args.wm_width == 2048
    legacy_default = parser().parse_args(["--output", "unused"])
    assert legacy_default.disable_torque_lag is False


@pytest.mark.parametrize("source_disabled,target_disabled", [(False, True), (True, False)])
def test_loading_cannot_switch_actuator_dynamics(source_disabled, target_disabled):
    saved = {"args": {"disable_torque_lag": source_disabled},
             "torque_lag_enabled": not source_disabled}
    with pytest.raises(ValueError, match="fresh cold-start"):
        validate_checkpoint_actuator(saved, SimpleNamespace(disable_torque_lag=target_disabled))


def test_legacy_lag_checkpoint_and_matching_no_lag_checkpoint_remain_loadable():
    validate_checkpoint_actuator({"args": {}}, SimpleNamespace(disable_torque_lag=False))
    validate_checkpoint_actuator(
        {"args": {"disable_torque_lag": True}, "torque_lag_enabled": False},
        SimpleNamespace(disable_torque_lag=True),
    )
    with pytest.raises(ValueError, match="fresh cold-start"):
        validate_checkpoint_actuator({"args": {}}, SimpleNamespace(disable_torque_lag=True))


@pytest.mark.parametrize("saved", [
    {"args": {"disable_torque_lag": "False"}},
    {"args": {"disable_torque_lag": True}, "torque_lag_enabled": True},
])
def test_inconsistent_saved_actuator_contract_is_rejected(saved):
    with pytest.raises(ValueError):
        validate_checkpoint_actuator(saved, SimpleNamespace(disable_torque_lag=True))


@pytest.mark.parametrize("iteration,wm_only,save", [
    (499, 500, False), (500, 500, True), (501, 500, True), (502, 500, False),
    (0, 0, False), (1, 0, False), (1, 1, True), (2, 1, True),
])
def test_checkpoint_boundaries_do_not_depend_on_periodic_save_interval(iteration, wm_only, save):
    assert warmup_boundary_checkpoint(iteration, wm_only) is save
