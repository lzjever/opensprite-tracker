"""Checkpoint routing and cold-start boundaries between the two training recipes."""

import pytest
import torch

from opensprite_tracker.supermimic.evaluate import load_policy
from opensprite_tracker.supermimic.networks import Critic, GoalCritic, Policy, WorldModel
from opensprite_tracker.supermimic.robot import CONTEXT_DIM, MEM_DIM, physical_contract
from opensprite_tracker.supermimic.train import parser
from opensprite_tracker.supermimic.training_variants import (
    GOAL_SIM, GOAL_SIM2REAL, checkpoint_variant, observation_contract,
    validate_variant_load,
)
from opensprite_tracker.supermimic.validate_model import load_models


def saved_model(variant, *, legacy=False):
    dim = 105 if variant == GOAL_SIM else 99
    actor = Policy(dim, 77, hist=2, hidden=(8, 4))
    critic = Critic(90, hidden=(8, 4)) if variant == GOAL_SIM else GoalCritic(90, 77, (8, 4))
    world = WorldModel((8, 8, 8), context_dim=CONTEXT_DIM, memory_dim=MEM_DIM)
    result = {
        "pi": actor.state_dict(), "critic": critic.state_dict(), "wm": world.state_dict(),
        "args": {"hist": 2, "goal_frames": 1, "motion": "data/motions/lafan_dance.npz",
                 "dataset_revision": "opensprite-umr-geno-v1-20261031", "disable_torque_lag": True},
        "method": "SuperMimic-OpenSprite-mjlab", "robot_contract": physical_contract(torque_lag_enabled=False),
        "torque_lag_enabled": False, "iteration": 17,
    }
    if not legacy:
        result.update(training_variant=variant, observation_contract=observation_contract(variant))
        result["args"]["training_variant"] = variant
    return result


@pytest.mark.parametrize("variant,legacy", [(GOAL_SIM, True), (GOAL_SIM, False), (GOAL_SIM2REAL, False)])
def test_policy_and_world_model_loaders_route_saved_recipe(tmp_path, variant, legacy):
    model = saved_model(variant, legacy=legacy)
    path = tmp_path / "checkpoint.pt"
    torch.save(model, path)
    policy, cfg, iteration, frames = load_policy(path)
    diagnostic_policy, world, diagnostic_cfg, diagnostic_iteration, _ = load_models(path, "cpu")
    assert cfg["training_variant"] == diagnostic_cfg["training_variant"] == variant
    assert iteration == diagnostic_iteration == 17 and frames == 1
    assert policy.obs_dim == observation_contract(variant)["actor_obs_dim"]
    sensor = torch.randn(2, 2, policy.obs_dim)
    goal = torch.randn(2, 77)
    torch.testing.assert_close(policy(sensor, goal), diagnostic_policy(sensor, goal), rtol=0, atol=0)
    assert not next(world.parameters()).requires_grad


@pytest.mark.parametrize("source,target", [(GOAL_SIM, GOAL_SIM2REAL), (GOAL_SIM2REAL, GOAL_SIM)])
def test_loading_across_recipes_fails_before_any_module_is_changed(source, target):
    with pytest.raises(ValueError, match="fresh cold-start"):
        validate_variant_load(saved_model(source), target)


@pytest.mark.parametrize("tamper", ["label", "obs", "contract", "missing_contract", "critic", "goal"])
def test_inconsistent_recipe_metadata_or_networks_are_rejected(tamper):
    ck = saved_model(GOAL_SIM2REAL)
    if tamper == "label": ck["training_variant"] = GOAL_SIM
    if tamper == "obs": ck["pi"]["obs_norm.mean"] = torch.zeros(105)
    if tamper == "contract": ck["observation_contract"]["actor_obs_dim"] = 105
    if tamper == "missing_contract": ck.pop("observation_contract")
    if tamper == "critic": ck["critic"] = Critic(90).state_dict()
    if tamper == "goal": ck["critic"]["enc.0.weight"] = torch.zeros(8, 78)
    with pytest.raises(ValueError): checkpoint_variant(ck)


def test_an_unlabelled_99_observation_checkpoint_is_not_guessed_as_legacy():
    ck = saved_model(GOAL_SIM2REAL)
    ck.pop("training_variant")
    ck.pop("observation_contract")
    ck["args"].pop("training_variant")
    with pytest.raises(ValueError, match="actor observations"):
        checkpoint_variant(ck)


def test_cli_defaults_preserve_old_recipe_and_new_choice_is_cold():
    old = parser().parse_args(["--output", "unused"])
    new = parser().parse_args(["--output", "unused", "--training-variant", GOAL_SIM2REAL])
    assert old.training_variant == GOAL_SIM and new.training_variant == GOAL_SIM2REAL
    assert new.resume is new.initialize_from is new.world_model_from is None
    assert new.wm_only_iters == 500
