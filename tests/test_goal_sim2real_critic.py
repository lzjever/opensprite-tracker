"""CPU contracts for the privileged goal-encoder value network."""

import pytest
import torch
from torch import nn

from opensprite_tracker.supermimic import engine
from opensprite_tracker.supermimic.engine import Learner
from opensprite_tracker.supermimic.networks import GoalCritic, Policy, RunningNorm

from test_engine_microbatch import assert_modules_equal, make_learner


def linear_shapes(module):
    return [(layer.in_features, layer.out_features)
            for layer in module if isinstance(layer, nn.Linear)]


def test_production_structure_parameter_count_and_independent_actor():
    critic = GoalCritic(1268)
    actor = Policy(obs_dim=99, hidden=(1024, 512, 256))
    assert critic.privileged_dim == 498
    assert linear_shapes(critic.enc) == linear_shapes(actor.enc) == [
        (770, 1024), (1024, 512), (512, 256), (256, 32)]
    assert linear_shapes(critic.dec) == [
        (530, 1024), (1024, 512), (512, 256), (256, 1)]
    assert sum(p.numel() for p in critic.parameters()) == 2_653_985
    assert sum(isinstance(m, nn.ELU) for m in critic.modules()) == 6
    assert not ({p.data_ptr() for p in critic.parameters()} &
                {p.data_ptr() for p in actor.parameters()})
    actor_goal_mean = actor.goal_norm.mean.clone()
    critic.norm.update(torch.randn(7, 1268))
    torch.testing.assert_close(actor.goal_norm.mean, actor_goal_mean, rtol=0, atol=0)
    assert actor.goal_norm.count == 0


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_forward_shapes_and_explicit_raw_input_normalization(leading):
    torch.manual_seed(101)
    critic = GoalCritic(8, goal_dim=3, hidden=(6, 4), z_dim=2)
    critic.norm.update(torch.randn(7, 8) * 2 + 3)
    x = torch.randn(*leading, 8)
    normalized = (x - critic.norm.mean) / critic.norm.std()
    expected = critic.dec(torch.cat((normalized[..., :5],
        torch.tanh(critic.enc(normalized[..., 5:]))), -1)).squeeze(-1)
    actual = critic(x)
    assert actual.shape == leading
    torch.testing.assert_close(actual, expected)
    assert critic.norm.count == 7  # Forward calls never update statistics.


@pytest.mark.parametrize("kwargs", [
    {"in_dim": 770}, {"in_dim": 769}, {"in_dim": 8, "goal_dim": 0},
    {"in_dim": 8, "goal_dim": 3, "z_dim": 0},
])
def test_invalid_feature_layouts_fail_early(kwargs):
    with pytest.raises(ValueError):
        GoalCritic(**kwargs)


def test_frozen_critic_preserves_privileged_and_goal_input_derivatives():
    torch.manual_seed(102)
    critic = GoalCritic(8, goal_dim=3, hidden=(6, 4), z_dim=2).double()
    critic.requires_grad_(False)
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(critic, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
    critic(x).sum().backward()
    assert x.grad[..., :5].abs().sum() > 0
    assert x.grad[..., 5:].abs().sum() > 0
    assert all(p.grad is None for p in critic.parameters())
    critic.requires_grad_(True)
    critic(x.detach()).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in critic.parameters())
    assert sum(p.grad.abs().sum() for p in critic.enc.parameters()) > 0


def test_full_logical_norm_equals_independent_raw_feature_normalizers():
    torch.manual_seed(103)
    critic = GoalCritic(8, goal_dim=3, hidden=(6, 4), z_dim=2)
    privileged, goal = RunningNorm(5), RunningNorm(3)
    for batch in (torch.randn(7, 8), torch.randn(4, 9, 8) * 2 + 1):
        Learner._update_full_norm(critic.norm, batch)
        privileged.update(batch[..., :5])
        goal.update(batch[..., 5:])
    probe = torch.randn(2, 8)
    torch.testing.assert_close(critic.norm(probe),
        torch.cat((privileged(probe[..., :5]), goal(probe[..., 5:])), -1))
    assert critic.norm.count == privileged.count == goal.count == 43


def goal_learner(micro):
    learner = make_learner(micro)
    torch.manual_seed(104)
    learner.critic = GoalCritic(15, goal_dim=3, hidden=(12, 8), z_dim=2)
    learner.critic.norm.update(torch.randn(11, 15))
    learner.opt_v = torch.optim.Adam(learner.critic.parameters(), lr=learner.args.lr)
    original = learner.opt_v.step

    def step(*args, **kwargs):
        learner.steps["v"] += 1
        return original(*args, **kwargs)

    learner.opt_v.step = step
    return learner


def test_existing_engine_microbatches_keep_fixed_targets_norm_and_tail_gradients(monkeypatch):
    whole, partitioned = goal_learner(False), goal_learner(True)
    targets = []
    original = engine.lambda_returns

    def record(*args):
        result = original(*args)
        targets.append(result.clone())
        return result

    monkeypatch.setattr(engine, "lambda_returns", record)
    torch.manual_seed(105)
    expected = whole.policy_update()
    torch.manual_seed(105)
    actual = partitioned.policy_update()
    assert len(targets) == 2
    torch.testing.assert_close(targets[0], targets[1], rtol=2e-5, atol=2e-6)
    assert whole.steps == partitioned.steps == {"wm": 0, "pi": 1, "v": 20}
    assert partitioned.critic.norm.count == 11 + (3 + 1) * 9
    assert actual["actor_grad_norm"] > 0
    for module in ("pi", "critic", "wm"):
        assert_modules_equal(getattr(whole, module), getattr(partitioned, module))
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=2e-5, atol=2e-6)


def test_engine_terminal_value_alone_transmits_gradient_to_actor(monkeypatch):
    learner = goal_learner(False)
    # Eliminate immediate reward and survival gradients to isolate the tail.
    monkeypatch.setattr(learner, "reward_term", lambda state, frame, action, previous:
        (torch.zeros_like(frame, dtype=torch.float32),
         torch.zeros_like(frame, dtype=torch.float32)))
    learner.wm.requires_grad_(False)
    learner.critic.requires_grad_(False)
    start, history = learner.sample_valid_starts()
    objective, *_ = learner._policy_rollout(start, history)
    objective.sum().backward()
    assert sum(p.grad.abs().sum() for p in learner.pi.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in learner.critic.parameters())
    assert all(p.grad is None for p in learner.wm.parameters())
