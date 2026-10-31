"""CPU equivalence checks: logical batches survive unequal memory partitions."""

from types import SimpleNamespace

import pytest
import torch

from opensprite_tracker.supermimic import engine
from opensprite_tracker.supermimic.engine import Learner, lambda_returns
from opensprite_tracker.supermimic.networks import Critic, Policy, RunningNorm, WorldModel
from opensprite_tracker.supermimic.state import unflatten_state


class Motion:
    def __init__(self):
        self.clip_id = torch.zeros(128, dtype=torch.long)
        self.clip_end = torch.tensor([127])

    def goal(self, frame, count):
        t = frame.float() * 0.01
        return torch.stack((t, torch.sin(t), torch.cos(t)), -1)


class Mdp:
    reward_scale = 1.0
    uses_transition_metrics = False

    def obs(self, state, action, xi, frame):
        return torch.cat((state[0], state[4][..., :4]), -1)

    def add_noise(self, observation):
        return observation  # Fixed rollouts isolate partition arithmetic.

    def reward(self, state, frame, action, previous):
        return 1 - state[4].square().mean(-1) - 0.02 * action.square().mean(-1)

    def soft_termination(self, state, frame):
        return torch.sigmoid((0.2 - state[0][..., 2]) * 3)

    def critic_features(self, state, frame, xi):
        return torch.cat((state[0], state[2], state[4][..., :4], xi), -1)


def flat_state(*shape):
    value = torch.randn(*shape, 75) * 0.02
    value[..., 2] += 0.5
    value[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return value


def make_learner(micro):
    torch.manual_seed(123)
    args = SimpleNamespace(lr=2e-4, memory_dim=0, compile=False, hist=2,
        horizon=3, goal_frames=1, pi_batch=9, wm_batch=7, wm_window=3,
        gamma=0.99, lam=0.95, critic_epochs=5, critic_minibatches=4,
        wm_microbatch=3 if micro else 0, pi_microbatch=4 if micro else 0,
        critic_microbatch=3 if micro else 0, critic_storage="cpu" if micro else "device")
    wm = WorldModel(hidden=(16, 16, 16), context_dim=2)
    torch.nn.init.normal_(wm.net[-1].weight, std=0.005)
    actor = Policy(obs_dim=7, goal_dim=3, hist=2, hidden=(12, 8))
    critic = Critic(15, hidden=(12, 8))
    critic.norm.update(torch.randn(11, 15))  # Nonzero previous count/variance.
    start = {"s": flat_state(9), "f": torch.arange(9), "xi": torch.randn(9, 2),
             "a_prev": torch.randn(9, 31) * 0.01, "ep_step": torch.arange(9)}
    history = {k: v[:, None].expand(-1, 2, *v.shape[1:]).clone() for k, v in start.items()}
    window = {"s": flat_state(7, 3), "s1": flat_state(7, 3),
              "a": torch.randn(7, 3, 31), "xi": torch.randn(7, 3, 2),
              "mask": torch.tensor([[1., 1, 1], [1, 0, 0], [1, 1, 0],
                                    [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]])}
    buffer = SimpleNamespace(sample_windows=lambda batch, length: window)
    learner = Learner(wm, actor, critic, Mdp(), Motion(), buffer, lambda xi: xi, args)
    learner.start_calls = 0

    def starts():
        learner.start_calls += 1
        return start, history

    learner.sample_valid_starts = starts
    learner.steps = {"wm": 0, "pi": 0, "v": 0}
    for key, optimizer in (("wm", learner.opt_wm), ("pi", learner.opt_pi), ("v", learner.opt_v)):
        original = optimizer.step

        def step(*args, key=key, original=original, **kwargs):
            learner.steps[key] += 1
            return original(*args, **kwargs)

        optimizer.step = step
    return learner


def assert_modules_equal(left, right):
    for (name, a), (_, b) in zip(left.state_dict().items(), right.state_dict().items()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, msg=name)


def test_wm_unequal_microbatches_preserve_mask_denominator_and_one_step():
    whole, partitioned = make_learner(False), make_learner(True)
    expected = whole.wm_update()
    actual = partitioned.wm_update()
    assert whole.steps["wm"] == partitioned.steps["wm"] == 1
    assert_modules_equal(whole.wm, partitioned.wm)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-5)
    for key in whole.last_model_metrics:
        torch.testing.assert_close(whole.last_model_metrics[key], partitioned.last_model_metrics[key])


def test_actor_and_critic_partitions_preserve_logical_mean_fixed_targets_and_20_steps(monkeypatch):
    whole, partitioned = make_learner(False), make_learner(True)
    fixed_targets = []
    original = engine.lambda_returns

    def record(*args):
        target = original(*args)
        fixed_targets.append(target.clone())
        return target

    monkeypatch.setattr(engine, "lambda_returns", record)
    torch.manual_seed(456)
    expected = whole.policy_update()
    torch.manual_seed(456)
    actual = partitioned.policy_update()
    assert whole.start_calls == partitioned.start_calls == 1
    assert whole.steps == partitioned.steps == {"wm": 0, "pi": 1, "v": 20}
    assert len(fixed_targets) == 2  # Once per whole update, not once per epoch/chunk.
    torch.testing.assert_close(fixed_targets[0], fixed_targets[1], rtol=2e-5, atol=2e-6)
    assert_modules_equal(whole.pi, partitioned.pi)
    assert_modules_equal(whole.critic, partitioned.critic)
    assert_modules_equal(whole.wm, partitioned.wm)
    assert partitioned.critic.norm.count == 11 + (3 + 1) * 9
    assert actual["actor_grad_norm"] > 0  # Frozen WM still transmits actor gradients.
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=2e-5, atol=2e-6)


def test_full_batch_norm_and_true_death_bootstrap_semantics():
    torch.manual_seed(23)
    reference, staged = RunningNorm(5), RunningNorm(5)
    first = torch.randn(7, 5)
    reference.update(first)
    staged.load_state_dict(reference.state_dict())
    logical = torch.randn(4, 9, 5)
    reference.update(logical)
    Learner._update_full_norm(staged, logical)
    assert_modules_equal(reference, staged)
    rewards = torch.tensor([[1.], [2.], [3.]])
    deaths = torch.tensor([[0.], [1.], [0.]])
    values = torch.tensor([[5.], [6.], [7.], [8.]])
    targets = lambda_returns(rewards, deaths, values, gamma=0.9, lam=0.95)
    assert targets[1].item() == 2.0
    assert targets[0].item() == pytest.approx(1 + 0.9 * (0.05 * 6 + 0.95 * 2))


def test_final_source_frame_keeps_arrival_reward_but_has_no_tail_bootstrap(monkeypatch):
    learner = make_learner(True)
    # Last sampled start is f=8 and H=3: arrival f=11 is the inclusive end.
    learner.motion.clip_end = torch.tensor([11])
    tail = torch.tensor(37.0, requires_grad=True)
    monkeypatch.setattr(learner.critic, "forward", lambda x: tail.expand(x.shape[:-1]))
    start, history = learner.sample_valid_starts()
    objective, alive_now, _, rewards, deaths, _, _ = learner._policy_rollout(start, history)
    assert deaths[-1, -1].item() == 1
    assert deaths[-1, -2].item() < 1  # Neighboring rollout has not reached end.
    assert rewards[-1, -1].item() > 0
    discount = learner.args.gamma ** torch.arange(learner.args.horizon)
    torch.testing.assert_close(objective[-1], (discount * alive_now[:, -1] * rewards[:, -1]).sum())
    assert torch.autograd.grad(objective[-1], tail)[0].item() == 0
