"""CPU warm initialization is validated completely and cannot load a bad actor."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opensprite_tracker.supermimic.initialization import initialize_world_model, METHOD
from opensprite_tracker.supermimic.networks import RunningNorm


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_norm = RunningNorm(3)
        self.net = nn.Linear(3, 2)

    def forward(self, value):
        return self.net(self.in_norm(value))


def learner(seed, heads=True):
    torch.manual_seed(seed)
    result = SimpleNamespace(wm=TinyModel(), pi=nn.Linear(3, 2), critic=nn.Linear(3, 1),
                             transition_model=TinyModel() if heads else None,
                             sampler=torch.arange(4).float())
    result.opt_wm = torch.optim.Adam(result.wm.parameters(), lr=2e-4)
    result.opt_pi = torch.optim.Adam(result.pi.parameters(), lr=2e-4)
    result.opt_v = torch.optim.Adam(result.critic.parameters(), lr=2e-4)
    result.opt_transition = torch.optim.Adam(result.transition_model.parameters(), lr=2e-4) if heads else None
    return result


def payload(source):
    torch.manual_seed(111)
    for model, optimizer in [(source.wm, source.opt_wm), (source.transition_model, source.opt_transition)]:
        if model is None:
            continue
        for _ in range(3):
            value = torch.randn(5, 3)
            model.in_norm.update(value)
            optimizer.zero_grad()
            model(value).square().mean().backward()
            optimizer.step()
    result = dict(method=METHOD, robot_contract={"robot": "sprite", "memory_dim": 66},
                  dataset_revision="test-umr-v1", args={"full_body_collision": True}, iteration=289,
                  wm=source.wm.state_dict(), opt_wm=source.opt_wm.state_dict(),
                  pi=source.pi.state_dict(), critic=source.critic.state_dict())
    if source.transition_model is not None:
        result.update(transition_model=source.transition_model.state_dict(), opt_transition=source.opt_transition.state_dict())
    return deepcopy(result)


def initialize(target, checkpoint):
    return initialize_world_model(target, checkpoint, robot_contract={"robot": "sprite", "memory_dim": 66},
                                  dataset_revision="test-umr-v1", full_body_collision=True)


def snapshot(target):
    return deepcopy({k: getattr(target, k).state_dict() if hasattr(getattr(target, k), "state_dict") else getattr(target, k)
                     for k in ["wm", "opt_wm", "transition_model", "opt_transition", "pi", "critic", "opt_pi", "opt_v", "sampler"]})


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("heads", [True, False])
def test_loads_only_models_norms_and_adam_and_preserves_source(heads):
    source, target = learner(1, heads), learner(2, heads)
    checkpoint = payload(source)
    source_before, target_before = deepcopy(checkpoint), snapshot(target)
    # Actor/critic payloads are deliberately unusable and must not be read.
    checkpoint["pi"]["weight"].fill_(torch.nan)
    checkpoint["critic"] = {"wrong": torch.tensor(float("inf"))}
    metadata = initialize(target, checkpoint)
    equal(target.wm.state_dict(), checkpoint["wm"])
    equal(target.opt_wm.state_dict(), checkpoint["opt_wm"])
    if heads:
        equal(target.transition_model.state_dict(), checkpoint["transition_model"])
        equal(target.opt_transition.state_dict(), checkpoint["opt_transition"])
    after = snapshot(target)
    for name in ("pi", "critic", "opt_pi", "opt_v", "sampler"):
        equal(after[name], target_before[name])
    for name in ("wm", "opt_wm") + (("transition_model", "opt_transition") if heads else ()):
        equal(checkpoint[name], source_before[name])
    assert metadata["source_iteration"] == 289
    assert metadata["opt_wm_steps"] == [3]
    assert metadata["opt_transition_steps"] == ([3] if heads else [])
    assert metadata["loaded_modules"] == (["wm", "opt_wm", "transition_model", "opt_transition"] if heads else ["wm", "opt_wm"])


@pytest.mark.parametrize("corruption", ["method", "robot", "dataset", "collision", "head_missing", "shape", "dtype", "nonfinite_weight", "norm", "group", "order", "missing_moment", "moment_shape", "moment_dtype", "nonfinite_moment", "step", "extra_state"])
def test_rejects_every_invalid_source_before_any_live_mutation(corruption):
    target = learner(2)
    checkpoint = payload(learner(1))
    moment = checkpoint["opt_transition"]["state"][0]
    if corruption == "method": checkpoint["method"] = "other"
    elif corruption == "robot": checkpoint["robot_contract"]["memory_dim"] = 62
    elif corruption == "dataset": checkpoint["dataset_revision"] = "other"
    elif corruption == "collision": checkpoint["args"]["full_body_collision"] = False
    elif corruption == "head_missing": del checkpoint["opt_transition"]
    elif corruption == "shape": checkpoint["transition_model"]["net.weight"] = torch.zeros(2, 4)
    elif corruption == "dtype": checkpoint["wm"]["net.weight"] = checkpoint["wm"]["net.weight"].double()
    elif corruption == "nonfinite_weight": checkpoint["transition_model"]["net.weight"][0, 0] = torch.nan
    elif corruption == "norm": checkpoint["wm"]["in_norm.var"][0] = -1
    elif corruption == "group": checkpoint["opt_transition"]["param_groups"][0]["betas"] = (.8, .99)
    elif corruption == "order": checkpoint["opt_wm"]["param_groups"][0]["params"].reverse()
    elif corruption == "missing_moment": del moment["exp_avg"]
    elif corruption == "moment_shape": moment["exp_avg_sq"] = torch.zeros(2, 4)
    elif corruption == "moment_dtype": moment["exp_avg"] = moment["exp_avg"].double()
    elif corruption == "nonfinite_moment": moment["exp_avg_sq"][0, 0] = torch.inf
    elif corruption == "step": moment["step"] = torch.tensor(.5)
    elif corruption == "extra_state": checkpoint["opt_transition"]["state"][99] = deepcopy(moment)
    before = snapshot(target)
    with pytest.raises(ValueError):
        initialize(target, checkpoint)
    equal(snapshot(target), before)


def test_unexpected_source_head_is_rejected_when_current_has_none():
    target = learner(2, heads=False)
    before = snapshot(target)
    with pytest.raises(ValueError, match="presence"):
        initialize(target, payload(learner(1, heads=True)))
    equal(snapshot(target), before)


def test_future_adam_steps_do_not_mutate_checkpoint_payload_via_tensor_aliases():
    target = learner(2)
    checkpoint = payload(learner(1))
    initialize(target, checkpoint)
    source_before = deepcopy(checkpoint)
    for model, optimizer in [(target.wm, target.opt_wm), (target.transition_model, target.opt_transition)]:
        optimizer.zero_grad()
        model(torch.ones(5, 3)).square().mean().backward()
        optimizer.step()
    equal(checkpoint, source_before)
