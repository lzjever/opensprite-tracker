"""Strict world-model-only warm initialization, distinct from training resume.

All source model/normalizer and Adam states are checked before any live module
is changed. Actor, critic, their optimizers, replay, and sampling are untouched.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import torch

METHOD = "SuperMimic-OpenSprite-mjlab"


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a state mapping")
    return value


def _finite_tensor(value, label):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")


def _module_state(module, source, label):
    source = _mapping(source, label)
    expected = module.state_dict()
    if set(source) != set(expected):
        missing, extra = sorted(set(expected) - set(source)), sorted(set(source) - set(expected))
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")
    for name, target in expected.items():
        value = source[name]
        _finite_tensor(value, f"{label}.{name}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(
                f"{label}.{name} shape/dtype differs: "
                f"{tuple(value.shape)}/{value.dtype} vs {tuple(target.shape)}/{target.dtype}"
            )
        if name.endswith((".var", ".count")) and bool((value < 0).any()):
            raise ValueError(f"{label}.{name} is a negative normalization statistic")
    return source


def _step(value, label):
    if isinstance(value, torch.Tensor):
        _finite_tensor(value, label)
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{label} must be a scalar Adam step")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a numeric Adam step")
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f"{label} must be a finite nonnegative integer")
    return int(value)


def _adam_state(optimizer, source, label):
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError(f"{label}: current optimizer must be Adam")
    source = _mapping(source, label)
    if set(source) != {"state", "param_groups"}:
        raise ValueError(f"{label} must contain exactly state and param_groups")
    states = _mapping(source["state"], f"{label}.state")
    groups = source["param_groups"]
    current = optimizer.state_dict()["param_groups"]
    if not isinstance(groups, (list, tuple)) or len(groups) != len(current):
        raise ValueError(f"{label} parameter-group count differs")
    all_ids, unique_steps = [], set()
    for index, (saved, expected, live) in enumerate(zip(groups, current, optimizer.param_groups)):
        saved = _mapping(saved, f"{label}.param_groups[{index}]")
        if set(saved) != set(expected):
            raise ValueError(f"{label} parameter-group keys differ")
        # PyTorch serializes IDs in parameter-group order. Matching IDs and
        # group sizes prevents silently remapping moments to reordered weights.
        ids = saved["params"]
        if (not isinstance(ids, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in ids)
                or list(ids) != list(expected["params"])):
            raise ValueError(f"{label} parameter-group order/size differs")
        for key, expected_value in expected.items():
            if key in {"params", "lr"}:
                continue
            if saved[key] != expected_value:
                raise ValueError(f"{label} Adam setting {key!r} differs")
        lr = saved["lr"]
        if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr < 0:
            raise ValueError(f"{label} learning rate is invalid")
        all_ids.extend(ids)
        for identifier, parameter in zip(ids, live["params"]):
            if identifier not in states:
                raise ValueError(f"{label} missing Adam state for parameter {identifier}")
            state = _mapping(states[identifier], f"{label}.state[{identifier}]")
            required = {"step", "exp_avg", "exp_avg_sq"}
            if saved["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(state) != required:
                raise ValueError(f"{label} Adam moment keys differ for parameter {identifier}")
            unique_steps.add(_step(state["step"], f"{label}.state[{identifier}].step"))
            for name in required - {"step"}:
                moment = state[name]
                _finite_tensor(moment, f"{label}.state[{identifier}].{name}")
                if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError(f"{label} {name} shape/dtype differs for parameter {identifier}")
                if name in {"exp_avg_sq", "max_exp_avg_sq"} and bool((moment < 0).any()):
                    raise ValueError(f"{label} {name} is negative")
    if len(set(all_ids)) != len(all_ids) or set(states) != set(all_ids):
        raise ValueError(f"{label} state IDs do not exactly cover its parameters")
    return sorted(unique_steps)


def initialize_world_model(
    learner, checkpoint, *, robot_contract, dataset_revision, full_body_collision
):
    """Load only WM/cost heads, including their norms and corresponding Adam.

    ``checkpoint`` is an already loaded (normally CPU-mapped) training dict.
    Different model dimensions, shape-changing state/memory/context layouts, normalization
    buffers, robots, datasets, collisions, or optimizer contracts fail before
    mutation. Source LR is retained with Adam; the new run's scheduler owns any
    explicitly configured subsequent LR. This does not resume source clocks.
    """
    checkpoint = _mapping(checkpoint, "checkpoint")
    if checkpoint.get("method") != METHOD:
        raise ValueError("Checkpoint method is not native OpenSprite SuperMimic")
    if checkpoint.get("robot_contract") != robot_contract:
        raise ValueError("Checkpoint robot/physics contract differs")
    if not isinstance(dataset_revision, str) or not dataset_revision or checkpoint.get("dataset_revision") != dataset_revision:
        raise ValueError("Checkpoint UMR dataset revision differs")
    args = _mapping(checkpoint.get("args"), "checkpoint.args")
    collision = args.get("full_body_collision")
    if not isinstance(full_body_collision, bool) or not isinstance(collision, bool) or collision != full_body_collision:
        raise ValueError("Checkpoint full-body collision contract differs")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 0:
        raise ValueError("Checkpoint source iteration must be a nonnegative integer")
    has_head = learner.transition_model is not None
    if has_head != (learner.opt_transition is not None):
        raise ValueError("Current cost head and optimizer presence differ")
    if ("transition_model" in checkpoint) != has_head or ("opt_transition" in checkpoint) != has_head:
        raise ValueError("Checkpoint/current cost-head presence differs")

    # Preflight every target, including the last cost-head optimizer, before
    # loading the first world-model weight. Do not partially initialize on a
    # later shape, moment, or finite-value failure.
    _module_state(learner.wm, checkpoint.get("wm"), "wm")
    wm_steps = _adam_state(learner.opt_wm, checkpoint.get("opt_wm"), "opt_wm")
    transition_steps = []
    if has_head:
        _module_state(learner.transition_model, checkpoint["transition_model"], "transition_model")
        transition_steps = _adam_state(learner.opt_transition, checkpoint["opt_transition"], "opt_transition")

    # Adam keeps an uncapturable step tensor on the CPU even for CUDA
    # parameters. Clone before the first load so future optimizer steps cannot
    # mutate the caller's source checkpoint through shared scalar/moment tensors.
    wm_optimizer_state = deepcopy(checkpoint["opt_wm"])
    transition_optimizer_state = deepcopy(checkpoint["opt_transition"]) if has_head else None
    learner.wm.load_state_dict(checkpoint["wm"], strict=True)
    learner.opt_wm.load_state_dict(wm_optimizer_state)
    loaded = ["wm", "opt_wm"]
    if has_head:
        learner.transition_model.load_state_dict(checkpoint["transition_model"], strict=True)
        learner.opt_transition.load_state_dict(transition_optimizer_state)
        loaded.extend(("transition_model", "opt_transition"))
    return {
        "mode": "world_model_and_cost_heads_only",
        "source_iteration": int(iteration),
        "opt_wm_steps": wm_steps,
        "opt_transition_steps": transition_steps,
        "loaded_modules": loaded,
        "untouched_modules": ["pi", "critic", "opt_pi", "opt_v", "replay", "sampling"],
    }
