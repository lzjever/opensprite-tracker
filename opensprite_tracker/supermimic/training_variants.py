"""Versioned actor observations and critic architecture for selectable recipes."""

from collections.abc import Mapping

GOAL_SIM = "goal-sim"
GOAL_SIM2REAL = "goal-sim2real"
TRAINING_VARIANTS = (GOAL_SIM, GOAL_SIM2REAL)


def observation_contract(variant=GOAL_SIM):
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"Unknown training variant: {variant!r}")
    return {
        "schema": 1,
        "training_variant": variant,
        "actor_observation_schema": (
            "proprio_reference_attitude_v1" if variant == GOAL_SIM
            else "proprio_imu_gravity_v1"
        ),
        "actor_obs_dim": 105 if variant == GOAL_SIM else 99,
        "actor_goal_schema": "reference_local_future77_v1",
        "critic_architecture": (
            "privileged_mlp_v1" if variant == GOAL_SIM
            else "privileged_goal_encoder_mlp_v1"
        ),
        "critic_uses_privileged_state": True,
    }


def checkpoint_variant(checkpoint):
    """Legacy unlabelled checkpoints are 105-observation goal-sim only."""
    cfg = checkpoint.get("args", {})
    cfg = cfg if isinstance(cfg, Mapping) else vars(cfg)
    variant = cfg.get("training_variant", GOAL_SIM)
    contract = observation_contract(variant)
    if checkpoint.get("training_variant", variant) != variant:
        raise ValueError("Checkpoint training variant metadata is inconsistent")
    saved = checkpoint.get("observation_contract")
    if saved is None:
        if variant != GOAL_SIM:
            raise ValueError("goal-sim2real checkpoint requires its observation contract")
    elif saved != contract:
        raise ValueError("Checkpoint observation contract is inconsistent")
    policy = checkpoint.get("pi", {})
    if "obs_norm.mean" in policy:
        if policy["obs_norm.mean"].numel() != contract["actor_obs_dim"]:
            raise ValueError("Checkpoint actor observations disagree with its training variant")
    critic = checkpoint.get("critic", {})
    if critic:
        expected = "net.0.weight" if variant == GOAL_SIM else "enc.0.weight"
        if expected not in critic:
            raise ValueError("Checkpoint critic architecture disagrees with its training variant")
        if variant == GOAL_SIM2REAL:
            if "dec.0.weight" not in critic or "norm.mean" not in critic:
                raise ValueError("Checkpoint goal-sim2real critic interface is incomplete")
            if "goal_norm.mean" in policy:
                goal_dim = policy["goal_norm.mean"].numel()
                if critic["enc.0.weight"].shape[1] != goal_dim:
                    raise ValueError("Checkpoint critic and actor goal dimensions disagree")
    return variant


def validate_variant_load(checkpoint, requested_variant):
    saved = checkpoint_variant(checkpoint)
    observation_contract(requested_variant)
    if saved != requested_variant:
        raise ValueError("Training variant changed; use a fresh cold-start run")
    return saved
