"""OpenSprite-only physical/context contract for the reference SuperMimic loop."""

from dataclasses import replace
from pathlib import Path
import mujoco
import torch
from opensprite_tracker.robot import (
    BODY_JOINT_NAMES,
    TRACKED_BODY_NAMES,
    JOINT_CONTROL,
    ROBOT_XML,
    ROOT,
)
from opensprite_tracker.task import robot_cfg

MJCF_PATH = ROBOT_XML
MOTION_PATH = ROOT / "data/motions/lafan_dance.npz"
LAFAN_PATH = MOTION_PATH
JOINT_NAMES = BODY_JOINT_NAMES
TRACKED_BODIES = TRACKED_BODY_NAMES
NUM_J = len(JOINT_NAMES)
PHYSICS_DT = 0.005
CONTROL_DT = 0.020
DECIMATION = 4
ACTION_SCALE = 0.25
MEM_DIM = 2 * NUM_J + 4
CONTEXT_DIM = 7 * NUM_J + 4
MEMORY_SLICES = {
    "filtered_effort_over_limit": slice(0, NUM_J),
    "previous_target_relative": slice(NUM_J, 2 * NUM_J),
    "delay_lag": slice(2 * NUM_J, 2 * NUM_J + 1),
    "delay_update_phase_over_3": slice(2 * NUM_J + 1, 2 * NUM_J + 2),
    "filter_initialized": slice(2 * NUM_J + 2, 2 * NUM_J + 3),
    "queue_valid": slice(2 * NUM_J + 3, 2 * NUM_J + 4),
}
CONTEXT_SLICES = {
    "foot_friction": slice(0, 1),
    "encoder_bias": slice(1, 1 + NUM_J),
    "torso_com_offset": slice(1 + NUM_J, 4 + NUM_J),
}
for i, key in enumerate(
    ("kp", "kd", "force_limit", "armature", "frictionloss", "torque_time_constant")
):
    CONTEXT_SLICES[key] = slice(4 + (i + 1) * NUM_J, 4 + (i + 2) * NUM_J)


def default_joint_pos(device="cpu", dtype=torch.float32):
    return torch.tensor(
        [JOINT_CONTROL[n].default_pos_rad for n in JOINT_NAMES],
        device=device,
        dtype=dtype,
    )


def velocity_limits(device="cpu", dtype=torch.float32):
    return torch.tensor(
        [JOINT_CONTROL[n].urdf_velocity_rad_s for n in JOINT_NAMES],
        device=device,
        dtype=dtype,
    )


def context_normalization(device="cpu", dtype=torch.float32, *, torque_lag_enabled=True):
    offset = torch.zeros(CONTEXT_DIM, device=device, dtype=dtype)
    scale = torch.ones_like(offset)
    offset[0] = 0.775
    scale[0] = 0.425
    scale[CONTEXT_SLICES["encoder_bias"]] = 0.01
    scale[CONTEXT_SLICES["torso_com_offset"]] = torch.tensor(
        [0.00588, 0.01225, 0.01102], device=device, dtype=dtype
    )
    for key, field in (
        ("kp", "kp"),
        ("kd", "kd"),
        ("force_limit", "command_effort_nm"),
    ):
        nominal = torch.tensor(
            [getattr(JOINT_CONTROL[n], field) for n in JOINT_NAMES],
            device=device,
            dtype=dtype,
        )
        offset[CONTEXT_SLICES[key]] = nominal * (0.925 if key == "force_limit" else 1.0)
        scale[CONTEXT_SLICES[key]] = nominal.abs().clamp_min(0.01) * (
            0.075 if key == "force_limit" else 0.1
        )
    m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    ids = [int(m.jnt_dofadr[m.joint(n).id]) for n in JOINT_NAMES]
    for key, values in (
        ("armature", m.dof_armature[ids]),
        ("frictionloss", m.dof_frictionloss[ids]),
    ):
        nominal = torch.tensor(values, device=device, dtype=dtype)
        offset[CONTEXT_SLICES[key]] = nominal
        scale[CONTEXT_SLICES[key]] = nominal.abs().clamp_min(0.001) * 0.3
    offset[CONTEXT_SLICES["torque_time_constant"]] = 0.0075 if torque_lag_enabled else 0.0
    scale[CONTEXT_SLICES["torque_time_constant"]] = 0.0025 if torque_lag_enabled else 1.0
    return offset, scale


def make_env_cfg(num_envs=1, seed=42, randomize=True, full_body_collision=True,
                 *, torque_lag_enabled=True):
    from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": robot_cfg(torque_lag_enabled=torque_lag_enabled)}
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.scene.terrain = replace(
        cfg.scene.terrain, terrain_type="plane", terrain_generator=None
    )
    cfg.commands = {}
    cfg.observations = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.metrics = {}
    cfg.curriculum = {}
    cfg.events = {}
    if randomize:
        from mjlab.envs.mdp import dr
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        cfg.events = {
            "foot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", geom_names=r"^(left|right)_foot_collision$"
                    ),
                    "operation": "abs",
                    "ranges": (0.35, 1.2),
                    "shared_random": True,
                },
            ),
            "encoder_bias": EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "bias_range": (-0.01, 0.01),
                },
            ),
            "base_com": EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("waist_yaw_link",)
                    ),
                    "operation": "add",
                    "ranges": {
                        i: (-v, v) for i, v in enumerate((0.00588, 0.01225, 0.01102))
                    },
                },
            ),
            "pd_gains": EventTermCfg(
                mode="startup",
                func=dr.pd_gains,
                params={
                    "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                    "kp_range": (0.9, 1.1),
                    "kd_range": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            "motor_strength": EventTermCfg(
                mode="startup",
                func=dr.effort_limits,
                params={
                    "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                    "effort_limit_range": (0.85, 1.0),
                    "operation": "scale",
                },
            ),
            "joint_armature": EventTermCfg(
                mode="startup",
                func=dr.joint_armature,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "ranges": (0.7, 1.3),
                    "operation": "scale",
                },
            ),
            "joint_friction": EventTermCfg(
                mode="startup",
                func=dr.joint_friction,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "ranges": (0.5, 1.5),
                    "operation": "scale",
                },
            ),
        }
    cfg.auto_reset = False
    cfg.actions["joint_pos"].scale = ACTION_SCALE
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="support_foot_contact",
            primary=ContactMatch(
                mode="body", entity="robot", pattern=r"^(left|right)_ankle_roll_link$"
            ),
            secondary=ContactMatch(mode="subtree", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=DECIMATION,
        ),
    )
    if full_body_collision:
        cfg.scene.sensors = cfg.scene.sensors + (
            ContactSensorCfg(
                name="nonfoot_ground_contact",
                primary=ContactMatch(
                    mode="body",
                    entity="robot",
                    pattern=r"^(?!(left|right)_ankle_roll_link$).+$",
                ),
                secondary=ContactMatch(mode="subtree", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                history_length=DECIMATION,
            ),
        )
    cfg.sim.mujoco.timestep = PHYSICS_DT
    cfg.decimation = DECIMATION
    cfg.sim.mujoco.iterations = 50
    cfg.sim.mujoco.ls_iterations = 25
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 600
    return cfg


def make_kinematics(device="cpu", full_body_collision=True):
    from .kinematics import OpenSpriteKinematics

    return OpenSpriteKinematics(
        device,
        model=mujoco.MjModel.from_xml_path(str(MJCF_PATH)),
        active_collision_only=True,
    )


def physical_contract(*, torque_lag_enabled=True):
    from opensprite_tracker.constants import (
        ACTUATOR_TEMPORAL_REVISION,
        TORQUE_TIME_CONSTANT_RANGE_S,
        COMMAND_DELAY_MIN_TICKS,
        COMMAND_DELAY_MAX_TICKS,
        COMMAND_DELAY_HOLD_PROBABILITY,
        COMMAND_DELAY_REFRESH_TICKS,
    )

    m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    contract = {
        "robot": "opensprite0825",
        "joints": JOINT_NAMES,
        "bodies": TRACKED_BODIES,
        "joint_axes": m.jnt_axis.tolist(),
        "hard_limits": m.jnt_range.tolist(),
        "mass": m.body_mass.tolist(),
        "inertia": m.body_inertia.tolist(),
        "com": m.body_ipos.tolist(),
        "armature": m.dof_armature.tolist(),
        "frictionloss": m.dof_frictionloss.tolist(),
        "geom_type": m.geom_type.tolist(),
        "geom_pos": m.geom_pos.tolist(),
        "geom_size": m.geom_size.tolist(),
        "geom_quat": m.geom_quat.tolist(),
        "excludes": m.exclude_signature.tolist(),
        "control": [(n, vars(JOINT_CONTROL[n])) for n in JOINT_NAMES],
        "physics_dt": PHYSICS_DT,
        "policy_dt": CONTROL_DT,
        "action_scale": ACTION_SCALE,
        "temporal_model": ACTUATOR_TEMPORAL_REVISION if torque_lag_enabled else ACTUATOR_TEMPORAL_REVISION + "_no_torque_lag_v1",
        "torque_tau_range": TORQUE_TIME_CONSTANT_RANGE_S if torque_lag_enabled else (0.0, 0.0),
        "delay_physics_ticks": (COMMAND_DELAY_MIN_TICKS, COMMAND_DELAY_MAX_TICKS),
        "delay_hold_probability": COMMAND_DELAY_HOLD_PROBABILITY,
        "delay_refresh_physics_ticks": COMMAND_DELAY_REFRESH_TICKS,
    }
    # Keep the exact legacy dictionary when lag is on, for old checkpoints.
    if not torque_lag_enabled:
        contract["torque_lag_enabled"] = False
    return contract
