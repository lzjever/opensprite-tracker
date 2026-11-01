"""Sprite0825 physical entity and user-requested temporal actuator model."""

from dataclasses import dataclass, replace
import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec import create_motor_actuator
from mjlab.utils.spec_config import CollisionCfg
from .robot import (
    BODY_JOINT_NAMES,
    TRACKED_BODY_NAMES,
    JOINT_CONTROL,
    ROBOT_XML,
    INITIAL_ROOT_HEIGHT,
    FULL_BODY_COLLISION_PATTERN,
    TORQUE_LAG_GROUPS,
)
from .constants import (
    TORQUE_TIME_CONSTANT_RANGE_S,
    COMMAND_DELAY_MIN_TICKS,
    COMMAND_DELAY_MAX_TICKS,
    COMMAND_DELAY_HOLD_PROBABILITY,
    COMMAND_DELAY_REFRESH_TICKS,
)

DECIMATION = 4


@dataclass(kw_only=True)
class FilteredIdealPdActuatorCfg(IdealPdActuatorCfg):
    torque_time_constant_range: tuple[float, float] = TORQUE_TIME_CONSTANT_RANGE_S
    torque_lag_enabled: bool = True

    def build(self, entity, target_ids, target_names):
        return FilteredIdealPdActuator(self, entity, target_ids, target_names)


class FilteredIdealPdActuator(IdealPdActuator[FilteredIdealPdActuatorCfg]):
    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._controls = tuple((JOINT_CONTROL[name] for name in target_names))
        group_by_name = {
            name: group_id
            for group_id, names in enumerate(TORQUE_LAG_GROUPS)
            for name in names
        }
        self._torque_group_ids = tuple((group_by_name[name] for name in target_names))

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            logical_name = target_name.rsplit("/", 1)[-1]
            control = JOINT_CONTROL[logical_name]
            self._mjs_actuators.append(
                create_motor_actuator(
                    spec,
                    target_name,
                    effort_limit=control.command_effort_nm,
                    armature=self.cfg.armature,
                    frictionloss=self.cfg.frictionloss,
                    viscous_damping=self.cfg.viscous_damping,
                    transmission_type=self.cfg.transmission_type,
                )
            )

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        physics_dt = float(mj_model.opt.timestep)
        Actuator.initialize(self, mj_model, model, data, device)
        num_envs = data.nworld

        def parameter(field: str) -> torch.Tensor:
            values = torch.tensor(
                [getattr(control, field) for control in self._controls],
                dtype=torch.float,
                device=device,
            )
            return values[None].repeat(num_envs, 1)

        self.stiffness = parameter("kp")
        self.damping = parameter("kd")
        self.force_limit = parameter("command_effort_nm")
        self.default_stiffness = self.stiffness.clone()
        self.default_damping = self.damping.clone()
        self.default_force_limit = self.force_limit.clone()
        low, high = self.cfg.torque_time_constant_range if self.cfg.torque_lag_enabled else (0.0, 0.0)
        if not 0.0 <= low <= high:
            raise ValueError("torque time constants must satisfy 0 <= low <= high")
        per_group = (
            low + (high - low) * torch.rand((num_envs, len(TORQUE_LAG_GROUPS)), device=device)
            if self.cfg.torque_lag_enabled
            else torch.zeros((num_envs, len(TORQUE_LAG_GROUPS)), device=device)
        )
        self.extended_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        group_gain = torch.ones_like(per_group)
        group_ids = torch.tensor(self._torque_group_ids, device=device)
        self.torque_time_constant_per_group = per_group
        self.torque_time_constant = per_group[:, group_ids]
        self.motor_output_gain = group_gain[:, group_ids]
        self._dynamics_identification_targets: torch.Tensor | None = None
        self.filter_alpha = torch.where(
            self.torque_time_constant == 0,
            torch.ones_like(self.torque_time_constant),
            1.0 - torch.exp(-physics_dt / self.torque_time_constant.clamp_min(1e-09)),
        )
        self.filtered_effort = torch.zeros(
            (num_envs, len(self.target_names)), device=device
        )
        self.sent_target = torch.zeros_like(self.filtered_effort)
        self.speed_peak = torch.zeros_like(self.filtered_effort)
        self.raw_peak = torch.zeros_like(self.filtered_effort)
        self.applied_peak = torch.zeros_like(self.filtered_effort)
        self.filter_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self.filter_initialized[env_ids] = False
        self.sent_target[env_ids] = 0.0

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        self.sent_target.copy_(cmd.position_target)
        raw = pd_torque(self.stiffness, self.damping, cmd)
        self.speed_peak.copy_(torch.maximum(self.speed_peak, cmd.vel.abs()))
        self.raw_peak.copy_(torch.maximum(self.raw_peak, raw.abs()))
        requested = super().compute(cmd)
        initialize = ~self.filter_initialized
        self.filtered_effort.copy_(
            torch.where(initialize[:, None], requested, self.filtered_effort)
        )
        self.filter_initialized.fill_(True)
        # Command delay is processed upstream and remains enabled. In zero-lag
        # mode memory stores the latest applied PD effort with the same layout.
        provisional = requested if not self.cfg.torque_lag_enabled else torch.where(
            self.filter_alpha == 1,
            requested,
            self.filtered_effort
            + self.filter_alpha * (requested - self.filtered_effort),
        )
        assert self.force_limit is not None
        self.filtered_effort[:] = torch.clamp(
            provisional, min=-self.force_limit, max=self.force_limit
        )
        self.applied_peak.copy_(
            torch.maximum(self.applied_peak, self.filtered_effort.abs())
        )
        return self.filtered_effort


ACTUATORS = (
    FilteredIdealPdActuatorCfg(
        target_names_expr=tuple((f"^{name}$" for name in BODY_JOINT_NAMES)),
        stiffness=0.0,
        damping=0.0,
        effort_limit=max((c.command_effort_nm for c in JOINT_CONTROL.values())),
        armature=None,
        frictionloss=None,
        viscous_damping=None,
        delay_min_lag=COMMAND_DELAY_MIN_TICKS,
        delay_max_lag=COMMAND_DELAY_MAX_TICKS,
        delay_hold_prob=COMMAND_DELAY_HOLD_PROBABILITY,
        delay_update_period=COMMAND_DELAY_REFRESH_TICKS,
        torque_time_constant_range=TORQUE_TIME_CONSTANT_RANGE_S,
    ),
)


def get_spec():
    return mujoco.MjSpec.from_file(str(ROBOT_XML))


def robot_cfg(*, torque_lag_enabled=True):
    actuators = ACTUATORS if torque_lag_enabled else tuple(
        replace(actuator, torque_lag_enabled=False, torque_time_constant_range=(0.0, 0.0))
        for actuator in ACTUATORS
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, INITIAL_ROOT_HEIGHT),
            joint_pos={name: c.default_pos_rad for name, c in JOINT_CONTROL.items()},
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.9
        ),
        sort_actuators=True,
        collisions=(
            CollisionCfg(
                geom_names_expr=(FULL_BODY_COLLISION_PATTERN,),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=1,
                friction=(0.8,),
            ),
        ),
    )
