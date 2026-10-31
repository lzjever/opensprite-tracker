"""Analytic tracking heads, shared by real transitions and imagined transitions.

Adapted from zita-ch/supermimic_vibed; see THIRD_PARTY_LICENSE.txt.
Only the critic/world model receive simulator context and actuator memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import mat_to_6d, rot_angle_sq, yaw_mat
from .state import flatten_state, invariant_features, unflatten_state
from .robot import JOINT_NAMES, CONTEXT_SLICES
from opensprite_tracker.constants import (PELVIS_HEIGHT_THRESHOLD_M,
    PELVIS_GRAVITY_SQUARED_THRESHOLD)

NUM_J = len(JOINT_NAMES)
OBS_DIM = 3*NUM_J + 12
INV_DIM = 13 + 2*NUM_J  # without the optional actuator-memory tail
CONTROL_DT = 0.02


@dataclass
class RewardCfg:
    profile: str = "paper"
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    w_foot_height: float = -20.0
    w_foot_tilt: float = -1.0
    w_joint_speed: float = -1.0
    soft_limit_factor: float = 0.9
    term_height: float = PELVIS_HEIGHT_THRESHOLD_M
    term_tilt: float = PELVIS_GRAVITY_SQUARED_THRESHOLD


class MDP:
    uses_transition_metrics = False
    reward_scale = 0.1  # Maximum positive tracking reward per control step.
    def __init__(self, kin, motion, default_q: torch.Tensor,
                 velocity_limits: torch.Tensor, cfg: RewardCfg | None = None):
        self.kin, self.motion = kin, motion
        self.cfg = cfg or RewardCfg()
        self.default_q, self.velocity_limits = default_q, velocity_limits
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = mid - half * self.cfg.soft_limit_factor
        self.soft_hi = mid + half * self.cfg.soft_limit_factor
        self.feet = [kin.tracked_names.index(n) for n in
                     ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.obs_noise = torch.tensor(
            [0.01] * NUM_J + [0.5] * NUM_J + [0.2] * 3 + [0.05] * 3
            + [0.0] * NUM_J + [0.05] * 6, device=default_q.device)

    def obs(self, S, a_prev, xi, f):
        """No root position/velocity or privileged parameters in actor input.

        Encoder bias is sensor calibration error: same sign/convention as the
        source observation convention, also included in the model context.
        """
        _, _, _, _, q, qd = S[:6]
        imu_R, gyro, gravity = self.kin.imu(*S[:6])
        torso_id = self.kin.tracked_names.index("waist_yaw_link")
        reference_imu_R = self.motion.body_R[f, torso_id] @ self.kin.imu_site_R
        rel = mat_to_6d(imu_R.transpose(-1, -2) @ reference_imu_R)
        return torch.cat((q - self.default_q + xi[..., CONTEXT_SLICES["encoder_bias"]], qd,
                          gyro, gravity, a_prev, rel), -1)

    def add_noise(self, obs):
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def _bodies_and_target(self, S, f):
        p, R, v, w, q, qd = S[:6]
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        m = self.motion
        rp, rR = m.root_p[f], m.root_R[f]
        # Same body-frame reanchoring as the paper/reference implementation.
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        target_p = anchor.unsqueeze(-2) + (
            dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        target_R = dR.unsqueeze(-3) @ m.body_R[f]
        return bp, bR, bv, bw, target_p, target_R

    def reward(self, S, f, a, a_prev, terms=False):
        """Reward of the *arrived-at* state and corresponding reference frame.

        Contact force is deliberately absent: a learned unconstrained force head
        would provide the actor an easy way to invent cheap, nonphysical contact.
        Real contact peaks remain an evaluation metric, not a fictional gradient.
        """
        c, m = self.cfg, self.motion
        p, R, _, _, q, qd = S[:6]
        bp, bR, bv, bw, target_p, target_R = self._bodies_and_target(S, f)
        e_pos = (target_p - bp).square().sum(-1).mean(-1)
        e_ori = rot_angle_sq(target_R, bR).mean(-1)
        e_lv = (m.body_v[f] - bv).square().sum(-1).mean(-1)
        e_av = (m.body_w[f] - bw).square().sum(-1).mean(-1)
        e_root = (m.root_p[f] - p).square().sum(-1)
        e_rot = rot_angle_sq(m.root_R[f], R)
        tracking = (torch.exp(-e_pos / c.body_pos_std**2)
                    + torch.exp(-e_ori / c.body_ori_std**2)
                    + torch.exp(-e_lv / c.body_linvel_std**2)
                    + torch.exp(-e_av / c.body_angvel_std**2)
                    + 0.5 * torch.exp(-e_root / c.root_pos_std**2)
                    + 0.5 * torch.exp(-e_rot / c.root_ori_std**2))
        rate = (a - a_prev).square().sum(-1)
        limits = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        sole_z = self.kin.sole_bottom_z(bp[..., self.feet, :], bR[..., self.feet, :, :])
        reference_z = self.kin.sole_bottom_z(target_p[..., self.feet, :], target_R[..., self.feet, :, :])
        foot_height = (reference_z - sole_z).square().mean(-1)
        foot_tilt = (self.kin.sole_normals_world(target_R[..., self.feet, :, :])
                     - self.kin.sole_normals_world(bR[..., self.feet, :, :])).square().sum(-1).mean(-1)
        speed = (qd.abs() / self.velocity_limits - 1).clamp_min(0).square().sum(-1)
        reward = (tracking + c.w_action_rate * rate + c.w_joint_limit * limits
                  + c.w_foot_height * foot_height + c.w_foot_tilt * foot_tilt
                  + c.w_joint_speed * speed) * CONTROL_DT
        if not terms:
            return reward
        return reward, {"tracking": tracking, "body_pos_error_m": e_pos.sqrt(),
                        "root_pos_error_m": e_root.sqrt(), "root_angle_error_rad": e_rot.sqrt(),
                        "joint_error_rad": (m.q[f] - q).square().mean(-1).sqrt(),
                        "foot_height_error_m": foot_height.sqrt(),
                        "foot_tilt_error": foot_tilt,
                        "joint_speed_excess": speed,
                        "action_rate": rate, "joint_limit": limits}

    def termination_errors(self, S, f):
        p, R = S[:2]
        # Complete reference-relative gravity accepts all matching lying,
        # rolling and inverted poses, including rotations with identical Z/Z.
        height = (self.motion.root_p[f, 2] - p[..., 2]).abs()
        tilt = (self.motion.root_R[f, 2, :] - R[..., 2, :]).square().sum(-1)
        return height, tilt

    def termination(self, S, f):
        height, tilt = self.termination_errors(S, f)
        return (height > self.cfg.term_height) | (tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz=0.02, tt=0.04):
        height, tilt = self.termination_errors(S, f)
        alive = (1 - torch.sigmoid((height - self.cfg.term_height) / tz))
        alive = alive * (1 - torch.sigmoid((tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi_normalized):
        p, R, _, _, q, _ = S[:6]
        Ht = yaw_mat(R).transpose(-1, -2)
        bp, _, _, _, target_p, _ = self._bodies_and_target(S, f)
        body_error = (Ht.unsqueeze(-3) @ (target_p - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        root_error = (Ht @ (self.motion.root_p[f] - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ self.motion.root_R[f])
        return torch.cat((invariant_features(S), self.motion.q[f] - q,
                          body_error, root_error, rel, xi_normalized), -1)


def reward_config(profile="paper", recorded=None):
    """Checkpoint reward identity, not whichever default the source now uses."""
    values = dict(recorded or {})
    name = values.pop("profile", profile)
    if name == "paper":
        return RewardCfg(**values)
    if name == "legacy_full":
        from .legacy_rewards import LegacyRewardCfg
        return LegacyRewardCfg(**values)
    raise ValueError(f"Unknown reward profile: {name}")


def make_mdp(kin, motion, default_q, limits, cfg=None):
    cfg = cfg or RewardCfg()
    if cfg.profile == "paper":
        return MDP(kin, motion, default_q, limits, cfg)
    if cfg.profile == "legacy_full":
        from .legacy_rewards import LegacyMDP
        return LegacyMDP(kin, motion, default_q, limits, cfg)
    raise ValueError(f"Unknown reward profile: {cfg.profile}")
