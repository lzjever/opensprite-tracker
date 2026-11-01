"""Whole-body tracking rewards for the single-dance OpenSprite task.

Tracking, support, command effort and contact costs use OpenSprite kinematics,
per-joint limits and recorded simulator transition measurements.
No pose classes, frame windows, upright target, or fixed root-height target.

Conventions:
* All targets use the supplied arrived-at frame and current torso anchor, not
  cached pre-step alignment or stale body kinematics.
* Immutable-reference non-foot collision-geometry clearance, provided by
  kin.reference_nonfoot_clearance and cached once, fades every support
  preference out over 2--5 cm. The reward does not prescribe an AABB or an
  exact primitive/mesh support query. Proximity indicates support ambiguity,
  NOT measured contact or load.
* Both reference sole normals must also face the ground's upward normal:
  confidence ramps from zero at normal Z <= 0 to one at Z >= .5. This only
  qualifies support preferences; foot orientation targets remain the reference,
  including inverted poses. It is not an upright posture reward.
* Flat foot stability scores XY velocity without a contact bonus or gate.
  It scores kinematic agreement, not proof of actual contact.
* Impact ramp is an explicit training-progress parameter.
* References retain MotionLib's FK-consistent body velocities. Sprite uses its
  own full-body primitives, official box soles and nonlocal self contacts.

Transition metrics MUST be supplied (measured or supervised predictions), in
BODY_JOINT_NAMES order: speed_peak includes the final boundary; torque peaks
are absolute maxima over four 5-ms substeps; contact_mean_n is mean(abs(Fz)),
contact_peak_n is max(clamp(-Fz, 0)). xi is raw 221-D context, not normalized.
Using predicted peaks is a model approximation, not an analytic force law.
"""
from dataclasses import dataclass

import torch

from .mdp import CONTROL_DT, MDP, RewardCfg
from .rotations import rot_angle_sq, yaw_mat
from .robot import CONTEXT_DIM, CONTEXT_SLICES


@dataclass
class LegacyRewardCfg(RewardCfg):
    profile: str = "legacy_full"
    w_root_pos: float = .5
    w_root_ori: float = .5
    w_body_pos: float = 1.
    w_body_ori: float = 1.
    w_body_linvel: float = 1.
    w_body_angvel: float = 1.
    w_joint_limit: float = -10.
    w_foot_pos: float = 1.
    w_foot_linvel: float = .5
    w_swing_clearance: float = -20.
    w_leg_joint: float = .35
    w_arm_joint: float = .35
    w_support_share: float = -2.
    w_support_stability: float = .4
    w_support_torso: float = -.3
    w_motor: float = -30.
    w_request: float = -3.
    w_impact: float = -146.14
    foot_pos_std: float = .1
    foot_linvel_std: float = .5
    leg_joint_std: float = .3
    arm_joint_std: float = .35
    ankle_tracking_factor: float = .75
    support_pos_std_scale: float = 1.5
    support_ori_std_scale: float = 1.35
    swing_blend_start: float = .015
    swing_blend_end: float = .025
    swing_tolerance: float = .005
    support_contact_full: float = .005
    support_contact_end: float = .020
    flat_rise_full: float = .015
    flat_rise_end: float = .045
    toe_rise_start: float = .040
    toe_rise_full: float = .100
    support_share: float = .6
    minimum_toe_force_n: float = 5.
    toe_contact_weight: float = .1
    support_linvel_std: float = .25
    support_angvel_std: float = 1.
    support_normal_std: float = .2
    support_height_std: float = .02
    support_torso_deadband: float = .2
    nonfoot_fade_start: float = .02
    nonfoot_fade_end: float = .05
    support_sole_normal_start: float = 0.
    support_sole_normal_full: float = .5
    impact_bodyweights: float = 3.
    impact_reset_grace_s: float = .2
    include_body_impact: bool = False


def _ramp(x, lo, hi):
    return ((x - lo) / (hi - lo)).clamp(0, 1)


class LegacyMDP(MDP):
    uses_transition_metrics = True
    reward_scale = .152  # .02 * (5 + 1 + .5 + .35 + .35 + .4)

    def __init__(self, kin, motion, default_q, velocity_limits, cfg=None):
        super().__init__(kin, motion, default_q, velocity_limits, cfg or LegacyRewardCfg())
        c = self.cfg
        self.torso = kin.tracked_names.index("waist_yaw_link")
        self.num_joints = len(kin.joint_names)
        names = kin.joint_names
        self.legs = [i for i, n in enumerate(names) if any(s in n for s in ("hip_", "knee_", "ankle_"))]
        self.arms = [i for i, n in enumerate(names) if any(s in n for s in ("shoulder_", "elbow_", "wrist_"))]
        self.pitch = [names.index(s + "_ankle_pitch_joint") for s in ("left", "right")]
        self.roll = [names.index(s + "_ankle_roll_joint") for s in ("left", "right")]
        self.other = [i for i in range(len(names)) if i not in self.pitch]  # includes roll
        self.leg_weights = default_q.new_tensor([
            c.ankle_tracking_factor if "ankle" in names[i] else 1. for i in self.legs])
        self.soft_ratio = torch.full_like(default_q, .9)
        self.soft_ratio[self.pitch] = .8
        self.impact_threshold_n = kin.total_mass_kg * 9.81 * c.impact_bodyweights
        self.reward_scale = CONTROL_DT * (c.w_root_pos + c.w_root_ori + c.w_body_pos
            + c.w_body_ori + c.w_body_linvel + c.w_body_angvel + c.w_foot_pos
            + c.w_foot_linvel + c.w_leg_joint + c.w_arm_joint + c.w_support_stability)

        # Same heel/toe rise rule, using Sprite's real box edges. No capsule
        # index or foot geometry from the source robot survives this adapter.
        self.heel_minus_toe = kin.heel_minus_toe
        # Reference only: cache once, never run reference FK inside rollout heads.
        with torch.no_grad():
            f = torch.arange(motion.num_frames, device=default_q.device)
            self.reference_nonfoot_clearance = kin.reference_nonfoot_clearance(*motion.ref_state(f)).detach()
            self.reference_support_confidence = _ramp(self.reference_nonfoot_clearance,
                c.nonfoot_fade_start, c.nonfoot_fade_end)
            p, R = motion.body_p[:, self.feet], motion.body_R[:, self.feet]
            bottom = kin.sole_bottom_z(p, R)
            rise = (R @ self.heel_minus_toe[..., None]).squeeze(-1)[..., 2]
            contact = 1 - _ramp(bottom, c.support_contact_full, c.support_contact_end)
            flat = 1 - _ramp(rise.abs(), c.flat_rise_full, c.flat_rise_end)
            toe = _ramp(rise, c.toe_rise_start, c.toe_rise_full)
            self.reference_toe_gates = torch.stack((contact[:, 0] * flat[:, 0] * contact[:, 1] * toe[:, 1],
                contact[:, 1] * flat[:, 1] * contact[:, 0] * toe[:, 0]), -1)
            self.reference_sole_confidence = _ramp(kin.sole_normals_world(R)[..., 2],
                c.support_sole_normal_start, c.support_sole_normal_full)
            # Each legacy gate assumes BOTH a flat plantar foot and a toe foot.
            sole_pair = self.reference_sole_confidence.prod(-1)
            self.reference_support_gates = self.reference_toe_gates * (
                self.reference_support_confidence * sole_pair)[:, None]

    def _bodies_and_target(self, S, f):
        bp, bR, bv, bw = self.kin.forward(*S[:6])
        rp, rR = self.motion.body_p[f, self.torso], self.motion.body_R[f, self.torso]
        ap, aR = bp[..., self.torso, :], bR[..., self.torso, :, :]
        dR = yaw_mat(aR @ rR.transpose(-1, -2))
        anchor = torch.cat((ap[..., :2], rp[..., 2:3]), -1)
        tp = anchor.unsqueeze(-2) + (dR.unsqueeze(-3)
            @ (self.motion.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        return bp, bR, bv, bw, tp, dR.unsqueeze(-3) @ self.motion.body_R[f]

    def _analytic(self, S, f, a, prev):
        c, m = self.cfg, self.motion
        bp, bR, bv, bw, tp, tR = self._bodies_and_target(S, f)
        q = S[4]
        gates = self.reference_support_gates[f]
        phase = gates.amax(-1)
        body_pos_sq = (tp - bp).square().sum(-1)
        body_ori_sq = rot_angle_sq(tR, bR)
        lv_sq = (m.body_v[f] - bv).square().sum(-1)
        av_sq = (m.body_w[f] - bw).square().sum(-1)
        root_sq = (m.body_p[f, self.torso] - bp[..., self.torso, :]).square().sum(-1)
        root_rot_sq = rot_angle_sq(m.body_R[f, self.torso], bR[..., self.torso, :, :])
        rp_std = c.root_pos_std * (1 + phase * (c.support_pos_std_scale - 1))
        ro_std = c.root_ori_std * (1 + phase * (c.support_ori_std_scale - 1))
        rate = (a - prev).square().sum(-1)
        limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        actual_z = self.kin.sole_bottom_z(bp[..., self.feet, :], bR[..., self.feet, :, :])
        ref_z = self.kin.sole_bottom_z(tp[..., self.feet, :], tR[..., self.feet, :, :])
        swing = _ramp(ref_z, c.swing_blend_start, c.swing_blend_end)
        clearance_cost = (swing * (ref_z - actual_z - c.swing_tolerance).clamp_min(0).square()).sum(-1) / swing.sum(-1).clamp_min(1)
        normal_sq = (self.kin.sole_normals_world(tR[..., self.feet, :, :])
                     - self.kin.sole_normals_world(bR[..., self.feet, :, :])).square().sum(-1)
        foot_vxy = (bv[..., self.feet, :2] - m.body_v[f][..., self.feet, :2]).square().sum(-1)
        stability = (gates * torch.exp(-foot_vxy / c.support_linvel_std**2
            - av_sq[..., self.feet] / c.support_angvel_std**2
            - normal_sq / c.support_normal_std**2
            - (actual_z - ref_z).square() / c.support_height_std**2)).sum(-1)
        torso_speed = (bw[..., self.torso, :] - m.body_w[f, self.torso]).norm(dim=-1)
        joint_sq = (m.q[f] - q).square()
        weighted = {
            "motion_global_root_pos": c.w_root_pos * torch.exp(-root_sq / rp_std.square()),
            "motion_global_root_ori": c.w_root_ori * torch.exp(-root_rot_sq / ro_std.square()),
            "motion_body_pos": c.w_body_pos * torch.exp(-body_pos_sq.mean(-1) / c.body_pos_std**2),
            "motion_body_ori": c.w_body_ori * torch.exp(-body_ori_sq.mean(-1) / c.body_ori_std**2),
            "motion_body_lin_vel": c.w_body_linvel * torch.exp(-lv_sq.mean(-1) / c.body_linvel_std**2),
            "motion_body_ang_vel": c.w_body_angvel * torch.exp(-av_sq.mean(-1) / c.body_angvel_std**2),
            "action_rate_l2": c.w_action_rate * rate,
            "joint_limit": c.w_joint_limit * limit,
            "motion_foot_tilt": c.w_foot_tilt * normal_sq.mean(-1),
            "motion_foot_pos": c.w_foot_pos * torch.exp(-body_pos_sq[..., self.feet].mean(-1) / c.foot_pos_std**2),
            "motion_foot_lin_vel": c.w_foot_linvel * torch.exp(-lv_sq[..., self.feet].mean(-1) / c.foot_linvel_std**2),
            "motion_swing_foot_clearance": c.w_swing_clearance * clearance_cost,
            "motion_leg_joint_pos": c.w_leg_joint * torch.exp(-(joint_sq[..., self.legs] * self.leg_weights).mean(-1) / c.leg_joint_std**2),
            "motion_arm_joint_pos": c.w_arm_joint * torch.exp(-joint_sq[..., self.arms].mean(-1) / c.arm_joint_std**2),
            "flat_support_foot_stability": c.w_support_stability * stability,
            "support_torso_excess_ang_vel": c.w_support_torso * phase * (torso_speed - c.support_torso_deadband).clamp_min(0).square(),
        }
        diagnostics = {"tracking": sum(list(weighted.values())[:6]),
            "body_pos_error_m": body_pos_sq.mean(-1).sqrt(), "root_pos_error_m": root_sq.sqrt(),
            "root_angle_error_rad": root_rot_sq.sqrt(), "joint_error_rad": joint_sq.mean(-1).sqrt(),
            "foot_height_error_m": (actual_z - ref_z).square().mean(-1).sqrt(),
            "foot_tilt_error": normal_sq.mean(-1), "action_rate": rate, "joint_limit": limit,
            "support_confidence": self.reference_support_confidence[f]}
        return weighted, diagnostics

    def reward_terms(self, S, f, a, prev):
        """Sixteen state/action terms, weighted before dt; no metrics or forces."""
        return self._analytic(S, f, a, prev)[0]

    def transition_cost_terms(self, metrics, f, xi, ep_step, impact_ramp=1.):
        """Five weighted-before-dt terms; callers may take per-term head minima.

        ep_step is the ARRIVED-AT episode age; no local rollout-age substitute.
        speed_peak must already include max(substeps, endpoint abs(qd)).
        """
        if metrics is None or xi is None or ep_step is None:
            raise ValueError("legacy_full requires metrics, raw xi, and arrived-at ep_step")
        required = ("speed_peak", "raw_torque_peak", "applied_torque_peak", "contact_mean_n", "contact_peak_n")
        for key in required:
            if key not in metrics:
                raise ValueError(f"Missing transition metric: {key}")
            width = 2 if key.startswith("contact_") else self.num_joints
            if metrics[key].shape != f.shape + (width,):
                raise ValueError(f"Wrong transition metric shape: {key}")
        if xi.shape != f.shape + (CONTEXT_DIM,):
            raise ValueError(f"Expected raw Sprite context (...,{CONTEXT_DIM})")
        c = self.cfg
        force_limit = xi[..., CONTEXT_SLICES["force_limit"]]
        def motor_cost(peak, raw):
            threshold = force_limit if raw else self.soft_ratio * force_limit
            values = (peak / threshold - 1).clamp_min(0).square()
            roll_threshold = force_limit[..., self.roll] * (1. if raw else .8)
            roll_cost = (peak[..., self.roll] / roll_threshold - 1).clamp_min(0).square().mean(-1)
            return .75 * values[..., self.pitch].mean(-1) + .25 * values[..., self.other].mean(-1) + .5 * roll_cost
        F = metrics["contact_mean_n"]
        share = F / F.sum(-1, keepdim=True).clamp_min(1)
        share_cost = (c.support_share - share).clamp_min(0).square()
        toe_cost = ((c.minimum_toe_force_n - F.flip(-1)).clamp_min(0) / c.minimum_toe_force_n).square()
        load_share = (self.reference_support_gates[f] * (share_cost + c.toe_contact_weight * toe_cost)).sum(-1)
        impact = (metrics["contact_peak_n"] / self.impact_threshold_n - 1).clamp_min(0).square().mean(-1)
        age = ep_step * CONTROL_DT > c.impact_reset_grace_s
        ramp = impact_ramp.clamp(0, 1) if isinstance(impact_ramp, torch.Tensor) else min(1., max(0., impact_ramp))
        result = {
            "joint_velocity_limit": c.w_joint_speed * (metrics["speed_peak"] / self.velocity_limits - 1).clamp_min(0).square().sum(-1),
            "load_motor_excess": c.w_motor * motor_cost(metrics["applied_torque_peak"], False),
            "load_request_excess": c.w_request * motor_cost(metrics["raw_torque_peak"], True),
            "load_contact_excess": c.w_impact * impact * age * ramp,
            "flat_support_load_share": c.w_support_share * load_share,
        }
        if c.include_body_impact:
            peak = metrics["body_contact_peak_n"]
            if peak.shape != f.shape+(1,):
                raise ValueError("Expected maximum non-foot body force per interval (...,1)")
            body_cost = (peak[..., 0]/self.impact_threshold_n-1).clamp_min(0).square()
            result["load_body_contact_excess"] = c.w_impact*body_cost*age*ramp
        return result

    def reward(self, S, f, a, prev, terms=False, *, metrics=None, xi=None,
               ep_step=None, impact_ramp=1.):
        costs = self.transition_cost_terms(metrics, f, xi, ep_step, impact_ramp)
        weighted, diagnostics = self._analytic(S, f, a, prev)
        weighted.update(costs)
        reward = sum(weighted.values()) * CONTROL_DT
        if not terms:
            return reward
        diagnostics["joint_speed_excess"] = (metrics["speed_peak"] / self.velocity_limits - 1).clamp_min(0).square().sum(-1)
        diagnostics.update({"per_reward/" + name: value for name, value in weighted.items()})
        return reward, diagnostics
