"""CPU sensor contracts and invariances for both selectable actor interfaces."""

from types import SimpleNamespace

import pytest
import torch

from opensprite_tracker.supermimic.kinematics import OpenSpriteKinematics
from opensprite_tracker.supermimic.mdp import MDP, OBS_DIM, make_mdp
from opensprite_tracker.supermimic.motion import MotionLib
from opensprite_tracker.supermimic.observations import sensor_observation
from opensprite_tracker.supermimic.robot import CONTEXT_DIM, CONTEXT_SLICES
from opensprite_tracker.supermimic.rotations import mat_to_6d, so3_exp


@pytest.fixture
def case():
    torch.manual_seed(8)
    kin = OpenSpriteKinematics(dtype=torch.float64)
    p = torch.randn(6, 3, dtype=torch.float64)
    p[:, 2] += 1
    R = so3_exp(torch.randn(6, 3, dtype=torch.float64))
    v, w = torch.randn(2, 6, 3, dtype=torch.float64)
    q = kin.default_q + torch.randn(6, 31, dtype=torch.float64) * .1
    qd = torch.randn(6, 31, dtype=torch.float64) * .1
    S = (p, R, v, w, q, qd, torch.randn(6, 66, dtype=torch.float64))
    bp, bR, bv, bw = kin.forward(*S[:6])
    motion = SimpleNamespace(root_p=p.clone(), root_R=R.clone(),
        root_v=v.clone(), root_w=w.clone(), q=q.clone(), qd=qd.clone(),
        body_p=bp.clone(), body_R=bR.clone(), body_v=bv.clone(), body_w=bw.clone())
    xi = torch.randn(6, CONTEXT_DIM, dtype=torch.float64) * .01
    a = torch.randn(6, 31, dtype=torch.float64)
    f = torch.arange(6)
    limits = torch.full((31,), 10., dtype=torch.float64)
    old = make_mdp(kin, motion, kin.default_q, limits)
    new = make_mdp(kin, motion, kin.default_q, limits, training_variant="goal-sim2real")
    return kin, motion, S, xi, a, f, old, new


def test_legacy_105_keeps_exact_observation_arithmetic(case):
    kin, motion, S, xi, a, f, old, new = case
    imu_R, gyro, gravity = kin.imu(*S[:6])
    torso = kin.tracked_names.index("waist_yaw_link")
    rel = mat_to_6d(imu_R.transpose(-1, -2) @ (motion.body_R[f, torso] @ kin.imu_site_R))
    expected = torch.cat((S[4] - kin.default_q + xi[..., CONTEXT_SLICES["encoder_bias"]],
                          S[5], gyro, gravity, a, rel), -1)
    torch.testing.assert_close(old.obs(S, a, xi, f), expected, rtol=0, atol=0)
    assert old.training_variant == "goal-sim" and old.obs_dim == OBS_DIM == 105
    assert new.obs_dim == 99
    assert old.obs_noise.shape == (105,) and new.obs_noise.shape == (99,)
    torch.testing.assert_close(old.obs_noise[:99], new.obs_noise)
    assert new.add_noise(new.obs(S, a, xi, f)).shape == (6, 99)


def test_raw_sensor_matches_simulation_without_double_bias(case):
    kin, _, S, xi, a, f, _, new = case
    _, gyro, gravity = kin.imu(*S[:6])
    q_sensor = S[4] + xi[..., CONTEXT_SLICES["encoder_bias"]]
    raw = sensor_observation(q_sensor, S[5], gyro, gravity, a, kin.default_q)
    torch.testing.assert_close(raw, new.obs(S, a, xi, f), rtol=0, atol=0)
    torch.testing.assert_close(raw[:, :31], q_sensor - kin.default_q, rtol=0, atol=0)
    history = tuple(x[:, None].expand(-1, 3, *x.shape[1:]) for x in (q_sensor, S[5], gyro, gravity, a))
    packed = sensor_observation(*history, kin.default_q)
    assert packed.shape == (6, 3, 99)
    torch.testing.assert_close(packed[:, 0], raw)


def test_actual_yaw_and_translation_do_not_reveal_global_pose(case):
    _, _, S, xi, a, f, old, new = case
    Z = so3_exp(torch.tensor([0., 0., 1.2], dtype=torch.float64))
    rotate = lambda x: (Z @ x[..., None]).squeeze(-1)
    transformed = (rotate(S[0]) + S[0].new_tensor([6., -3., 0.]),
                   Z @ S[1], rotate(S[2]), rotate(S[3]), *S[4:])
    torch.testing.assert_close(new.obs(S, a, xi, f), new.obs(transformed, a, xi, f))
    assert not torch.allclose(old.obs(S, a, xi, f)[..., -6:], old.obs(transformed, a, xi, f)[..., -6:])


def test_reference_pose_frame_and_privileged_state_are_not_sensor_inputs(case):
    _, motion, S, xi, a, f, _, new = case
    expected = new.obs(S, a, xi, f)
    changed_xi = xi + 4
    changed_xi[..., CONTEXT_SLICES["encoder_bias"]] = xi[..., CONTEXT_SLICES["encoder_bias"]]
    changed_state = (S[0] + 9, S[1], S[2] + 13, *S[3:6], S[6] + 5)
    motion.body_R = so3_exp(S[0].new_tensor([0., 0., 2.])) @ motion.body_R
    # No reference frame lookup is needed by the sensor-only adapter.
    torch.testing.assert_close(new.obs(changed_state, a, changed_xi, f + 1000), expected)


def test_measured_angular_velocity_and_tilt_remain_observable(case):
    _, _, S, xi, a, f, _, new = case
    original = new.obs(S, a, xi, f)
    faster = (*S[:3], S[3] + 1, *S[4:])
    assert not torch.allclose(new.obs(faster, a, xi, f)[..., 62:65], original[..., 62:65])
    tilt = so3_exp(S[0].new_tensor([.3, -.2, 0.]))
    tilted = (S[0], tilt @ S[1], *S[2:])
    assert not torch.allclose(new.obs(tilted, a, xi, f)[..., 65:68], original[..., 65:68])


def test_gravity_estimate_is_preserved_not_reconstructed_from_acceleration(case):
    kin, _, S, _, a, _, _, _ = case
    gyro = torch.zeros(6, 3, dtype=torch.float64)
    gravity = torch.tensor([.2, .1, -.9], dtype=torch.float64).expand(6, 3)
    packed = sensor_observation(S[4], S[5], gyro, gravity, a, kin.default_q)
    # A supplied estimator sample (including its small error) is not normalized
    # or reinterpreted as accelerometer data by the packing function.
    torch.testing.assert_close(packed[..., 65:68], gravity, rtol=0, atol=0)


def test_variant_does_not_change_reward_or_failures(case):
    _, _, S, _, a, f, old, new = case
    displaced = (S[0] + .25, S[1], *S[2:])
    for state in (S, displaced):
        torch.testing.assert_close(old.reward(state, f, a, a), new.reward(state, f, a, a), rtol=0, atol=0)
        torch.testing.assert_close(old.termination(state, f), new.termination(state, f))
        torch.testing.assert_close(old.soft_termination(state, f), new.soft_termination(state, f), rtol=0, atol=0)


def test_legacy_reward_factory_selects_sensor_mode_without_changing_defaults(case):
    from opensprite_tracker.supermimic.legacy_rewards import LegacyMDP, LegacyRewardCfg

    kin, motion, S, xi, a, f, _, new = case
    motion.num_frames = len(f)
    motion.ref_state = lambda frames: tuple(x[frames] for x in S[:6])
    limits = torch.full((31,), 10., dtype=torch.float64)
    default = LegacyMDP(kin, motion, kin.default_q, limits)
    selected = make_mdp(kin, motion, kin.default_q, limits, LegacyRewardCfg(),
                        training_variant="goal-sim2real")
    assert default.training_variant == "goal-sim" and default.obs_dim == 105
    assert selected.training_variant == "goal-sim2real" and selected.obs_dim == 99
    torch.testing.assert_close(selected.obs(S, a, xi, f), new.obs(S, a, xi, f))
    for name, value in default.reward_terms(S, f, a, a).items():
        torch.testing.assert_close(value, selected.reward_terms(S, f, a, a)[name], rtol=0, atol=0)
    torch.testing.assert_close(default.termination(S, f), selected.termination(S, f))
    torch.testing.assert_close(default.soft_termination(S, f), selected.soft_termination(S, f))


def test_reference_goal_is_invariant_to_reference_translation_and_yaw(case):
    _, source, _, _, _, f, _, _ = case
    def motion_from(p, R, v, w):
        m = MotionLib.__new__(MotionLib)
        m.root_p, m.root_R = p, R
        m.root_v_local = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
        m.root_w_local = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
        m.q, m.qd = source.q, source.qd
        m.clip_id, m.clip_end = torch.zeros(6, dtype=torch.long), torch.tensor([5])
        return m
    # Include body-X vertical and inverted orientations used by lying dances.
    R = source.root_R.clone()
    R[0] = so3_exp(R.new_tensor([0., torch.pi / 2, 0.]))
    R[1] = so3_exp(R.new_tensor([torch.pi, 0., 0.]))
    Z = so3_exp(R.new_tensor([0., 0., -1.4]))
    rotate = lambda x: (Z @ x[..., None]).squeeze(-1)
    before = motion_from(source.root_p, R, source.root_v, source.root_w)
    after = motion_from(rotate(source.root_p) + 4, Z @ R, rotate(source.root_v), rotate(source.root_w))
    goal = before.goal(f, 10)
    assert goal.shape == (6, 770) and torch.isfinite(goal).all()
    torch.testing.assert_close(goal, after.goal(f, 10))


def test_unknown_variant_and_malformed_sensor_layout_fail(case):
    kin, motion, S, _, a, _, _, _ = case
    with pytest.raises(ValueError, match="Unknown training variant"):
        MDP(kin, motion, kin.default_q, torch.ones(31), training_variant="typo")
    with pytest.raises(ValueError, match="qd must have shape"):
        sensor_observation(S[4], S[5][..., :30], S[3], S[3], a, kin.default_q)
