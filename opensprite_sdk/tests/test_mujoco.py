"""Real CPU physics, frame, watchdog and packaged-asset integration checks."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import os
import ctypes.util

import mujoco
import numpy as np
import pytest

from opensprite_sdk.backends.mujoco import MujocoBackend, PACKAGE_ROOT, load_profile
from opensprite_sdk.constants import JOINT_NAMES
from opensprite_sdk.messages import JointCmd, LowCmd_, message_to_dict


def command(backend, *, mode=1):
    state = backend.read()
    return LowCmd_(profile_id=backend.profile_id, clock_domain="sim", seq=state.seq, session_id="physics-test-controller",
        stamp_ns=state.stamp_ns, source_state_seq=state.seq, source_state_session_id=state.session_id,
        joint_cmd=[JointCmd(mode, float(q), 0., float(kp), float(kd), 0.) for q, kp, kd in zip(backend.profile["default_q"], backend.profile["kp"], backend.profile["kd"])])


def test_packaged_model_limits_and_profile():
    with MujocoBackend() as backend:
        assert (backend.model.nq, backend.model.nv, backend.model.nu) == (38, 37, 0)
        assert tuple(backend.model.joint(i).name for i in range(1, 32)) == JOINT_NAMES
        assert backend.model.opt.timestep == .005
        assert backend.model.opt.iterations == 50
        assert backend.model.opt.ls_iterations == 25
        assert backend.profile["torque_lag_seconds"] == 0
        assert backend.policy_steps == 4
        state = backend.read()
        assert state.seq >= 1 and state.stamp_ns == 0 and state.clock_domain == "sim"
        assert state.imu_state.attached_link == "waist_yaw_link"
        assert all(x.sample_time_kind == "simulation" for x in state.joint_state)
        assert state.imu_state.sample_time_kind == "simulation"
        assert not any(x in message_to_dict(state)["data"] for x in ("root_position", "root_quaternion", "global_yaw", "qpos", "qvel"))
        np.testing.assert_allclose([j.q for j in state.joint_state], backend.profile["default_q"])


def test_nominal_stand_real_physics_ten_seconds():
    with MujocoBackend() as backend:
        backend.enable()
        minimum_height = math.inf
        for _ in range(500):
            backend.write(command(backend))
            backend.step(backend.policy_steps)
            truth = backend.debug_truth()
            minimum_height = min(minimum_height, truth["root_position"][2])
            assert np.all(np.abs(truth["applied_torque"]) <= np.asarray(backend.profile["effort_limit"]) + 1e-12)
        assert backend.read().status == "enabled"
        assert minimum_height > .40
        assert backend.data.time == pytest.approx(10)
        assert backend.read().seq >= 1000
        assert backend.read().stamp_ns == 10_000_000_000


def test_command_requires_explicit_enable_and_disable_is_zero_torque():
    with MujocoBackend() as backend:
        cmd = command(backend)
        with pytest.raises(RuntimeError, match="enable"):
            backend.write(cmd)
        backend.enable()
        cmd.joint_cmd[0].tau = 3.
        backend.write(cmd)
        backend.step()
        assert backend.debug_truth()["applied_torque"][0] != 0
        backend.write(command(backend, mode=0))
        assert backend.read().status == "disabled"
        assert not backend.data.qfrc_applied.any()
        backend.step()
        assert not backend.debug_truth()["applied_torque"].any()


def test_effort_limit_is_applied_to_complete_pd_and_feedforward():
    with MujocoBackend() as backend:
        backend.enable()
        cmd = command(backend)
        for item in cmd.joint_cmd:
            item.q += 10
            item.tau = 1e5
        backend.write(cmd)
        backend.step()
        truth = backend.debug_truth()
        assert np.max(truth["requested_torque"]) > 1e4
        np.testing.assert_array_equal(truth["applied_torque"], backend.profile["effort_limit"])
        np.testing.assert_array_equal(backend.data.qfrc_applied[backend.vids], backend.profile["effort_limit"])


def test_watchdog_expires_without_refresh_and_enable_cannot_refresh_it():
    with MujocoBackend() as backend:
        backend.enable()
        backend.write(command(backend))
        for _ in range(21):
            backend.enable()
            backend.step()
        assert backend.read().status == "fault"
        assert backend.read().faults == ["command_watchdog"]
        assert not backend.debug_truth()["applied_torque"].any()
        with pytest.raises(RuntimeError, match="reset"):
            backend.enable()
        backend.reset()
        backend.enable()
        assert backend.enabled


@pytest.mark.parametrize("mutation,match", [
    (lambda c: setattr(c, "profile_id", "other"), "profile"),
    (lambda c: setattr(c, "clock_domain", "monotonic"), "clock"),
    (lambda c: setattr(c, "stamp_ns", 1), "future"),
    (lambda c: setattr(c, "source_state_seq", 0), "source_state"),
    (lambda c: setattr(c, "source_state_seq", 2**32), "source_state"),
    (lambda c: setattr(c.joint_cmd[0], "mode", 0), "mixed"),
    (lambda c: setattr(c.joint_cmd[0], "kp", 501), "gains"),
    (lambda c: setattr(c.joint_cmd[0], "q", float("nan")), "finite"),
])
def test_malformed_or_mismatched_command_never_reaches_physics(mutation, match):
    with MujocoBackend() as backend:
        backend.enable()
        cmd = command(backend)
        mutation(cmd)
        with pytest.raises(ValueError, match=match):
            backend.write(cmd)
        assert backend._cmd is None
        assert not backend.data.qfrc_applied.any()


def test_sequence_replay_and_controller_ownership_are_rejected():
    with MujocoBackend() as backend:
        backend.enable()
        cmd = command(backend)
        cmd.seq = 1
        cmd.session_id = "controller_a"
        backend.write(cmd)
        with pytest.raises(ValueError, match="replayed"):
            backend.write(cmd)
        cmd.seq = 2
        cmd.session_id = "controller_b"
        with pytest.raises(ValueError, match="owns"):
            backend.write(cmd)


def test_stale_state_rejected_even_with_fresh_command_timestamp():
    with MujocoBackend() as backend:
        cmd = command(backend)
        backend.step(21)  # disabled physics may advance without an active command
        backend.enable()
        cmd.stamp_ns = backend.now_ns
        with pytest.raises(ValueError, match="source state is stale"):
            backend.write(cmd)


def test_gravity_has_no_global_yaw_and_correct_body_roll():
    with MujocoBackend() as backend:
        base = backend.data.qpos.copy()
        identity = backend.read().imu_state.projected_gravity
        yaw = base.copy()
        yaw[3:7] = [math.cos(.7), 0, 0, math.sin(.7)]
        backend.reset(qpos=yaw)
        np.testing.assert_allclose(backend.read().imu_state.projected_gravity, identity, atol=1e-12)
        roll = base.copy()
        angle = .3
        roll[3:7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
        backend.reset(qpos=roll)
        np.testing.assert_allclose(backend.read().imu_state.projected_gravity, [0, -math.sin(angle), -math.cos(angle)], atol=1e-12)


def test_torso_imu_includes_waist_joint_motion_and_differs_from_pelvis():
    with MujocoBackend() as backend:
        qvel = np.zeros(37)
        index = JOINT_NAMES.index("waist_yaw_joint")
        qvel[backend.vids[index]] = 1.
        backend.reset(qvel=qvel)
        assert np.linalg.norm(backend.read().imu_state.gyroscope) == pytest.approx(1.)
    with MujocoBackend("runtime_timing_v4") as backend:
        qvel = np.zeros(37)
        qvel[backend.vids[index]] = 1.
        backend.reset(qvel=qvel)
        assert backend.read().imu_state.attached_link == "pelvis_link"
        np.testing.assert_allclose(backend.read().imu_state.gyroscope, [0, 0, 0], atol=1e-12)
        assert backend.dt == .002 and backend.policy_steps == 10


def test_delay_is_seeded_bounded_and_applied_at_physics_rate():
    traces = []
    for _ in range(2):
        with MujocoBackend(seed=123) as backend:
            backend.enable()
            trace = []
            for i in range(100):
                cmd = command(backend)
                cmd.joint_cmd[0].tau = float(i)
                backend.write(cmd)
                backend.step()
                trace.append((backend.debug_truth()["command_lag_ticks"], backend._last_applied_command[0, 4]))
            traces.append(trace)
    assert traces[0] == traces[1]
    assert {x[0] for x in traces[0]} <= {0, 1}
    for i, (lag, target) in enumerate(traces[0]):
        assert target == float(max(0, i - lag))


def test_packaged_assets_sha256_and_no_parent_path_dependencies():
    manifest = json.loads((PACKAGE_ROOT / "assets/manifest.json").read_text())
    assert len(manifest["files"]) >= 35
    for relative, row in manifest["files"].items():
        path = PACKAGE_ROOT / relative
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    source = (PACKAGE_ROOT / "backends/mujoco.py").read_text()
    assert "opensprite_tracker" not in source and ".reference" not in source


def test_reset_close_and_invalid_profile_are_explicit(tmp_path):
    backend = MujocoBackend()
    with pytest.raises(ValueError, match="38-vector"):
        backend.reset(qpos=np.zeros(31))
    bad = backend.data.qpos.copy()
    bad[3:7] = 0
    with pytest.raises(ValueError, match="unit norm"):
        backend.reset(qpos=bad)
    with pytest.raises(ValueError, match="steps"):
        backend.step(-1)
    profile = load_profile()
    profile["physics_dt"] = .003
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="whole number"):
        load_profile(path)
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.read()


def test_pre_reset_command_cannot_be_replayed_into_new_simulator_session():
    with MujocoBackend() as backend:
        stale = command(backend)
        backend.reset()
        backend.enable()
        with pytest.raises(ValueError, match="another simulator session"):
            backend.write(stale)


@pytest.mark.skipif(not ctypes.util.find_library("OSMesa"), reason="CPU OSMesa is not installed")
def test_headless_cpu_render_in_fresh_process():
    code = """from opensprite_sdk.backends.mujoco import MujocoBackend
import json
with MujocoBackend() as b:
 im=b.render(width=320,height=240)
 assert im.shape==(240,320,3) and im.dtype.name=='uint8'
 assert float(im.std())>10
 print(json.dumps({'shape':list(im.shape),'std':float(im.std())}))
"""
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYTHONPATH": str(PACKAGE_ROOT.parent)}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["std"] > 10


def test_disable_enable_invalidates_old_sources_and_retains_controller_epoch():
    with MujocoBackend() as backend:
        backend.enable()
        cmd = command(backend)
        backend.write(cmd)
        old_session = backend.read().session_id
        backend.disable()
        assert backend.read().session_id == old_session
        backend.enable()
        with pytest.raises(ValueError, match="source_state_seq"):
            backend.write(cmd)
        fresh = command(backend)
        fresh.session_id = "different-controller"
        with pytest.raises(ValueError, match="owns"):
            backend.write(fresh)
        backend.write(command(backend))


@pytest.mark.parametrize("field,value", [("seq", 0), ("session_id", "")])
def test_direct_backend_also_requires_writer_identity(field, value):
    with MujocoBackend() as backend:
        backend.enable()
        cmd = command(backend)
        setattr(cmd, field, value)
        with pytest.raises(ValueError, match="positive sequence"):
            backend.write(cmd)


def test_paused_simulator_republishes_unique_sequences_with_same_sample_time():
    with MujocoBackend() as backend:
        first, second = backend.read(), backend.read()
        assert second.seq > first.seq
        assert first.stamp_ns == second.stamp_ns == 0
        assert all(j.sample_ns == 0 and j.tau_est_valid for j in second.joint_state)
        backend.enable()
        cmd = command(backend)
        cmd.source_state_seq = first.seq
        backend.write(cmd)
        backend.step()
        assert backend.read().stamp_ns == 5_000_000
