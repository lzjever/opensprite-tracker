"""Interactive viewer control lifecycle through real SDK messages and physics."""
import importlib.util
from pathlib import Path

from opensprite_sdk.backends.mujoco import MujocoBackend
from opensprite_sdk.policy import PolicyBundle
from test_policy import bundle_fixture


spec = importlib.util.spec_from_file_location(
    "view_policy", Path(__file__).resolve().parents[1] / "examples/view_policy.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def make_session(tmp_path, *, fail=False):
    with MujocoBackend() as backend:
        qpos = backend.data.qpos.copy()

    def reference(data):
        data["qpos"][:] = qpos
        if fail:
            data["qpos"][1:, 2] += 1

    root, actor = bundle_fixture(tmp_path, mutate_reference=reference)
    return viewer.SimulationSession(PolicyBundle(root, session=actor))


def test_end_of_reference_holds_without_reset_or_wrap(tmp_path):
    session = make_session(tmp_path)
    try:
        session.start()
        for _ in range(4):
            assert session.step()
        state = session.backend.debug_truth()["qpos"].copy()
        assert session.finished and session.frame == 4
        assert not session.step()
        assert (session.backend.debug_truth()["qpos"] == state).all()
    finally:
        session.close()


def test_first_failure_does_not_stop_policy_or_physics(tmp_path):
    session = make_session(tmp_path, fail=True)
    try:
        session.start()
        assert session.step() and session.first_failure == .02
        for _ in range(3):
            assert session.step()
        assert session.finished and session.first_failure == .02
        assert session.backend.data.time == .08
    finally:
        session.close()


def test_manual_restart_renews_epochs_and_actor_history(tmp_path):
    session = make_session(tmp_path)
    try:
        session.start()
        session.step()
        old_state = session.backend.read()
        old_actor_epoch = session.bundle.session_id
        session.restart()
        new_state = session.backend.read()
        assert new_state.session_id != old_state.session_id
        assert session.bundle.session_id != old_actor_epoch
        assert session.bundle.history is None
        assert session.frame == 0 and session.first_failure is None
        assert session.step() and session.frame == 1
    finally:
        session.close()
