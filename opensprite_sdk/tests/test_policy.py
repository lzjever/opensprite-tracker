"""Deployment contracts and public-channel integration, with a CPU fake actor."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from opensprite_sdk.constants import JOINT_NAMES
from opensprite_sdk.messages import LowState_, JointState, IMUState
from opensprite_sdk.policy import PolicyBundle, sensor_observation
from opensprite_sdk.client import LowLevelClient, joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelServer
from opensprite_sdk.service import BackendService


class FakeSession:
    def __init__(self, output=None):
        self.output = np.full((1, 31), .1, np.float32) if output is None else output
        self.calls = []

    def run(self, outputs, inputs):
        assert outputs == ["action"]
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        return [self.output.copy()]


def state_fixture(*, sequence=1, stamp=20_000_000, session="robot-session", clock="sim"):
    return LowState_(seq=sequence, stamp_ns=stamp, session_id=session, clock_domain=clock,
        joint_state=[JointState(q=.01*i, dq=.001*i, valid=True, sample_ns=stamp,
                               sample_time_kind="simulation" if clock=="sim" else "host_receive") for i in range(31)],
        imu_state=IMUState(gyroscope=(.1, .2, .3), valid=True, sample_ns=stamp,
            sample_time_kind="simulation" if clock=="sim" else "host_read", attached_link="waist_yaw_link"),
        status="enabled")


def bundle_fixture(tmp_path, *, mutate_contract=None, mutate_reference=None, output=None):
    root = tmp_path / "bundle"
    root.mkdir()
    frames = 5
    reference = {"goals":np.zeros((frames, 770), np.float32),
                 "qpos":np.zeros((frames, 38), np.float64),
                 "qvel":np.zeros((frames, 37), np.float64),
                 "root_rotation":np.tile(np.eye(3), (frames, 1, 1)),
                 "joint_names":np.asarray(JOINT_NAMES)}
    reference["qpos"][:, 3] = 1
    reference["goals"][:, 0] = np.arange(frames)
    if mutate_reference:
        mutate_reference(reference)
    np.savez_compressed(root / "reference.npz", **reference)
    (root / "actor.onnx").write_bytes(b"test actor session is injected; no ONNX runtime required")
    contract = {"schema":"opensprite-policy-v1", "training_variant":"goal-sim2real",
                "profile_id":"legacy_v4", "joint_names":list(JOINT_NAMES),
                "obs_dim":99, "goal_dim":770, "hist":10, "policy_hz":50,
                "frames":frames, "duration_seconds":.08, "imu_link":"waist_yaw_link",
                "default_q":[0.]*31, "kp":[10.]*31, "kd":[.5]*31,
                "action_scale":.25, "hardware_qualified":False,
                "action_clip":20.,
                "files":{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ("actor.onnx", "reference.npz")}}
    if mutate_contract:
        mutate_contract(contract)
    (root / "contract.json").write_text(json.dumps(contract))
    session = FakeSession(output)
    return root, session


def test_actor_input_history_reference_and_previous_action(tmp_path):
    root, session = bundle_fixture(tmp_path)
    bundle = PolicyBundle(root, session=session)
    first = state_fixture()
    command = bundle.command(first, 0)
    assert command.source_state_seq == first.seq
    assert command.source_state_session_id == first.session_id
    assert command.clock_domain == "sim" and command.seq == 1
    np.testing.assert_allclose([x.q for x in command.joint_cmd], .025)
    assert set(session.calls[0]) == {"observation_history", "goal"}
    history = session.calls[0]["observation_history"]
    assert history.shape == (1, 10, 99) and history.dtype == np.float32
    np.testing.assert_allclose(history[0, :, 62:65], np.tile([.1,.2,.3], (10, 1)))
    np.testing.assert_allclose(history[0, :, 65:68], np.tile([0.,0.,-1.], (10, 1)))
    assert np.count_nonzero(history[0, :, 68:]) == 0
    second = state_fixture(sequence=2, stamp=40_000_000)
    bundle.command(second, 1)
    np.testing.assert_allclose(session.calls[1]["observation_history"][0,-1,68:], .1)
    assert session.calls[1]["goal"][0,0] == 1


def test_global_truth_is_not_part_of_sensor_actor_input():
    a, b = state_fixture(), state_fixture()
    a.global_position = [0, 0, .5]
    b.global_position = [500, -100, 30]
    a.global_yaw, b.global_yaw = .1, -2.5
    np.testing.assert_array_equal(sensor_observation(a, np.zeros(31), np.zeros(31)),
                                  sensor_observation(b, np.zeros(31), np.zeros(31)))


@pytest.mark.parametrize("mutation", [
    lambda s:setattr(s, "stamp_ns", 20_000_001),
    lambda s:setattr(s.joint_state[4], "sample_ns", 20_000_001),
    lambda s:setattr(s.joint_state[4], "sample_ns", 0),
    lambda s:setattr(s.joint_state[4], "valid", False),
    lambda s:setattr(s.joint_state[4], "sample_time_kind", "unknown"),
    lambda s:setattr(s.imu_state, "sample_ns", 20_000_001),
    lambda s:setattr(s.imu_state, "valid", False),
    lambda s:setattr(s.imu_state, "sample_time_kind", "unknown"),
    lambda s:setattr(s, "status", "fault"),
    lambda s:s.faults.append("transport_fault"),
])
def test_sensor_observation_rejects_invalid_or_future_sources(mutation):
    state = state_fixture()
    mutation(state)
    with pytest.raises((ValueError, RuntimeError)):
        sensor_observation(state, np.zeros(31), np.zeros(31), now_ns=20_000_000,
                           max_motor_age_ns=10_000_000)


def test_actual_monotonic_age_is_used_when_now_omitted():
    state = state_fixture(stamp=time.monotonic_ns()-1_000_000_000, clock="monotonic")
    with pytest.raises(RuntimeError, match="stale"):
        sensor_observation(state, np.zeros(31), np.zeros(31))


def test_imu_age_independent_of_fresh_robot_envelope():
    state = state_fixture(stamp=100_000_000)
    state.imu_state.sample_ns = 60_000_000
    with pytest.raises(RuntimeError, match="IMU"):
        sensor_observation(state, np.zeros(31), np.zeros(31), now_ns=100_000_000)


def test_mismatched_imu_link_and_dynamic_acceleration_are_not_silently_used():
    state = state_fixture()
    state.imu_state.attached_link = "pelvis_link"
    with pytest.raises(ValueError, match="links differ"):
        sensor_observation(state, np.zeros(31), np.zeros(31))
    state.imu_state.attached_link = "waist_yaw_link"
    state.imu_state.projected_gravity = (0,0,-2)
    with pytest.raises(ValueError, match="unit gravity"):
        sensor_observation(state, np.zeros(31), np.zeros(31))


@pytest.mark.parametrize("mutate", [
    lambda c:c.update(training_variant="goal-sim"),
    lambda c:c.update(joint_names=list(reversed(JOINT_NAMES))),
    lambda c:c.update(obs_dim=105), lambda c:c.update(hist=8),
    lambda c:c.update(frames=5.), lambda c:c.update(frames=1),
    lambda c:c.update(profile_id=""), lambda c:c.update(imu_link=""),
    lambda c:c.update(default_q=[0.]*30), lambda c:c.update(kp=[-1.]*31),
    lambda c:c.update(kd=[float("nan")]*31),
    lambda c:c.update(action_scale=0), lambda c:c.update(action_scale=float("inf")),
    lambda c:c.update(hardware_qualified="false"),
    lambda c:c.update(files={"../actor.onnx":"fake"}),
])
def test_malformed_policy_contract_fails_before_inference(tmp_path, mutate):
    root, session = bundle_fixture(tmp_path, mutate_contract=mutate)
    with pytest.raises(ValueError):
        PolicyBundle(root, session=session)
    assert session.calls == []


@pytest.mark.parametrize("mutate", [
    lambda r:r.update(goals=np.zeros((5,769))),
    lambda r:r["goals"].__setitem__((0,0), np.nan),
    lambda r:r.update(qpos=np.zeros((4,38))),
    lambda r:r.update(qvel=np.zeros((5,36))),
    lambda r:r.update(root_rotation=np.zeros((5,3,2))),
    lambda r:r.update(root_rotation=np.zeros((5,3,3))),
    lambda r:r["qpos"].__setitem__((0,3), 0),
])
def test_malformed_reference_fails_before_inference(tmp_path, mutate):
    root, session = bundle_fixture(tmp_path, mutate_reference=mutate)
    with pytest.raises(ValueError):
        PolicyBundle(root, session=session)
    assert session.calls == []


def test_hash_tamper_fails_before_session_use(tmp_path):
    root, session = bundle_fixture(tmp_path)
    (root / "actor.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        PolicyBundle(root, session=session)


def test_reference_frames_are_once_only_no_wrap_or_hold(tmp_path):
    root, session = bundle_fixture(tmp_path)
    bundle = PolicyBundle(root, session=session)
    for invalid in (-1, True, 1):
        with pytest.raises(ValueError):
            bundle.command(state_fixture(), invalid)
    for frame in range(4):
        bundle.command(state_fixture(sequence=frame+1, stamp=(frame+1)*20_000_000), frame)
    for invalid in (0, 3, 4, 5):
        with pytest.raises(ValueError):
            bundle.command(state_fixture(sequence=5, stamp=100_000_000), invalid)
    assert len(session.calls) == 4


@pytest.mark.parametrize("mutation", [
    lambda s:setattr(s, "seq", 1),
    lambda s:setattr(s, "stamp_ns", 20_000_000),
    lambda s:setattr(s, "session_id", "new-state-epoch"),
    lambda s:setattr(s, "clock_domain", "monotonic"),
    lambda s:setattr(s, "profile_id", "physical_v5"),
])
def test_state_replay_epoch_clock_and_profile_rejected(tmp_path, mutation):
    root, session = bundle_fixture(tmp_path)
    bundle = PolicyBundle(root, session=session)
    bundle.command(state_fixture(), 0)
    state = state_fixture(sequence=2, stamp=40_000_000)
    mutation(state)
    with pytest.raises((ValueError, RuntimeError)):
        bundle.command(state, 1, now_ns=40_000_000)
    assert len(session.calls) == 1 and bundle.last_frame == 0


def test_explicit_policy_reset_starts_new_actor_session(tmp_path):
    root, session = bundle_fixture(tmp_path)
    bundle = PolicyBundle(root, session=session)
    before = bundle.command(state_fixture(), 0)
    bundle.reset()
    after = bundle.command(state_fixture(session="new-state-epoch"), 0)
    assert before.session_id != after.session_id
    assert after.seq == 1 and after.source_state_session_id == "new-state-epoch"
    assert np.count_nonzero(session.calls[-1]["observation_history"][0,:,68:]) == 0


@pytest.mark.parametrize("output", [np.zeros((31,), np.float32),
    np.full((1,31), np.nan, np.float32), np.zeros((2,31), np.float32)])
def test_bad_actor_output_does_not_commit_policy_history(tmp_path, output):
    root, session = bundle_fixture(tmp_path, output=output)
    bundle = PolicyBundle(root, session=session)
    with pytest.raises(RuntimeError):
        bundle.command(state_fixture(), 0)
    assert bundle.history is None and bundle.last_frame == -1 and bundle.sequence == 0


def test_action_clip_matches_trainer_and_previous_sent_action(tmp_path):
    output = np.full((1,31), 50., np.float32)
    output[0,1] = -50.
    root, session = bundle_fixture(tmp_path, output=output)
    bundle = PolicyBundle(root, session=session)
    command = bundle.command(state_fixture(), 0)
    assert command.joint_cmd[0].q == 5.
    assert command.joint_cmd[1].q == -5.
    assert bundle.previous_action[0] == 20.
    assert bundle.previous_action[1] == -20.
    bundle.command(state_fixture(sequence=2, stamp=40_000_000), 1)
    np.testing.assert_allclose(session.calls[-1]["observation_history"][0,-1,68:70], [20.,-20.])


@pytest.mark.parametrize("field,value", [("seq", 0), ("session_id", "")])
def test_missing_first_state_identity_fails_before_inference(tmp_path, field, value):
    root, session = bundle_fixture(tmp_path)
    bundle = PolicyBundle(root, session=session)
    state = state_fixture()
    setattr(state, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        bundle.command(state, 0)
    assert session.calls == []


class FakeBackend:
    def __init__(self):
        self.closed = False
        self.started = False
        self.enabled = False
        self.sequence = 0
        self.stamp = 0
        self.commands = []

    def start(self):
        self.started = True

    def read(self):
        self.sequence += 1
        return state_fixture(sequence=self.sequence, stamp=self.stamp)

    def write(self, command):
        self.commands.append(deepcopy(command))

    def step(self, count):
        self.stamp += count*5_000_000

    def close(self):
        self.closed = True


@pytest.mark.parametrize("transport", ["memory", "unix"])
def test_common_client_service_command_state_routing(tmp_path, transport):
    endpoint = str(tmp_path / "sdk.sock")
    server = ChannelServer(endpoint, domain_id=66) if transport == "unix" else None
    if server:
        server.start()
    ChannelFactoryInitialize(66, transport=transport, endpoint=endpoint if server else None)
    backend = FakeBackend()
    try:
        with BackendService(backend) as service, LowLevelClient() as client:
            service.publish_state()
            state = client.Read()
            command = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
            client.Write(command)
            deadline = time.monotonic()+2
            received = False
            while not received and time.monotonic() < deadline:
                received = service.tick(steps=4, wait_for_first_command=True)
                if not received:
                    time.sleep(.001)
            assert received and len(backend.commands) == 1
            received_command = backend.commands[0]
            assert received_command.source_state_session_id == state.session_id
            assert received_command.source_state_seq == state.seq
            assert received_command.seq == 1 and received_command.session_id
            assert backend.stamp == 20_000_000
            assert backend.started and not backend.enabled
            assert client.Read().joint_state[0].valid
    finally:
        if server:
            server.close()
    assert backend.closed


def test_service_start_failure_closes_backend(tmp_path):
    ChannelFactoryInitialize(1, transport="unix", endpoint=str(tmp_path/"missing.sock"))
    backend = FakeBackend()
    with pytest.raises(OSError):
        BackendService(backend).start()
    assert backend.started and backend.closed


def test_shadow_service_routes_to_preview_without_actuating_write():
    class ShadowBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.previews = []

        def preview(self, command):
            self.previews.append(deepcopy(command))

        def write(self, command):
            raise AssertionError("shadow routing must never call actuating write")

    ChannelFactoryInitialize(67)
    backend = ShadowBackend()
    with BackendService(backend, shadow=True) as service, LowLevelClient() as client:
        service.publish_state()
        state = client.Read()
        client.Write(joint_command(state, [0.]*31, [2.]*31, [.1]*31))
        assert service.tick()
        assert len(backend.previews) == 1
        assert backend.previews[0].source_state_session_id == state.session_id
        assert not backend.enabled and backend.commands == []
    assert backend.closed


def test_subframe_duration_does_not_leak_a_simulator(tmp_path, monkeypatch):
    """A CLI boundary may reject an empty run or close its constructed backend."""
    root, session = bundle_fixture(tmp_path)
    example_path = Path(__file__).resolve().parents[1] / "examples" / "run_policy.py"
    spec = importlib.util.spec_from_file_location("sdk_policy_example_test", example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    monkeypatch.setattr(example, "PolicyBundle", lambda path: PolicyBundle(path, session=session))
    from opensprite_sdk.backends import mujoco as simulation
    actual_backend = simulation.MujocoBackend
    created = []

    class TrackedBackend(actual_backend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(simulation, "MujocoBackend", TrackedBackend)
    args = SimpleNamespace(bundle=root, output=tmp_path/"run", backend="mujoco", seed=42,
                           duration=.001, video=False, actuate=False, hardware_config=None,
                           realtime=False, continue_after_failure=True)
    try:
        summary = example.run(args)
    except ValueError:
        pass
    else:
        assert summary["transitions"] == 0
        assert not summary["standard_full_success"]
    assert all(backend._closed for backend in created)
    assert session.calls == []
