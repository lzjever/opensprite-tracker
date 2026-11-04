"""Control ownership, robot-local watchdog and management acknowledgements."""
from copy import deepcopy
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from opensprite_sdk.backends.mujoco import MujocoBackend
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.messages import ControlRequest_
from opensprite_sdk.runtime import ManagementClient, RobotRuntime


@pytest.fixture
def runtime():
    ChannelFactoryInitialize(147, transport="memory")
    value = RobotRuntime(MujocoBackend(), allow_enable=True)
    value.start()
    try:
        yield value
    finally:
        value.close()


def request(runtime, operation, *, owner="controller", request_id=None):
    state = runtime.last_state
    runtime._test_request_seq = getattr(runtime, "_test_request_seq", 0)+1
    return ControlRequest_(profile_id=state.profile_id, stamp_ns=state.stamp_ns,
        clock_domain=state.clock_domain, session_id=owner, seq=runtime._test_request_seq,
        request_id=request_id or uuid.uuid4().hex, controller_session=owner,
        robot_session=state.session_id, operation=operation)


def arm(runtime):
    assert runtime.handle_request(request(runtime, "claim")).accepted
    assert runtime.handle_request(request(runtime, "enable")).accepted


def test_claim_never_enables_and_foreign_controller_cannot_take_over(runtime):
    assert runtime.handle_request(request(runtime, "claim")).accepted
    assert not runtime.enabled and not runtime.backend.enabled
    for operation in ("claim", "enable", "disable", "release"):
        reply = runtime.handle_request(request(runtime, operation, owner="foreign"))
        assert not reply.accepted and not reply.execution_verified
    assert runtime.owner == "controller"


def test_enable_requires_local_permission_and_matching_robot_epoch(runtime):
    assert runtime.handle_request(request(runtime, "claim")).accepted
    runtime.allow_enable = False
    assert not runtime.handle_request(request(runtime, "enable")).accepted
    runtime.allow_enable = True
    invalid = request(runtime, "enable")
    invalid.robot_session = "an-old-robot-session"
    assert not runtime.handle_request(invalid).accepted
    assert not runtime.backend.enabled


def test_request_retries_are_idempotent_and_changed_intent_is_rejected(runtime):
    value = request(runtime, "claim")
    reply = runtime.handle_request(value)
    assert reply.accepted
    retried = deepcopy(value)
    retried.seq += 10
    retried.stamp_ns += 1
    assert runtime.handle_request(retried) == reply
    changed = deepcopy(value)
    changed.operation = "enable"
    assert not runtime.handle_request(changed).accepted
    assert not runtime.backend.enabled


def test_discovery_and_enable_do_not_advance_initial_physics(runtime):
    for _ in range(10):
        runtime.tick(publish=True)
    assert runtime.backend.data.time == 0
    arm(runtime)
    for _ in range(5):
        runtime.tick(publish=True)
    assert runtime.backend.data.time == 0


def test_owner_command_starts_independent_physics_and_disable_is_confirmed(runtime):
    with ManagementClient(controller_session="controller") as client:
        runtime.publish_state()
        state = client.Read()
        arm(runtime)
        command = joint_command(state, [j.q for j in state.joint_state], [2.]*31, [.1]*31)
        client.Write(command)
        runtime.tick(publish=True)
        assert runtime.accepted_commands == 1 and runtime.backend.data.time == .005
        # No additional command and no diagnostic consumer is required per tick.
        runtime.tick(publish=True)
        assert runtime.backend.data.time == .01
        reply = runtime.handle_request(request(runtime, "disable"))
        assert reply.accepted and reply.execution_verified
        assert not runtime.backend.enabled and runtime.stop_requested


def test_forged_owner_command_is_dropped_without_disabling_legitimate_owner(runtime):
    arm(runtime)
    with ManagementClient(controller_session="foreign") as client:
        runtime.publish_state()
        state = client.Read()
        client.Write(joint_command(state, [0.]*31, [0.]*31, [0.]*31))
        runtime.tick()
    assert runtime.rejected_commands == 1
    assert runtime.enabled and runtime.backend.data.time == 0


def test_robot_local_wall_watchdog_expires_even_if_physics_did_not_start(runtime):
    current = [10.]
    runtime.clock = lambda: current[0]
    arm(runtime)
    current[0] += .101
    runtime.tick()
    assert runtime.fault == "controller_stream_watchdog"
    assert not runtime.backend.enabled and runtime.stop_requested


def test_management_client_correlates_ack_and_shares_controller_identity(runtime):
    errors = []
    finished = threading.Event()

    def client_program():
        try:
            with ManagementClient(controller_session="controller") as client:
                client.Read()
                assert client.request("claim", timeout=1.).status == "claimed"
                assert client.request("enable", timeout=1.).accepted
                assert client.request("disable", timeout=1.).execution_verified
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=client_program)
    thread.start()
    deadline = time.monotonic()+2.
    while not finished.is_set() and time.monotonic() < deadline:
        runtime.tick(publish=True)
        time.sleep(.001)
    thread.join(timeout=.1)
    assert finished.is_set() and not errors


def test_simulation_state_phase_and_command_acceptance_are_explicit(runtime):
    with ManagementClient(controller_session="controller") as client:
        runtime.publish_state()
        state = client.Read()
        initial_sequence = runtime.last_state.seq
        assert state.accepted_command_seq == 0
        assert state.accepted_controller_session == ""
        arm(runtime)
        client.Write(joint_command(state, [0.]*31, [0.]*31, [0.]*31))
        for _ in range(3):
            runtime.tick(publish=True)
            assert runtime.last_state.seq == initial_sequence
        runtime.tick(publish=True)
        arrived = client.Read()
        assert arrived.stamp_ns == 20_000_000
        assert arrived.accepted_command_seq == 1
        assert arrived.accepted_controller_session == "controller"


def test_management_replay_stays_rejected_after_response_cache_eviction(runtime):
    initial = request(runtime, "claim")
    assert runtime.handle_request(initial).accepted
    for _ in range(260):
        assert runtime.handle_request(request(runtime, "claim")).accepted
    assert (initial.controller_session, initial.request_id) not in runtime._cache
    replay = runtime.handle_request(initial)
    assert not replay.accepted and "replayed" in replay.detail


def test_enable_waits_for_low_command_dds_discovery_before_publishing(runtime):
    with ManagementClient(controller_session="controller") as client:
        runtime.publish_state()
        client.Read()
        assert runtime.handle_request(request(runtime, "claim")).accepted
        client.robot.publisher._config = SimpleNamespace(transport="dds")
        client.robot.publisher.WaitForMatch = lambda timeout: False
        with pytest.raises(TimeoutError, match="readiness"):
            client.request("enable", timeout=.01)
        assert runtime.requests.Read(0) is None
        assert not runtime.backend.enabled


def test_close_attempts_all_resources_and_preserves_first_failure(runtime, monkeypatch):
    closed = []
    original_backend = runtime.backend.close
    originals = []

    def backend_failure():
        original_backend()
        closed.append("backend")
        raise RuntimeError("backend close failed")

    monkeypatch.setattr(runtime.backend, "close", backend_failure)
    for channel in (runtime.command, runtime.requests, runtime.state_writer, runtime.replies):
        original = channel.Close
        originals.append((channel, original))

        def close(original=original, topic=channel.topic):
            original()
            closed.append(topic)
            if topic == "rt/lowcmd":
                raise ValueError("command close also failed")

        monkeypatch.setattr(channel, "Close", close)
    try:
        with pytest.raises(RuntimeError, match="backend close failed"):
            runtime.close()
        assert closed == ["backend", "rt/lowcmd", "rt/control_request", "rt/lowstate", "rt/control_reply"]
        assert len(runtime.cleanup_errors) == 2
        assert not runtime._reply_sender.is_alive()
    finally:
        monkeypatch.setattr(runtime.backend, "close", original_backend)
        for channel, original in originals:
            monkeypatch.setattr(channel, "Close", original)


def test_run_preserves_primary_error_when_cleanup_also_fails(monkeypatch):
    ChannelFactoryInitialize(148, transport="memory")
    runtime = RobotRuntime(MujocoBackend())
    original_backend = runtime.backend.close

    def tick(**kwargs):
        raise ValueError("primary control failure")

    def close():
        original_backend()
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(runtime, "tick", tick)
    monkeypatch.setattr(runtime.backend, "close", close)
    try:
        with pytest.raises(ValueError, match="primary control failure"):
            runtime.run(duration=.1)
        assert len(runtime.cleanup_errors) == 1
        assert all(not channel._active for channel in (
            runtime.command, runtime.requests, runtime.state_writer, runtime.replies))
    finally:
        original_backend()


def test_slow_reliable_reply_does_not_block_physics(runtime, monkeypatch):
    blocked = threading.Event()
    release = threading.Event()
    original_write = runtime.replies.Write

    def slow_reply(message, **kwargs):
        blocked.set()
        assert release.wait(1.)
        return original_write(message, **kwargs)

    monkeypatch.setattr(runtime.replies, "Write", slow_reply)
    try:
        arm(runtime)
        runtime._queue_reply(runtime.handle_request(request(runtime, "claim")))
        assert blocked.wait(1.)
        with ManagementClient(controller_session="controller") as client:
            runtime.publish_state()
            state = client.Read()
            client.Write(joint_command(state, [0.]*31, [0.]*31, [0.]*31))
            for _ in range(4):
                runtime.tick()
            assert runtime.backend.data.time == .02
            assert not release.is_set()
    finally:
        release.set()
