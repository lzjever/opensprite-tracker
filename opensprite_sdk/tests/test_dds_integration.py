"""Real DDS processes, native type support, isolation and bounded readers.

Only absence of the optional native library skips these tests. A configured but
broken library, discovery failure or transport failure is a test failure.
"""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from opensprite_sdk import messages
from opensprite_sdk.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def native_dds_available():
    configured = os.environ.get("OPENSPRITE_DDS_LIBRARY")
    library = Path(configured) if configured else ROOT / "native-dds/build/libopensprite_dds.so"
    if not library.is_file():
        if configured:
            pytest.fail(f"configured DDS library does not exist: {library}")
        pytest.skip("optional native DDS library is not built; run tools/build_dds.sh")
    # Import/load failures intentionally propagate after the dependency exists.
    from opensprite_sdk.core.dds import DdsEndpoint
    # Concurrent reviewers may run this suite; robot runtime topics deliberately
    # use fixed names, so serialize integration cases across pytest processes.
    with open("/tmp/opensprite-sdk-dds-integration.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield DdsEndpoint


def domain():
    return 200 + int(uuid.uuid4().hex[:8], 16) % 20


def topic():
    return "rt/acceptance_" + uuid.uuid4().hex


def message(name):
    result = getattr(messages, name)()
    result.profile_id = "legacy_v4"
    if name == "LowCmd_":
        result.joint_cmd[30].q = .25
        result.source_state_seq = 2**53 + 3
        result.source_state_session_id = "source-robot"
    elif name == "LowState_":
        result.joint_state[30].q = .25
        result.imu_state.gyroscope = (.25, -.5, .75)
        result.faults = ["DDS诊断"]
    else:
        result.request_id = "request-1"
        result.controller_session = "controller-1"
        result.robot_session = "robot-1"
        if name == "ControlRequest_":
            result.operation = "claim"
        else:
            result.accepted = True
            result.status = "claimed"
            result.detail = "DDS诊断"
    return result


CHILD_READER = r'''
import json,sys
from opensprite_sdk import messages
from opensprite_sdk.core.channel import ChannelFactoryInitialize,ChannelSubscriber
ChannelFactoryInitialize(int(sys.argv[1]),interface="lo",transport="dds")
subscriber=ChannelSubscriber(sys.argv[2],getattr(messages,sys.argv[3]))
subscriber.Init()
print("ready",flush=True)
try:
    message=subscriber.Read(12)
    assert message is not None,repr(subscriber.last_error)
    print(json.dumps(messages.message_to_dict(message)),flush=True)
finally:
    subscriber.Close()
'''


def start_reader(dds_domain, channel, name):
    child = subprocess.Popen([sys.executable, "-u", "-c", CHILD_READER,
                              str(dds_domain), channel, name],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready, _, _ = select.select([child.stdout], [], [], 10)
    if not ready:
        cleanup(child)
        raise AssertionError("DDS reader subprocess did not initialize")
    line = child.stdout.readline().strip()
    if line != "ready":
        stdout, stderr = child.communicate(timeout=2)
        raise AssertionError((line, stdout, stderr))
    return child


def cleanup(child):
    if child.poll() is None:
        if hasattr(signal, "SIGCONT"):
            os.kill(child.pid, signal.SIGCONT)
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
    child.stdout.close()
    child.stderr.close()


@pytest.mark.parametrize("name", ["LowCmd_", "LowState_", "ControlRequest_", "ControlReply_"])
def test_real_dds_cross_process_public_channel(name):
    dds_domain, channel = domain(), topic()
    child = start_reader(dds_domain, channel, name)
    ChannelFactoryInitialize(dds_domain, interface="lo", transport="dds")
    publisher = ChannelPublisher(channel, getattr(messages, name))
    publisher.Init()
    expected = message(name)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            publisher.Write(expected, timeout=.1)
            available, _, _ = select.select([child.stdout], [], [], .025)
            if available:
                received = messages.message_from_dict(json.loads(child.stdout.readline()))
                break
        else:
            raise AssertionError("real DDS discovery/delivery did not complete")
        assert type(received) is type(expected)
        assert received.seq > 0 and received.session_id
        expected.seq, expected.stamp_ns, expected.session_id = (
            received.seq, received.stamp_ns, received.session_id)
        assert messages.message_to_dict(received) == messages.message_to_dict(expected)
        child.wait(timeout=3)
        assert child.returncode == 0, child.stderr.read()
    finally:
        publisher.Close()
        cleanup(child)


def test_dds_domain_isolation_has_positive_delivery_control():
    first_domain = domain()
    channel = topic()
    ChannelFactoryInitialize(first_domain, interface="lo", transport="dds")
    subscriber = ChannelSubscriber(channel, messages.LowCmd_)
    subscriber.Init()
    correct = ChannelPublisher(channel, messages.LowCmd_)
    correct.Init()
    ChannelFactoryInitialize(first_domain + 1, interface="lo", transport="dds")
    wrong = ChannelPublisher(channel, messages.LowCmd_)
    wrong.Init()
    try:
        deadline = time.monotonic() + 5
        actual = None
        while time.monotonic() < deadline and actual is None:
            bad = messages.LowCmd_()
            bad.joint_cmd[0].q = -99.
            wrong.Write(bad)
            good = messages.LowCmd_()
            good.joint_cmd[0].q = .5
            correct.Write(good)
            actual = subscriber.Read(.05)
        assert actual is not None and actual.joint_cmd[0].q == .5
        correct.Close()
        while subscriber.Read(.1) is not None:
            pass
        # Wrong-domain delivery is not merely obscured by the good latest value.
        for _ in range(10):
            wrong.Write(bad)
        assert subscriber.Read(.2) is None
    finally:
        wrong.Close()
        correct.Close()
        subscriber.Close()


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX process suspension")
def test_paused_dds_reader_does_not_block_publisher_or_healthy_reader():
    dds_domain, channel = domain(), topic()
    holding_reader = r"""
import sys,time
from opensprite_sdk.core.dds import DdsEndpoint
reader=DdsEndpoint(int(sys.argv[1]),"lo",sys.argv[2],"LowState_","subscriber")
print("ready",flush=True)
assert reader.take(5.) is not None
print("matched-and-received",flush=True)
time.sleep(30.)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", holding_reader, str(dds_domain), channel],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert select.select([child.stdout], [], [], 5.)[0]
    assert child.stdout.readline().strip() == "ready"
    ChannelFactoryInitialize(dds_domain, interface="lo", transport="dds")
    healthy = ChannelSubscriber(channel, messages.LowState_)
    healthy.Init()
    publisher = ChannelPublisher(channel, messages.LowState_)
    publisher.Init()
    try:
        # Confirm both readers matched and the soon-paused reader received a
        # sample. Stopping it before discovery would not exercise Data-sharing
        # coupling between a writer pool and its matched reader history.
        deadline = time.monotonic() + 5
        while publisher.MatchedCount() < 2:
            assert time.monotonic() < deadline
            time.sleep(.005)
        publisher.Write(messages.LowState_())
        assert select.select([child.stdout], [], [], 3.)[0]
        assert child.stdout.readline().strip() == "matched-and-received"
        os.kill(child.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            publisher.Write(messages.LowState_(), timeout=.1)
            if healthy.Read(.05) is not None:
                break
        else:
            raise AssertionError("healthy DDS peer did not discover")
        durations = []
        started = time.monotonic()
        for index in range(300):
            state = messages.LowState_()
            state.joint_state[0].q = float(index)
            before = time.monotonic()
            publisher.Write(state, timeout=.1)
            durations.append(time.monotonic() - before)
        assert time.monotonic() - started < 5
        assert max(durations) < .2
        final = messages.LowState_()
        final.joint_state[0].q = 299.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            publisher.Write(final, timeout=.1)
            state = healthy.Read(.05)
            if state is not None and state.joint_state[0].q == 299.:
                break
        else:
            raise AssertionError("healthy reader did not receive the newest state")
        assert healthy.last_error is None
    finally:
        publisher.Close()
        healthy.Close()
        cleanup(child)


def test_dds_close_wakes_a_blocked_read_without_live_threads():
    ChannelFactoryInitialize(domain(), interface="lo", transport="dds")
    subscriber = ChannelSubscriber(topic(), messages.LowState_)
    subscriber.Init()
    result = []
    blocked = threading.Thread(target=lambda: result.append(subscriber.Read()))
    blocked.start()
    subscriber.Close()
    blocked.join(timeout=2)
    assert not blocked.is_alive() and result == [None]


def test_native_malformed_envelope_is_rejected_without_poisoning_endpoint(native_dds_available):
    endpoint = native_dds_available(domain(), "lo", topic(), "LowCmd_", "publisher")
    try:
        valid = messages.message_to_dict(message("LowCmd_"))
        malformed = deepcopy(valid)
        malformed["data"].pop("joint_cmd")
        with pytest.raises((ValueError, RuntimeError)):
            endpoint.write(malformed, timeout=.1)
        # The same native endpoint remains usable after a conversion error.
        endpoint.write(valid, timeout=.1)
    finally:
        endpoint.close()


def test_native_blocked_take_does_not_starve_matches_or_close(native_dds_available):
    endpoint = native_dds_available(domain(), "lo", topic(), "LowState_", "subscriber")
    result = []
    reading = threading.Event()

    def read():
        reading.set()
        result.append(endpoint.take(timeout=.2))

    worker = threading.Thread(target=read)
    worker.start()
    assert reading.wait(1)
    time.sleep(.02)
    try:
        started = time.monotonic()
        assert endpoint.matches() == 0
        assert time.monotonic() - started < .1
        started = time.monotonic()
        endpoint.close()
        assert time.monotonic() - started < .5
        worker.join(timeout=1)
        assert not worker.is_alive() and result == [None]
        with pytest.raises(RuntimeError, match="closed"):
            endpoint.take(timeout=0)
    finally:
        endpoint.close()
        worker.join(timeout=1)


def test_invalid_interface_and_type_conflict_do_not_poison_native_context(native_dds_available):
    dds_domain, channel = domain(), topic()
    with pytest.raises(RuntimeError, match="interface"):
        native_dds_available(dds_domain, "opensprite_no_such_interface", channel,
                             "LowCmd_", "publisher")
    first = native_dds_available(dds_domain, "lo", channel, "LowCmd_", "publisher")
    second = None
    try:
        with pytest.raises(RuntimeError):
            native_dds_available(dds_domain, "lo", channel, "LowState_", "publisher")
        second = native_dds_available(dds_domain, "lo", channel, "LowCmd_", "subscriber")
        deadline = time.monotonic() + 3
        while first.matches() == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        first.write(messages.message_to_dict(message("LowCmd_")))
        assert second.take(timeout=1) is not None
    finally:
        if second is not None:
            second.close()
        first.close()


RUNTIME_CLIENT = r'''
import json,sys,time
from opensprite_sdk.client import joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.runtime import ManagementClient
ChannelFactoryInitialize(int(sys.argv[1]),interface="lo",transport="dds")
role=sys.argv[2]
with ManagementClient(controller_session=role) as client:
    state=client.Read(timeout=8)
    if role=="unclaimed":
        client.Write(joint_command(state,[j.q for j in state.joint_state],[2.]*31,[.1]*31))
        print("sent",flush=True)
        time.sleep(.2)
    elif role=="foreign":
        denied=[]
        for operation in ("claim","enable","disable","release"):
            try:client.request(operation,timeout=2)
            except RuntimeError:denied.append(operation)
        state=client.Read(timeout=2)
        for _ in range(8):
            client.Write(joint_command(state,[0.]*31,[0.]*31,[0.]*31))
            time.sleep(.03)
        print(json.dumps({"denied":denied}),flush=True)
        time.sleep(.1)
    else:
        client.request("claim",state=state)
        client.request("enable")
        print("streaming",flush=True)
        while True:
            state=client.Read(timeout=2)
            client.Write(joint_command(state,[j.q for j in state.joint_state],[2.]*31,[.1]*31))
'''


def start_runtime(dds_domain):
    from opensprite_sdk.backends.mujoco import MujocoBackend
    from opensprite_sdk.runtime import RobotRuntime
    ChannelFactoryInitialize(dds_domain, interface="lo", transport="dds")
    runtime = RobotRuntime(MujocoBackend(), allow_enable=True)
    stop = threading.Event()
    errors = []

    def run():
        try:
            runtime.run(duration=12., stop_event=stop)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 3
    while runtime.last_state is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert runtime.last_state is not None, errors
    return runtime, worker, stop, errors


def runtime_client(dds_domain, role):
    return subprocess.Popen([sys.executable, "-u", "-c", RUNTIME_CLIENT,
                              str(dds_domain), role], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)


def read_line(child, timeout=8.):
    available, _, _ = select.select([child.stdout], [], [], timeout)
    assert available, "DDS runtime client produced no result"
    line = child.stdout.readline().strip()
    if not line and child.poll() is not None:
        raise AssertionError(child.stderr.read())
    return line


def test_real_dds_unclaimed_command_cannot_enable_or_advance_robot():
    dds_domain = domain()
    runtime, worker, stop, errors = start_runtime(dds_domain)
    child = runtime_client(dds_domain, "unclaimed")
    try:
        assert read_line(child) == "sent"
        child.wait(timeout=3)
        assert child.returncode == 0, child.stderr.read()
        deadline = time.monotonic() + 2
        while runtime.rejected_commands == 0 and time.monotonic() < deadline:
            time.sleep(.005)
        assert runtime.rejected_commands >= 1
        assert not runtime.enabled and not runtime.backend.enabled
        assert runtime.accepted_commands == 0 and runtime.backend.data.time == 0
    finally:
        cleanup(child)
        stop.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and not errors


def test_real_dds_foreign_controller_rejected_and_owner_crash_trips_local_watchdog():
    dds_domain = domain()
    runtime, worker, stop, errors = start_runtime(dds_domain)
    owner = runtime_client(dds_domain, "owner")
    foreign = None
    try:
        assert read_line(owner) == "streaming"
        deadline = time.monotonic() + 3
        while runtime.accepted_commands < 3 and time.monotonic() < deadline:
            time.sleep(.005)
        assert runtime.accepted_commands >= 3 and runtime.enabled, errors
        foreign = runtime_client(dds_domain, "foreign")
        denied = json.loads(read_line(foreign))
        assert denied["denied"] == ["claim", "enable", "disable", "release"]
        foreign.wait(timeout=3)
        assert foreign.returncode == 0, foreign.stderr.read()
        assert runtime.owner == "owner" and runtime.enabled
        assert runtime.rejected_commands >= 1
        # SIGTERM deliberately bypasses SDK cleanup/disable management.
        killed = time.monotonic()
        owner.terminate()
        owner.wait(timeout=2)
        worker.join(timeout=2)
        assert not worker.is_alive() and time.monotonic() - killed < 1.
        assert runtime.fault in ("controller_stream_watchdog", "simulator_watchdog_or_physics_fault")
        assert not runtime.enabled and not runtime.backend.enabled
        assert runtime.accepted_controller_session == "owner"
    finally:
        if foreign is not None:
            cleanup(foreign)
        cleanup(owner)
        stop.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and not errors
