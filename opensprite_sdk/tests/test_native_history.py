"""Actual200Hz C++ observations retained across bounded actor CPU stalls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import select
import sys
import time

import pytest

from opensprite_sdk.client import LowLevelClient
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.core.dds import DdsEndpoint
from opensprite_sdk.runtime import ManagementClient
from test_dds_integration import native_dds_available
from test_native_sdk import installed_cpp_peer, stop

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_history_policy", ROOT/"examples/remote_policy.py")
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def producer(peer, domain):
    return subprocess.Popen([str(peer), str(domain), "rt/lowstate", "state-stream"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_sent(child):
    assert select.select([child.stdout], [], [], 3.)[0], "producer did not send its observations"
    line = child.stdout.readline()
    assert json.loads(line)["samples"] == 120
    assert child.poll() is None  # the source stays alive during history inspection


def block_python_reader(seconds=.03):
    # C++ sends in a separate process, so this truly prevents the Python DDS
    # reader thread from draining its native history while observations arrive.
    previous = sys.getswitchinterval()
    sys.setswitchinterval(.1)
    try:
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            pass
    finally:
        sys.setswitchinterval(previous)


@pytest.mark.parametrize("client_type", [LowLevelClient, ManagementClient])
def test_real_observation_history_preserves_exact_policy_grid_after_cpu_stall(
        native_dds_available, installed_cpp_peer, client_type):
    ChannelFactoryInitialize(229, "lo", transport="dds")
    client = client_type(state_queue_len=64).Init()
    child = producer(installed_cpp_peer, 229)
    try:
        first = client.Read(timeout=3.)
        assert first.stamp_ns == 0
        block_python_reader()
        second = POLICY.next_state(client, first, target_stamp_ns=20_000_000)
        assert second.stamp_ns == 20_000_000 and second.seq == 5
        assert second.joint_state[0].q == .004
        assert all(j.sample_ns == second.stamp_ns for j in second.joint_state)
        third = POLICY.next_state(client, second, target_stamp_ns=40_000_000)
        assert third.stamp_ns == 40_000_000 and third.seq == 9
        assert third.joint_state[0].q == .008
    finally:
        client.Close()
        stop(child)


def test_bounded_history_does_not_hide_a_genuinely_lost_policy_window(native_dds_available, installed_cpp_peer):
    ChannelFactoryInitialize(230, "lo", transport="dds")
    client = LowLevelClient(state_queue_len=64).Init()
    child = producer(installed_cpp_peer, 230)
    try:
        first = client.Read(timeout=3.)
        assert first.stamp_ns == 0
        wait_sent(child)
        time.sleep(.03)
        with pytest.raises(RuntimeError, match="phase|missed|window"):
            POLICY.next_state(client, first, target_stamp_ns=20_000_000)
        drained = []
        while True:
            try:
                drained.append(client.Read(timeout=0))
            except TimeoutError:
                break
        assert len(drained) <= 63  # one expired-window observation was consumed
        assert drained[-1].seq == 120
        assert drained[-1].stamp_ns == 595_000_000
    finally:
        client.Close()
        stop(child)


def test_default_subscriber_remains_latest_state(native_dds_available, installed_cpp_peer):
    ChannelFactoryInitialize(231, "lo", transport="dds")
    client = LowLevelClient().Init()
    child = producer(installed_cpp_peer, 231)
    try:
        wait_sent(child)
        time.sleep(.03)
        latest = client.Read(timeout=.1)
        assert latest.seq == 120 and latest.stamp_ns == 595_000_000
        with pytest.raises(TimeoutError):
            client.Read(timeout=0)
    finally:
        client.Close()
        stop(child)


@pytest.mark.parametrize("depth,expected_count", [(1, 1), (64, 64)])
def test_native_reader_depth_is_explicit_bounded_and_chronological(
        native_dds_available, installed_cpp_peer, depth, expected_count):
    # No Python receiver thread calls take while the producer sends120 samples.
    # The retained data must therefore come from the actual DDS reader history.
    endpoint = DdsEndpoint(232, "lo", "rt/lowstate", "LowState_", "subscriber", history_depth=depth)
    child = producer(installed_cpp_peer, 232)
    try:
        wait_sent(child)
        retained = []
        while True:
            envelope = endpoint.take(timeout=0)
            if envelope is None:
                break
            retained.append(envelope["data"])
        assert len(retained) == expected_count
        assert [state["seq"] for state in retained] == list(range(121-expected_count, 121))
        assert all(state["stamp_ns"] == (state["seq"]-1)*5_000_000 for state in retained)
    finally:
        endpoint.close()
        stop(child)


def test_installed_cpp_subscriber_can_select_bounded_history(installed_cpp_peer):
    reader = subprocess.Popen([str(installed_cpp_peer), "232", "rt/lowstate", "state-history"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    try:
        assert select.select([reader.stdout], [], [], 3.)[0]
        assert reader.stdout.readline().strip() == "ready"
        child = producer(installed_cpp_peer, 232)
        wait_sent(child)
        stdout, stderr = reader.communicate(timeout=3.)
        assert reader.returncode == 0, stderr
        assert json.loads(stdout)["retained"] == 64
    finally:
        if child is not None:
            stop(child)
        stop(reader)
