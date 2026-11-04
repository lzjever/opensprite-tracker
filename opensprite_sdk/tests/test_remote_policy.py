"""A dropped deployment observation cannot silently shift reference phase."""
import importlib.util
from pathlib import Path

import pytest

from opensprite_sdk.messages import LowState_


spec = importlib.util.spec_from_file_location(
    "remote_policy", Path(__file__).resolve().parents[1] / "examples/remote_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Client:
    def __init__(self, states):
        self.states = list(states)

    def Read(self, timeout):
        return self.states.pop(0)


def state(sequence, stamp, clock="sim", session="robot"):
    return LowState_(seq=sequence, stamp_ns=stamp, clock_domain=clock, session_id=session)


def test_idle_packets_are_skipped_and_exact20ms_observation_is_used():
    previous = state(1, 0)
    expected = state(7, 20_000_000)
    client = Client([state(2, 0), state(3, 0), expected])
    assert policy.next_state(client, previous) is expected


def test_lost_simulation_observation_fails_instead_of_shifting_reference():
    with pytest.raises(RuntimeError, match="phase drift"):
        policy.next_state(Client([state(12, 40_000_000)]), state(1, 0))


@pytest.mark.parametrize("following", [state(3, 132_000_000, "monotonic"),
    state(2, 141_000_000, "monotonic")])
def test_missing_or_late_hardware_observation_is_rejected(following):
    with pytest.raises(RuntimeError, match="hardware policy deadline"):
        policy.next_state(Client([following]), state(1, 100_000_000, "monotonic"))


def test_normal_native_schedule_jitter_is_accepted():
    following = state(2, 121_000_000, "monotonic")
    assert policy.next_state(Client([following]), state(1, 100_000_000, "monotonic")) is following


@pytest.mark.parametrize('clock', ['sim','monotonic'])
def test_500hz_observations_are_decimated_to50hz(clock):
    previous=state(1,100_000_000,clock)
    stream=[state(i+1,100_000_000+i*2_000_000,clock) for i in range(1,11)]
    expected=stream[-1]
    assert policy.next_state(Client(stream),previous) is expected
    assert expected.seq==11 and expected.stamp_ns==120_000_000


def test_hardware_reference_uses_fixed_anchor_instead_of_accumulating_jitter():
    previous=state(11,121_000_000,'monotonic')
    following=state(21,140_500_000,'monotonic')
    assert policy.next_state(Client([following]),previous,target_stamp_ns=140_000_000) is following


def test_source_restart_requires_explicit_policy_history_reset():
    with pytest.raises(RuntimeError, match="restart required"):
        policy.next_state(Client([state(2, 20_000_000, session="new")]), state(1, 0))
