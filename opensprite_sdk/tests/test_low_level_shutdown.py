"""SDK write acceptance must not be mistaken for execution acknowledgement."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensprite_sdk.messages import LowState_, JointState


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("sdk_low_level_shutdown",ROOT/"examples/low_level.py")
EXAMPLE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def state(sequence, status="enabled", session="sim-session"):
    return LowState_(seq=sequence, clock_domain="sim", session_id=session,
                     status=status, joint_state=[JointState(valid=True) for _ in range(31)])


def exercise(monkeypatch, acknowledgements):
    clock=SimpleNamespace(now=0.)
    client=SimpleNamespace(closed=False, written=[])
    values=iter([state(1),state(2),*acknowledgements])
    class FakeClient:
        def __enter__(self):return self
        def __exit__(self,*args):client.closed=True
        def Read(self,timeout=1.):return next(values)
        def Write(self,message):client.written.append(deepcopy(message));return True
    monkeypatch.setattr(EXAMPLE,"LowLevelClient",FakeClient)
    monkeypatch.setattr(EXAMPLE,"ChannelFactoryInitialize",lambda *a,**k:None)
    monkeypatch.setattr(EXAMPLE.time,"monotonic",lambda:clock.now)
    monkeypatch.setattr(EXAMPLE.time,"sleep",lambda value:setattr(clock,"now",clock.now+value))
    args=SimpleNamespace(domain=1,endpoint="unused.sock",kp=2.,kd=.12,gains_profile=None,
                         gains=None,duration=.01,joint=30,amplitude=0.)
    return client,args


def test_final_disable_waits_past_queued_enabled_snapshot(monkeypatch):
    client,args=exercise(monkeypatch,[state(3),state(4,"disabled")])
    result=EXAMPLE.run(args)
    assert result["status"]=="disabled" and result["last_state_sequence"]==4
    assert client.closed and len(client.written)==2
    assert all(j.mode==0 for j in client.written[-1].joint_cmd)


def test_stale_disabled_snapshot_is_not_a_new_execution_ack(monkeypatch):
    client,args=exercise(monkeypatch,[state(2,"disabled"),state(3,"disabled")])
    assert EXAMPLE.run(args)["last_state_sequence"]==3
    assert client.closed


def test_changed_epoch_during_shutdown_requires_explicit_restart(monkeypatch):
    client,args=exercise(monkeypatch,[state(3,"disabled","different-sim-session")])
    with pytest.raises(RuntimeError,match="session changed"):
        EXAMPLE.run(args)
    assert client.closed
