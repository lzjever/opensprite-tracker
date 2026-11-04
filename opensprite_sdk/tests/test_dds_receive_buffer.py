"""Real typed DDS receives remain isolated when sharing a bounded buffer."""
from copy import deepcopy
import threading
import time
import uuid

from opensprite_sdk.core.dds import DdsEndpoint
from opensprite_sdk.messages import LowState_, message_to_dict
from test_dds_integration import native_dds_available


def test_concurrent_takes_keep_long_and_short_payloads_isolated(native_dds_available):
    topic="rt/buffer_"+uuid.uuid4().hex
    reader=DdsEndpoint(180,"lo",topic,"LowState_","subscriber",history_depth=64)
    writer=DdsEndpoint(180,"lo",topic,"LowState_","publisher")
    workers=[]
    try:
        deadline=time.monotonic()+3
        while writer.matches()==0 and time.monotonic()<deadline:
            time.sleep(.002)
        assert writer.matches()==1
        for sequence in range(1,33):
            state=LowState_(seq=sequence,stamp_ns=sequence*5_000_000,
                clock_domain="sim",session_id="buffer-owner")
            state.joint_state[0].q=float(sequence)
            state.faults=["history "*100] if sequence%2 else []
            writer.write(message_to_dict(state))
            time.sleep(.001)
        received,errors=[],[]
        lock=threading.Lock()
        ready=threading.Barrier(3)

        def consume():
            ready.wait()
            try:
                while True:
                    item=reader.take(timeout=.05)
                    if item is None:
                        return
                    with lock:
                        received.append(deepcopy(item["data"]))
            except BaseException as error:
                errors.append(error)

        workers=[threading.Thread(target=consume) for _ in range(2)]
        for worker in workers:
            worker.start()
        ready.wait()
        for worker in workers:
            worker.join(timeout=2)
        assert all(not worker.is_alive() for worker in workers)
        assert not errors
        assert sorted(item["seq"] for item in received)==list(range(1,33))
        for item in received:
            assert item["joint_state"][0]["q"]==float(item["seq"])
            assert item["faults"]==(["history "*100] if item["seq"]%2 else [])
    finally:
        reader.close()
        writer.close()
        for worker in workers:
            worker.join(timeout=1)


def test_poll_does_not_wait_behind_another_blocked_take(native_dds_available):
    reader=DdsEndpoint(181,"lo","rt/poll_"+uuid.uuid4().hex,"LowState_","subscriber")
    started=threading.Event()
    results=[]

    def wait():
        started.set()
        results.append(reader.take(timeout=.25))

    worker=threading.Thread(target=wait)
    worker.start()
    try:
        assert started.wait(timeout=1)
        time.sleep(.01)
        before=time.monotonic()
        assert reader.take(timeout=0) is None
        assert time.monotonic()-before<.05
        # Discovery uses the independent lifetime guard, not the receive lock.
        assert reader.matches()==0
        reader.close()
        worker.join(timeout=1)
        assert not worker.is_alive() and results==[None]
    finally:
        reader.close()
        worker.join(timeout=1)
