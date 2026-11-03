"""A slow diagnostics process must not become a control publisher barrier."""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from opensprite_sdk.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
    ChannelSubscriber, ChannelServer)
from opensprite_sdk.messages import LowState_


READER = r'''
import json,sys,time
from opensprite_sdk.core.channel import ChannelFactoryInitialize,ChannelSubscriber
from opensprite_sdk.messages import LowState_
ChannelFactoryInitialize(46, transport="unix", endpoint=sys.argv[1])
subscriber=ChannelSubscriber("rt/lowstate",LowState_)
subscriber.Init(queueLen=1)
print("ready",flush=True)
expected=int(sys.stdin.readline())
sequences=[]
deadline=time.monotonic()+3
while time.monotonic()<deadline:
    message=subscriber.Read(.1)
    if message is not None:
        sequences.append(message.seq)
        if message.seq==expected:break
print(json.dumps({"sequences":sequences,"error":str(subscriber.last_error)}),flush=True)
subscriber.Close()
'''


def _paused_reader(endpoint):
    child = subprocess.Popen([sys.executable, "-c", READER, str(endpoint)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "ready"
    os.kill(child.pid, signal.SIGSTOP)
    return child


def _cleanup_child(child):
    if child.poll() is None:
        os.kill(child.pid, signal.SIGCONT)
        child.terminate()
        child.wait(timeout=3)
    for stream in (child.stdin, child.stdout, child.stderr):
        stream.close()


def _read_through(subscriber, sequence):
    deadline = time.monotonic()+2
    messages = []
    while time.monotonic()<deadline:
        message = subscriber.Read(.1)
        if message is not None:
            messages.append(message)
            if message.seq == sequence:
                return messages
    raise AssertionError(f"healthy subscriber did not receive sequence {sequence}")


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX process suspension")
def test_paused_sdk_reader_does_not_block_publisher_or_healthy_reader(tmp_path):
    endpoint=tmp_path/"backpressure.sock"
    with ChannelServer(endpoint, domain_id=46) as server:
        ChannelFactoryInitialize(46,transport="unix",endpoint=endpoint)
        healthy=ChannelSubscriber("rt/lowstate",LowState_)
        publisher=ChannelPublisher("rt/lowstate",LowState_)
        healthy.Init(queueLen=1)
        child=_paused_reader(endpoint)
        publisher.Init()
        try:
            received=[]
            for sequence in range(1,201):
                assert publisher.Write(LowState_(seq=sequence, session_id="controller"), timeout=.1)
                if sequence%20==0:
                    received.extend(_read_through(healthy,sequence))
            assert healthy.last_error is None
            assert len(received)>=10
            # The paused peer retains at most one pending userspace frame;
            # repeated writes must coalesce rather than accumulating memory.
            with server._lock:
                peers=tuple(server._subscribers["rt/lowstate"].values())
            assert any(peer.dropped_frames>0 for peer in peers)
            for peer in peers:
                with peer._condition:
                    assert peer._pending is None or isinstance(peer._pending,bytes)
            os.kill(child.pid,signal.SIGCONT)
            child.stdin.write("200\n")
            child.stdin.flush()
            stdout,stderr=child.communicate(timeout=4)
            assert child.returncode==0,stderr
            resumed=json.loads(stdout.strip())
            assert resumed["error"]=="None"
            assert resumed["sequences"][-1]==200
            assert all(a<b for a,b in zip(resumed["sequences"],resumed["sequences"][1:]))
        finally:
            publisher.Close()
            healthy.Close()
            _cleanup_child(child)


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX process suspension")
def test_shutdown_flushes_healthy_final_state_without_waiting_for_paused_peer(tmp_path):
    endpoint=tmp_path/"close.sock"
    server=ChannelServer(endpoint,domain_id=46).start()
    ChannelFactoryInitialize(46,transport="unix",endpoint=endpoint)
    healthy=ChannelSubscriber("rt/lowstate",LowState_)
    publisher=ChannelPublisher("rt/lowstate",LowState_)
    healthy.Init()
    child=_paused_reader(endpoint)
    publisher.Init()
    try:
        for sequence in range(1,101):
            assert publisher.Write(LowState_(seq=sequence,session_id="controller"),timeout=.1)
        assert publisher.Write(LowState_(seq=101,session_id="controller",status="disabled"),timeout=.1)
        publisher.Close()
        # Create/close descriptors concurrently with worker cancellation, so
        # select/close races cannot send through a recycled descriptor.
        recycling=threading.Event()
        stop=threading.Event()
        errors=[]
        def recycle_descriptors():
            recycling.set()
            while not stop.is_set():
                a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
                try:
                    b.setblocking(False)
                    try:
                        if b.recv(65536):errors.append("unexpected write on recycled descriptor")
                    except BlockingIOError:
                        pass
                finally:
                    a.close();b.close()
        recycler=threading.Thread(target=recycle_descriptors)
        recycler.start()
        assert recycling.wait(1)
        threads=tuple(server._threads)
        started=time.monotonic()
        try:
            server.close()
        finally:
            stop.set();recycler.join(1)
        assert time.monotonic()-started<1
        assert not errors and not recycler.is_alive()
        assert all(not thread.is_alive() for thread in threads)
        assert _read_through(healthy,101)[-1].status=="disabled"
        assert not endpoint.exists()
    finally:
        server.close()
        publisher.Close()
        healthy.Close()
        _cleanup_child(child)


def test_final_frame_survives_immediate_publisher_and_server_close(tmp_path):
    # Repeated scheduling variations exercise the pending-slot flush rather
    # than relying on the sender worker winning a one-off timing race.
    for index in range(12):
        endpoint=tmp_path/f"final-{index}.sock"
        server=ChannelServer(endpoint,domain_id=46).start()
        ChannelFactoryInitialize(46,transport="unix",endpoint=endpoint)
        publisher=ChannelPublisher("rt/lowstate",LowState_)
        subscriber=ChannelSubscriber("rt/lowstate",LowState_)
        subscriber.Init();publisher.Init()
        try:
            publisher.Write(LowState_(status="disabled"),timeout=.1)
            publisher.Close();server.close()
            message=subscriber.Read(1)
            assert message is not None and message.status=="disabled"
        finally:
            publisher.Close();subscriber.Close();server.close()
