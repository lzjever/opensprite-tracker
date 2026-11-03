import multiprocessing
import socket
import threading
import time
import pytest
from opensprite_sdk.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
    ChannelSubscriber, ChannelServer, FRAME_CAP, _encode, _decode)
from opensprite_sdk.messages import LowCmd_, LowState_
from opensprite_sdk.utils.thread import RecurrentThread


@pytest.fixture
def memory():
    ChannelFactoryInitialize(72)


def test_memory_latest_snapshot_and_single_writer(memory):
    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    subscriber.Init()
    publisher.Init()
    second = ChannelPublisher("rt/lowcmd", LowCmd_)
    try:
        with pytest.raises(RuntimeError, match="already has"):
            second.Init()
        message = LowCmd_()
        for value in range(5):
            message.motor_cmd[0].q = value
            publisher.Write(message)
        message.motor_cmd[0].q = 100
        read = subscriber.Read(.1)
        assert read.motor_cmd[0].q == 4
        assert read.seq == 5 and read.stamp_ns > 0 and read.session_id
        assert subscriber.Read(0) is None
        assert message.seq == 0 and message.session_id == ""
    finally:
        publisher.Close()
        second.Close()
        subscriber.Close()
    replacement = ChannelPublisher("rt/lowcmd", LowCmd_)
    replacement.Init()
    replacement.Close()


def test_memory_domain_isolation(memory):
    subscriber = ChannelSubscriber("test", LowCmd_)
    subscriber.Init()
    ChannelFactoryInitialize(73)
    publisher = ChannelPublisher("test", LowCmd_)
    publisher.Init()
    try:
        publisher.Write(LowCmd_())
        assert subscriber.Read(.01) is None
    finally:
        subscriber.Close()
        publisher.Close()


def test_callback_errors_are_visible_and_do_not_prevent_close(memory):
    called = threading.Event()
    def callback(message):
        called.set()
        raise RuntimeError("user callback error")
    subscriber = ChannelSubscriber("callback", LowState_)
    subscriber.Init(callback, 2)
    publisher = ChannelPublisher("callback", LowState_)
    publisher.Init()
    try:
        publisher.Write(LowState_())
        assert called.wait(1)
        deadline = time.monotonic() + 1
        while subscriber.last_error is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert isinstance(subscriber.last_error, RuntimeError)
    finally:
        publisher.Close()
        subscriber.Close()
    assert subscriber._threads == []


def test_read_wait_is_woken_by_close(memory):
    subscriber = ChannelSubscriber("blocked", LowCmd_)
    subscriber.Init()
    result = []
    thread = threading.Thread(target=lambda: result.append(subscriber.Read()))
    thread.start()
    subscriber.Close()
    thread.join(1)
    assert result == [None] and not thread.is_alive()


def test_publisher_rejects_replay_session_change_and_wrong_type(memory):
    publisher = ChannelPublisher("replay", LowCmd_)
    publisher.Init()
    try:
        publisher.Write(LowCmd_(seq=3, session_id="fixed"))
        with pytest.raises(ValueError, match="increase"):
            publisher.Write(LowCmd_(seq=3, session_id="fixed"))
        with pytest.raises(ValueError, match="session changed"):
            publisher.Write(LowCmd_(seq=4, session_id="other"))
        with pytest.raises(ValueError, match="wrong message"):
            publisher.Write(LowState_())
    finally:
        publisher.Close()


@pytest.mark.parametrize("transport", ["memory", "unix"])
def test_state_backend_epoch_changes_are_visible_without_reopening_publisher(tmp_path, transport):
    endpoint = str(tmp_path / "epochs.sock")
    server = ChannelServer(endpoint, domain_id=19) if transport == "unix" else None
    if server:
        server.start()
    ChannelFactoryInitialize(19, transport=transport, endpoint=endpoint if server else None)
    publisher = ChannelPublisher("rt/lowstate", LowState_)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    try:
        publisher.Init()
        subscriber.Init()
        publisher.Write(LowState_(seq=8, session_id="before-disable"))
        first = subscriber.Read(1)
        assert first.seq == 8 and first.session_id == "before-disable"
        publisher.Write(LowState_(seq=1, session_id="after-disable"))
        second = subscriber.Read(1)
        assert second.seq == 1 and second.session_id == "after-disable"
        with pytest.raises(ValueError, match="increase"):
            publisher.Write(LowState_(seq=1, session_id="after-disable"))
    finally:
        publisher.Close()
        subscriber.Close()
        if server:
            server.close()


def _child_publish(endpoint, result):
    try:
        ChannelFactoryInitialize(7, transport="unix", endpoint=endpoint)
        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        command = LowCmd_(clock_domain="sim", stamp_ns=0)
        command.motor_cmd[0].q = .25
        publisher.Write(command)
        publisher.Close()
        result.put("ok")
    except Exception as error:
        result.put(repr(error))


def test_unix_multiprocess_delivery_and_reconnect(tmp_path):
    endpoint = str(tmp_path / "channel.sock")
    context = multiprocessing.get_context("spawn")
    with ChannelServer(endpoint, domain_id=7):
        ChannelFactoryInitialize(7, transport="unix", endpoint=endpoint)
        subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        subscriber.Init()
        result = context.Queue()
        child = context.Process(target=_child_publish, args=(endpoint, result))
        child.start()
        try:
            state = subscriber.Read(5)
            assert state is not None and state.joint_cmd[0].q == .25
            assert state.clock_domain == "sim" and state.stamp_ns == 0
            assert result.get(timeout=5) == "ok"
            child.join(5)
            assert child.exitcode == 0
            replacement = ChannelPublisher("rt/lowcmd", LowCmd_)
            deadline = time.monotonic() + 2
            while True:
                try:
                    replacement.Init()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.01)
            try:
                replacement.Write(LowCmd_())
                second = subscriber.Read(1)
                assert second.seq == 1 and second.session_id != state.session_id
                assert subscriber.last_error is None
            finally:
                replacement.Close()
        finally:
            subscriber.Close()
            if child.is_alive():
                child.terminate()
                child.join(1)
            result.close()
    assert not (tmp_path / "channel.sock").exists()


def test_unix_rejects_second_publisher_and_wrong_domain(tmp_path):
    endpoint = str(tmp_path / "channel.sock")
    with ChannelServer(endpoint):
        ChannelFactoryInitialize(transport="unix", endpoint=endpoint)
        first = ChannelPublisher("rt/lowcmd", LowCmd_)
        first.Init()
        try:
            with pytest.raises(RuntimeError, match="already has"):
                ChannelPublisher("rt/lowcmd", LowCmd_).Init()
            ChannelFactoryInitialize(1, transport="unix", endpoint=endpoint)
            with pytest.raises(RuntimeError, match="domain mismatch"):
                ChannelSubscriber("rt/lowcmd", LowCmd_).Init()
        finally:
            first.Close()


def test_codec_and_server_reject_invalid_frames(tmp_path):
    for payload in (b"not json", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b"x"*(FRAME_CAP+1)):
        with pytest.raises(ValueError):
            _decode(payload)
    endpoint = str(tmp_path / "channel.sock")
    with ChannelServer(endpoint):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(1)
        client.connect(endpoint)
        try:
            client.send(b'{"op":"subscriber","topic":[],"type":"LowCmd_","domain":0}')
            assert "error" in _decode(client.recv(FRAME_CAP))
        finally:
            client.close()


def test_server_never_unlinks_preexisting_file(tmp_path):
    endpoint = tmp_path / "important"
    endpoint.write_text("preserve")
    server = ChannelServer(endpoint)
    with pytest.raises(FileExistsError):
        server.start()
    server.close()
    assert endpoint.read_text() == "preserve"


def test_recurrent_thread_exposes_exception_and_stops():
    def fail():
        raise RuntimeError("control failed")
    thread = RecurrentThread(.01, fail)
    thread.Start()
    thread.Wait(1)
    thread.Stop()
    assert isinstance(thread.last_error, RuntimeError)
    assert not thread._thread.is_alive()
