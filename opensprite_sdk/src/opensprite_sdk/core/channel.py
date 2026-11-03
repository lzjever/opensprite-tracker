"""Unitree-style local channels; memory and UNIX transports, not DDS wire compatible.

The UNIX broker protocol is SDK-specific JSON over SOCK_SEQPACKET. Its socket
must never be confused with the official native controller's binary IPC socket.
There is one publisher per domain/topic. These local channels are not a network
authentication mechanism; use a private UNIX socket directory for deployment.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import uuid
from ..messages import MESSAGE_TYPES, LowState_, message_from_dict, message_to_dict

FRAME_CAP = 65536


def _encode(value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > FRAME_CAP:
        raise ValueError("channel frame exceeds size limit")
    return payload


def _decode(payload):
    if len(payload) > FRAME_CAP:
        raise ValueError("channel frame exceeds size limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        result = json.loads(payload, object_pairs_hook=pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("invalid channel JSON") from error
    if not isinstance(result, dict):
        raise ValueError("channel frame must be an object")
    return result


def _send(sock, value):
    data = _encode(value)
    if sock.send(data) != len(data):
        raise ConnectionError("short channel write")


def _receive(sock):
    payload = sock.recv(FRAME_CAP + 1)
    if not payload:
        raise ConnectionError("channel disconnected")
    return _decode(payload)


class _MemoryBroker:
    def __init__(self):
        self.lock = threading.RLock()
        self.publishers = {}
        self.subscribers = {}

    def add_publisher(self, topic, publisher):
        with self.lock:
            if topic in self.publishers:
                raise RuntimeError(f"topic {topic} already has a publisher")
            self.publishers[topic] = publisher

    def publish(self, topic, publisher, message):
        with self.lock:
            if self.publishers.get(topic) is not publisher:
                raise RuntimeError("publisher does not own topic")
            targets = tuple(self.subscribers.get(topic, ()))
        for target in targets:
            target._deliver(message)


@dataclass(frozen=True)
class _Factory:
    domain: int = 0
    transport: str = "memory"
    endpoint: str | None = None


_factory = _Factory()
_brokers = {}
_factory_lock = threading.Lock()


def ChannelFactoryInitialize(domain=0, interface=None, *, transport="memory", endpoint=None):
    """Set defaults for subsequently initialized channels; never opens CAN.

    ``interface`` accepts None/lo for familiar SDK call sites. This local
    implementation does not configure a DDS network interface.
    """
    if type(domain) is not int or domain < 0 or domain > 232:
        raise ValueError("domain must be an integer in 0..232")
    if interface not in (None, "lo", "localhost"):
        raise ValueError("local channels only support the loopback interface")
    if transport not in ("memory", "unix"):
        raise ValueError("transport must be memory or unix")
    if transport == "unix" and (not endpoint or not isinstance(endpoint, (str, Path))):
        raise ValueError("unix transport requires an endpoint")
    global _factory
    with _factory_lock:
        _factory = _Factory(domain, transport, str(endpoint) if endpoint else None)


def _config():
    with _factory_lock:
        config = _factory
        broker = _brokers.setdefault((config.domain, config.endpoint), _MemoryBroker())
    return config, broker


def _topic(topic, message_type):
    if not isinstance(topic, str) or not 0 < len(topic) <= 128 or "\0" in topic:
        raise ValueError("invalid channel topic")
    if message_type not in MESSAGE_TYPES.values():
        raise ValueError("unsupported channel message type")


def _connect(config, role, topic, message_type):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.settimeout(2)
    try:
        sock.connect(config.endpoint)
        _send(sock, {"op": role, "domain": config.domain, "topic": topic,
                     "type": message_type.__name__})
        response = _receive(sock)
        if response != {"ok": True}:
            raise RuntimeError(response.get("error", "channel registration failed"))
        return sock
    except BaseException:
        sock.close()
        raise


class ChannelPublisher:
    def __init__(self, topic, message_type):
        _topic(topic, message_type)
        self.topic, self.message_type = topic, message_type
        self._active = False
        self._socket = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._session = uuid.uuid4().hex

    def Init(self):
        with self._lock:
            if self._active:
                return
            self._config, self._broker = _config()
            if self._config.transport == "memory":
                self._broker.add_publisher(self.topic, self)
            else:
                self._socket = _connect(self._config, "publisher", self.topic, self.message_type)
            self._active = True

    def Write(self, message, timeout=None):
        with self._lock:
            if not self._active:
                raise RuntimeError("publisher is not initialized")
            if type(message) is not self.message_type:
                raise ValueError("wrong message type for publisher")
            outgoing = deepcopy(message)
            if outgoing.seq == 0:
                outgoing.seq = self._sequence + 1
            if not outgoing.session_id:
                outgoing.session_id = self._session
            if outgoing.stamp_ns == 0 and outgoing.clock_domain == "monotonic":
                outgoing.stamp_ns = time.monotonic_ns()
            outgoing.validate()
            state_epoch_changed = self.message_type is LowState_ and outgoing.session_id != self._session
            if outgoing.seq <= self._sequence and not state_epoch_changed:
                raise ValueError("publisher sequence must increase")
            if outgoing.session_id != self._session:
                # A producer may supply its own session, fixed for this publisher.
                if self._sequence and not state_epoch_changed:
                    raise ValueError("publisher session changed")
                self._session = outgoing.session_id
            if self._socket is None:
                self._broker.publish(self.topic, self, outgoing)
            else:
                if timeout is not None:
                    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 10:
                        raise ValueError("invalid write timeout")
                self._socket.settimeout(2 if timeout is None else timeout)
                _send(self._socket, {"op": "write", "message": message_to_dict(outgoing)})
                response = _receive(self._socket)
                if response != {"ok": True}:
                    raise RuntimeError(response.get("error", "channel write failed"))
            self._sequence = outgoing.seq
            return True

    def Close(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            else:
                with self._broker.lock:
                    if self._broker.publishers.get(self.topic) is self:
                        del self._broker.publishers[self.topic]


class ChannelSubscriber:
    def __init__(self, topic, message_type):
        _topic(topic, message_type)
        self.topic, self.message_type = topic, message_type
        self._condition = threading.Condition()
        self._queue = deque(maxlen=1)
        self._active = False
        self._socket = None
        self._threads = []
        self.last_error = None

    def Init(self, handler=None, queueLen=1):
        if handler is not None and not callable(handler):
            raise ValueError("handler must be callable")
        if type(queueLen) is not int or not 1 <= queueLen <= 1024:
            raise ValueError("queueLen must be in 1..1024")
        if self._active:
            raise RuntimeError("subscriber is already initialized")
        self._config, self._broker = _config()
        self._queue = deque(maxlen=queueLen)
        self._active = True
        self._handler = handler
        try:
            if self._config.transport == "memory":
                with self._broker.lock:
                    self._broker.subscribers.setdefault(self.topic, set()).add(self)
            else:
                self._socket = _connect(self._config, "subscriber", self.topic, self.message_type)
                self._socket.settimeout(.1)
                self._start(self._read_socket)
            if handler is not None:
                self._start(self._dispatch)
        except BaseException:
            self.Close()
            raise

    def _start(self, function):
        thread = threading.Thread(target=function, daemon=True, name="opensprite-channel")
        self._threads.append(thread)
        thread.start()

    def _deliver(self, message):
        if type(message) is not self.message_type:
            self.last_error = ValueError("topic message type mismatch")
            return
        with self._condition:
            if self._active:
                self._queue.append(deepcopy(message))
                self._condition.notify_all()

    def _read_socket(self):
        sock = self._socket
        while self._active:
            try:
                response = _receive(sock)
                if set(response) != {"message"}:
                    raise ValueError("invalid subscription frame")
                self._deliver(message_from_dict(response["message"]))
            except socket.timeout:
                continue
            except (OSError, ValueError, RuntimeError) as error:
                if self._active:
                    self.last_error = error
                with self._condition:
                    self._active = False
                    self._condition.notify_all()
                return

    def _dispatch(self):
        while self._active:
            message = self.Read(.1)
            if message is not None:
                try:
                    self._handler(message)
                except Exception as error:
                    self.last_error = error

    def Read(self, timeout=None):
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("invalid read timeout")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._queue and self._active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._queue.popleft() if self._queue else None

    def Close(self):
        with self._condition:
            self._active = False
            self._queue.clear()
            self._condition.notify_all()
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        elif hasattr(self, "_broker"):
            with self._broker.lock:
                self._broker.subscribers.get(self.topic, set()).discard(self)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)
        self._threads.clear()


class ChannelServer:
    """Local SDK channel broker, independent of the native motor-controller IPC."""
    def __init__(self, endpoint, domain_id=0):
        self.endpoint = str(endpoint)
        if type(domain_id) is not int or not 0 <= domain_id <= 232:
            raise ValueError("invalid domain")
        self.domain_id = domain_id
        self._lock = threading.RLock()
        self._publishers = {}
        self._subscribers = {}
        self._clients = set()
        self._threads = []
        self._active = False
        self._owns_endpoint = False

    def start(self):
        if self._active:
            return self
        if os.path.lexists(self.endpoint):
            raise FileExistsError("channel endpoint already exists; refusing to unlink it")
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self._socket.bind(self.endpoint)
            self._owns_endpoint = True
            self._endpoint_inode = os.stat(self.endpoint).st_ino
            os.chmod(self.endpoint, 0o600)
            self._socket.listen(32)
            self._socket.settimeout(.1)
        except BaseException:
            self._socket.close()
            raise
        self._active = True
        self._start(self._accept)
        return self

    def _start(self, function, *args):
        thread = threading.Thread(target=function, args=args, daemon=True,
                                  name="opensprite-channel-server")
        self._threads.append(thread)
        thread.start()

    def _accept(self):
        while self._active:
            try:
                client, _ = self._socket.accept()
                client.settimeout(.2)
                with self._lock:
                    self._clients.add(client)
                self._start(self._serve, client)
            except socket.timeout:
                pass
            except OSError:
                return

    def _serve(self, client):
        topic = role = None
        try:
            registration = _receive(client)
            if set(registration) != {"op", "domain", "topic", "type"}:
                raise ValueError("invalid registration")
            role, topic = registration["op"], registration["topic"]
            message_type = MESSAGE_TYPES.get(registration["type"])
            _topic(topic, message_type)
            if type(registration["domain"]) is not int or registration["domain"] != self.domain_id:
                raise ValueError("domain mismatch")
            if role not in ("publisher", "subscriber"):
                raise ValueError("invalid channel role")
            with self._lock:
                if role == "publisher":
                    if topic in self._publishers:
                        raise RuntimeError("topic already has a publisher")
                    self._publishers[topic] = (client, message_type)
                else:
                    self._subscribers.setdefault(topic, {})[client] = message_type
            _send(client, {"ok": True})
            previous_sequence, session_id = 0, None
            while self._active:
                try:
                    frame = _receive(client)
                except socket.timeout:
                    continue
                if role != "publisher" or set(frame) != {"op", "message"} or frame["op"] != "write":
                    raise ValueError("unexpected channel operation")
                message = message_from_dict(frame["message"])
                if type(message) is not message_type:
                    raise ValueError("publisher message type mismatch")
                state_epoch_changed = message_type is LowState_ and message.session_id != session_id
                if not message.session_id or message.seq <= 0 or (message.seq <= previous_sequence and not state_epoch_changed):
                    raise ValueError("publisher sequence/session invariant failed")
                if session_id is not None and message.session_id != session_id and not state_epoch_changed:
                    raise ValueError("publisher session changed")
                previous_sequence, session_id = message.seq, message.session_id
                with self._lock:
                    targets = tuple(self._subscribers.get(topic, {}).items())
                for subscriber, expected in targets:
                    if expected is message_type:
                        try:
                            _send(subscriber, {"message": frame["message"]})
                        except OSError:
                            subscriber.close()
                _send(client, {"ok": True})
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            try:
                _send(client, {"error": str(error)[:512]})
            except OSError:
                pass
        finally:
            with self._lock:
                if isinstance(topic, str) and topic in self._publishers and self._publishers[topic][0] is client:
                    del self._publishers[topic]
                if isinstance(topic, str):
                    self._subscribers.get(topic, {}).pop(client, None)
                self._clients.discard(client)
            client.close()

    def close(self):
        self._active = False
        if hasattr(self, "_socket"):
            self._socket.close()
        with self._lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)
        self._threads.clear()
        if self._owns_endpoint and os.path.exists(self.endpoint):
            if os.stat(self.endpoint).st_ino == self._endpoint_inode:
                os.unlink(self.endpoint)
            self._owns_endpoint = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
