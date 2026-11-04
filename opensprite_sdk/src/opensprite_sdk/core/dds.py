"""Narrow ctypes bridge to typed native Fast DDS (no JSON over DDS)."""
from __future__ import annotations

import ctypes
import ctypes.util
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import threading
import sys
import time

_library_lock = threading.Lock()
_libraries = {}


def _library():
    override = os.environ.get("OPENSPRITE_DDS_LIBRARY")
    sdk = Path(__file__).resolve().parents[3]
    candidates = ([Path(override)] if override else [
        Path(__file__).with_name("libopensprite_dds.so"),
        Path(sys.prefix) / "lib/libopensprite_dds.so",
        sdk / "native-dds/build/libopensprite_dds.so",
    ])
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None and not override:
        path = ctypes.util.find_library("opensprite_dds")
    if path is None:
        raise RuntimeError("Fast DDS native adapter unavailable; run bash tools/build_dds.sh "
                           "in the SDK or set OPENSPRITE_DDS_LIBRARY")
    with _library_lock:
        if path not in _libraries:
            try:
                lib = ctypes.CDLL(str(path))
            except OSError as error:
                raise RuntimeError(f"cannot load Fast DDS adapter {path}: {error}") from error
            lib.osdds_error.argtypes = []
            lib.osdds_error.restype = ctypes.c_char_p
            lib.osdds_open.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_char_p]
            lib.osdds_open.restype = ctypes.c_void_p
            if hasattr(lib, "osdds_open_with_history"):
                lib.osdds_open_with_history.argtypes = lib.osdds_open.argtypes+[ctypes.c_uint]
                lib.osdds_open_with_history.restype = ctypes.c_void_p
            lib.osdds_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.osdds_write.restype = ctypes.c_int
            lib.osdds_take.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            lib.osdds_take.restype = ctypes.c_int
            lib.osdds_matches.argtypes = [ctypes.c_void_p]
            lib.osdds_matches.restype = ctypes.c_int
            lib.osdds_close.argtypes = [ctypes.c_void_p]
            lib.osdds_close.restype = None
            _libraries[path] = lib
        return _libraries[path]


def _timeout(value, *, default=.1):
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError("DDS timeout must be finite seconds in 0..10")
    return int(math.ceil(value * 1000))


class DdsEndpoint:
    """One reader/writer; ROS 2 .msg definitions are the wire authority.

    Command/state: best effort, keep last one, volatile. Lifecycle messages:
    reliable, keep last one, volatile, maximum native write blocking 10ms.
    Matching is observable; accepting a write does not acknowledge execution.
    An explicit history_depth retains at most64 real received samples; the
    default remains one and sender history is unchanged. Buffered readers use
    independent transport copies rather than writer-coupled Data-sharing loans.
    """

    def __init__(self, domain, interface, topic, message_type, role, *, history_depth=1):
        if type(domain) is not int or not 0 <= domain <= 232:
            raise ValueError("DDS domain must be an integer in 0..232")
        if message_type not in ("LowCmd_", "LowState_", "ControlRequest_", "ControlReply_"):
            raise ValueError("unsupported DDS message type")
        if role not in ("publisher", "subscriber"):
            raise ValueError("invalid DDS endpoint role")
        if type(history_depth) is not int or not 1 <= history_depth <= 64:
            raise ValueError("DDS reader history must be in1..64")
        if role == "publisher" and history_depth != 1:
            raise ValueError("DDS publisher history remains one")
        values = [interface or "lo", topic, message_type, role]
        if any(not isinstance(v, str) or not v or "\0" in v or len(v.encode("utf8")) > 128 for v in values):
            raise ValueError("invalid DDS endpoint string")
        self._lib = _library()
        self._lifetime = threading.Condition()
        self._inflight = 0
        self._take_lock = threading.Lock()
        # One bounded native receive allocation per reader, not per sensor
        # sample. Publisher endpoints never need a receive buffer.
        self._receive_buffer = ctypes.create_string_buffer(1048576) if role == "subscriber" else None
        encoded = [v.encode("utf8") for v in values]
        if history_depth == 1:
            self._handle = self._lib.osdds_open(domain, *encoded)
        else:
            if not hasattr(self._lib, "osdds_open_with_history"):
                raise RuntimeError("native DDS adapter lacks bounded history; rebuild with tools/build_dds.sh")
            self._handle = self._lib.osdds_open_with_history(domain, *encoded, history_depth)
        if not self._handle:
            raise RuntimeError(self._error())
        self._role = role

    def _error(self):
        return (self._lib.osdds_error() or b"Fast DDS adapter failed").decode("utf8", errors="replace")

    @contextmanager
    def _using(self):
        # Native DDS query/read/write calls are thread safe. Only lifetime is
        # serialized: a blocking reader must not starve match queries or close.
        with self._lifetime:
            if not self._handle:
                raise RuntimeError("DDS endpoint is closed")
            handle = self._handle
            self._inflight += 1
        try:
            yield handle
        finally:
            with self._lifetime:
                self._inflight -= 1
                self._lifetime.notify_all()

    def write(self, envelope, timeout=None):
        _timeout(timeout)
        payload = json.dumps(envelope, allow_nan=False, separators=(",", ":")).encode("utf8")
        if len(payload) > 1048576:
            raise ValueError("DDS bridge payload exceeds 1MiB")
        with self._using() as handle:
            if self._lib.osdds_write(handle, payload, len(payload)) < 0:
                raise RuntimeError(self._error())
        return True

    def take(self, timeout=.1):
        milliseconds = _timeout(timeout)
        if self._role != "subscriber":
            raise RuntimeError("DDS endpoint is not a subscriber")
        deadline = time.monotonic()+milliseconds/1000.
        acquired = (self._take_lock.acquire(timeout=milliseconds/1000.) if milliseconds
                    else self._take_lock.acquire(blocking=False))
        if not acquired:
            return None
        try:
            # The buffer is shared only by take calls. Lifetime/matches remain
            # independent, so a blocked reader cannot starve discovery/close.
            with self._using() as handle:
                buffer = self._receive_buffer
                remaining = max(0, int(math.ceil((deadline-time.monotonic())*1000))) if milliseconds else 0
                size = self._lib.osdds_take(handle, buffer, len(buffer), remaining)
                if size < 0:
                    raise RuntimeError(self._error())
                # buffer.raw would copy the entire1MiB before slicing. Copy
                # only the returned sample while the buffer is still owned.
                payload = ctypes.string_at(buffer, size) if size else None
            return json.loads(payload) if payload is not None else None
        finally:
            self._take_lock.release()

    def matches(self):
        with self._using() as handle:
            count = self._lib.osdds_matches(handle)
            if count < 0:
                raise RuntimeError(self._error())
            return count

    def close(self):
        with self._lifetime:
            handle = self._handle
            self._handle = None
            while self._inflight:
                self._lifetime.wait()
            self._receive_buffer = None
        if handle:
            self._lib.osdds_close(handle)
