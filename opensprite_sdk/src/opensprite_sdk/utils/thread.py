"""Periodic best-effort Python callbacks, never a hard real-time scheduler."""
import math
import threading
import time


class RecurrentThread:
    def __init__(self, interval, target, name="opensprite-control"):
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        if not callable(target):
            raise ValueError("target must be callable")
        self.interval, self.target, self.name = interval, target, name
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None
        self.overruns = 0

    def Start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("recurrent thread is already running")
        self._stop.clear()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _run(self):
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self.target()
            except Exception as error:
                self.last_error = error
                self._stop.set()
                return
            deadline += self.interval
            remaining = deadline - time.monotonic()
            if remaining < 0:
                self.overruns += 1
                deadline = time.monotonic()
            else:
                self._stop.wait(remaining)

    def Stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, 2 * self.interval))

    def Wait(self, timeout=None):
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
