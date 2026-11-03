"""Bridge public SDK channels to an owned robot backend."""
from __future__ import annotations

import math
from pathlib import Path
import time

from .client import LOWCMD, LOWSTATE
from .core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, ChannelServer
from .messages import LowCmd_, LowState_


class BackendService:
    """The backend owns control permissions; channels never silently enable it."""
    def __init__(self, backend, *, shadow=False):
        self.backend = backend
        self.shadow = shadow
        self.command = ChannelSubscriber(LOWCMD, LowCmd_)
        self.state = ChannelPublisher(LOWSTATE, LowState_)
        self.stop_requested = False

    def start(self):
        self.backend.start()
        try:
            self.command.Init(queueLen=1)
            self.state.Init()
        except BaseException:
            self.command.Close()
            self.state.Close()
            self.backend.close()
            raise
        return self

    def publish_state(self):
        state = self.backend.read()
        self.state.Write(state, timeout=.1)
        return state

    def tick(self, *, steps=0, wait_for_first_command=False):
        command = self.command.Read(0)
        if self.command.last_error:
            raise ConnectionError(f"command channel: {self.command.last_error}")
        if command is not None:
            if self.shadow:
                self.backend.preview(command)
            else:
                self.backend.write(command)
            if all(j.mode == 0 for j in command.joint_cmd):
                self.stop_requested = True
                # Hardware closes the native lease; actual motor disable is
                # verified by the native report, not fabricated by the SDK.
                if hasattr(self.backend, "step"):
                    self.publish_state()
                return True
        if steps and (command is not None or not wait_for_first_command):
            self.backend.step(steps)
        self.publish_state()
        return command is not None

    def close(self):
        try:
            self.backend.close()
        finally:
            self.command.Close()
            self.state.Close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


def run_sim_server(profile="legacy_v4", endpoint="/tmp/opensprite-sdk.sock", domain_id=1,
                   realtime=True, duration=10., enable=False):
    from .backends.mujoco import MujocoBackend
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    Path(endpoint).parent.mkdir(parents=True, exist_ok=True)
    backend = MujocoBackend(profile=profile)
    started = False
    with ChannelServer(endpoint, domain_id=domain_id):
        ChannelFactoryInitialize(domain_id, transport="unix", endpoint=endpoint)
        with BackendService(backend) as service:
            if enable:
                backend.enable()
            deadline = time.monotonic()
            idle_deadline = deadline+30
            while backend.now_ns < int(duration*1e9):
                received = service.tick(steps=round(backend.policy_dt/backend.dt),
                                        wait_for_first_command=not started)
                if service.stop_requested:
                    return
                started = started or received
                if not started and time.monotonic() > idle_deadline:
                    raise TimeoutError("no control client connected within 30 seconds")
                if started and backend.read().faults:
                    raise RuntimeError("simulator control watchdog/fault triggered")
                if realtime or not started:
                    deadline += backend.policy_dt
                    time.sleep(max(0., deadline-time.monotonic()))
