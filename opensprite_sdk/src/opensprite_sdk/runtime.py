"""Explicitly owned robot runtime, independent of policy and network callbacks.

MuJoCo advances at its physics rate. Hardware retains the native CAN scheduler;
its blocking feedback acquisition runs separately from command delivery. This
module promises neither Python hard real time nor equivalent sim/real dynamics.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import copy, deepcopy
from dataclasses import asdict
import json
import math
import threading
import time
import uuid

from .client import LowLevelClient, LOWCMD, LOWSTATE
from .core.channel import ChannelPublisher, ChannelSubscriber
from .messages import LowCmd_, LowState_, ControlRequest_, ControlReply_

CONTROL_REQUEST = "rt/control_request"
CONTROL_REPLY = "rt/control_reply"


class ManagementClient:
    """Correlated, retryable management; a successful Write never means enabled."""

    def __init__(self, *, controller_session=None):
        self.controller_session = controller_session or uuid.uuid4().hex
        self.robot = LowLevelClient()
        self.requests = ChannelPublisher(CONTROL_REQUEST, ControlRequest_)
        self.replies = ChannelSubscriber(CONTROL_REPLY, ControlReply_)
        self.state = None

    def Init(self):
        try:
            self.robot.Init()
            self.replies.Init(queueLen=32)
            self.requests.Init()
        except BaseException:
            try:
                self.Close()
            except BaseException:
                pass
            raise
        return self

    def Read(self, timeout=1.):
        self.state = self.robot.Read(timeout)
        return self.state

    def Write(self, command):
        if command.session_id and command.session_id != self.controller_session:
            raise ValueError("command session differs from the claimed controller")
        command.session_id = self.controller_session
        return self.robot.Write(command)

    def request(self, operation, *, timeout=3., state=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("management timeout must be finite and positive")
        deadline = time.monotonic()+timeout
        state = state or self.state
        if state is None:
            state = self.Read(timeout=max(0., deadline-time.monotonic()))
        if operation == "enable":
            # A claim reply proves only management-topic discovery. The first
            # best-effort policy command must already have a matched reader
            # before the robot starts its local command watchdog.
            for endpoint in (self.robot.publisher, self.robot.subscriber,
                             self.requests, self.replies):
                config = getattr(endpoint, "_config", None)
                if config is not None and config.transport == "dds":
                    remaining = max(0., deadline-time.monotonic())
                    if not endpoint.WaitForMatch(remaining):
                        raise TimeoutError("DDS endpoint readiness missing before enable")
        request = ControlRequest_(profile_id=state.profile_id,
            joint_order_hash=state.joint_order_hash, stamp_ns=state.stamp_ns,
            clock_domain=state.clock_domain, session_id=self.controller_session,
            request_id=uuid.uuid4().hex, controller_session=self.controller_session,
            robot_session=state.session_id, operation=operation)
        retry_at = 0.
        while time.monotonic() < deadline:
            if time.monotonic() >= retry_at:
                # Publication sequence is transport metadata, not request identity.
                self.requests.Write(request)
                retry_at = time.monotonic()+.2
            reply = self.replies.Read(min(.05, max(0., deadline-time.monotonic())))
            if reply is None:
                if self.replies.last_error:
                    raise ConnectionError(self.replies.last_error)
                continue
            if (reply.request_id != request.request_id
                    or reply.controller_session != self.controller_session
                    or reply.robot_session != state.session_id):
                continue
            if not reply.accepted:
                raise RuntimeError(f"{operation} rejected: {reply.detail}")
            return reply
        raise TimeoutError(f"no correlated {operation} acknowledgement")

    def Close(self):
        first_error = None
        for endpoint in (self.requests, self.replies, self.robot.publisher, self.robot.subscriber):
            try:
                endpoint.Close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self.Init()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.Close()
        except BaseException:
            if exc is None:
                raise


class RobotRuntime:
    """One robot owner. Network readers only deliver bounded message queues.

    Local startup permission and a correlated claim/enable exchange are both
    required. A controller cannot enable by publishing LowCmd alone. Hardware
    disable acknowledges the requested native disconnect, never measured motor
    status; the existing native supervisor/report remains authoritative.
    """

    def __init__(self, backend, *, allow_enable=False, shadow=False,
                 command_timeout=.1, clock=time.monotonic):
        if not math.isfinite(command_timeout) or not .02 <= command_timeout <= 1:
            raise ValueError("command timeout must be in [.02,1] seconds")
        if type(allow_enable) is not bool or type(shadow) is not bool:
            raise ValueError("local enable and shadow permissions must be explicit booleans")
        if shadow and hasattr(backend, "step"):
            raise ValueError("shadow mode is only supported by the hardware backend")
        self.backend = backend
        self.allow_enable = bool(allow_enable)
        self.shadow = bool(shadow)
        self.command_timeout = command_timeout
        self.clock = clock
        self.is_sim = hasattr(backend, "step")
        self.command = ChannelSubscriber(LOWCMD, LowCmd_)
        self.state_writer = ChannelPublisher(LOWSTATE, LowState_)
        self.requests = ChannelSubscriber(CONTROL_REQUEST, ControlRequest_)
        self.replies = ChannelPublisher(CONTROL_REPLY, ControlReply_)
        self.owner = None
        self.enabled = False
        self.started_motion = False
        self.stop_requested = False
        self.fault = None
        self.last_state = None
        self.last_command_wall = None
        self.enabled_wall = None
        self.accepted_commands = 0
        self.accepted_command_seq = 0
        self.accepted_controller_session = ""
        self.rejected_commands = 0
        self.deadline_misses = 0
        self._cache = OrderedDict()
        self._management_highwater = {}
        self.physics_steps = 0
        self._stop = threading.Event()
        self._reader = None
        self._reader_error = None
        self._reply_sender = None
        self._reply_condition = threading.Condition()
        self._reply_queue = deque(maxlen=32)
        self.reply_drops = 0
        self.reply_errors = 0
        self.cleanup_errors = []
        self._latest = None
        self._latest_lock = threading.Lock()
        self._backend_lock = threading.RLock()
        self._last_published_sequence = -1
        self._pending = None
        self._active = False

    def start(self):
        self.backend.start()
        try:
            self.command.Init(queueLen=1)
            self.requests.Init(queueLen=32)
            self.state_writer.Init()
            self.replies.Init()
            self._active = True
            self._reply_sender = threading.Thread(target=self._send_replies,
                name="opensprite-control-replies", daemon=True)
            self._reply_sender.start()
            if self.is_sim:
                self.publish_state()
            else:
                self._reader = threading.Thread(target=self._read_hardware,
                    name="opensprite-feedback", daemon=True)
                self._reader.start()
        except BaseException:
            try:
                self.close()
            except BaseException:
                # Preserve the startup failure; close records cleanup failures.
                pass
            raise
        return self

    def _send_replies(self):
        while not self._stop.is_set():
            with self._reply_condition:
                while not self._reply_queue and not self._stop.is_set():
                    self._reply_condition.wait(.1)
                if self._stop.is_set():
                    return
                reply = self._reply_queue.popleft()
            try:
                self.replies.Write(reply, timeout=.01)
            except (OSError, ValueError, RuntimeError):
                # A slow reliable peer cannot stall physics or native control.
                # Requests are correlated and retryable through the ACK cache.
                self.reply_errors += 1

    def _queue_reply(self, reply):
        with self._reply_condition:
            if len(self._reply_queue) == self._reply_queue.maxlen:
                self.reply_drops += 1
            self._reply_queue.append(reply)
            self._reply_condition.notify()

    def _read_hardware(self):
        try:
            while not self._stop.is_set():
                state = self.backend.read()
                with self._latest_lock:
                    self._latest = state
        except BaseException as error:
            if not self._stop.is_set():
                self._reader_error = error

    def publish_state(self):
        if self.is_sim:
            with self._backend_lock:
                state = self.backend.read()
        else:
            with self._latest_lock:
                state = deepcopy(self._latest)
            if state is None or state.seq == self._last_published_sequence:
                return None
        state.accepted_command_seq = self.accepted_command_seq
        state.accepted_controller_session = self.accepted_controller_session
        self.state_writer.Write(state, timeout=.01)
        self.last_state = deepcopy(state)
        self._last_published_sequence = state.seq
        return state

    @staticmethod
    def _request_identity(request):
        semantic = asdict(request)
        # Retries have new publication sequence/time but identical intent.
        semantic.pop("seq")
        semantic.pop("stamp_ns")
        return json.dumps(semantic, sort_keys=True, separators=(",", ":"))

    def handle_request(self, request):
        """Return a semantic acknowledgement; callers may republish cached replies."""
        request.validate()
        state = self.last_state
        reply = ControlReply_(profile_id=self.backend.profile_id,
            clock_domain=state.clock_domain if state else request.clock_domain,
            stamp_ns=state.stamp_ns if state else 0,
            request_id=request.request_id, controller_session=request.controller_session,
            robot_session=state.session_id if state else request.robot_session,
            accepted=False, status="rejected", detail="", execution_verified=False)
        key = (request.controller_session, request.request_id)
        identity = self._request_identity(request)
        if key in self._cache:
            previous_identity, previous_reply = self._cache[key]
            if previous_identity != identity:
                reply.detail = "request id reused with different intent"
                return reply
            if request.controller_session in self._management_highwater:
                self._management_highwater[request.controller_session] = max(
                    request.seq, self._management_highwater[request.controller_session])
            return deepcopy(previous_reply)
        try:
            if state is None:
                raise RuntimeError("robot feedback is not ready")
            if (request.profile_id != state.profile_id
                    or request.joint_order_hash != state.joint_order_hash
                    or request.clock_domain != state.clock_domain
                    or request.robot_session != state.session_id):
                raise ValueError("robot session/profile/order/clock mismatch")
            if not request.controller_session or request.session_id != request.controller_session:
                raise ValueError("request publisher must declare the controller session")
            if request.seq <= self._management_highwater.get(request.controller_session, 0):
                raise ValueError("management sequence was replayed or reordered")
            if self.fault or self.stop_requested:
                raise RuntimeError("runtime is stopping or faulted")
            if request.operation == "claim":
                if self.owner not in (None, request.controller_session):
                    raise RuntimeError("another controller owns this robot")
                if (request.controller_session not in self._management_highwater
                        and len(self._management_highwater) >= 256):
                    raise RuntimeError("controller history is full; restart the endpoint explicitly")
                self.owner = request.controller_session
                reply.status = "claimed"
            else:
                if self.owner != request.controller_session:
                    raise RuntimeError("requesting controller does not own this robot")
                if request.operation == "enable":
                    if not self.allow_enable:
                        raise RuntimeError("local startup permission does not allow enable")
                    if not self.enabled:
                        with self._backend_lock:
                            if self.shadow:
                                with self.backend._state_lock:
                                    self.backend._require_fresh_state()
                            else:
                                self.backend.enable()
                        self.enabled = True
                        self.enabled_wall = self.clock()
                        self.last_command_wall = None
                    reply.status = "shadow_ready" if self.shadow else "enable_requested"
                    reply.execution_verified = self.is_sim and self.backend.enabled
                elif request.operation == "disable":
                    with self._backend_lock:
                        self.backend.disable()
                    self.enabled = False
                    self.stop_requested = True
                    reply.status = "disabled" if self.is_sim else "disable_requested"
                    reply.execution_verified = self.is_sim
                elif request.operation == "release":
                    if self.enabled or self.started_motion:
                        raise RuntimeError("release requires an idle, never-started controller")
                    self.owner = None
                    reply.status = "released"
                else:
                    raise ValueError("unsupported management operation")
            reply.accepted = True
        except (ValueError, RuntimeError) as error:
            reply.detail = str(error)[:1024]
        if self.owner == request.controller_session:
            self._management_highwater[request.controller_session] = max(
                request.seq, self._management_highwater.get(request.controller_session, 0))
        elif reply.accepted and request.operation == "release":
            self._management_highwater[request.controller_session] = request.seq
        self._cache[key] = (identity, deepcopy(reply))
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        return reply

    def _hardware_execution_feedback_fresh(self):
        # Preserve the stricter 10ms ankle execution gate. Native aggregate
        # feedback is50Hz: a late command waits for the next fresh snapshot,
        # potentially adding one period. Do not relabel cached feedback as new.
        with self.backend._state_lock:
            try:
                self.backend._require_fresh_state()
                return True
            except RuntimeError:
                return False

    def _trip(self, reason):
        self.fault = reason
        self.enabled = False
        self.stop_requested = True
        with self._backend_lock:
            self.backend.disable()

    def tick(self, *, physics_steps=1, publish=False):
        if self._reader_error is not None:
            raise RuntimeError("hardware feedback worker failed") from self._reader_error
        if not self.is_sim:
            self.publish_state()
        # Bound lifecycle work per physics tick. Reliable DDS sends happen on
        # the dedicated reply worker and cannot hold up command execution.
        request = self.requests.Read(0)
        if request is not None:
            reply = self.handle_request(request)
            self._queue_reply(reply)
        if self.requests.last_error or self.command.last_error:
            raise ConnectionError(self.requests.last_error or self.command.last_error)
        if self.stop_requested:
            return
        command = self.command.Read(0)
        if command is not None:
            if self.owner is None or command.session_id != self.owner:
                self.rejected_commands += 1
            elif not self.enabled:
                self.rejected_commands += 1
            else:
                self._pending = command
        if self._pending is not None:
            if self.is_sim or self._hardware_execution_feedback_fresh():
                command, self._pending = self._pending, None
                try:
                    with self._backend_lock:
                        if self.shadow:
                            self.backend.preview(command)
                        else:
                            self.backend.write(command)
                    self.accepted_commands += 1
                    self.accepted_command_seq = command.seq
                    self.accepted_controller_session = command.session_id
                    self.last_command_wall = self.clock()
                    self.started_motion = True
                    if all(j.mode == 0 for j in command.joint_cmd):
                        self.enabled = False
                        self.stop_requested = True
                except BaseException:
                    self._trip("owner_command_rejected")
                    raise
        if self.enabled:
            last = self.last_command_wall if self.last_command_wall is not None else self.enabled_wall
            if self.clock()-last >= self.command_timeout:
                self._trip("controller_stream_watchdog")
        # Discovery/claim/enable happen at the exact initial reference state.
        # After the first command the physics cadence is independent of readers.
        if self.is_sim and self.started_motion and not self.stop_requested:
            with self._backend_lock:
                self.backend.step(physics_steps)
                self.physics_steps += physics_steps
                if self.backend._faults:
                    self._trip("simulator_watchdog_or_physics_fault")
        if self.is_sim and not self.stop_requested:
            # Idle discovery publications cannot shift policy phase. Once
            # started, publish only at actual20ms simulation boundaries.
            if ((self.started_motion and self.physics_steps % self.backend.policy_steps == 0)
                    or (not self.started_motion and publish)):
                self.publish_state()

    def run(self, *, duration, stop_event=None, viewer=False, on_state=None):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        self.start()
        began = self.clock()
        period = self.backend.dt if self.is_sim else .002
        publish_every = self.backend.policy_steps if self.is_sim else 1
        iteration = 0
        deadline = self.clock()
        view = None
        failure = None
        try:
            if viewer:
                if not self.is_sim:
                    raise ValueError("local truth viewer is simulation only")
                view = _RuntimeViewer(self)
                view.start()
            while self.clock()-began < duration and not self.stop_requested:
                if stop_event is not None and stop_event.is_set():
                    break
                iteration += 1
                previous_sequence = self.last_state.seq if self.last_state is not None else -1
                self.tick(publish=iteration % publish_every == 0)
                if (on_state is not None and self.last_state is not None
                        and self.last_state.seq != previous_sequence):
                    on_state(self)
                deadline += period
                remaining = deadline-self.clock()
                if remaining < -period:
                    self.deadline_misses += 1
                    deadline = self.clock()
                self._stop.wait(max(0., remaining))
        except BaseException as error:
            failure = error
            raise
        finally:
            # Leave enough time for the already-written final management reply
            # to reach a healthy peer; this is not an execution confirmation.
            if self.stop_requested and failure is None:
                time.sleep(.05)
            cleanup_error = None
            for cleanup in ([view.close] if view is not None else [])+[self.close]:
                try:
                    cleanup()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if failure is None and cleanup_error is not None:
                raise cleanup_error
        return self.summary(self.clock()-began)

    def summary(self, wall_seconds=0.):
        return {"profile_id":self.backend.profile_id, "backend":"mujoco" if self.is_sim else "hardware",
            "accepted_commands":self.accepted_commands, "rejected_commands":self.rejected_commands,
            "accepted_command_seq":self.accepted_command_seq,
            "physics_steps":self.physics_steps,
            "simulated_seconds":self.physics_steps*self.backend.dt if self.is_sim else None,
            "fault":self.fault, "wall_seconds":wall_seconds,
            "deadline_misses":self.deadline_misses,
            "management_reply_drops":self.reply_drops,
            "management_reply_errors":self.reply_errors,
            "cleanup_errors":list(self.cleanup_errors),
            "native_motor_disable_verified":False if not self.is_sim else None,
            "hardware_tested":False}

    def close(self):
        failures = []

        def attempt(label, action):
            try:
                action()
            except BaseException as error:
                failures.append(error)
                self.cleanup_errors.append(f"{label}: {type(error).__name__}: {error}")

        self._stop.set()
        with self._reply_condition:
            self._reply_condition.notify_all()
        # Request local/native disable before waiting for any network worker.
        attempt("backend", self.backend.close)
        if self._reply_sender is not None and self._reply_sender is not threading.current_thread():
            attempt("reply worker", lambda: self._reply_sender.join(timeout=1.))
        for channel in (self.command, self.requests, self.state_writer, self.replies):
            attempt(f"channel {channel.topic}", channel.Close)
        if self._reader is not None and self._reader is not threading.current_thread():
            attempt("feedback worker", lambda: self._reader.join(timeout=1.))
        self._active = False
        if failures:
            raise failures[0]


class _RuntimeViewer:
    """Copy truth into an independent render-only model/data on another thread."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="opensprite-viewer", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15.):
            raise TimeoutError("MuJoCo viewer did not become ready")
        if self.error:
            raise RuntimeError("MuJoCo viewer failed") from self.error

    def _run(self):
        try:
            import mujoco
            import mujoco.viewer
            import numpy as np
            backend = self.runtime.backend
            # MuJoCo's UI may edit model options. An independent MjData alone
            # would still let GUI controls change the live physics model.
            model = copy(backend.model)
            data = mujoco.MjData(model)
            with mujoco.viewer.launch_passive(model, data,
                    show_left_ui=False, show_right_ui=False) as handle:
                with handle.lock():
                    handle.cam.distance = 2.1
                    handle.cam.azimuth = 135
                    handle.cam.elevation = -15
                self.ready.set()
                while handle.is_running() and not self.stop.is_set():
                    with self.runtime._backend_lock:
                        qpos = backend.data.qpos.copy()
                        qvel = backend.data.qvel.copy()
                        stamp = float(backend.data.time)
                    with handle.lock():
                        data.qpos[:] = qpos
                        data.qvel[:] = qvel
                        data.time = stamp
                        mujoco.mj_forward(model, data)
                        handle.cam.lookat[:] = data.xpos[1]+np.array([0.,0.,.05])
                    handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "OpenSprite SDK / Fast DDS runtime\nTime\nCommands",
                        f"{backend.profile_id} / render-only observer\n{stamp:.2f} s\n"
                        f"{self.runtime.accepted_commands}"))
                    handle.sync()
                    self.stop.wait(.02)
        except BaseException as error:
            self.error = error
            self.ready.set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)
