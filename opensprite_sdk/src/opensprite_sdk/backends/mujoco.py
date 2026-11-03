"""Deterministic CPU MuJoCo backend with joint-space SDK messages.

The public state is sensor-only. ``debug_truth`` and ``reset`` are simulation
utilities and are deliberately absent from LowState_. No trainer is imported.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from ..constants import JOINT_NAMES, JOINT_ORDER_HASH
from ..messages import IMUState, JointState, LowCmd_, LowState_

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_profile(profile="legacy_v4"):
    """Load an independently packaged simulator profile or an explicit JSON file."""
    candidate = Path(profile)
    path = candidate if candidate.is_file() else PACKAGE_ROOT / "profiles" / f"{profile}.json"
    data = json.loads(path.read_text())
    if data.get("schema") != "opensprite_sdk_sim_profile_v1":
        raise ValueError("unsupported simulation profile schema")
    if tuple(data.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("profile joint order does not match SDK")
    for name in ("default_q", "kp", "kd", "effort_limit", "velocity_limit", "joint_lower", "joint_upper"):
        value = np.asarray(data.get(name), dtype=float)
        if value.shape != (31,) or not np.isfinite(value).all():
            raise ValueError(f"profile {name} must be a finite 31-vector")
    if np.any(np.asarray(data["effort_limit"]) <= 0) or np.any(np.asarray(data["velocity_limit"]) <= 0):
        raise ValueError("profile effort and velocity limits must be positive")
    if np.any(np.asarray(data["kp"]) < 0) or np.any(np.asarray(data["kd"]) < 0):
        raise ValueError("profile gains must be nonnegative")
    if np.any(np.asarray(data["joint_lower"]) >= np.asarray(data["joint_upper"])):
        raise ValueError("invalid joint position limits")
    for name in ("physics_dt", "policy_dt", "maximum_command_age_seconds", "maximum_source_state_age_seconds", "maximum_kp", "maximum_kd"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid profile {name}")
    ratio = data["policy_dt"] / data["physics_dt"]
    if not math.isclose(ratio, round(ratio), abs_tol=1e-10):
        raise ValueError("policy period must be a whole number of physics steps")
    delay = data["command_delay"]
    if any(type(delay[k]) is not int for k in ("min_ticks", "max_ticks", "refresh_ticks")):
        raise ValueError("delay ticks must be integers")
    if not 0 <= delay["min_ticks"] <= delay["max_ticks"] <= 100 or delay["refresh_ticks"] < 1:
        raise ValueError("invalid delay tick range")
    if not 0 <= delay["hold_probability"] <= 1 or type(delay["random_phase"]) is not bool:
        raise ValueError("invalid delay probability or phase")
    if data.get("torque_lag_seconds") != 0:
        raise ValueError("this backend requires the zero torque-lag execution profile")
    return data


class MujocoBackend:
    """A manually stepped simulator; a server may pace ``step`` in real time.

    Command and sensor timestamps use integer simulation nanoseconds. RNG is
    seeded, but its draws are not bit-identical to a GPU training RNG. Targets
    are not clipped to mechanical position ranges: PD effort is explicitly
    limited, as in the legacy tracker. Velocity limits are reporting bounds,
    not an artificial hard cap on simulated velocity.
    """

    def __init__(self, profile="legacy_v4", *, seed=42, model_path=None):
        self.profile = load_profile(profile)
        self.profile_id = self.profile["profile_id"]
        self.dt = float(self.profile["physics_dt"])
        self.policy_dt = float(self.profile["policy_dt"])
        self.policy_steps = round(self.policy_dt / self.dt)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._renderer = None
        self._closed = False
        self._enabled = False
        self._faults = []
        self._session = uuid.uuid4().hex
        self._owner = None
        self._last_cmd_seq = 0
        self._last_command_ns = None
        self._enabled_ns = None
        self._tick = 0
        self._state_seq = 0
        self._source_states = {}
        self._cmd = None
        self._last_applied_command = None
        self._tau = np.zeros(31)
        self._raw_tau = np.zeros(31)
        delay = self.profile["command_delay"]
        self._queue = deque(maxlen=delay["max_ticks"] + 1)
        self._lag = 0
        self._phase = int(self._rng.integers(delay["refresh_ticks"])) if delay["random_phase"] else 0
        filename = Path(model_path) if model_path else PACKAGE_ROOT / self.profile["robot_model"]
        self.model = self._build_model(filename)
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.qids = self.model.jnt_qposadr[ids].copy()
        self.vids = self.model.jnt_dofadr[ids].copy()
        if self.model.nu != 0 or self.model.nq != 38 or self.model.nv != 37:
            raise ValueError("OpenSprite simulator expects a free base and 31 external-PD joints")
        np.testing.assert_allclose(self.model.jnt_range[ids, 0], self.profile["joint_lower"], atol=1e-9)
        np.testing.assert_allclose(self.model.jnt_range[ids, 1], self.profile["joint_upper"], atol=1e-9)
        self._effort = np.asarray(self.profile["effort_limit"])
        self._imu_body = self.model.body(self.profile["imu_body_name"]).id
        self._gyro = self.model.sensor("sdk_gyro").adr[0]
        self._accel = self.model.sensor("sdk_accelerometer").adr[0]
        self.reset()

    def _build_model(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((filename.parent / compiler.get("meshdir", ".")).resolve()))
        world = root.find("worldbody")
        assets = root.find("asset")
        if assets is None:
            assets = ET.SubElement(root, "asset")
        ET.SubElement(assets, "texture", name="sdk_checker", type="2d", builtin="checker", rgb1="0.26 0.28 0.32", rgb2="0.55 0.57 0.60", width="512", height="512")
        ET.SubElement(assets, "material", name="sdk_floor", texture="sdk_checker", texrepeat="20 20", texuniform="true", reflectance="0.05")
        ET.SubElement(world, "geom", name="sdk_ground", type="plane", size="10 10 0.1", material="sdk_floor", friction="0.8 0.02 0.001", condim="3")
        ET.SubElement(world, "light", name="sdk_keylight", pos="1 -2 4", dir="-0.2 0.4 -1", diffuse="0.8 0.8 0.8", castshadow="true")
        body = next((b for b in world.iter("body") if b.get("name") == self.profile["imu_body_name"]), None)
        if body is None:
            raise ValueError("configured IMU body is not in the robot")
        ET.SubElement(body, "site", name="sdk_imu", pos="0 0 0", size="0.002", rgba="0 0 0 0")
        sensor = root.find("sensor")
        if sensor is None:
            sensor = ET.SubElement(root, "sensor")
        ET.SubElement(sensor, "gyro", name="sdk_gyro", site="sdk_imu")
        ET.SubElement(sensor, "accelerometer", name="sdk_accelerometer", site="sdk_imu")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        model.opt.timestep = self.dt
        model.opt.iterations = int(self.profile["solver_iterations"])
        model.opt.ls_iterations = int(self.profile["solver_ls_iterations"])
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1080
        return model

    @property
    def now_ns(self):
        return int(round(self._tick * self.dt * 1e9))

    @property
    def enabled(self):
        return self._enabled

    def _open(self):
        if self._closed:
            raise RuntimeError("simulator is closed")

    def start(self):
        self._open()
        return self

    def reset(self, *, qpos=None, qvel=None):
        """Reset simulation only. Full MuJoCo qpos/qvel are accepted in SI units."""
        self._open()
        for name, value, size in (("qpos", qpos, self.model.nq), ("qvel", qvel, self.model.nv)):
            if value is not None and (np.asarray(value).shape != (size,) or not np.isfinite(value).all()):
                raise ValueError(f"{name} must be a finite {size}-vector")
        if qpos is not None and not math.isclose(float(np.linalg.norm(np.asarray(qpos)[3:7])), 1.0, abs_tol=1e-5):
            raise ValueError("root quaternion must have unit norm in wxyz order")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = self.profile["default_q"]
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        self._tick = 0
        self._state_seq = 0
        self._source_states.clear()
        self._enabled = False
        self._faults.clear()
        self._owner = None
        self._last_cmd_seq = 0
        self._last_command_ns = None
        self._enabled_ns = None
        self._cmd = None
        self._last_applied_command = None
        self._queue.clear()
        self._lag = 0
        self._tau.fill(0)
        self._raw_tau.fill(0)
        self._session = uuid.uuid4().hex
        mujoco.mj_forward(self.model, self.data)
        return self.read()

    def enable(self):
        self._open()
        if self._faults:
            raise RuntimeError("reset simulator before enabling after a fault")
        if not self._enabled:
            self._enabled = True
            self._enabled_ns = self.now_ns

    def disable(self):
        self._open()
        self._enabled = False
        self._cmd = None
        self._queue.clear()
        self._tau.fill(0)
        self.data.qfrc_applied.fill(0)
        # Stop invalidates causal source snapshots, but retains the stream
        # epoch, writer identity and sequence high-water mark. Only reset
        # starts a new controller/robot epoch.
        self._source_states.clear()
        self._last_command_ns = None

    def _fault(self, reason):
        self.disable()
        self._faults.append(reason)

    def write(self, command):
        """Latch a validated command; this never implicitly enables the robot."""
        self._open()
        if type(command) is not LowCmd_:
            raise ValueError("expected LowCmd_")
        command.validate()
        if command.profile_id != self.profile_id or command.clock_domain != "sim":
            raise ValueError("command profile or clock domain does not match simulator")
        modes = {j.mode for j in command.joint_cmd}
        if len(modes) != 1:
            raise ValueError("mixed per-joint enable modes are unsupported")
        if command.seq < 1 or not command.session_id:
            raise ValueError("active SDK writers require a positive sequence and nonempty controller session")
        if command.source_state_session_id != self._session:
            raise ValueError("command source state belongs to another simulator session")
        if command.stamp_ns > self.now_ns:
            raise ValueError("command timestamp is in the future")
        if self.now_ns - command.stamp_ns > self.profile["maximum_command_age_seconds"] * 1e9:
            raise ValueError("command is stale")
        source = command.source_state_seq
        if source not in self._source_states:
            raise ValueError("command source_state_seq is absent, expired or in the future")
        source_ns = self._source_states[source]
        if self.now_ns - source_ns > self.profile["maximum_source_state_age_seconds"] * 1e9:
            raise ValueError("command source state is stale")
        owner = command.session_id
        if self._owner is not None and owner != self._owner:
            raise ValueError("another controller owns this simulator")
        if command.seq <= self._last_cmd_seq:
            raise ValueError("command sequence was replayed or reordered")
        if modes == {0}:
            self.disable()
            return
        if not self._enabled:
            raise RuntimeError("call enable() before active simulation commands")
        if self._faults:
            raise RuntimeError("simulator is faulted")
        values = np.array([[j.q, j.dq, j.kp, j.kd, j.tau] for j in command.joint_cmd], dtype=float)
        if np.any(values[:, 2] > self.profile["maximum_kp"]) or np.any(values[:, 3] > self.profile["maximum_kd"]):
            raise ValueError("requested gains exceed simulator profile bounds")
        self._owner = owner
        self._last_cmd_seq = command.seq
        self._last_command_ns = command.stamp_ns
        self._cmd = values.copy()

    def _delayed_command(self):
        self._queue.append(self._cmd.copy())
        delay = self.profile["command_delay"]
        if (self._tick + self._phase) % delay["refresh_ticks"] == 0:
            candidate = int(self._rng.integers(delay["min_ticks"], delay["max_ticks"] + 1))
            if self._rng.random() >= delay["hold_probability"]:
                self._lag = candidate
        # Startup backfills with the first available frame, as the training delay buffer does.
        return self._queue[-1 - min(self._lag, len(self._queue) - 1)]

    def step(self, steps=1):
        self._open()
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            if self._enabled:
                newest = self._last_command_ns if self._last_command_ns is not None else self._enabled_ns
                if self.now_ns - newest >= self.profile["maximum_command_age_seconds"] * 1e9:
                    self._fault("command_watchdog")
            self._raw_tau.fill(0)
            self._tau.fill(0)
            if self._enabled and self._cmd is not None:
                cmd = self._delayed_command()
                self._last_applied_command = cmd.copy()
                self._raw_tau = cmd[:, 2] * (cmd[:, 0] - self.data.qpos[self.qids]) + cmd[:, 3] * (cmd[:, 1] - self.data.qvel[self.vids]) + cmd[:, 4]
                if not np.isfinite(self._raw_tau).all():
                    self._fault("nonfinite_requested_torque")
                    raise ValueError("requested PD torque overflowed")
                self._tau = np.clip(self._raw_tau, -self._effort, self._effort)
            self.data.qfrc_applied.fill(0)
            self.data.qfrc_applied[self.vids] = self._tau
            mujoco.mj_step(self.model, self.data)
            self._tick += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self._fault("nonfinite_physics_state")
                raise RuntimeError("MuJoCo produced a nonfinite state")
        # mj_step leaves some kinematics/sensors at the pre-integration state.
        mujoco.mj_forward(self.model, self.data)
        return self.read()

    def read(self):
        self._open()
        stamp = self.now_ns
        self._state_seq += 1
        self._source_states[self._state_seq] = stamp
        if len(self._source_states) > 512:
            del self._source_states[next(iter(self._source_states))]
        rotation = self.data.xmat[self._imu_body].reshape(3, 3)
        imu = IMUState(
            gyroscope=tuple(float(x) for x in self.data.sensordata[self._gyro:self._gyro + 3]),
            projected_gravity=tuple(float(x) for x in -rotation[2, :]),
            accelerometer=tuple(float(x) for x in self.data.sensordata[self._accel:self._accel + 3]),
            valid=True, sample_ns=stamp, attached_link=self.profile["imu_body_name"], sample_time_kind="simulation",
        )
        result = LowState_(profile_id=self.profile_id, joint_order_hash=JOINT_ORDER_HASH, seq=self._state_seq,
            stamp_ns=stamp, clock_domain="sim", session_id=self._session,
            joint_state=[JointState(float(q), float(v), float(t), True, stamp, sample_time_kind="simulation", tau_est_valid=True) for q, v, t in zip(self.data.qpos[self.qids], self.data.qvel[self.vids], self._tau)],
            imu_state=imu, status="fault" if self._faults else ("enabled" if self._enabled else "disabled"), faults=list(self._faults))
        result.validate()
        return result

    def debug_truth(self):
        """Return privileged diagnostics separately from deployment observations."""
        self._open()
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "root_position": self.data.qpos[:3].copy(), "root_quaternion_wxyz": self.data.qpos[3:7].copy(), "time": float(self.data.time), "requested_torque": self._raw_tau.copy(), "applied_torque": self._tau.copy(), "command_lag_ticks": self._lag}

    def render(self, *, width=1280, height=720):
        """Return RGB pixels. Set MUJOCO_GL=osmesa before import for CPU rendering."""
        self._open()
        if not 64 <= width <= 1920 or not 64 <= height <= 1080:
            raise ValueError("render dimensions must fit the 1920x1080 framebuffer")
        if self._renderer is not None and (self._renderer.width != width or self._renderer.height != height):
            self._renderer.close()
            self._renderer = None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[1] + np.array([0.0, 0.0, 0.05])
        camera.distance = 1.9
        camera.azimuth = 135
        camera.elevation = -15
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._closed:
            return
        self.disable()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
