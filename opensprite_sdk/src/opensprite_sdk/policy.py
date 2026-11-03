"""Reference-conditioned deployment without simulator state in the actor input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .constants import JOINT_NAMES, JOINT_ORDER_HASH
from .messages import JointCmd, LowCmd_, LowState_


def quaternion_matrix(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("expected finite wxyz quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def _axis_rotation(axis, angle):
    x, y, z = np.asarray(axis, dtype=float)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


def remap_imu(q, dq, gyro, gravity, chain):
    """Move pelvis measurements to a descendant link using measured q/dq only.

    The relative angular velocity is essential: a fixed rotation alone gives
    the wrong torso gyro while a waist joint moves. No heading is reconstructed.
    """
    rotation, relative_omega = np.eye(3), np.zeros(3)
    for entry in chain:
        index = JOINT_NAMES.index(entry["joint"])
        before = rotation @ quaternion_matrix(entry["body_quat_wxyz"])
        axis = np.asarray(entry["axis"], dtype=float)
        relative_omega += before @ axis * dq[index]
        rotation = before @ _axis_rotation(axis, q[index]-entry["qref"])
    return rotation.T @ (np.asarray(gyro)+relative_omega), rotation.T @ gravity


def sensor_observation(state: LowState_, default_q, previous_action, *,
                       imu_link="waist_yaw_link", imu_remap=None,
                       now_ns=None, max_motor_age_ns=60_000_000,
                       max_imu_age_ns=30_000_000):
    state.validate()
    if state.status == "fault" or state.faults:
        raise RuntimeError(f"robot fault: {state.faults}")
    now = ((time.monotonic_ns() if state.clock_domain == "monotonic" else state.stamp_ns)
           if now_ns is None else now_ns)
    if state.stamp_ns > now or now-state.stamp_ns > max_motor_age_ns:
        raise RuntimeError("stale or future robot state")
    for joint in state.joint_state:
        if (not joint.valid or joint.sample_time_kind == "unknown"
                or joint.sample_ns > now or now-joint.sample_ns > max_motor_age_ns):
            raise RuntimeError("stale, future or unknown joint measurement")
    imu = state.imu_state
    if (not imu.valid or imu.sample_time_kind == "unknown"
            or imu.sample_ns > now or now-imu.sample_ns > max_imu_age_ns):
        raise RuntimeError("stale, future or unknown IMU measurement")
    q = np.asarray([j.q for j in state.joint_state])
    dq = np.asarray([j.dq for j in state.joint_state])
    gyro, gravity = np.asarray(imu.gyroscope), np.asarray(imu.projected_gravity)
    if not np.isclose(np.linalg.norm(gravity), 1, atol=.02):
        raise ValueError("IMU must provide estimated unit gravity")
    if imu.attached_link != imu_link:
        if (not imu_remap or imu_remap.get("source_link") != imu.attached_link
                or imu_remap.get("target_link") != imu_link):
            raise ValueError("policy and sensor IMU links differ without a calibrated kinematic transform")
        gyro, gravity = remap_imu(q, dq, gyro, gravity, imu_remap["chain"])
    default_q, previous_action = np.asarray(default_q), np.asarray(previous_action)
    if default_q.shape != (31,) or previous_action.shape != (31,):
        raise ValueError("expected 31 default positions and previous actions")
    obs = np.concatenate((q-default_q, dq, gyro, gravity, previous_action)).astype(np.float32)
    if obs.shape != (99,) or not np.isfinite(obs).all():
        raise ValueError("invalid actor observation")
    return obs


class PolicyBundle:
    """A hash-checked ONNX actor, frozen reference goals and explicit contract."""

    def __init__(self, path, *, threads=1, session=None):
        self.root = Path(path)
        self.contract = json.loads((self.root/"contract.json").read_text())
        c = self.contract
        if (c.get("schema") != "opensprite-policy-v1" or c.get("training_variant") != "goal-sim2real"
                or c.get("joint_names") != list(JOINT_NAMES) or c.get("obs_dim") != 99
                or c.get("goal_dim") != 770 or c.get("hist") != 10 or c.get("policy_hz") != 50):
            raise ValueError("unsupported policy contract")
        if (not isinstance(c.get("profile_id"), str) or not c["profile_id"]
                or not isinstance(c.get("imu_link"), str) or not c["imu_link"]
                or type(c.get("frames")) is not int or c["frames"] < 2):
            raise ValueError("invalid policy identity or frame count")
        for name in ("default_q", "kp", "kd"):
            values=np.asarray(c.get(name), dtype=float)
            if values.shape != (31,) or not np.isfinite(values).all() or (name != "default_q" and np.any(values<0)):
                raise ValueError(f"invalid policy {name}")
        if isinstance(c.get("action_scale"), bool) or not isinstance(c.get("action_scale"), (int,float)) or not np.isfinite(c["action_scale"]) or c["action_scale"]<=0:
            raise ValueError("invalid action scale")
        if c.get("action_clip") != 20 or isinstance(c["action_clip"], bool):
            raise ValueError("supported tracker action_clip must be 20")
        if type(c.get("hardware_qualified", False)) is not bool:
            raise ValueError("hardware_qualified must be boolean")
        if not isinstance(c.get("files"), dict):
            raise ValueError("bundle file manifest missing")
        for name, expected in c["files"].items():
            if Path(name).name != name or hashlib.sha256((self.root/name).read_bytes()).hexdigest() != expected:
                raise ValueError("bundle file path or SHA256 mismatch")
        if not {"actor.onnx", "reference.npz"}.issubset(c["files"]):
            raise ValueError("bundle needs actor and reference hashes")
        with np.load(self.root/"reference.npz", allow_pickle=False) as data:
            self.reference = {k: np.array(data[k]) for k in data.files}
        goals = self.reference["goals"]
        if goals.ndim != 2 or goals.shape[1] != 770 or not np.isfinite(goals).all():
            raise ValueError("invalid reference goals")
        if len(goals) < 2 or len(goals) != c["frames"]:
            raise ValueError("reference length differs from contract")
        for name, shape in (("qpos", (len(goals),38)), ("qvel", (len(goals),37)),
                            ("root_rotation", (len(goals),3,3))):
            values = self.reference.get(name)
            if values is None or values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"invalid reference {name}")
        if not np.allclose(np.linalg.norm(self.reference["qpos"][:,3:7], axis=-1),1.,atol=1e-5):
            raise ValueError("reference quaternion must have unit norm")
        rotation = self.reference["root_rotation"]
        if not np.allclose(rotation, quaternion_matrix(self.reference["qpos"][:,3:7]), atol=1e-5):
            raise ValueError("reference rotation must match its unit quaternion")
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(str(self.root/"actor.onnx"), sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        self.session = session
        self.reset()

    def reset(self):
        self.history = None
        self.previous_action = np.zeros(31, np.float32)
        self.last_state_sequence = -1
        self.last_state_session = None
        self.last_stamp = -1
        self.clock_domain = None
        self.sequence = 0
        self.last_frame = -1
        self.session_id = uuid.uuid4().hex

    def command(self, state, frame, *, now_ns=None):
        c = self.contract
        state.validate()
        if state.seq < 1 or not state.session_id:
            raise ValueError("state needs a positive sequence and nonempty source session")
        if state.profile_id != c["profile_id"]:
            raise ValueError("policy/profile mismatch; dynamics cannot be silently substituted")
        if type(frame) is not int or not 0 <= frame < len(self.reference["goals"])-1:
            raise ValueError("reference ended; no wrapping or holding the final frame")
        if frame != self.last_frame+1:
            raise ValueError("reference frames must advance once per policy step")
        if self.last_state_session is not None and state.session_id != self.last_state_session:
            raise RuntimeError("state session changed; policy history needs an explicit reset")
        if self.clock_domain is not None and state.clock_domain != self.clock_domain:
            raise RuntimeError("state clock domain changed")
        if state.seq <= self.last_state_sequence or state.stamp_ns <= self.last_stamp:
            raise RuntimeError("state sequence/time did not advance")
        obs = sensor_observation(state, c["default_q"], self.previous_action,
                                 imu_link=c["imu_link"], imu_remap=c.get("imu_remap"), now_ns=now_ns)
        history = np.repeat(obs[None], c["hist"], axis=0) if self.history is None else np.concatenate((self.history[1:], obs[None]))
        raw = self.session.run(["action"], {"observation_history":history[None],
                                           "goal":self.reference["goals"][frame:frame+1].astype(np.float32)})[0]
        if raw.shape != (1, 31) or not np.isfinite(raw).all():
            raise RuntimeError("nonfinite or incorrectly shaped policy output")
        # Match the training collector and evaluator. Previous action is the
        # command sent before hidden actuator delay, after this clipping.
        raw = np.clip(raw, -c["action_clip"], c["action_clip"]).astype(np.float32)
        command = LowCmd_(profile_id=state.profile_id, joint_order_hash=JOINT_ORDER_HASH,
                          seq=self.sequence+1, stamp_ns=state.stamp_ns,
                          clock_domain=state.clock_domain, session_id=self.session_id,
                          source_state_seq=state.seq, source_state_session_id=state.session_id)
        for i in range(31):
            command.joint_cmd[i] = JointCmd(mode=1,
                q=float(c["default_q"][i]+c["action_scale"]*raw[0, i]),
                dq=0., kp=float(c["kp"][i]), kd=float(c["kd"][i]), tau=0.)
        command.validate()
        self.history, self.previous_action = history, raw[0].astype(np.float32)
        self.last_state_sequence, self.last_state_session = state.seq, state.session_id
        self.last_stamp, self.clock_domain = state.stamp_ns, state.clock_domain
        self.sequence += 1
        self.last_frame = frame
        return command
