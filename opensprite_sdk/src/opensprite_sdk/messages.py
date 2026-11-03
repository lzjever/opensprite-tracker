"""Mutable SDK messages in OpenSprite joint coordinates, SI units.

``motor_cmd`` and ``motor_state`` are Unitree-style spelling aliases. They still
refer to joints, including pitch/roll of coupled mechanisms, never CAN motors.
These messages are not Unitree DDS wire types.
"""
from dataclasses import asdict, dataclass, field, fields
import math
from .constants import JOINT_COUNT, JOINT_ORDER_HASH


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum or value > 2**64 - 1:
        raise ValueError(f"{label} must be an unsigned integer")


def _finite(value, label):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite numeric data")


def _triple(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{label} must have three components")
    for component in value:
        _finite(component, label)


@dataclass
class JointCmd:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tau: float = 0.0

    def validate(self):
        if type(self.mode) is not int or self.mode not in (0, 1):
            raise ValueError("joint mode must be 0 (disabled) or 1 (command)")
        for name in ("q", "dq", "kp", "kd", "tau"):
            _finite(getattr(self, name), name)
        if self.kp < 0 or self.kd < 0:
            raise ValueError("Kp/Kd must be nonnegative")


@dataclass
class JointState:
    """Joint q/dq validity is separate from availability of estimated torque.

    ``sample_ns`` is a simulation timestamp or host receive timestamp according
    to ``sample_time_kind``; it is not necessarily a drive measurement clock.
    """
    q: float = 0.0
    dq: float = 0.0
    tau_est: float = 0.0
    valid: bool = False
    sample_ns: int = 0
    sample_time_kind: str = "unknown"
    tau_est_valid: bool = False

    def validate(self):
        for name in ("q", "dq", "tau_est"):
            _finite(getattr(self, name), name)
        if type(self.valid) is not bool:
            raise ValueError("joint validity must be boolean")
        if type(self.tau_est_valid) is not bool:
            raise ValueError("torque estimate validity must be boolean")
        _integer(self.sample_ns, "joint sample_ns")
        if self.sample_time_kind not in ("unknown", "simulation", "host_receive"):
            raise ValueError("invalid joint sample_time_kind")


@dataclass
class IMUState:
    gyroscope: tuple = (0.0, 0.0, 0.0)
    projected_gravity: tuple = (0.0, 0.0, -1.0)
    accelerometer: tuple = (0.0, 0.0, 0.0)
    valid: bool = False
    sample_ns: int = 0
    frame: str = "body"
    attached_link: str = "pelvis_link"
    sample_time_kind: str = "unknown"

    @property
    def frame_id(self):
        return self.attached_link

    @frame_id.setter
    def frame_id(self, value):
        self.attached_link = value

    def validate(self):
        for name in ("gyroscope", "projected_gravity", "accelerometer"):
            _triple(getattr(self, name), name)
        if type(self.valid) is not bool:
            raise ValueError("IMU validity must be boolean")
        _integer(self.sample_ns, "IMU sample_ns")
        if self.sample_time_kind not in ("unknown", "simulation", "host_read"):
            raise ValueError("invalid IMU sample_time_kind")
        if self.frame != "body" or not isinstance(self.attached_link, str) or not 0 < len(self.attached_link) <= 128:
            raise ValueError("IMU requires body coordinates and a named attached link")


@dataclass
class _Header:
    profile_id: str = "legacy_v4"
    joint_order_hash: int = JOINT_ORDER_HASH
    seq: int = 0
    stamp_ns: int = 0
    clock_domain: str = "monotonic"
    session_id: str = ""

    def _validate_header(self):
        if not isinstance(self.profile_id, str) or not 0 < len(self.profile_id) <= 128:
            raise ValueError("invalid profile_id")
        if type(self.joint_order_hash) is not int or self.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("joint order does not match OpenSprite SDK")
        _integer(self.seq, "sequence")
        _integer(self.stamp_ns, "stamp_ns")
        if self.clock_domain not in ("sim", "monotonic"):
            raise ValueError("clock_domain must be sim or monotonic")
        if not isinstance(self.session_id, str) or len(self.session_id) > 128:
            raise ValueError("invalid session_id")


@dataclass
class LowCmd_(_Header):
    joint_cmd: list = field(default_factory=lambda: [JointCmd() for _ in range(JOINT_COUNT)])
    source_state_seq: int = 0
    source_state_session_id: str = ""

    @property
    def motor_cmd(self):
        return self.joint_cmd

    @motor_cmd.setter
    def motor_cmd(self, value):
        self.joint_cmd = value

    def validate(self):
        self._validate_header()
        _integer(self.source_state_seq, "source_state_seq")
        if not isinstance(self.source_state_session_id, str) or len(self.source_state_session_id) > 128:
            raise ValueError("invalid source_state_session_id")
        if not isinstance(self.joint_cmd, list) or len(self.joint_cmd) != JOINT_COUNT:
            raise ValueError("expected 31 joint commands")
        for command in self.joint_cmd:
            if type(command) is not JointCmd:
                raise ValueError("invalid joint command type")
            command.validate()


@dataclass
class LowState_(_Header):
    joint_state: list = field(default_factory=lambda: [JointState() for _ in range(JOINT_COUNT)])
    imu_state: IMUState = field(default_factory=IMUState)
    status: str = "disabled"
    faults: list = field(default_factory=list)

    @property
    def motor_state(self):
        return self.joint_state

    @motor_state.setter
    def motor_state(self, value):
        self.joint_state = value

    def validate(self):
        self._validate_header()
        if not isinstance(self.joint_state, list) or len(self.joint_state) != JOINT_COUNT:
            raise ValueError("expected 31 joint states")
        for state in self.joint_state:
            if type(state) is not JointState:
                raise ValueError("invalid joint state type")
            state.validate()
        if type(self.imu_state) is not IMUState:
            raise ValueError("invalid IMU state type")
        self.imu_state.validate()
        if self.status not in ("disabled", "enabled", "fault", "shadow", "ready"):
            raise ValueError("invalid robot status")
        if not isinstance(self.faults, list) or len(self.faults) > 64 or any(
            not isinstance(x, str) or len(x) > 1024 for x in self.faults
        ):
            raise ValueError("invalid fault list")


MESSAGE_TYPES = {"LowCmd_": LowCmd_, "LowState_": LowState_}


def message_to_dict(message):
    if type(message) not in MESSAGE_TYPES.values():
        raise ValueError("unsupported message type")
    message.validate()
    return {"schema": 1, "type": type(message).__name__, "data": asdict(message)}


def _construct(cls, data):
    if not isinstance(data, dict) or set(data) != {x.name for x in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    if cls is IMUState:
        data = dict(data)
        for name in ("gyroscope", "projected_gravity", "accelerometer"):
            if not isinstance(data[name], (list, tuple)):
                raise ValueError("invalid IMU vector")
            data[name] = tuple(data[name])
    return cls(**data)


def message_from_dict(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {"schema", "type", "data"}:
        raise ValueError("invalid message envelope")
    if type(envelope["schema"]) is not int or envelope["schema"] != 1:
        raise ValueError("unsupported message schema")
    if not isinstance(envelope["type"], str):
        raise ValueError("invalid message type name")
    cls = MESSAGE_TYPES.get(envelope["type"])
    if cls is None:
        raise ValueError("unsupported message type")
    raw = envelope["data"]
    if not isinstance(raw, dict):
        raise ValueError("invalid message data")
    data = dict(raw)
    try:
        if cls is LowCmd_:
            data["joint_cmd"] = [_construct(JointCmd, x) for x in data["joint_cmd"]]
        else:
            data["joint_state"] = [_construct(JointState, x) for x in data["joint_state"]]
            data["imu_state"] = _construct(IMUState, data["imu_state"])
        result = _construct(cls, data)
        result.validate()
        return result
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError("malformed message data") from error


MotorCmd_ = JointCmd
MotorState_ = JointState
IMUState_ = IMUState
LowCmd = LowCmd_
LowState = LowState_
