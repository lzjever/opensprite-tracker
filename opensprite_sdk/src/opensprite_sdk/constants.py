"""OpenSprite joint order, independent of transport and physical motor order."""
from enum import IntEnum
JOINT_NAMES = ('left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'head_pitch_joint', 'head_roll_joint', 'head_yaw_joint')
JOINT_COUNT = 31
JointIndex = IntEnum("JointIndex", {
    "".join(part.capitalize() for part in name.removesuffix("_joint").split("_")): index
    for index, name in enumerate(JOINT_NAMES)
})

def ordered_name_hash(names):
    names = tuple(names)
    if len(names) != JOINT_COUNT or len(set(names)) != JOINT_COUNT:
        raise ValueError("expected 31 unique joint names")
    value = 0xcbf29ce484222325
    for name in names:
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("invalid joint name")
        for byte in name.encode("utf-8") + b"\0":
            value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return value

JOINT_ORDER_HASH = ordered_name_hash(JOINT_NAMES)
