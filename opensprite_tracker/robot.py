"""Sprite0825 contract; physical limits come from the vendored official URDF.

PD gains and torque lag are explicit simulation choices: the public asset does
not publish tuned controller gains or an identified actuator response model.
No X2 physical constants are used.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "assets/opensprite/opensprite.xml"
URDF = ROOT / "assets/opensprite/vendor/sprite0825/sprite0825_sanitized_v4.urdf"
PHYSICS_DT_S = .005
POLICY_DT_S = .020
DECIMATION = 4
INITIAL_ROOT_HEIGHT = .46224576
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_COLLISION_PATTERN = r"^(left|right)_foot_collision$"
BODY_PRIMITIVE_COLLISION_PATTERN = r"^.*_primitive_collision$"
FULL_BODY_COLLISION_PATTERN = r"^.*_collision$"

_joints = tuple(j for j in ET.parse(URDF).getroot().findall("joint")
                if j.get("type") != "fixed")
BODY_JOINT_NAMES = tuple(j.get("name") for j in _joints)
HEAD_JOINT_NAMES = tuple(n for n in BODY_JOINT_NAMES if n.startswith("head_"))
LEG_JOINT_NAMES = tuple(n for n in BODY_JOINT_NAMES
                        if any(x in n for x in ("hip_", "knee_", "ankle_")))
ARM_JOINT_NAMES = tuple(n for n in BODY_JOINT_NAMES
                        if any(x in n for x in ("shoulder_", "elbow_", "wrist_")))
# Track the actual complete Sprite tree, including both waist links, all three
# head links, intermediate hip/shoulder housings and every wrist link. Its
# ordering is the MJCF DFS order used by MuJoCo, with the floating pelvis first.
TRACKED_BODY_NAMES = tuple(body.get("name") for body in
                          ET.parse(ROBOT_XML).getroot().find("worldbody").iter("body"))


@dataclass(frozen=True)
class JointControl:
    kp: float
    kd: float
    command_effort_nm: float
    urdf_velocity_rad_s: float
    default_pos_rad: float = 0.


def _gains(name):
    if "ankle_" in name:
        return 60., 1.5
    if "hip_" in name or "knee_" in name:
        return 60., 2.
    if "waist_" in name:
        return 20., .7
    if "wrist_pitch" in name or "wrist_roll" in name or name.startswith("head_"):
        return 2., .12
    return 8., .3


JOINT_CONTROL = {}
JOINT_LIMITS = {}
for joint in _joints:
    name = joint.get("name")
    limits = joint.find("limit")
    nominal = .12 if name == "left_shoulder_roll_joint" else (
        -.12 if name == "right_shoulder_roll_joint" else 0.)
    JOINT_CONTROL[name] = JointControl(*_gains(name), float(limits.get("effort")),
                                       float(limits.get("velocity")), nominal)
    JOINT_LIMITS[name] = (float(limits.get("lower")), float(limits.get("upper")))

# The public model gives no shared actuator-response groups. Each of its 31
# physical joints is independent; no grouping from another robot is retained.
TORQUE_LAG_GROUPS = tuple((name,) for name in BODY_JOINT_NAMES)
assert len(BODY_JOINT_NAMES) == 31
assert len(TRACKED_BODY_NAMES) == 32 and TRACKED_BODY_NAMES[0] == "pelvis_link"
assert sorted(n for group in TORQUE_LAG_GROUPS for n in group) == sorted(BODY_JOINT_NAMES)
