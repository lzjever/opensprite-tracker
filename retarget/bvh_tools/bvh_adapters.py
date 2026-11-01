"""BVH channel parsing, transforms, and repository-owned Geno skeleton validation."""
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


HUMAN = ['Hips', 'LeftLeg', 'LeftShin', 'LeftFoot', 'RightLeg', 'RightShin',
         'RightFoot', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm',
         'RightForeArm', 'RightHand']
LAFAN1_JOINTS = (
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe',
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
)
LAFAN1_PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12,
                  11, 14, 15, 16, 11, 18, 19, 20)
LAFAN1_ROOT_CHANNELS = ('Xposition', 'Yposition', 'Zposition',
                        'Zrotation', 'Yrotation', 'Xrotation')
LAFAN1_JOINT_CHANNELS = ('Zrotation', 'Yrotation', 'Xrotation')
TENNIS_JOINTS = (
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
)
TENNIS_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11,
                  12, 13, 0, 15, 16, 17, 0, 19, 20, 21)
TENNIS_ROTATION_ORDERS = (
    ('Yrotation', 'Xrotation', 'Zrotation'),
    ('Xrotation', 'Yrotation', 'Zrotation'),
)
SOMA_UNIFORM_OFFSET_ATOL_M = 5e-4
SOMA_UNIFORM_POSITION_CHANNELS = ('Xposition', 'Yposition', 'Zposition')
MAPPING = {'Hips': 'Hips', 'Spine1': 'Spine', 'Chest': 'Spine1',
           'Neck1': 'Neck', 'Neck2': 'Neck', 'Head': 'Head'}
for side in ('Left', 'Right'):
    for part in ('Shoulder', 'Arm', 'ForeArm', 'Hand', 'Foot', 'ToeBase'):
        MAPPING[side + part] = side + part
    MAPPING[side + 'Leg'] = side + 'UpLeg'
    MAPPING[side + 'Shin'] = side + 'Leg'
    for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
        for i in range(1, 4):
            MAPPING[f'{side}Hand{finger}{i}'] = f'{side}Hand{finger}{i}'
TENNIS_MAPPING = dict(MAPPING, Spine2='Spine2', Chest='Spine3')
GENO_STANCE = Path(__file__).resolve().parents[2] / 'data/sources/lafan1-resolved/Geno_stance.bvh'


@lru_cache(maxsize=1)
def geno_stance():
    """Reference hierarchy for the Geno native-surface input contract."""
    return read_transforms(GENO_STANCE)


class SkeletonSignature(NamedTuple):
    """The complete metric BVH rest skeleton consumed by UMR."""

    joint_names: tuple
    parents: tuple
    local_rest_offsets_m: np.ndarray

def _hierarchy(hierarchy):
    names, parents, offsets, channels, stack = [], [], [], [], []
    current = -1
    for line in hierarchy.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] in ('ROOT', 'JOINT'):
            parents.append(stack[-1] if stack else -1)
            current = len(names)
            names.append(p[1])
            offsets.append([0., 0., 0.])
            channels.append([])
        elif p[:2] == ['End', 'Site']:
            current = -1
        elif p[0] == '{':
            stack.append(current)
        elif p[0] == '}':
            stack.pop()
            current = stack[-1] if stack else -1
        elif p[0] == 'OFFSET' and current >= 0:
            offsets[current] = list(map(float, p[1:4]))
        elif p[0] == 'CHANNELS':
            channels[current] = p[2:]
    return names, parents, np.array(offsets), channels


def bvh_skeleton_signature(path, metres_per_unit=.01):
    """Return complete joint order, parents and metre-scale local rest offsets."""
    hierarchy_lines = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() == 'MOTION':
                break
            hierarchy_lines.append(line)
        else:
            raise ValueError(f'BVH has no MOTION section: {path}')
    names, parents, offsets, _ = _hierarchy(''.join(hierarchy_lines))
    return SkeletonSignature(tuple(names), tuple(parents),
                             np.asarray(offsets, dtype=np.float64) * metres_per_unit)


def assert_skeleton_signature(actual, expected, rtol=1e-5, atol=1e-6,
                              ignore_offsets=()):
    """Reject a skeleton that is not the same canonical metric rest skeleton."""
    if actual.joint_names != expected.joint_names:
        raise ValueError('BVH joint order does not match the canonical skeleton')
    if actual.parents != expected.parents:
        raise ValueError('BVH parent indices do not match the canonical skeleton')
    ignored = set(ignore_offsets)
    compared = np.asarray([
        name not in ignored for name in expected.joint_names
    ], dtype=bool)
    if not np.allclose(actual.local_rest_offsets_m[compared],
                       expected.local_rest_offsets_m[compared],
                       rtol=rtol, atol=atol):
        difference = float(np.max(np.abs(
            actual.local_rest_offsets_m[compared]
            - expected.local_rest_offsets_m[compared])))
        raise ValueError(
            f'BVH local rest offsets do not match the canonical skeleton '
            f'(max absolute difference {difference:g} m)')
    return actual


def soma_uniform_ignored_offsets(info):
    """Ignore Hips rest placement only when frames provide its XYZ position."""
    index = info['joint_names'].index('Hips')
    channels = info['channels'][index]
    return ('Hips',) if channels[:3] == SOMA_UNIFORM_POSITION_CHANNELS else ()


def validate_tennis_skeleton(info, offsets_m):
    """Validate this capture convention while allowing actor-specific bone lengths."""
    if (tuple(info['joint_names']) != TENNIS_JOINTS
            or info['parents'] != TENNIS_PARENTS
            or info['channels'][0][:3] != SOMA_UNIFORM_POSITION_CHANNELS
            or info['channels'][0][3:] not in TENNIS_ROTATION_ORDERS
            or any(ch != info['channels'][0][3:] for ch in info['channels'][1:])):
        raise ValueError('Tennis BVH hierarchy or channels differ from the validated rig')
    offsets = np.asarray(offsets_m)
    if offsets.shape != (len(TENNIS_JOINTS), 3) or not np.isfinite(offsets).all():
        raise ValueError('Tennis BVH offsets are invalid')
    index = {name: i for i, name in enumerate(TENNIS_JOINTS)}
    if (not all(.05 < offsets[index[name], 1] < .2
                for name in ('Spine', 'Spine1', 'Spine2', 'Spine3'))
            or not all(-.7 < offsets[index[name], 1] < -.2
                       for name in ('LeftLeg', 'LeftFoot', 'RightLeg', 'RightFoot'))
            or not (.05 < offsets[index['LeftUpLeg'], 0] < .2
                    and -.2 < offsets[index['RightUpLeg'], 0] < -.05)
            or not all(
                .1 < side * offsets[index[side_name + part], 0] < .35
                and np.linalg.norm(offsets[index[side_name + part], 1:])
                < .25 * abs(offsets[index[side_name + part], 0])
                for side_name, side in (('Left', 1), ('Right', -1))
                for part in ('Arm', 'ForeArm', 'Hand'))
            or not all(offsets[index[name], 2] > .08
                       for name in ('LeftToeBase', 'RightToeBase'))):
        raise ValueError('Tennis BVH rest axes or centimetre scale differ from the validated rig')


def validate_geno_skeleton(info, offsets_m):
    _, names, parents, _, channels, _, _ = geno_stance()
    if (info['joint_names'] != names or info['parents'] != tuple(parents)
            or info['channels'] != tuple(map(tuple, channels))):
        raise ValueError('Geno BVH hierarchy or six position/rotation channels differ from the validated rig')
    offsets = np.asarray(offsets_m)
    if not np.isfinite(offsets).all():
        raise ValueError('Geno BVH offsets must be finite')
    if not all(
            .2 < np.linalg.norm(offsets[names.index(side + part)]) < .7
            for side in ('Left', 'Right') for part in ('Leg', 'Foot')):
        raise ValueError('Geno BVH bone lengths differ from the centimetre convention')


def inspect_bvh(path):
    """Read timing and choose an adapter from joint names and parent links."""
    hierarchy_lines = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() == 'MOTION':
                break
            hierarchy_lines.append(line)
        else:
            raise ValueError(f'BVH has no MOTION section: {path}')
        frames_line = next(stream, '')
        frame_time_line = next(stream, '')
    hierarchy = ''.join(hierarchy_lines)
    names, parents, offsets, channels = _hierarchy(hierarchy)
    frames = int(frames_line.split(':', 1)[1])
    frame_time = float(frame_time_line.split(':', 1)[1])
    parent = {n: names[p] if p >= 0 else None for n, p in zip(names, parents)}
    soma_core = set(HUMAN) | {'Root', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head'}
    standard_core = {MAPPING[n] for n in HUMAN} | {'Spine', 'Spine1', 'Neck', 'Head'}
    index = {name: i for i, name in enumerate(names)}
    lafan = (tuple(names) == LAFAN1_JOINTS
             and tuple(parents) == LAFAN1_PARENTS
             and tuple(channels[0]) == LAFAN1_ROOT_CHANNELS
             and all(tuple(value) == LAFAN1_JOINT_CHANNELS for value in channels[1:])
             and all(abs(offsets[index[name], 0]) > .95 * np.linalg.norm(offsets[index[name]])
                     for name in ('Spine1', 'Spine2', 'LeftLeg', 'RightLeg'))
             and offsets[index['LeftUpLeg'], 2] * offsets[index['RightUpLeg'], 2] < 0
             and all(20. <= np.linalg.norm(offsets[index[name]]) <= 70.
                     for name in ('LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot')))
    _, geno_names, geno_parents, _, _, _, _ = geno_stance()
    if names == geno_names and parents == geno_parents:
        adapter = 'orangeduck_geno'
    elif (tuple(names) == TENNIS_JOINTS and tuple(parents) == TENNIS_PARENTS):
        adapter = 'tennis_bvh'
    elif (soma_core <= set(names) and parent['Hips'] == 'Root'
            and all(parent[s + 'Shin'] == s + 'Leg' for s in ('Left', 'Right'))):
        adapter = 'soma'
    elif (lafan and standard_core <= set(names) and parent['Hips'] is None
          and parent['Spine1'] == 'Spine' and parent['Spine2'] == 'Spine1'
          and all(parent[s + 'Leg'] == s + 'UpLeg' for s in ('Left', 'Right'))):
        adapter = 'lafan1'
    elif (standard_core <= set(names) and parent['Hips'] is None
          and parent['Spine1'] == 'Spine'
          and all(parent[s + 'Leg'] == s + 'UpLeg' for s in ('Left', 'Right'))):
        adapter = 'hips_spine_up_leg'
    else:
        raise ValueError(f'Unsupported BVH skeleton in {path}: {names}. Add an explicit adapter for its convention.')
    return dict(fps=1 / frame_time, frame_time=frame_time, frames=frames,
                joint_names=names, parents=tuple(parents),
                channels=tuple(tuple(value) for value in channels), adapter=adapter)


def read_transforms(path):
    """Return declared BVH world transforms in the input units and axes."""
    hierarchy, motion = Path(path).read_text().split('MOTION', 1)
    names, parents, offsets, channels = _hierarchy(hierarchy)
    lines = motion.strip().splitlines()
    declared_frames = int(lines[0].split(':', 1)[1])
    data = np.loadtxt(lines[2:], ndmin=2)
    if data.shape != (declared_frames, sum(map(len, channels))):
        raise ValueError(f'BVH frame count or channel width differs from header: {path}')
    rotations = np.empty((len(data), len(names), 3, 3))
    positions = np.empty((len(data), len(names), 3))
    cursor = 0
    for j, ch in enumerate(channels):
        local = np.tile(offsets[j], (len(data), 1))
        order, angles = '', []
        for k, c in enumerate(ch):
            if c.endswith('position'):
                local[:, 'XYZ'.index(c[0])] = data[:, cursor + k]
            else:
                order += c[0]
                angles.append(data[:, cursor + k])
        r = (Rotation.from_euler(order, np.array(angles).T, degrees=True).as_matrix()
             if order else np.tile(np.eye(3), (len(data), 1, 1)))
        p = parents[j]
        rotations[:, j] = r if p < 0 else rotations[:, p] @ r
        positions[:, j] = (local if p < 0 else positions[:, p]
                           + np.einsum('tij,tj->ti', rotations[:, p], local))
        cursor += len(ch)
    return hierarchy, names, parents, offsets, channels, positions, rotations


def adapt_bvh(source: Path, destination: Path, template: Path,
              repair_first_frame_root_jump_m=None):
    """Copy SOMA or map a supported convention into SOMA bind axes.

    The tennis profile may replace a displaced first frame with frame one.
    No frames are removed or resampled; retargeting keeps the source rate.
    """
    source, destination = Path(source), Path(destination)
    info = inspect_bvh(source)
    if info['adapter'] == 'orangeduck_geno':
        raise ValueError('Geno requires its native FBX surface; select orangeduck-geno')
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_fps=info['fps'], source_frames=info['frames'],
                    source_frame_time=info['frame_time'], adapter=info['adapter'])
    if info['adapter'] == 'soma':
        shutil.copy2(source, destination)
        return dict(metadata, human_joint_map={n: n for n in HUMAN})
    if info['adapter'] == 'lafan1':
        raise ValueError(
            'LaFAN1 must be converted to SMPL-X before UMR retargeting; '
            'select the lafan1-smplx source profile')
    if info['adapter'] == 'tennis_bvh':
        validate_tennis_skeleton(info, bvh_skeleton_signature(source).local_rest_offsets_m)
    if repair_first_frame_root_jump_m is not None:
        if (info['adapter'] != 'tennis_bvh'
                or not np.isfinite(repair_first_frame_root_jump_m)
                or repair_first_frame_root_jump_m <= 0):
            raise ValueError('First-frame root repair requires a positive tennis threshold')

    _, sn, sp, so, _, spos, sr = read_transforms(source)
    first_frame_replaced = False
    first_root_step_m = 0.
    if repair_first_frame_root_jump_m is not None:
        hips = sn.index('Hips')
        if len(spos) < 2:
            raise ValueError('Tennis first-frame repair requires at least two frames')
        first_root_step_m = float(np.linalg.norm(spos[1, hips] - spos[0, hips]) * .01)
        if first_root_step_m > repair_first_frame_root_jump_m:
            spos[0] = spos[1]
            sr[0] = sr[1]
            first_frame_replaced = True
    hierarchy, tn, tp, to, channels, tpos, tr = read_transforms(template)
    hips = tn.index('Hips')
    # Both retargeters need the same translation: static Root, moving Hips.
    if not any(c.endswith('position') for c in channels[hips]):
        channels[hips] = ['Xposition', 'Yposition', 'Zposition'] + channels[hips]
        before, after = hierarchy.split('JOINT Hips', 1)
        after = re.sub(r'CHANNELS\s+3\s+', 'CHANNELS 6 Xposition Yposition Zposition ', after, count=1)
        hierarchy = before + 'JOINT Hips' + after
    rest_s = np.empty_like(so)
    for j, p in enumerate(sp):
        rest_s[j] = so[j] if p < 0 else rest_s[p] + so[j]
    src_leg = np.mean([np.linalg.norm(rest_s[sn.index(s+'UpLeg')] - rest_s[sn.index(s+'Leg')])
                       + np.linalg.norm(rest_s[sn.index(s+'Leg')] - rest_s[sn.index(s+'Foot')])
                       for s in ('Left', 'Right')])
    dst_leg = np.mean([np.linalg.norm(tpos[0, tn.index(s+'Leg')] - tpos[0, tn.index(s+'Shin')])
                       + np.linalg.norm(tpos[0, tn.index(s+'Shin')] - tpos[0, tn.index(s+'Foot')])
                       for s in ('Left', 'Right')])
    scale = float(dst_leg / src_leg)
    n = len(spos)
    mapping = TENNIS_MAPPING if info['adapter'] == 'tennis_bvh' else MAPPING
    world = np.empty((n, len(tn), 3, 3))
    delta = np.tile(np.eye(3), (n, len(tn), 1, 1))
    for j, name in enumerate(tn):
        if name in mapping and mapping[name] in sn:
            delta[:, j] = sr[:, sn.index(mapping[name])]
        elif name == 'Spine2':
            a, b = sr[:, sn.index('Spine')], sr[:, sn.index('Spine1')]
            half = Rotation.from_matrix(np.swapaxes(a, 1, 2) @ b).as_rotvec() * .5
            delta[:, j] = a @ Rotation.from_rotvec(half).as_matrix()
        elif tp[j] >= 0:
            # Missing terminal/face/finger joints inherit the parent's motion.
            delta[:, j] = delta[:, tp[j]]
        world[:, j] = delta[:, j] @ tr[0, j]
    values = []
    for j, ch in enumerate(channels):
        p = tp[j]
        local_r = world[:, j] if p < 0 else np.swapaxes(world[:, p], 1, 2) @ world[:, j]
        order = ''.join(c[0] for c in ch if c.endswith('rotation'))
        angles = Rotation.from_matrix(local_r).as_euler(order, degrees=True)
        local_t = np.tile(to[j], (n, 1))
        if p < 0:
            local_t = np.zeros((n, 3))
        elif j == hips:
            local_t = spos[:, sn.index('Hips')] * scale
        k = 0
        for c in ch:
            if c.endswith('position'):
                values.append(local_t[:, 'XYZ'.index(c[0])])
            else:
                values.append(angles[:, k])
                k += 1
    with destination.open('w') as f:
        f.write(hierarchy + f"MOTION\nFrames: {n}\nFrame Time: {info['frame_time']:.17g}\n")
        np.savetxt(f, np.array(values).T, fmt='%.7f')
    return dict(metadata, source_to_soma_leg_scale=scale,
                **({'first_frame_replaced': first_frame_replaced,
                    'first_root_step_m': first_root_step_m}
                   if repair_first_frame_root_jump_m is not None else {}),
                human_joint_map={n: mapping[n] for n in HUMAN})
