#!/usr/bin/env python3
"""Prepare a portable robot scene. Runtime needs no Python or actor bundle."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from opensprite_sdk.backends.mujoco import MujocoBackend, PACKAGE_ROOT
from opensprite_sdk.constants import JOINT_NAMES


def prepare(args):
    output=args.output.resolve()
    if output.exists():
        raise ValueError("scene output already exists; select a fresh directory")
    with MujocoBackend(args.profile,seed=args.seed) as backend:
        if not np.isclose(backend.dt,.005):
            raise ValueError('simulation physics must be200Hz; command dispatch is independently500Hz')
        if args.motion:
            with np.load(args.motion,allow_pickle=False) as motion:
                if 'qpos' in motion:
                    pose,velocity=motion['qpos'][0],motion['qvel'][0]
                else:
                    names=motion['joint_names'].astype(str).tolist()
                    ids=[names.index(n) for n in JOINT_NAMES]
                    root=motion['body_names'].astype(str).tolist().index('pelvis_link')
                    quat=np.asarray(motion['body_quat_w'][0,root],dtype=float)
                    matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quat)
                    pose=np.concatenate((motion['body_pos_w'][0,root],quat,motion['joint_pos'][0,ids]))
                    velocity=np.concatenate((motion['body_lin_vel_w'][0,root],
                        matrix.reshape(3,3).T@motion['body_ang_vel_w'][0,root],motion['joint_vel'][0,ids]))
                backend.reset(qpos=pose,qvel=velocity)
        floor=backend.model.geom('sdk_ground').id
        distances=[c.dist for c in backend.data.contact if floor in c.geom]
        lift=max(0.,.001-min(distances)) if distances else 0.
        qpos=backend.data.qpos.copy();qpos[2]+=lift
        qvel=backend.data.qvel.copy()
        output.mkdir(parents=True)
        try:
            (output/'meshes').mkdir()
            model=output/'robot.xml'
            mujoco.mj_saveLastXML(str(model),backend.model)
            tree=ET.parse(model);compiler=tree.getroot().find('compiler')
            compiler.set('meshdir','meshes')
            source=PACKAGE_ROOT/backend.profile['robot_model']
            original_tree=ET.parse(source)
            meshdir=source.parent/original_tree.getroot().find('compiler').get('meshdir','.')
            meshes={m.get('name'):(meshdir/m.get('file')).resolve()
                    for m in original_tree.getroot().findall('asset/mesh')}
            for mesh in tree.getroot().findall('asset/mesh'):
                original=meshes[mesh.get('name')]
                shutil.copy2(original,output/'meshes'/original.name)
                mesh.set('file',original.name)
            tree.write(model,encoding='unicode')
            scene={'schema':'opensprite-native-sim-v1','profile':backend.profile,
                'joint_execution_hz':500,'physics_hz':200,'policy_hz':50,
                'model':'robot.xml','initial_qpos':qpos.tolist(),'initial_qvel':qvel.tolist(),
                'initial_lift_m':lift,'seed':args.seed}
            (output/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
            shutil.copy2(PACKAGE_ROOT/'assets/robot/LICENSE',output/'ROBOT_LICENSE')
        except BaseException:
            shutil.rmtree(output)
            raise
    return {'scene':str(output/'scene.json'),'profile':args.profile,
            'runtime_requires_actor':False,'runtime_requires_python':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',default='legacy_v4')
    parser.add_argument('--motion',type=Path,help='optional retargeted qpos/qvel; no actor needed')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',required=True,type=Path)
    print(json.dumps(prepare(parser.parse_args()),indent=2))
