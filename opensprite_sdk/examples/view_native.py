#!/usr/bin/env python3
"""Render native simulation diagnostics without owning policy or physics."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from opensprite_sdk.visualization import passive_viewer


def run(args):
    scene=json.loads(args.scene.read_text())
    model=mujoco.MjModel.from_xml_path(str(args.scene.parent/scene['model']))
    data=mujoco.MjData(model)
    data.qpos[:]=scene['initial_qpos'];data.qvel[:]=scene['initial_qvel']
    mujoco.mj_forward(model,data)
    stream=None;stamp=0.;dispatches=0
    try:
        with passive_viewer(model,data,show_left_ui=False,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.distance=2.1;viewer.cam.azimuth=135;viewer.cam.elevation=-15
            start=time.monotonic()
            while viewer.is_running():
                if stream is None and args.truth.exists():stream=args.truth.open()
                latest=None
                if stream is not None:
                    while True:
                        position=stream.tell();line=stream.readline()
                        if not line:break
                        if not line.endswith('\n'):
                            stream.seek(position);break
                        latest=json.loads(line)
                with viewer.lock():
                    if latest is not None:
                        qpos=np.asarray(latest['qpos'],dtype=float)
                        qvel=np.asarray(latest['qvel'],dtype=float)
                        if qpos.shape!=(38,) or qvel.shape!=(37,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                            raise ValueError('invalid native diagnostic state')
                        data.qpos[:]=qpos;data.qvel[:]=qvel
                        stamp=float(latest['simulated_seconds']);data.time=stamp
                        dispatches=latest.get('joint_command_dispatches',0)
                        mujoco.mj_forward(model,data)
                    viewer.cam.lookat[:]=data.xpos[1]+np.array([0.,0.,.05])
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'OpenSprite native simulation\nClocks\nTime\nJoint dispatches\nStatus',
                    f'Diagnostic observer / no control\n500 Hz commands | 200 Hz physics | 50 Hz actor\n'
                    f'{stamp:.2f} s\n{dispatches}\nUnscored diagnostics | no video recording'))
                viewer.sync()
                if args.duration is not None and time.monotonic()-start>=args.duration:break
                time.sleep(.02)
    finally:
        if stream is not None:stream.close()
    return {'last_simulated_seconds':stamp,'video_recorded':False,'owns_physics':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--truth',required=True,type=Path)
    parser.add_argument('--duration',type=float,help='optional finite GUI check in wall seconds')
    args=parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration<=0):
        parser.error('duration must be finite and positive')
    print(json.dumps(run(args),indent=2))
