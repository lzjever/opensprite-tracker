#!/usr/bin/env python3
"""Watch the deployed ONNX actor in an interactive, real-time MuJoCo window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

from opensprite_sdk.backends.mujoco import MujocoBackend
from opensprite_sdk.client import LowLevelClient
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.policy import PolicyBundle, quaternion_matrix
from opensprite_sdk.service import BackendService
from opensprite_sdk.visualization import passive_viewer


class SimulationSession:
    """Use the ordinary SDK messages; viewer truth is only for diagnostics."""
    def __init__(self, bundle, *, seed=42):
        self.bundle=bundle
        self.backend=MujocoBackend(profile=bundle.contract["profile_id"],seed=seed)
        self.service=BackendService(self.backend)
        self.client=None
        self.frame=0
        self.first_failure=None
        self.height_error=0.
        self.gravity_error=0.

    def start(self):
        ChannelFactoryInitialize(22,transport="memory")
        self.service.start()
        self.restart()
        return self

    def restart(self):
        # A manual restart is a new run, with new source and controller epochs.
        # It never wraps a running reference or silently clears its first failure.
        if self.client is not None:
            self.client.Close()
        reference=self.bundle.reference
        self.backend.reset(qpos=reference["qpos"][0],qvel=reference["qvel"][0])
        floor=self.backend.model.geom("sdk_ground").id
        distances=[c.dist for c in self.backend.data.contact if floor in c.geom]
        lift=max(0.,.001-min(distances)) if distances else 0.
        if lift:
            qpos=self.backend.data.qpos.copy()
            qpos[2]+=lift
            self.backend.reset(qpos=qpos,qvel=reference["qvel"][0])
        self.backend.enable()
        self.bundle.reset()
        self.client=LowLevelClient().Init()
        self.service.publish_state()
        self.frame=0
        self.first_failure=None
        self.height_error=0.
        self.gravity_error=0.

    @property
    def finished(self):
        return self.frame>=self.bundle.contract["frames"]-1

    def step(self):
        if self.finished:
            return False
        state=self.client.Read()
        self.client.Write(self.bundle.command(state,self.frame,now_ns=state.stamp_ns))
        if not self.service.tick(steps=self.backend.policy_steps):
            raise RuntimeError("SDK did not deliver the actor command")
        self.frame+=1
        truth=self.backend.debug_truth()
        ref=self.bundle.reference
        rotation=quaternion_matrix(truth["qpos"][3:7])
        self.height_error=float(abs(truth["qpos"][2]-ref["qpos"][self.frame,2]))
        self.gravity_error=float(np.sum((rotation[2]-ref["root_rotation"][self.frame,2])**2))
        if self.first_failure is None and (self.height_error>.17 or self.gravity_error>1.6):
            self.first_failure=self.frame/50.
            print(json.dumps({"event":"first_failure","seconds":self.first_failure,
                              "continue_policy_and_physics":True}),flush=True)
        return True

    def close(self):
        try:
            if self.client is not None:
                self.client.Close()
        finally:
            self.service.close()


def run(args):
    import mujoco.viewer
    bundle=PolicyBundle(args.bundle)
    session=SimulationSession(bundle,seed=args.seed)
    keys=queue.SimpleQueue()
    paused=args.paused
    try:
        session.start()
        with passive_viewer(session.backend.model,session.backend.data,
                key_callback=keys.put,show_left_ui=False,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.distance=2.1
                viewer.cam.azimuth=135
                viewer.cam.elevation=-15
                viewer.cam.lookat[:]=session.backend.data.xpos[1]+np.array([0.,0.,.05])
            reference_data=mujoco.MjData(session.backend.model)
            reference_options=mujoco.MjvOption()
            reference_options.geomgroup[:]=0
            reference_options.geomgroup[2]=1
            print(json.dumps({"event":"viewer_ready","backend":"mujoco",
                "profile":bundle.contract["profile_id"],"model":bundle.contract.get("checkpoint_iteration"),
                "policy_hz":50,"video_recording":False,"paused":paused}),flush=True)
            deadline=time.monotonic()
            while viewer.is_running():
                while not keys.empty():
                    key=keys.get()
                    if key==32:
                        paused=not paused
                        deadline=time.monotonic()
                    elif key in (ord('R'),ord('r')):
                        session.restart()
                        paused=False
                        deadline=time.monotonic()
                        print(json.dumps({"event":"manual_restart","frame":0}),flush=True)
                    elif key==256:  # GLFW_KEY_ESCAPE
                        viewer.close()
                        break
                if not viewer.is_running():
                    break
                if not paused and not session.finished:
                    session.step()
                    if session.frame%250==0:
                        print(json.dumps({"event":"progress","seconds":session.frame/50,
                                          "first_failure":session.first_failure}),flush=True)
                status="FINISHED - R to restart" if session.finished else ("PAUSED" if paused else "RUNNING")
                if session.first_failure is not None:
                    status+=f" | POST FAILURE / UNSCORED (first {session.first_failure:.2f}s)"
                with viewer.lock():
                    viewer.cam.lookat[:]=session.backend.data.xpos[1]+np.array([0.,0.,.05])
                    if not args.no_reference:
                        reference_data.qpos[:]=bundle.reference["qpos"][session.frame]
                        mujoco.mj_forward(session.backend.model,reference_data)
                        mujoco.mjv_updateScene(session.backend.model,reference_data,reference_options,None,
                            viewer.cam,mujoco.mjtCatBit.mjCAT_DYNAMIC,viewer.user_scn)
                        for index in range(viewer.user_scn.ngeom):
                            viewer.user_scn.geoms[index].rgba[:]=[.12,.55,1.,.22]
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "OpenSprite SDK sim2sim\nTime\nState\nHeight / gravity error\nControls",
                    f"ONNX model {bundle.contract.get('checkpoint_iteration')} | blue = reference\n"
                    f"{session.frame/50:.2f} / {(bundle.contract['frames']-1)/50:.2f} s\n"
                    f"{status}\n{session.height_error:.3f} m / {session.gravity_error:.3f}\n"
                    "SPACE pause/resume | R new run | ESC close"))
                viewer.sync()
                if args.exit_at_end and session.finished:
                    break
                if args.duration is not None and session.frame/50>=args.duration:
                    break
                if paused or session.finished:
                    deadline=time.monotonic()
                    time.sleep(.02)
                else:
                    deadline+=session.backend.policy_dt
                    remaining=deadline-time.monotonic()
                    if remaining<-.1:
                        # GUI stalls must not produce a burst of catch-up steps.
                        deadline=time.monotonic()
                    time.sleep(max(0.,remaining))
    finally:
        session.close()
    return {"frames":session.frame,"seconds":session.frame/50,
            "first_failure_seconds":session.first_failure,"video_recorded":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--paused",action="store_true")
    parser.add_argument("--no-reference",action="store_true")
    parser.add_argument("--exit-at-end",action="store_true")
    parser.add_argument("--duration",type=float,help="optional finite smoke-test duration")
    args=parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration<.02):
        parser.error("duration must be finite and at least .02 seconds")
    print(json.dumps(run(args),indent=2))
