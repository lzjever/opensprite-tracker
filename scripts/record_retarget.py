#!/usr/bin/env python3
"""Render retargeted qpos directly, independent of a policy/checkpoint."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import imageio.v2 as imageio
from opensprite_tracker.robot import ROBOT_XML


def record(path, output, start=0., duration=0., stride=2):
    with np.load(path, allow_pickle=False) as archive:
        qpos = archive["qpos"]
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    first = max(0, round(start*fps))
    last = min(len(qpos), round((start+duration)*fps)) if duration > 0 else len(qpos)
    if first >= last:
        raise ValueError("Requested interval is outside the motion")
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0,0,.05], rgba=[.24,.29,.35,1])
    spec.worldbody.add_light(pos=[0,0,3], dir=[0,0,-1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    model = spec.compile()
    model.vis.global_.offheight = 720
    model.vis.global_.offwidth = 960
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 125., -15.
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(output, fps=fps/stride, codec="libx264",
                                quality=7, macro_block_size=1) as writer:
            for frame in range(first, last, stride):
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [*data.qpos[:2], .40]
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
    print(f"Recorded {path.stem} [{first/fps:.2f}, {last/fps:.2f})s -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float, default=0., help="0 records to the real end")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1 or args.start < 0 or args.duration < 0:
        parser.error("stride must be positive; start/duration must be nonnegative")
    record(args.motion, args.output, args.start, args.duration, args.stride)
