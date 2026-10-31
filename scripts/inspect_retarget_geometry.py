#!/usr/bin/env python3
"""Measure exact primitive-ground support heights, optionally render a sheet."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from opensprite_tracker.robot import ROBOT_XML


def collision_bottom(model, data, ids):
    heights = data.geom_xpos[ids, 2].copy()
    for i, gid in enumerate(ids):
        kind, size = model.geom_type[gid], model.geom_size[gid]
        rz = data.geom_xmat[gid].reshape(3, 3)[2]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            heights[i] -= abs(rz) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            heights[i] -= size[0] + abs(rz[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            heights[i] -= size[0]
        else:
            raise ValueError(f"Unhandled collision type {kind}")
    return heights


def support_gaps(mask, fps):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends-starts
    longest = int(np.argmax(lengths)) if len(lengths) else None
    return dict(fraction=float(np.mean(mask)), total_seconds=float(np.sum(mask)/fps),
                longest_seconds=float(lengths[longest]/fps) if longest is not None else 0.,
                longest_start_s=float(starts[longest]/fps) if longest is not None else None,
                longest_end_s=float(ends[longest]/fps) if longest is not None else None)


def inspect(path, output, render=False, self_collision=False):
    with np.load(path, allow_pickle=False) as archive:
        trajectory = archive["qpos"]
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    ids = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    heights = np.empty((len(trajectory), len(ids)))
    self_minimum = np.zeros(len(trajectory))
    self_pairs = {}
    for i, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        if self_collision:
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if contact.dist >= 0:
                    continue
                self_minimum[i] = min(self_minimum[i], contact.dist)
                pair = " / ".join(sorted(model.geom(g).name for g in contact.geom))
                self_pairs[pair] = min(self_pairs.get(pair, 0.), float(contact.dist))
        else:
            mujoco.mj_kinematics(model, data)
        heights[i] = collision_bottom(model, data, ids)
    minimum = heights.min(1)
    foot_columns = [j for j,gid in enumerate(ids) if model.geom(gid).name in
                    ("left_foot_collision", "right_foot_collision")]
    foot_minimum = heights[:, foot_columns].min(1)
    result = dict(motion=path.stem, frames=len(trajectory), seconds=len(trajectory)/fps,
                  minimum_collision_z_m=float(minimum.min()),
                  worst_time_s=float(np.argmin(minimum)/fps),
                  fraction_below_minus_1cm=float(np.mean(minimum < -.01)),
                  fraction_below_minus_3cm=float(np.mean(minimum < -.03)),
                  fraction_below_minus_10cm=float(np.mean(minimum < -.10)),
                  all_body_support_gap_over_2cm=support_gaps(minimum > .02, fps),
                  both_feet_gap_over_2cm=support_gaps(foot_minimum > .02, fps),
                  by_geometry={model.geom(gid).name: float(heights[:,j].min())
                               for j, gid in enumerate(ids)})
    if self_collision:
        result["self_collision"] = dict(
            minimum_signed_distance_m=float(self_minimum.min()),
            fraction_below_minus_1cm=float(np.mean(self_minimum < -.01)),
            fraction_below_minus_3cm=float(np.mean(self_minimum < -.03)),
            worst_time_s=float(np.argmin(self_minimum)/fps),
            pairs=dict(sorted(self_pairs.items(), key=lambda item:item[1])[:20]))
    output.mkdir(exist_ok=True, parents=True)
    (output / f"{path.stem}_geometry.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if render:
        extra = [int(np.argmin(trajectory[:, 2]))]
        if self_collision:
            extra.append(int(np.argmin(self_minimum)))
        render_sheet(path.stem, model, data, trajectory, minimum, fps, output, extra)


def render_sheet(name, model, data, trajectory, minimum, fps, output, extra=()):
    from PIL import Image, ImageDraw
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0,0,.05], rgba=[.24,.29,.35,1])
    spec.worldbody.add_light(pos=[0,0,3], dir=[0,0,-1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    model = spec.compile()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 125.
    camera.elevation = -15.
    frames = sorted(set(np.linspace(0, len(trajectory)-1, 9, dtype=int).tolist()
                        + [int(np.argmin(minimum))] + list(extra)))
    sheet = Image.new("RGB", (4*480, 3*390), "white")
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for k, frame in enumerate(frames[:12]):
            data.qpos[:] = trajectory[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [*data.qpos[:2], .40]
            renderer.update_scene(data, camera)
            image = Image.fromarray(renderer.render())
            x, y = (k%4)*480, (k//4)*390
            sheet.paste(image, (x,y))
            ImageDraw.Draw(sheet).text((x+8,y+365),
                f"{name} t={frame/fps:.2f}s min_z={minimum[frame]:.3f}m",fill="black")
    sheet.save(output / f"{name}_sheet.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/retarget/previews")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--self-collision", action="store_true")
    args = parser.parse_args()
    for path in args.motions:
        inspect(path, args.output, args.render, args.self_collision)
