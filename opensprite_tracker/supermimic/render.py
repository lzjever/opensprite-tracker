"""Offline CPU MuJoCo FK + EGL mesh rendering of a recorded REAL rollout.

python -m opensprite_tracker.supermimic.render --rollout EVAL/rollout.npz \
    --output outputs/supermimic/video-15000 --env-index 0
python -m opensprite_tracker.supermimic.render --rollout EVAL/rollout.npz \
    --start-seconds 32 --end-seconds 34 --output outputs/supermimic/video-jump

No policy/world model or simulator stepping is used. Following
scripts/record_agibot_x2_policy_comparison.py and mjlab's MotionCommand ghost
renderer, hide robot collision geoms, clone the same mesh model, recolor its
visual geoms, then append the reference with mjv_addGeoms. Both robots retain
their recorded world positions; no per-frame alignment hides tracking drift.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ["MUJOCO_GL"] = "egl"

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def build_models(path, width, height):
    spec = mujoco.MjSpec.from_file(str(path))
    # The standalone robot asset has no scene floor/light; add visual scenery only.
    spec.worldbody.add_geom(
        name="render_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.26, 0.28, 0.30, 1],
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        name="render_key", pos=[0, -3, 5], dir=[0, 0.5, -1], diffuse=[0.8, 0.8, 0.8]
    )
    model = spec.compile()
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    # The standalone MJCF disables some contact flags; scene CollisionCfg turns
    # soles back on in training. They are still collision shapes, not visuals.
    collision_names = np.array(
        [(model.geom(i).name or "").endswith("_collision") for i in range(model.ngeom)]
    )
    collisions = (
        (model.geom_contype != 0) | (model.geom_conaffinity != 0) | collision_names
    ) & (model.geom_bodyid != 0)
    model.geom_rgba[collisions, 3] = 0
    ghost = copy.deepcopy(model)
    for i in range(ghost.ngeom):
        if collisions[i] or ghost.geom_bodyid[i] == 0:
            ghost.geom_rgba[i, 3] = 0
        else:
            ghost.geom_rgba[i] = [0.35, 0.75, 1.0, 0.45]
    return model, ghost


def qpos_addresses(model, joint_names):
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("render model must have exactly one floating root")
    names = {model.joint(i).name.rsplit("/", 1)[-1]: i for i in range(model.njnt)}
    joints = [int(model.jnt_qposadr[names[name]]) for name in joint_names]
    return int(model.jnt_qposadr[free[0]]), joints


def set_pose(model, data, root_adr, joint_adrs, p, quat, q):
    data.qpos[:] = model.qpos0
    data.qpos[root_adr : root_adr + 3] = p
    data.qpos[root_adr + 3 : root_adr + 7] = quat
    data.qpos[joint_adrs] = q
    mujoco.mj_forward(model, data)


def render(args):
    rollout = args.rollout.resolve(strict=True)
    with np.load(rollout, allow_pickle=False) as archive:
        keys = (
            "root_pos",
            "root_quat_wxyz",
            "joint_pos",
            "ref_frame",
            "live",
            "ref_root_pos",
            "ref_root_quat_wxyz",
            "ref_joint_pos",
            "first_failure_frame",
            "joint_names",
            "fps",
            "metadata_json",
        )
        arrays = {key: archive[key] for key in keys}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    ref_start = int(metadata.get("ref_start", 0))
    if (
        args.checkpoint is not None
        and str(args.checkpoint.resolve()) != metadata["checkpoint"]
    ):
        raise ValueError("--checkpoint does not match the recorded rollout")
    env_id = args.env_index
    if not 0 <= env_id < arrays["root_pos"].shape[1]:
        raise ValueError("env-index is outside the recorded batch")
    fps = float(arrays["fps"])
    start = max(0, int(math.ceil(args.start_seconds * fps)))
    end = (
        len(arrays["root_pos"])
        if args.end_seconds is None
        else min(len(arrays["root_pos"]), int(math.floor(args.end_seconds * fps)) + 1)
    )
    if start >= end:
        raise ValueError("empty recording interval")
    model_path = Path(args.model or metadata["model_path"]).resolve(strict=True)
    if args.width < 1 or args.height < 1:
        raise ValueError("width/height must be positive")
    if args.output is None:
        output = Path(
            tempfile.mkdtemp(
                prefix="render-" + time.strftime("%Y%m%d-%H%M%S") + "-",
                dir=rollout.parent,
            )
        )
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    model, ghost = build_models(model_path, args.width, args.height)
    data, ghost_data = mujoco.MjData(model), mujoco.MjData(ghost)
    root_adr, joint_adrs = qpos_addresses(
        model, arrays["joint_names"].astype(str).tolist()
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = (
        args.camera_distance,
        args.camera_azimuth,
        args.camera_elevation,
    )
    opt, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    opt.sitegroup[:] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    failure = int(arrays["first_failure_frame"][env_id])
    video = output / "comparison.mp4"
    with mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
        max_geom=max(10000, 3 * model.ngeom),
    ) as renderer:
        with imageio.get_writer(
            str(video), fps=fps, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for t in range(start, end):
                p = arrays["root_pos"][t, env_id]
                ref_p = arrays["ref_root_pos"][t]
                set_pose(
                    model,
                    data,
                    root_adr,
                    joint_adrs,
                    p,
                    arrays["root_quat_wxyz"][t, env_id],
                    arrays["joint_pos"][t, env_id],
                )
                set_pose(
                    ghost,
                    ghost_data,
                    root_adr,
                    joint_adrs,
                    ref_p,
                    arrays["ref_root_quat_wxyz"][t],
                    arrays["ref_joint_pos"][t],
                )
                camera.lookat[:] = 0.5 * (p + ref_p)
                camera.lookat[2] = max(0.5, camera.lookat[2])
                renderer.update_scene(data, camera=camera, scene_option=opt)
                # Same mesh IDs and visual geometry as the solid robot.
                mujoco.mjv_addGeoms(
                    ghost,
                    ghost_data,
                    opt,
                    perturb,
                    mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
                    renderer.scene,
                )
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                frame = int(arrays["ref_frame"][t, env_id])
                live = bool(arrays["live"][t, env_id])
                local_frame = frame - ref_start
                status = (
                    "LIVE"
                    if live
                    else f"FAILED at clip frame {failure - ref_start} ({(failure - ref_start) / fps:.2f}s)"
                )
                text = (
                    f"SuperMimic OpenSprite | clip frame {local_frame} | {local_frame / fps:.2f}s | env {env_id}\n"
                    f"solid: real MuJoCo policy | blue transparent: reference mesh\n"
                    f"{status} | post-failure transitions excluded from scores"
                )
                draw.rectangle((8, 8, min(args.width - 1, 650), 67), fill=(20, 22, 26))
                draw.multiline_text(
                    (15, 14),
                    text,
                    fill=(230, 235, 240) if live else (255, 130, 110),
                    spacing=4,
                )
                writer.append_data(np.asarray(image))
                if (t - start + 1) % 250 == 0 or t + 1 == end:
                    print(f"RENDER {t - start + 1}/{end - start}", flush=True)
    info = {
        "rollout": str(rollout),
        "checkpoint": metadata["checkpoint"],
        "model": str(model_path),
        "env_index": env_id,
        "reference_start_frame": ref_start,
        "time_display": "clip-local seconds",
        "start_state_index": start,
        "end_state_index_exclusive": end,
        "fps": fps,
        "frames": end - start,
        "video": str(video),
        "rendering": "CPU MuJoCo forward kinematics, EGL; no physics steps",
        "ghost": "identical model meshes; collision geoms hidden; no position realignment",
        "source_complete_clip": metadata["complete_clip"],
    }
    (output / "render.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"SAVED {video}", flush=True)
    return video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="optional consistency check against rollout metadata",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="new directory; default is a unique directory beside rollout",
    )
    p.add_argument(
        "--model", type=Path, help="override recorded MJCF path if checkout moved"
    )
    p.add_argument("--env-index", type=int, default=0)
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--end-seconds", type=float)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera-distance", type=float, default=3.2)
    p.add_argument("--camera-azimuth", type=float, default=120.0)
    p.add_argument("--camera-elevation", type=float, default=-10.0)
    render(p.parse_args())


if __name__ == "__main__":
    main()
