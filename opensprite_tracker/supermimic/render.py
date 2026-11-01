# SPDX-License-Identifier: MIT
"""Offline official-mesh comparison of a recorded REAL OpenSprite rollout.

Render the simulated robot and reference ghost in their recorded world
coordinates using the repository's OpenSprite meshes. Rendering advances no
physics and performs no resets.

python -m opensprite_tracker.supermimic.render --rollout EVAL/rollout.npz --output VIDEO
python -m opensprite_tracker.supermimic.render --rollout EVAL/rollout.npz \
    --preview-frame 750 --output PREVIEW
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def continuation_mode(metadata, requested=None):
    enabled = metadata.get("continue_after_failure", False)
    if not isinstance(enabled, bool):
        raise ValueError("continue_after_failure metadata must be boolean")
    if requested is not None and bool(requested) != enabled:
        raise ValueError("Post-failure render assertion must match the actual recording metadata")
    if enabled:
        if metadata.get("post_failure_policy", "continue_actor") != "continue_actor":
            raise ValueError("Continued recording must use the real actor after failure")
        if metadata.get("no_reset") is False or metadata.get("no_reference_wrapping") is False:
            raise ValueError("Continued recording must not reset or wrap the reference")
    return enabled


def recording_frame_plan(arrays, metadata, *, env, source_fps, video_fps,
                         start_seconds=0., end_seconds=None, preview_frame=None,
                         max_render_frames=None, failure_hold_seconds=1.5,
                         include_post_failure=None):
    """Pure frame selection, preserving genuine continued states after failure."""
    if not (math.isfinite(source_fps) and math.isfinite(video_fps)
            and source_fps > 0 and video_fps > 0):
        raise ValueError("Source and video FPS must be positive")
    if failure_hold_seconds < 0 or not math.isfinite(failure_hold_seconds):
        raise ValueError("failure-hold-seconds must be finite and nonnegative")
    count = len(arrays["root_pos"])
    if count < 1 or not 0 <= env < arrays["root_pos"].shape[1] or len(arrays["live"]) != count:
        raise ValueError("Invalid recorded state/live dimensions or env-index")
    continuing = continuation_mode(metadata, include_post_failure)
    failed = np.flatnonzero(~arrays["live"][:, env])
    first_failure = int(failed[0]) if len(failed) else None
    recorded_end = count-1 if continuing or first_failure is None else first_failure
    if continuing:
        if np.any(arrays["live"][1:, env] & ~arrays["live"][:-1, env]):
            raise ValueError("First-failure live mask must remain sticky without a reset")
        if metadata.get("executed_transitions", count-1) != count-1:
            raise ValueError("Continued metadata and recorded state count disagree")
        if "ref_frame" in arrays and np.any(np.diff(arrays["ref_frame"][:, env]) != 1):
            raise ValueError("Continued reference must advance once per state without freezing or wrapping")
        if metadata.get("complete_clip") and count-1 != metadata.get("expected_transitions", count-1):
            raise ValueError("Complete-source metadata cannot describe a shortened continued recording")
    start = max(0, int(math.ceil(start_seconds * source_fps)))
    end = recorded_end + 1
    if end_seconds is not None:
        end = min(end, int(math.floor(end_seconds * source_fps)) + 1)
    if start >= end:
        raise ValueError("empty recording interval, possibly after physical failure")
    if preview_frame is not None:
        if not start <= preview_frame < end:
            raise ValueError("preview-frame is outside the recorded interval")
        indices = [preview_frame]
    else:
        output_count = max(1, int(math.floor((end-1-start)/source_fps*video_fps))+1)
        indices = np.minimum(end-1, np.rint(start + np.arange(output_count)*source_fps/video_fps).astype(int)).tolist()
        if max_render_frames is not None:
            if max_render_frames < 1:
                raise ValueError("max-render-frames must be positive")
            indices = indices[:max_render_frames]
    hold = (max(0, round(failure_hold_seconds*video_fps))
            if not continuing and preview_frame is None and first_failure is not None
            and indices[-1] == recorded_end else 0)
    return {"indices": indices, "interval": (start, end), "recorded_end": recorded_end,
            "first_failure": first_failure, "continue_after_failure": continuing,
            "hold_frames": hold,
            "full_recorded_span_rendered": preview_frame is None and indices[0] == 0 and indices[-1] == count-1,
            "full_source_recorded": bool(metadata.get("complete_clip")
                and count-1 == metadata.get("expected_transitions")
                and metadata.get("requested_transitions") == metadata.get("expected_transitions"))}


def failure_annotation(live, continuing, first_failure_seconds, *, held=False):
    if live:
        return "LIVE", None
    if continuing:
        first = f"{first_failure_seconds:.2f}s" if first_failure_seconds is not None else "unavailable"
        return "POST-FAILURE CONTINUING", f"首次失败 {first} · 失败后持续仿真不重置 / POST-FAILURE CONTINUING"
    return "FAILED", ("真实终止 · 视频停帧回看 · 未继续仿真" if held else "真实终止 · 记录在此截断")


def policy_label(metadata):
    return "单动作策略" if metadata.get("training_clip_count") == 1 else "策略"


def build_models(path, width, height):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.add_texture(name="render_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.22, 0.25, 0.29], rgb2=[0.29, 0.32, 0.36], width=512, height=512)
    material = spec.add_material(name="render_floor_mat", texuniform=True,
                                 texrepeat=[1, 1], rgba=[1, 1, 1, 1],
                                 specular=0.12, shininess=0.18, reflectance=0.04)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "render_checker"
    spec.worldbody.add_geom(name="render_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[100, 100, 0.05], material="render_floor_mat",
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(name="render_key", pos=[-2, -3, 5], dir=[0.3, 0.5, -1],
                             diffuse=[0.8, 0.78, 0.74], ambient=[0.18, 0.18, 0.2],
                             specular=[0.25, 0.25, 0.25], castshadow=True)
    spec.worldbody.add_light(name="render_fill", pos=[3, 2, 4], dir=[-0.5, -0.3, -1],
                             diffuse=[0.35, 0.42, 0.5], castshadow=False)
    spec.visual.quality.offsamples = 4
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.active = True
    spec.visual.headlight.ambient = [0.22, 0.24, 0.28]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    collision_names = np.array([(model.geom(i).name or "").endswith("_collision")
                                for i in range(model.ngeom)])
    collisions = ((model.geom_contype != 0) | (model.geom_conaffinity != 0) | collision_names) & (model.geom_bodyid != 0)
    model.geom_rgba[collisions, 3] = 0
    for i in range(model.ngeom):
        if collisions[i] or model.geom_bodyid[i] == 0:
            continue
        body = model.body(int(model.geom_bodyid[i])).name or ""
        # Only presentation colours change; official meshes and poses stay exact.
        if body in ("pelvis_link", "waist_yaw_link", "waist_roll_link"):
            model.geom_rgba[i] = [0.78, 0.16, 0.21, 1]
        elif "ankle" in body or "wrist" in body:
            model.geom_rgba[i] = [0.28, 0.32, 0.38, 1]
        else:
            model.geom_rgba[i] = [0.9, 0.92, 0.94, 1]
    ghost = copy.deepcopy(model)
    for i in range(ghost.ngeom):
        ghost.geom_rgba[i] = ([0, 0, 0, 0] if collisions[i] or ghost.geom_bodyid[i] == 0
                              else [0.18, 0.65, 1.0, 0.22])
    return model, ghost


def qpos_addresses(model, joint_names):
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("render model must have exactly one floating root")
    names = {model.joint(i).name.rsplit("/", 1)[-1]: i for i in range(model.njnt)}
    return int(model.jnt_qposadr[free[0]]), [int(model.jnt_qposadr[names[n]]) for n in joint_names]


def set_pose(model, data, root_adr, joint_adrs, p, quat, q):
    data.qpos[:] = model.qpos0
    data.qpos[root_adr:root_adr + 3] = p
    data.qpos[root_adr + 3:root_adr + 7] = quat
    data.qpos[joint_adrs] = q
    mujoco.mj_forward(model, data)


class FollowCamera:
    """Frame both full robot link clouds without concealing world drift."""
    def __init__(self, args, model, fps):
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = args.camera_azimuth
        self.camera.elevation = args.camera_elevation
        self.minimum = args.camera_distance
        self.aspect, self.fovy = args.width / args.height, math.radians(float(model.vis.global_.fovy))
        self.alpha = 1 - math.exp(-1 / (fps * 0.2))
        self.initialized = False

    def update(self, actual, reference):
        points = np.concatenate((actual.xpos[1:], reference.xpos[1:]))
        lo, hi = points.min(0) - 0.10, points.max(0) + 0.10
        target = (lo + hi) / 2
        az, el = math.radians(self.camera.azimuth), math.radians(self.camera.elevation)
        right = np.array([-math.sin(az), math.cos(az), 0])
        outward = np.array([math.cos(az)*math.cos(el), math.sin(az)*math.cos(el), -math.sin(el)])
        up = np.cross(outward, right)
        target -= up * 0.06  # Centre the poses in the usable band above the HUD.
        if not self.initialized:
            self.camera.lookat[:] = target
        else:
            self.camera.lookat[:] += self.alpha * (target - self.camera.lookat)
        # Fit perspective depth too: the closer ghost can otherwise hide its
        # feet behind the HUD when pelvis drift reaches several metres.
        padding = np.array([[x, y, z] for x in (-.12, .12) for y in (-.12, .12) for z in (-.12, .12)])
        relative = (points[:, None] + padding - self.camera.lookat).reshape(-1, 3)
        tan_y = math.tan(self.fovy / 2)
        fit = np.maximum(np.abs(relative @ right) / (tan_y * self.aspect * .82),
                         np.abs(relative @ up) / (tan_y * .54))
        required = max(self.minimum, float(np.max(relative @ outward + fit)))
        if not self.initialized:
            self.camera.distance = required
            self.initialized = True
        else:
            smooth = self.camera.distance + self.alpha * (required - self.camera.distance)
            # Expand immediately if needed; zoom inward smoothly to avoid pumping.
            self.camera.distance = max(required, smooth)
        return self.camera


class Hud:
    def __init__(self, args, arrays, metadata, fps, recorded_end):
        self.args, self.a, self.m, self.fps = args, arrays, metadata, fps
        self.w, self.h, self.scale = args.width, args.height, args.height / 1080
        self.font_path = next((p for p in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ] if Path(p).is_file()), None)
        self.fonts = {}
        names = {"dance2_subject1": "舞蹈"}
        clip = metadata.get("clip_name", f"clip {metadata.get('clip', '?')}")
        self.title = args.title or f"{names.get(clip, '动作追踪')}  /  {clip}"
        self.source_seconds = metadata.get("expected_transitions", 0) / fps
        self.request_seconds = metadata.get("requested_transitions", len(arrays["root_pos"]) - 1) / fps
        self.actual_seconds = recorded_end / fps
        self.continuing = continuation_mode(metadata)
        failed = np.flatnonzero(~arrays["live"][:, args.env_index])
        self.first_failure_seconds = (
            (int(arrays["ref_frame"][failed[0], args.env_index])-int(metadata.get("ref_start", 0)))/fps
            if len(failed) else None
        )

    def font(self, size):
        size = max(10, round(size * self.scale))
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
        return self.fonts[size]

    def metric(self, name, t, env):
        value = self.a.get(name)
        # State t is the arrival of transition t-1. Initial state has no samples.
        if value is None or t == 0 or t - 1 >= len(value):
            return None
        sample = np.asarray(value[t - 1, env])
        return sample if np.isfinite(sample).all() else None

    def draw(self, rgb, t, env, held=False):
        image = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        s = self.scale
        xy = lambda box: tuple(round(v * s) for v in box)
        text = lambda pos, value, size, colour: draw.text(xy(pos), value, font=self.font(size), fill=colour)
        white, muted, cyan = (240, 245, 252, 255), (164, 181, 202, 255), (95, 197, 249, 255)
        pad = 36 * s
        draw.rounded_rectangle((pad, 28*s, self.w-pad, 166*s), radius=18*s,
                               fill=(12, 21, 34, 216), outline=(113, 148, 184, 70), width=max(1, round(s)))
        draw.rounded_rectangle((pad, self.h-182*s, self.w-pad, self.h-32*s), radius=18*s,
                               fill=(12, 21, 34, 215), outline=(113, 148, 184, 70), width=max(1, round(s)))
        text((58, 41), "OPENSPRITE  /  SUPERMIMIC", 17, cyan)
        text((58, 68), self.title, 31, white)
        iteration = self.m.get("iteration", "unavailable")
        details = f"{policy_label(self.m)} · checkpoint {iteration}    |    seed {self.m.get('seed', '?')} · env {env}    |    {self.args.fps:g} FPS · {self.w}×{self.h}"
        text((58, 114), details, 19, muted)
        frame = int(self.a["ref_frame"][t, env]) - int(self.m.get("ref_start", 0))
        seconds = frame / self.fps
        live = bool(self.a["live"][t, env])
        status, failure_message = failure_annotation(live, self.continuing, self.first_failure_seconds, held=held)
        status_colour = (113, 228, 181, 255) if live else (255, 132, 125, 255)
        x = self.w - (535 if not live and self.continuing else 225)*s
        draw.rounded_rectangle((x, 58*s, self.w-59*s, 110*s), radius=13*s, fill=(*status_colour[:3], 30))
        draw.ellipse((x+16*s, 78*s, x+26*s, 88*s), fill=status_colour)
        draw.text((x+39*s, 67*s), status, font=self.font(20 if self.continuing else 23), fill=status_colour)
        if failure_message:
            length = draw.textlength(failure_message, font=self.font(21))
            left = (self.w-length)/2
            draw.rounded_rectangle((left-20*s, 187*s, left+length+20*s, 233*s),
                                   radius=12*s, fill=(74, 26, 31, 224))
            draw.text((left, 191*s), failure_message, font=self.font(21), fill=status_colour)
        drift = float(np.linalg.norm(self.a["root_pos"][t, env] - self.a["ref_root_pos"][t]))
        joint = self.metric("metric_joint_error_rad", t, env)
        torque = self.metric("metric_applied_torque_ratio", t, env)
        speed = self.metric("metric_speed_ratio", t, env)
        blank = "—" if t == 0 else "unavailable"
        values = [f"{drift:.3f} m", f"{np.sqrt(np.mean(joint**2)):.3f} rad" if joint is not None else blank,
                  f"{100*np.max(torque):.1f}%" if torque is not None else blank,
                  f"{np.max(speed):.2f}×" if speed is not None else blank]
        labels = ["PELVIS · 世界位置漂移", "JOINT · 关节 RMSE", "TORQUE · 实际 / 上限", "SPEED · 速度 / 上限"]
        col = (self.w - 116*s) / 4
        for i, (label, value) in enumerate(zip(labels, values)):
            cx = 58*s + i*col
            draw.text((cx, self.h-165*s), label, font=self.font(17), fill=muted)
            colour = status_colour if i == 0 and not live else white
            if i == 3 and speed is not None and np.max(speed) > 1:
                colour = (255, 192, 115, 255)
            draw.text((cx, self.h-137*s), value, font=self.font(32), fill=colour)
            if i > 0:
                draw.line((cx-20*s, self.h-157*s, cx-20*s, self.h-86*s), fill=(120,150,185,55), width=1)
        note = "初始状态：尚无转移测量" if t == 0 else "力矩与速度：刚结束的 20 ms 控制周期内实测峰值"
        if not live and self.continuing:
            note = "真实持续仿真 · 标准评分含首次失败转移，其后仅作诊断（20 ms实测峰值）"
        draw.text((58*s, self.h-79*s), note, font=self.font(15), fill=muted)
        scope = ("完整源请求" if self.m.get("requested_transitions") == self.m.get("expected_transitions")
                 else f"请求前缀 {self.request_seconds:.2f}s")
        duration = f"{scope} · 实录 {self.actual_seconds:.2f}s · 源 {self.source_seconds:.2f}s"
        draw.text((self.w-58*s, self.h-78*s), duration, font=self.font(16), fill=muted, anchor="ra")
        legend_y = self.h - 228*s
        draw.rounded_rectangle((36*s, legend_y-8*s, self.w-36*s, legend_y+30*s), radius=10*s, fill=(12,21,34,160))
        for x, colour in [(58, white), (73, (213, 58, 72, 255)), (310, cyan)]:
            draw.ellipse((x*s, legend_y+6*s, (x+11)*s, legend_y+17*s), fill=colour)
        draw.text((96*s, legend_y-3*s), "白 / 红：实际机器人", font=self.font(17), fill=muted)
        draw.text((335*s, legend_y-3*s), "蓝色透明：UMR 参考", font=self.font(17), fill=muted)
        draw.text((570*s, legend_y-3*s), "世界坐标直接叠加 · 无位置对齐", font=self.font(17), fill=muted)
        draw.text((self.w-58*s, legend_y-3*s), f"{seconds:06.2f}s  /  {self.source_seconds:.2f}s", font=self.font(17), fill=white, anchor="ra")
        bar_y = self.h - 195*s
        draw.rounded_rectangle((36*s, bar_y, self.w-36*s, bar_y+4*s), radius=2*s, fill=(65,84,107,180))
        progress = min(1.0, seconds / self.source_seconds) if self.source_seconds else 0
        if progress > 0:
            draw.rounded_rectangle((36*s, bar_y, 36*s+(self.w-72*s)*progress, bar_y+4*s), radius=2*s, fill=cyan)
        return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render(args):
    rollout = args.rollout.resolve(strict=True)
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if args.checkpoint is not None and str(args.checkpoint.resolve()) != metadata["checkpoint"]:
        raise ValueError("--checkpoint does not match recorded rollout")
    env = args.env_index
    if not 0 <= env < arrays["root_pos"].shape[1]:
        raise ValueError("env-index is outside recorded batch")
    fps = float(arrays["fps"])
    if min(args.width, args.height, args.fps) <= 0 or args.camera_distance <= 0 or args.failure_hold_seconds < 0:
        raise ValueError("resolution, FPS and camera-distance must be positive")
    plan = recording_frame_plan(arrays, metadata, env=env, source_fps=fps, video_fps=args.fps,
        start_seconds=args.start_seconds, end_seconds=args.end_seconds, preview_frame=args.preview_frame,
        max_render_frames=args.max_render_frames, failure_hold_seconds=args.failure_hold_seconds,
        include_post_failure=getattr(args, "include_post_failure", None))
    recorded_end, interval, indices = plan["recorded_end"], plan["interval"], plan["indices"]
    fallback = ROOT / "assets/opensprite/opensprite.xml"
    recorded_model = Path(metadata.get("model_path", ""))
    model_path = Path(args.model or (recorded_model if recorded_model.is_file() else fallback)).resolve(strict=True)
    output = (Path(tempfile.mkdtemp(prefix="render-" + time.strftime("%Y%m%d-%H%M%S") + "-", dir=rollout.parent))
              if args.output is None else args.output.resolve())
    if args.output is not None:
        output.mkdir(parents=True, exist_ok=False)
    model, ghost = build_models(model_path, args.width, args.height)
    data, ghost_data = mujoco.MjData(model), mujoco.MjData(ghost)
    root_adr, joint_adrs = qpos_addresses(model, arrays["joint_names"].astype(str).tolist())
    follow = FollowCamera(args, model, args.fps)
    hud = Hud(args, arrays, metadata, fps, recorded_end)
    opt, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    opt.sitegroup[:] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    video = None if args.preview_frame is not None else output / "comparison.mp4"
    writer_context = (imageio.get_writer(str(video), fps=args.fps, codec="libx264", quality=None,
                      macro_block_size=1, ffmpeg_params=["-crf", "18", "-preset", "medium", "-threads", "4", "-movflags", "+faststart"])
                      if video is not None else nullcontext(None))
    camera_start = camera_end = None
    hold_frames = 0
    with mujoco.Renderer(model, height=args.height, width=args.width, max_geom=max(10000, 3*model.ngeom)) as renderer:
        with writer_context as writer:
            for n, t in enumerate(indices):
                set_pose(model, data, root_adr, joint_adrs, arrays["root_pos"][t, env], arrays["root_quat_wxyz"][t, env], arrays["joint_pos"][t, env])
                set_pose(ghost, ghost_data, root_adr, joint_adrs, arrays["ref_root_pos"][t], arrays["ref_root_quat_wxyz"][t], arrays["ref_joint_pos"][t])
                camera = follow.update(data, ghost_data)
                renderer.update_scene(data, camera=camera, scene_option=opt)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                mujoco.mjv_addGeoms(ghost, ghost_data, opt, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC.value, renderer.scene)
                image = hud.draw(renderer.render(), t, env)
                camera_end = {"lookat": camera.lookat.tolist(), "distance": float(camera.distance),
                              "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)}
                if n == 0:
                    Image.fromarray(image).save(output / "preview.png")
                    camera_start = camera_end
                if writer is not None:
                    writer.append_data(image)
                if (n+1) % 250 == 0 or n+1 == len(indices):
                    print(f"RENDER {n+1}/{len(indices)} state={t}", flush=True)
            if writer is not None and plan["hold_frames"]:
                hold_frames = plan["hold_frames"]
                # Repeat only the last already-recorded pose. Time/metrics stay
                # frozen and the added banner discloses the video-only hold.
                held_image = hud.draw(renderer.render(), indices[-1], env, held=True)
                for _ in range(hold_frames):
                    writer.append_data(held_image)
    info = {"rollout": str(rollout), "checkpoint": metadata["checkpoint"], "iteration": metadata.get("iteration"),
            "clip": metadata.get("clip"), "clip_name": metadata.get("clip_name"), "title": hud.title,
            "model": str(model_path), "env_index": env, "reference_start_frame": metadata.get("ref_start", 0),
            "start_state_index": indices[0], "end_state_index_exclusive": indices[-1]+1,
            "requested_state_interval": list(interval), "fps": args.fps, "source_fps": fps,
            "width": args.width, "height": args.height, "frames": len(indices) + hold_frames,
            "recorded_pose_frames": len(indices), "failure_hold_frames": hold_frames,
            "failure_hold_seconds": hold_frames / args.fps,
            "video": str(video) if video else None, "preview": str(output / "preview.png"),
            "preview_only": args.preview_frame is not None, "failure_state_index": plan["first_failure"],
            "first_failure_seconds": hud.first_failure_seconds,
            "continue_after_failure": plan["continue_after_failure"],
            "post_failure_policy": metadata.get("post_failure_policy"),
            "post_failure_frames_rendered": sum(t > plan["first_failure"] for t in indices) if plan["first_failure"] is not None else 0,
            "full_recorded_span_rendered": plan["full_recorded_span_rendered"],
            "full_source_recorded": plan["full_source_recorded"],
            "source_success_rate": metadata.get("success_rate"),
            "physical_failure_truncation": not plan["continue_after_failure"], "camera_start": camera_start, "camera_end": camera_end,
            "rendering": "offline CPU MuJoCo FK + EGL, no policy or physics stepping",
            "quality": {"msaa_samples": int(model.vis.quality.offsamples), "shadow_size": int(model.vis.quality.shadowsize), "codec": "H264 CRF18"},
            "ghost": "identical official meshes, blue alpha.22, recorded world positions, no realignment",
            "hud_metric_timing": "state t uses completed transition t-1; state0 unavailable; torque/speed are 20ms measured peaks",
            "hud_font": hud.font_path, "source_complete_clip": metadata.get("complete_clip", False),
            "expected_transitions": metadata.get("expected_transitions"), "requested_transitions": metadata.get("requested_transitions")}
    (output / "render.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"SAVED {video or output / 'preview.png'}", flush=True)
    return video or output / "preview.png"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="optional recorded checkpoint consistency check")
    p.add_argument("--output", type=Path, help="new output directory, otherwise unique beside rollout")
    p.add_argument("--model", type=Path, help="override recorded MJCF path if checkout moved")
    p.add_argument("--env-index", type=int, default=0)
    p.add_argument("--start-seconds", type=float, default=0)
    p.add_argument("--end-seconds", type=float)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=50)
    p.add_argument("--camera-distance", type=float, default=2.5)
    p.add_argument("--camera-azimuth", type=float, default=120)
    p.add_argument("--camera-elevation", type=float, default=-20)
    p.add_argument("--title", help="optional HUD title")
    p.add_argument("--preview-frame", type=int, help="render only this absolute recorded state index to preview.png")
    p.add_argument("--max-render-frames", type=int, help="limit frames for visual QA, never implies a full evaluation")
    p.add_argument("--include-post-failure", action=argparse.BooleanOptionalAction, default=None,
                   help="optional consistency assertion; actual continued recordings are detected automatically")
    p.add_argument("--failure-hold-seconds", type=float, default=1.5,
                   help="legacy video-only failure hold; automatically disabled for continued recordings")
    render(p.parse_args())


if __name__ == "__main__":
    main()
