"""Packed 50 Hz NPZ references with explicit, non-crossing clip boundaries.

files accepts a path or a sequence of paths. Each NPZ may additionally declare
clip_starts/clip_lengths (or LaFAN's motion_starts/motion_lengths). Declared
ranges are packed in order; unused frames are excluded, never interpolated.
clip_end is INCLUSIVE; bin_end is EXCLUSIVE and excludes terminal-only resets.
"""

from pathlib import Path

import numpy as np
import torch

from .robot import MOTION_PATH, JOINT_NAMES, CONTROL_DT
from .rotations import quat_to_mat, yaw_mat, mat_to_6d

BIN_FRAMES = 50


def _clip_ranges(data, frame_count):
    for prefix in ("clip", "motion"):
        start_key, length_key = f"{prefix}_starts", f"{prefix}_lengths"
        if start_key in data or length_key in data:
            if start_key not in data or length_key not in data:
                raise ValueError(
                    f"{start_key} and {length_key} must be provided together"
                )
            starts, lengths = np.asarray(data[start_key]), np.asarray(data[length_key])
            if (
                starts.ndim != 1
                or lengths.shape != starts.shape
                or not starts.size
                or not np.issubdtype(starts.dtype, np.integer)
                or not np.issubdtype(lengths.dtype, np.integer)
            ):
                raise ValueError(
                    "clip boundaries must be nonempty equal-length integer vectors"
                )
            ends = starts + lengths
            if (
                np.any(starts < 0)
                or np.any(lengths < 2)
                or np.any(ends > frame_count)
                or np.any(starts[1:] < ends[:-1])
            ):
                raise ValueError(
                    "clips need >=2 frames and ordered, disjoint, in-range boundaries"
                )
            return list(zip(starts.tolist(), lengths.tolist()))
    if frame_count < 2:
        raise ValueError("a clip needs at least two frames")
    return [(0, frame_count)]


class MotionLib:
    def __init__(self, files=None, kin=None, device="cpu", **kwargs):
        del kwargs
        paths = (
            [MOTION_PATH]
            if files is None
            else ([files] if isinstance(files, (str, Path)) else list(files))
        )
        if not paths:
            raise ValueError("at least one motion file is required")
        if kin is None:
            from .kinematics import OpenSpriteKinematics

            kin = OpenSpriteKinematics(device)
        self.device, self.names = str(device), [str(path) for path in paths]
        self.source_files = self.names.copy()
        fields = (
            "q",
            "qd",
            "root_p",
            "root_R",
            "root_v",
            "root_w",
            "body_p",
            "body_R",
            "body_v",
            "body_w",
        )
        chunks = {name: [] for name in fields}
        lengths, self.clip_names = [], []
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if not np.isclose(float(data["fps"].reshape(-1)[0]), 1 / CONTROL_DT):
                    raise ValueError("references must already be 50 Hz")
                joint_names = data["joint_names"].astype(str).tolist()
                body_names = data["body_names"].astype(str).tolist()
                jids = [joint_names.index(n) for n in JOINT_NAMES]
                bids = [body_names.index(n) for n in kin.tracked_names]
                root = body_names.index("pelvis_link")
                ranges = _clip_ranges(data, len(data["joint_pos"]))
                prefix = "clip" if "clip_starts" in data else "motion"
                names_key = f"{prefix}_names"
                clip_names = (
                    data[names_key].astype(str).tolist() if names_key in data else None
                )
                if clip_names is not None and len(clip_names) != len(ranges):
                    raise ValueError(f"{names_key} must contain one name per clip")
                for index, (start, length) in enumerate(ranges):
                    sl = slice(start, start + length)
                    values = (
                        tensor(data["joint_pos"][sl][:, jids]),
                        tensor(data["joint_vel"][sl][:, jids]),
                        tensor(data["body_pos_w"][sl, root]),
                        quat_to_mat(tensor(data["body_quat_w"][sl, root])),
                        tensor(data["body_lin_vel_w"][sl, root]),
                        tensor(data["body_ang_vel_w"][sl, root]),
                        tensor(data["body_pos_w"][sl][:, bids]),
                        quat_to_mat(tensor(data["body_quat_w"][sl][:, bids])),
                        tensor(data["body_lin_vel_w"][sl][:, bids]),
                        tensor(data["body_ang_vel_w"][sl][:, bids]),
                    )
                    for name, value in zip(fields, values):
                        chunks[name].append(value)
                    lengths.append(length)
                    self.clip_names.append(
                        clip_names[index]
                        if clip_names is not None
                        else str(path)
                        if len(ranges) == 1
                        else f"{path}#{index}"
                    )
        for name in fields:
            setattr(self, name, torch.cat(chunks[name], dim=0))
        self.num_frames = len(self.q)
        if (
            not torch.isfinite(self.q).all()
            or (self.q < kin.q_lo).any()
            or (self.q > kin.q_hi).any()
        ):
            raise ValueError(
                "OpenSprite reference violates an official joint hard limit"
            )
        with torch.no_grad():
            fk = kin.forward(
                self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd
            )
            self.source_fk_error = {
                name: {
                    "max_abs": float((actual - original).abs().max()),
                    "rms": float((actual - original).square().mean().sqrt()),
                }
                for name, actual, original in zip(
                    ("body_p", "body_R", "body_v", "body_w"),
                    fk,
                    (self.body_p, self.body_R, self.body_v, self.body_w),
                )
            }
            self.body_p, self.body_R, self.body_v, self.body_w = fk
        self.num_clips = len(lengths)
        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(
            torch.arange(self.num_clips, device=device), self.clip_len
        )
        starts, ends, clips, frame_bins, bin_offset, frame_offset = [], [], [], [], 0, 0
        for clip, length in enumerate(lengths):
            # Only frames with an outgoing transition are reset candidates.
            local = torch.arange(
                0, length - 1, BIN_FRAMES, device=device, dtype=torch.long
            )
            starts.append(local + frame_offset)
            ends.append(
                torch.minimum(local + BIN_FRAMES, torch.full_like(local, length - 1))
                + frame_offset
            )
            clips.append(torch.full_like(local, clip))
            frame_bins.append(
                (torch.arange(length, device=device) // BIN_FRAMES).clamp_max(
                    len(local) - 1
                )
                + bin_offset
            )
            bin_offset += len(local)
            frame_offset += length
        self.bin_start, self.bin_end = torch.cat(starts), torch.cat(ends)
        self.num_bins = len(self.bin_start)
        self.bin_clip = torch.cat(clips)
        self.frame_bin = torch.cat(frame_bins)
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v[..., None]).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w[..., None]).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]

    def ref_state(self, f):
        return tuple(
            x[f]
            for x in (
                self.root_p,
                self.root_R,
                self.root_v,
                self.root_w,
                self.q,
                self.qd,
            )
        )

    def future_frames(self, f, K):
        return torch.minimum(
            f[..., None] + torch.arange(1, K + 1, device=f.device),
            self.clip_end[self.clip_id[f]][..., None],
        )

    def goal(self, f, K):
        fut = self.future_frames(f, K)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        dp = (
            Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(-2))[..., None]
        ).squeeze(-1)
        feats = torch.cat(
            (
                dp,
                mat_to_6d(Ht @ self.root_R[fut]),
                self.root_v_local[fut],
                self.root_w_local[fut],
                self.q[fut],
                0.1 * self.qd[fut],
            ),
            -1,
        )
        return feats.flatten(-2)
