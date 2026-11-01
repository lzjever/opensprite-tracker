#!/usr/bin/env python3
"""Zero-phase anti-aliasing followed by resampling on a seconds-based grid.

qpos uses MuJoCo order: xyz, root quaternion wxyz, finite-range hinge angles.
Native solver trajectories remain the caller's responsibility to preserve.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve, firwin, kaiserord
from scipy.spatial.transform import Rotation, Slerp


def validate_motion(qpos, fps):
    """Reject invalid numerical trajectories; never repair NaN/Inf by clipping."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or not len(qpos) or qpos.shape[1] < 7:
        raise ValueError('Motion must have nonempty [frames, qpos] shape')
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Motion values and positive fps must be finite')
    quaternion_norm=np.linalg.norm(qpos[:,3:7],axis=1)
    if np.any(quaternion_norm==0) or not np.allclose(quaternion_norm,1.,rtol=0.,atol=1e-6):
        raise ValueError('Motion root quaternions must be normalized and nonzero')


def output_times(frame_count, source_fps, target_fps):
    duration = (frame_count - 1) / float(source_fps)
    # nextafter removes a one-ULP representation error at an integer boundary.
    count = int(np.floor(np.nextafter(duration * target_fps, np.inf))) + 1
    return np.arange(count, dtype=float) / target_fps


def _filter(source_fps, target_fps, antialias=True):
    if not (np.isfinite(source_fps) and np.isfinite(target_fps) and source_fps > 0 and target_fps > 0):
        raise ValueError("Both frame rates must be finite and positive")
    if not antialias:
        return np.ones(1), {
            "enabled": False, "applied": False, "reason": "Disabled by user",
            "taps": 1, "group_delay_seconds": 0.0,
        }
    if target_fps >= source_fps:
        return np.ones(1), {
            "enabled": True, "applied": False, "reason": "No downsampling", "taps": 1,
            "group_delay_seconds": 0.0,
        }
    nyquist = target_fps / 2
    passband = 0.8 * nyquist
    stopband = nyquist
    cutoff = (passband + stopband) / 2
    taps, beta = kaiserord(60, (stopband - passband) / (source_fps / 2))
    taps += 1 - taps % 2
    kernel = firwin(taps, cutoff, window=("kaiser", beta), fs=source_fps)
    return kernel, {
        "enabled": True, "applied": True, "kind": "symmetric Kaiser FIR",
        "passband_edge_hz": passband, "cutoff_hz": cutoff,
        "stopband_edge_hz": stopband, "design_attenuation_db": 60,
        "taps": taps, "kaiser_beta": float(beta),
        "boundary": "reflect without repeating endpoints",
        "boundary_affected_seconds_each_end": (taps // 2) / source_fps,
        "group_delay_seconds": 0.0,
        "delay_handling": "centered offline convolution removes the FIR group delay",
    }


def _linear_filter(values, kernel):
    values = np.asarray(values, dtype=float)
    if len(kernel) == 1 or len(values) == 1:
        return values.copy()
    half = len(kernel) // 2
    pads = [(half, half)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pads, mode="reflect")
    shape = (len(kernel),) + (1,) * (values.ndim - 1)
    return convolve(padded, kernel.reshape(shape), mode="valid", method="auto")


def _rotation_filter(quaternions_wxyz, kernel):
    rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
    if len(kernel) == 1 or len(rotations) == 1:
        return rotations, 0.0
    half = len(kernel) // 2
    indices = np.pad(np.arange(len(rotations)), (half, half), mode="reflect")
    inverse = rotations.inv()
    tangent = np.zeros((len(rotations), 3))
    max_relative_angle = 0.0
    for offset, weight in enumerate(kernel):
        relative = inverse * rotations[indices[offset:offset + len(rotations)]]
        local_log = relative.as_rotvec()
        max_relative_angle = max(max_relative_angle,
                                 float(np.linalg.norm(local_log, axis=1).max()))
        tangent += weight * local_log
    return rotations * Rotation.from_rotvec(tangent), max_relative_angle


def _interpolate(values, times, source_fps):
    if len(values) == 1:
        return np.repeat(values, len(times), axis=0)
    return interp1d(np.arange(len(values)) / source_fps, values,
                    axis=0, kind="linear")(times)


def _joint_limits(qpos, model):
    if model is None:
        return None
    # MuJoCo hinge type is 3; the root free joint is intentionally excluded.
    joints = np.flatnonzero((np.asarray(model.jnt_type) == 3) &
                            np.asarray(model.jnt_limited, dtype=bool))
    if len(joints) == 0:
        return {"outside_frame_joint_count": 0, "max_excess_radians": 0.0}
    angles = qpos[:, np.asarray(model.jnt_qposadr)[joints]]
    limits = np.asarray(model.jnt_range)[joints]
    excess = np.maximum(limits[:, 0] - angles, angles - limits[:, 1])
    return {"outside_frame_joint_count": int(np.count_nonzero(excess > 1e-8)),
            "max_excess_radians": float(max(0.0, excess.max()))}


def process_motion(qpos_mujoco, source_fps, target_fps=50.0, model=None, antialias: bool=True):
    """Return (resampled qpos, exact output times in seconds, metadata).

    Hinge angles are neither unwrapped nor clipped. SO(3) logs use a separate
    center-frame anchor for each FIR window, never a whole-clip global anchor.
    antialias=False skips smoothing but retains the target sampling time grid.
    """
    qpos = np.asarray(qpos_mujoco, dtype=float)
    validate_motion(qpos, source_fps)
    kernel, filter_info = _filter(source_fps, target_fps, antialias)
    times = output_times(len(qpos), source_fps, target_fps)
    linear_columns = np.r_[0:3, 7:qpos.shape[1]]
    smooth = _linear_filter(qpos[:, linear_columns], kernel)
    rotations, max_relative_angle = _rotation_filter(qpos[:, 3:7], kernel)
    result = np.empty((len(times), qpos.shape[1]))
    result[:, linear_columns] = _interpolate(smooth, times, source_fps)
    if len(qpos) == 1:
        interpolated = rotations
    else:
        interpolated = Slerp(np.arange(len(qpos)) / source_fps, rotations)(times)
    quat = interpolated.as_quat(canonical=False)
    # q and -q denote one rotation; keep the stored representation continuous.
    signs = np.r_[1.0, np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], axis=1)
                                         < 0, -1.0, 1.0))]
    result[:, 3:7] = (quat * signs[:, None])[:, [3, 0, 1, 2]]
    validate_motion(result, target_fps)
    metadata = {
        "source_fps": float(source_fps), "target_fps": float(target_fps),
        "source_frames": len(qpos), "output_frames": len(result),
        "source_last_sample_seconds": (len(qpos) - 1) / source_fps,
        "output_last_sample_seconds": float(times[-1]),
        "anti_alias_filter": filter_info,
        "rotation_filter": ("center-frame local SO(3) logarithm FIR, then exponential"
                            if filter_info["applied"] else "none; SLERP only"),
        "maximum_window_relative_rotation_degrees": float(np.rad2deg(max_relative_angle)),
        "rotation_log_near_branch_cut": bool(max_relative_angle > np.deg2rad(170)),
        "interpolation": "linear xyz/finite hinge angles; root quaternion SLERP",
        "hinge_unwrap": False, "joint_limit_clipping": False,
        "native_joint_limits": _joint_limits(qpos, model),
        "resampled_joint_limits": _joint_limits(result, model),
    }
    return result, times, metadata


def resample_positions(positions, source_fps, target_fps=50.0, antialias: bool=True):
    """Filter source FK marker positions and sample on the same output grid."""
    kernel, _ = _filter(source_fps, target_fps, antialias)
    positions = np.asarray(positions, dtype=float)
    times = output_times(len(positions), source_fps, target_fps)
    return _interpolate(_linear_filter(positions, kernel), times, source_fps)
