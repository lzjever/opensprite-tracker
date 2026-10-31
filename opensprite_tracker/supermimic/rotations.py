# SPDX-License-Identifier: MIT
# Adapted from supermimic_vibed (SuperMimic independent implementation).
# Copyright (c) 2026 ETH Zurich, Chong Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Batched differentiable rotations: quaternions wxyz, column-vector matrices."""

import torch
import torch.nn.functional as F


def quat_to_mat(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m):
    """Select the best-conditioned branch; finite derivatives at identity/pi.

    The principal quaternion has w >= 0. The principal logarithm necessarily
    has a branch cut at pi; finite local derivatives do not remove that cut.
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Keep every candidate finite, including unselected branches in backward.
    qw = (1 + m00 + m11 + m22).clamp_min(1e-12).sqrt() / 2
    qx = (1 + m00 - m11 - m22).clamp_min(1e-12).sqrt() / 2
    qy = (1 - m00 + m11 - m22).clamp_min(1e-12).sqrt() / 2
    qz = (1 - m00 - m11 + m22).clamp_min(1e-12).sqrt() / 2
    candidates = torch.stack((
        torch.stack((qw, (m21-m12)/(4*qw), (m02-m20)/(4*qw), (m10-m01)/(4*qw)), -1),
        torch.stack(((m21-m12)/(4*qx), qx, (m01+m10)/(4*qx), (m02+m20)/(4*qx)), -1),
        torch.stack(((m02-m20)/(4*qy), (m01+m10)/(4*qy), qy, (m12+m21)/(4*qy)), -1),
        torch.stack(((m10-m01)/(4*qz), (m02+m20)/(4*qz), (m12+m21)/(4*qz), qz), -1),
    ), -2)
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = candidates.gather(-2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v):
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = (theta2 + 1e-12).sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2/6 + theta2.square()/120, theta.sin()/theta)
    b = torch.where(small, 0.5 - theta2/24 + theta2.square()/720, (1-theta.cos())/(theta2+1e-12))
    K = skew(v)
    return torch.eye(3, device=v.device, dtype=v.dtype) + a[..., None]*K + b[..., None]*(K @ K)


def quat_to_rotvec(q):
    q = F.normalize(q, dim=-1)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    s = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    half = torch.atan2(s, q[..., :1])
    scale = torch.where(s < 1e-6, 2 / q[..., :1].clamp_min(1e-6), 2*half/s.clamp_min(1e-12))
    return v * scale


def so3_log(R):
    return quat_to_rotvec(mat_to_quat(R))


def rot_angle_sq(Ra, Rb):
    """Squared geodesic angle without the acos clamp's zero-angle dead zone."""
    return so3_log(Ra.transpose(-1, -2) @ Rb).square().sum(-1)


def yaw_mat(R):
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    # Body X has no horizontal heading when it points vertically. Dividing its
    # tiny projection by sqrt(norm²+eps) used to return a scaled/non-rotation
    # matrix and collapse relative reference geometry. In that chart use body
    # Y (rotated -90 degrees) instead. No globally continuous yaw exists for
    # every 3-D orientation; this explicit chart switch preserves SO(3), not a
    # claim of smooth heading through every inversion. Ordinary headings match
    # the previous convention.
    front_valid = fx.square()+fy.square() >= 1e-8
    fx, fy = (torch.where(front_valid, fx, R[..., 1, 1]),
              torch.where(front_valid, fy, -R[..., 0, 1]))
    norm = (fx.square() + fy.square()).sqrt()
    c, s = fx/norm, fy/norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R):
    """First two columns, interleaved row-major, matching the reference."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def axis_angle_to_mat(axis, angle):
    return so3_exp(axis * angle[..., None])


def quat_slerp(q0, q1, t):
    q0, q1 = F.normalize(q0, dim=-1), F.normalize(q1, dim=-1)
    dot = (q0*q1).sum(-1, keepdim=True)
    q1 = q1 * torch.where(dot < 0, -1.0, 1.0)
    # Avoid evaluating acos at one even on the unselected linear branch.
    eps = torch.finfo(q0.dtype).eps
    theta = dot.abs().clamp(max=1-eps).acos()
    s = theta.sin().clamp_min(eps)
    t = t[..., None]
    linear = dot.abs() > 1-1e-6
    w0 = torch.where(linear, 1-t, ((1-t)*theta).sin()/s)
    w1 = torch.where(linear, t, (t*theta).sin()/s)
    return F.normalize(w0*q0 + w1*q1, dim=-1)
