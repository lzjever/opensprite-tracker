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

"""Robot-independent state layout shared by dynamics and analytic MDP heads.

S = (p, R, v, w, q, qd[, h]); p/v/w are in the world frame, R is the
root-link rotation, and h is normalized actuator/delay memory. The adapter
must use the same root-link origin for position and velocity.

For 31 joints: flat = [p(3), quaternion_wxyz(4), v(3), w(3), q(31),
qd(31), h(M)] has dimension 75+M; tangent deltas have dimension 70+M.
No robot assets or actuator constants are defined here.
"""

import torch

from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_log, yaw_mat

NUM_J = 31
STATE_DIM = BASE_STATE_DIM = 13 + 2 * NUM_J
INV_DIM = BASE_INV_DIM = 13 + 2 * NUM_J
WM_OUT = BASE_OUT_DIM = 12 + 2 * NUM_J


def flatten_state(S):
    if len(S) not in (6, 7):
        raise ValueError("state must have six physical components and optional h")
    p, R, v, w, q, qd = S[:6]
    parts = (p, mat_to_quat(R), v, w, q, qd)
    return torch.cat(parts + tuple(S[6:]), -1)


def unflatten_state(x, num_joints=NUM_J, memory_dim=None):
    base = 13 + 2 * num_joints
    if memory_dim is None:
        memory_dim = x.shape[-1] - base
    if memory_dim < 0 or x.shape[-1] != base + memory_dim:
        raise ValueError("flat state dimension does not match joints/memory")
    S = (
        x[..., :3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13 : 13 + num_joints],
        x[..., 13 + num_joints : base],
    )
    return S + (x[..., base:],) if memory_dim else S


def invariant_features(S):
    """Heading/XY invariant physical features followed by all memory components.

    Only the network input removes heading/XY. Full p,R remain in S for state
    integration, observations, root tracking rewards and termination.
    """
    p, R, v, w, q, qd = S[:6]
    H, Rt = yaw_mat(R), R.transpose(-1, -2)
    parts = (
        p[..., 2:3],
        mat_to_6d(H.transpose(-1, -2) @ R),
        (Rt @ v.unsqueeze(-1)).squeeze(-1),
        (Rt @ w.unsqueeze(-1)).squeeze(-1),
        q,
        qd,
    )
    return torch.cat(parts + tuple(S[6:]), -1)


def state_delta(S, S1, include_memory=True):
    """Local delta from S to S1; also the error from a target to a prediction."""
    p, R, v, w, q, qd = S[:6]
    p1, R1, v1, w1, q1, qd1 = S1[:6]
    Ht, Rt = yaw_mat(R).transpose(-1, -2), R.transpose(-1, -2)
    parts = (
        (Ht @ (p1 - p).unsqueeze(-1)).squeeze(-1),
        so3_log(Rt @ R1),
        (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
        (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
        q1 - q,
        qd1 - qd,
    )
    if include_memory:
        if len(S) != len(S1):
            raise ValueError("both states must have the same memory layout")
        if len(S) == 7:
            parts += (S1[6] - S[6],)
    return torch.cat(parts, -1)
