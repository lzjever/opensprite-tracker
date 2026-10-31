"""Supervised 20 ms transition metrics, NOT differentiable contact physics.

Usage::

    model = TransitionMetricsModel().to(device)
    target = pack_targets(real_transition, xi_raw, limits, bodyweight_n)
    model.update_normalization(S, S1, executed_action, xi_normalized)
    prediction = model(S, S1, executed_action, xi_normalized)
    loss = model.loss(prediction, target, valid_transition_mask)
    metrics = decode_metrics(prediction, S, S1, xi_raw, limits, bodyweight_n)

forward/targets are log1p(normalized positive metrics), NOT unconstrained logits.
Each independent MLP predicts positive residuals above endpoint lower bounds;
applied torque instead uses a sigmoid over the remaining force-limit interval.
Loss bootstraps each head independently; freeze PARAMETERS, not forward/input
gradients, for actor optimization. Normalization updates only on explicit calls
with real transitions. Targets and normalization samples must precede reset;
valid masks include the terminal transition, exclude padding/post-reset mixing.

Order: speed31, requested torque31, applied torque31, mean force2, peak force2.
Speed uses URDF limits, torque uses actual per-env xi force limits, force uses
the SAME bodyweight convention as the physical reward. h[:31] is filtered
effort already divided by actual force limit. If predicted |h| exceeds one,
that invalid WM bound remains visible rather than being clipped away.
The contact peak >= mean(abs(Fz)) constraint assumes the current flat-ground
foot sensor's downward normal force (Fz <= 0); it is NOT a universal law for
full-body contacts or arbitrary contact normals. Labels are never 'repaired'.

The two heads only mitigate optimism: taking the larger cost cannot correct
shared underestimation by both heads. Nonlinear costs of metric predictions
are not expected costs, and joint low error does not certify actor gradients.
Compute each cost separately per head before taking the conservative cost;
never maximize force before evaluating a nonmonotone load-share cost. Planned
support is not observed contact; no contact selector or support model lives here.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .networks import RunningNorm, mlp, run_mlp
from .state import invariant_features, BASE_INV_DIM
from .robot import JOINT_NAMES, CONTEXT_SLICES, CONTEXT_DIM, MEM_DIM

NUM_JOINTS = len(JOINT_NAMES)
METRICS_DIM = 3*NUM_JOINTS+4
METRIC_SLICES = {
    "speed_peak": slice(0, NUM_JOINTS),
    "raw_torque_peak": slice(NUM_JOINTS, 2*NUM_JOINTS),
    "applied_torque_peak": slice(2*NUM_JOINTS, 3*NUM_JOINTS),
    "contact_mean_n": slice(3*NUM_JOINTS, 3*NUM_JOINTS+2),
    "contact_peak_n": slice(3*NUM_JOINTS+2, METRICS_DIM),
}
FORCE_LIMIT_SLICE = CONTEXT_SLICES["force_limit"]


def encode_metrics(normalized_metrics: torch.Tensor) -> torch.Tensor:
    """Encode nonnegative normalized labels without clipping/fabricating data."""
    return torch.log1p(normalized_metrics)


def metric_slices(include_body=False):
    return {**METRIC_SLICES, **({"body_contact_peak_n": slice(METRICS_DIM, METRICS_DIM+1)} if include_body else {})}


def _scales(xi_raw, velocity_limits, bodyweight_n, include_body=False):
    limits = torch.as_tensor(velocity_limits, device=xi_raw.device, dtype=xi_raw.dtype)
    force = xi_raw[..., FORCE_LIMIT_SLICE]
    speed = torch.broadcast_to(limits, force.shape)
    weight = torch.as_tensor(bodyweight_n, device=xi_raw.device, dtype=xi_raw.dtype)
    if weight.ndim == force.ndim - 1:
        weight = weight.unsqueeze(-1)
    weight = torch.broadcast_to(weight, force.shape[:-1] + (2,))
    parts = (speed, force, force, weight, weight)
    return torch.cat(parts+((weight[..., :1],) if include_body else ()), dim=-1)


def pack_targets(out_transition: Mapping, xi_raw, velocity_limits, bodyweight_n):
    """Actual simulator interval statistics -> detached encoded labels (...,97).

    Requires the five keys in METRIC_SLICES; no endpoint force reconstruction,
    no head-generated supervision, no terminal/reset masking hidden here.
    """
    include_body = "body_contact_peak_n" in out_transition
    physical = torch.cat(tuple(out_transition[key] for key in metric_slices(include_body)), dim=-1)
    if physical.shape[-1] != METRICS_DIM+int(include_body):
        raise ValueError("transition metrics must have widths 31,31,31,2,2[,1]")
    return encode_metrics(physical / _scales(xi_raw, velocity_limits, bodyweight_n, include_body)).detach()


def decode_metrics(prediction, pre_state, post_state, xi_raw, velocity_limits, bodyweight_n):
    """Encoded predictions -> five physical-unit tensors (heads,...,features).

    Bounds are already imposed by forward, not re-clipped during decoding.
    States are retained in the public signature for adapter compatibility.
    This also decodes packed targets (without a head dimension) exactly.
    """
    del pre_state, post_state
    include_body = prediction.shape[-1] == METRICS_DIM+1
    physical = torch.expm1(prediction) * _scales(xi_raw, velocity_limits, bodyweight_n, include_body)
    return {key: physical[..., sl] for key, sl in metric_slices(include_body).items()}


class TransitionMetricsModel(nn.Module):
    """Independent small heads sharing ONLY explicit real-input RunningNorm.

    Default input: 2*(invariant75 + h66) + action31 + context221 = 534.
    velocity_limits can be supplied to avoid robot imports or test other limits;
    the 97-output metric layout and h filtered-effort prefix are Sprite contracts.
    """

    def __init__(self, input_dim=None, hidden: Sequence[int] = (256, 256), heads=2,
                 *, velocity_limits=None, bootstrap_probability=0.8, include_body=False):
        super().__init__()
        if input_dim is None:
            input_dim = 2*(BASE_INV_DIM+MEM_DIM)+NUM_JOINTS+CONTEXT_DIM
        if heads < 1 or not 0 < bootstrap_probability <= 1:
            raise ValueError("need positive heads and bootstrap_probability in (0,1]")
        if velocity_limits is None:
            from .robot import velocity_limits as robot_velocity_limits
            velocity_limits = robot_velocity_limits()
        limits = torch.as_tensor(velocity_limits, dtype=torch.float32).clone()
        if limits.shape != (NUM_JOINTS,) or not bool((limits > 0).all()):
            raise ValueError("velocity_limits must contain 31 positive URDF limits")
        self.register_buffer("velocity_limits", limits)
        self.input_dim, self.num_heads = input_dim, heads
        self.include_body = include_body
        self.metric_dim = METRICS_DIM+int(include_body)
        self.metric_slices = metric_slices(include_body)
        self.bootstrap_probability = bootstrap_probability
        self.in_norm = RunningNorm(input_dim)
        self.heads = nn.ModuleList(mlp(input_dim, hidden, self.metric_dim, act=nn.SiLU)
                                   for _ in range(heads))

    def inputs(self, pre_state, post_state, action, xi_normalized):
        x = torch.cat((invariant_features(pre_state), invariant_features(post_state),
                       action, xi_normalized), dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"transition input width {x.shape[-1]} != {self.input_dim}")
        return x

    @torch.no_grad()
    def update_normalization(self, pre_state, post_state, action, xi_normalized, mask=None):
        x = self.inputs(pre_state, post_state, action, xi_normalized)
        if mask is not None:
            x = x[torch.broadcast_to(mask.bool(), x.shape[:-1])]
        self.in_norm.update(x)

    def forward(self, pre_state, post_state, action, xi_normalized):
        x = self.in_norm(self.inputs(pre_state, post_state, action, xi_normalized))
        logits = torch.stack([run_mlp(head, x) for head in self.heads], dim=0)
        residual = F.softplus(logits)
        speed_bound = torch.maximum(pre_state[5].abs(), post_state[5].abs()) / self.velocity_limits
        if len(post_state) != 7 or post_state[6].shape[-1] < NUM_JOINTS:
            raise ValueError("post_state needs h[:31] = filtered_effort / actual force_limit")
        applied_bound = post_state[6][..., :NUM_JOINTS].abs()
        remaining = (1. - applied_bound).clamp_min(0.)
        applied = applied_bound + remaining * torch.sigmoid(logits[..., METRIC_SLICES["applied_torque_peak"]])
        mean = residual[..., METRIC_SLICES["contact_mean_n"]]
        normalized = torch.cat((speed_bound + residual[..., :NUM_JOINTS],
                                residual[..., METRIC_SLICES["raw_torque_peak"]],
                                applied,
                                mean, mean + residual[..., METRIC_SLICES["contact_peak_n"]]), dim=-1)
        if self.include_body:
            normalized = torch.cat((normalized, residual[..., METRICS_DIM:METRICS_DIM+1]), -1)
        return encode_metrics(normalized)

    def loss(self, prediction, target, mask=None, *, bootstrap_mask=None):
        """Scalar group-balanced masked MSE in log1p metric space.

        Average features within each of the five physical groups, then average
        groups equally, then valid samples per head and finally heads. This
        removes dimensional imbalance, NOT reward-cost bias or shared optimism.

        mask: (...) valid TRANSITIONS, not ~done (terminal samples are valid).
        bootstrap_mask: optional (heads,...) boolean mask for reproducible or
        externally sampled membership; otherwise independent Bernoulli draws.
        Targets are detached. Entirely empty masks yield differentiable zero.
        """
        if prediction.shape[0] != self.num_heads or prediction.shape[-1] != self.metric_dim:
            raise ValueError(f"prediction must have shape (heads,...,{self.metric_dim})")
        if target.shape != prediction.shape[1:]:
            raise ValueError("target must have shape (...,97), with no head axis")
        shape = prediction.shape[:-1]
        valid = torch.ones(shape, dtype=torch.bool, device=prediction.device)
        if mask is not None:
            valid = valid & torch.broadcast_to(mask.bool(), shape[1:]).unsqueeze(0)
        if bootstrap_mask is None:
            bootstrap_mask = torch.rand(shape, device=prediction.device) < self.bootstrap_probability
        valid = valid & torch.broadcast_to(bootstrap_mask.bool(), shape)
        # Select before subtraction: invalid padding may contain NaNs.
        pred = torch.where(valid.unsqueeze(-1), prediction, 0.)
        labels = torch.where(valid.unsqueeze(-1), target.detach().unsqueeze(0), 0.)
        error = (pred - labels).square()
        grouped = torch.stack([error[..., sl].mean(-1) for sl in self.metric_slices.values()], dim=-1)
        squared = grouped.mean(-1).reshape(self.num_heads, -1).sum(-1)
        count = valid.reshape(self.num_heads, -1).sum(-1)
        return (squared / count.clamp_min(1)).mean()
