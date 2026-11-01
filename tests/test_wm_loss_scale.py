"""CPU gradient checks: preserve WM time-sum scale without scaling cost heads."""
from types import SimpleNamespace
import torch
from torch import nn
from opensprite_tracker.supermimic.engine import Learner


class CountSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=.1)
        self.calls = 0
        self.saved_grad = None

    def step(self, *args, **kwargs):
        self.calls += 1
        self.saved_grad = self.param_groups[0]['params'][0].grad.detach().clone()
        return super().step(*args, **kwargs)


class ConstantWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(.01))
        self.out_norm = SimpleNamespace(std=lambda: torch.ones(1))

    def state_error(self, prediction, truth):
        return prediction


class ConstantHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(.02))
        self.register_buffer('normalization_sentinel', torch.tensor(123.))
        self.loss_calls = 0

    def forward(self, before, after, action, xi):
        return self.bias.expand(2, *action.shape[:-1], 1)

    def loss(self, prediction, target, mask):
        self.loss_calls += 1
        return ((prediction-target).square().squeeze(-1)*mask).sum()/(2*mask.sum())


def learner(batch, horizon, *, scale=None, micro=0, masked=False):
    obj = Learner.__new__(Learner)
    obj.args = SimpleNamespace(wm_batch=batch, wm_window=horizon, wm_microbatch=micro)
    if scale is not None:
        obj.args.wm_loss_scale = scale
    obj.wm, obj.transition_model = ConstantWM(), ConstantHead()
    obj.opt_wm, obj.opt_transition = CountSGD(obj.wm.parameters()), CountSGD(obj.transition_model.parameters())
    obj.error_floor = torch.tensor([.001])
    obj.normalize_xi = lambda x: x
    state = torch.zeros(batch, horizon, 75)
    state[...,3] = 1.
    mask = torch.ones(batch,horizon)
    if masked:
        for b in range(batch):
            mask[b,1+(b % horizon):] = 0
    window = {'s':state.clone(), 's1':state.clone(), 'a':torch.zeros(batch,horizon,31),
              'xi':torch.zeros(batch,horizon,2), 'mask':mask,
              'transition_metrics':torch.full((batch,horizon,1),.05)}
    obj.sample_calls = []
    def sample(B,K):
        obj.sample_calls.append((B,K))
        return window
    obj.buffer = SimpleNamespace(sample_windows=sample)
    obj.wm_rollout = lambda flat, action, xi: obj.wm.theta.expand(len(flat),horizon,1)
    return obj


def test_wm_scale_quarters_gradient_but_preserves_reported_error_and_head_update():
    default, explicit, scaled = learner(7,8,masked=True), learner(7,8,scale=1.,masked=True), learner(7,8,scale=.25,masked=True)
    baseline = default.wm_update()
    unscaled = explicit.wm_update()
    quarter = scaled.wm_update()
    for obj in (default, explicit, scaled):
        assert obj.sample_calls == [(7,8)]
        assert obj.opt_wm.calls == obj.opt_transition.calls == obj.transition_model.loss_calls == 1
        assert obj.transition_model.normalization_sentinel.item() == 123.
        assert abs(obj.opt_wm.saved_grad) < 1  # No clipping obscures the scale check.
    torch.testing.assert_close(default.opt_wm.saved_grad, explicit.opt_wm.saved_grad)
    torch.testing.assert_close(scaled.opt_wm.saved_grad, explicit.opt_wm.saved_grad*.25)
    for left,right in [(baseline,unscaled),(unscaled,quarter)]:
        for x,y in zip(left,right):torch.testing.assert_close(x,y)
    for key in explicit.last_model_metrics:
        torch.testing.assert_close(explicit.last_model_metrics[key],scaled.last_model_metrics[key])
    torch.testing.assert_close(explicit.opt_transition.saved_grad,scaled.opt_transition.saved_grad)
    torch.testing.assert_close(explicit.transition_model.bias,scaled.transition_model.bias)


def test_k8_b1024_and_k32_b256_quarter_have_same_equal_per_step_wm_gradient():
    short,long = learner(1024,8,scale=1.), learner(256,32,scale=.25)
    short_error,long_error = short.wm_update(),long.wm_update()
    # Equal per-transition labels: both optimize 8 * mean error, not 32 * mean.
    torch.testing.assert_close(short.opt_wm.saved_grad,torch.tensor(.16))
    torch.testing.assert_close(long.opt_wm.saved_grad,short.opt_wm.saved_grad)
    torch.testing.assert_close(long.wm.theta,short.wm.theta)
    for x,y in zip(short_error,long_error):torch.testing.assert_close(x,y)
    torch.testing.assert_close(short.opt_transition.saved_grad,long.opt_transition.saved_grad)
    assert short.sample_calls == [(1024,8)] and long.sample_calls == [(256,32)]
    assert short.opt_wm.calls == long.opt_wm.calls == short.opt_transition.calls == long.opt_transition.calls == 1


def test_scaled_wm_unequal_microbatches_keep_logical_denominator_and_one_head_step():
    whole,partitioned = learner(7,8,scale=.25,masked=True),learner(7,8,scale=.25,micro=3,masked=True)
    left,right = whole.wm_update(),partitioned.wm_update()
    torch.testing.assert_close(whole.opt_wm.saved_grad,partitioned.opt_wm.saved_grad)
    torch.testing.assert_close(whole.wm.theta,partitioned.wm.theta)
    torch.testing.assert_close(whole.opt_transition.saved_grad,partitioned.opt_transition.saved_grad)
    for x,y in zip(left,right):torch.testing.assert_close(x,y)
    assert whole.sample_calls == partitioned.sample_calls == [(7,8)]
    assert whole.opt_wm.calls == partitioned.opt_wm.calls == whole.opt_transition.calls == partitioned.opt_transition.calls == 1
