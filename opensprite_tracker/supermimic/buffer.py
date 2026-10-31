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

"""Dynamic-shape (time, env) replay for real, pre-reset transitions.

done denotes any reset boundary (termination OR timeout), used only to prevent
windows/history crossing resets. It is NOT a critic death label. Optional
extra_fields can retain separate term/timeout labels without conflating them.
s1 must be the state BEFORE autoreset, including that transition's memory.

``device`` is the learner/sample destination. Optional ``storage_device``
places the replay and its sampling indices elsewhere, e.g. pageable CPU RAM.
Only gathered samples move to the learner; the complete replay never does.
Optional trailing validation environments are excluded from supervised WM
training windows. Actor start/history sampling still uses every environment;
real-data RunningNorm updates are managed by the collector and remain global.
This is a WM optimization holdout, not fully isolated normalization/data.
"""

import torch


class ReplayBuffer:
    def __init__(self, T, N, device, *, state_dim=75, action_dim=31,
                 context_dim=0, extra_fields=None, storage_device=None,
                 validation_envs=0):
        if T < 1 or N < 1 or state_dim < 1 or action_dim < 1 or context_dim < 0:
            raise ValueError("invalid replay dimensions")
        if validation_envs < 0 or validation_envs >= N:
            raise ValueError("validation_envs must be nonnegative and leave a training environment")
        self.T, self.N, self.device = T, N, device
        self.validation_envs = validation_envs
        self.storage_device = torch.device(device if storage_device is None else storage_device)
        self.state_dim, self.action_dim, self.context_dim = state_dim, action_dim, context_dim
        self.fields = {
            "s": (state_dim, torch.float32),
            "s1": (state_dim, torch.float32),
            "a": (action_dim, torch.float32),
            "a_prev": (action_dim, torch.float32),
            "xi": (context_dim, torch.float32),
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        if extra_fields:
            if self.fields.keys() & extra_fields.keys():
                raise ValueError("extra_fields must not override required replay fields")
            self.fields.update(extra_fields)
        self.data = {}
        for key, (shape, dtype) in self.fields.items():
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.data[key] = torch.zeros((T, N) + shape, device=self.storage_device, dtype=dtype)
        self.ptr = self.size = 0

    @torch.no_grad()
    def add(self, transition):
        # Validate all fields before changing a row; never store an autograd graph.
        for key, storage in self.data.items():
            if transition[key].shape != storage.shape[1:]:
                raise ValueError(f"{key}: expected {storage.shape[1:]}, got {transition[key].shape}")
        for key, storage in self.data.items():
            # Blocking cross-device copies make the complete row visible before
            # ptr/size advance. Do not pin the entire multi-GiB CPU replay.
            storage[self.ptr].copy_(transition[key], non_blocking=False)
        self.ptr = (self.ptr+1) % self.T
        self.size = min(self.size+1, self.T)

    def _row(self, age):
        return (self.ptr-self.size+age) % self.T

    def sample_windows(self, B, k, split="train", generator=None):
        if B < 1 or k < 1 or self.size < k:
            raise ValueError(f"need a positive batch and at least k={k} replay rows, have {self.size}")
        if split == "train":
            env_low, env_high = 0, self.N-self.validation_envs
        elif split == "validation":
            if self.validation_envs == 0:
                raise ValueError("no validation environments configured")
            env_low, env_high = self.N-self.validation_envs, self.N
        else:
            raise ValueError("replay split must be 'train' or 'validation'")
        age0 = torch.randint(self.size-k+1, (B,), device=self.storage_device,
                             generator=generator)
        env = torch.randint(env_low, env_high, (B,), device=self.storage_device,
                            generator=generator)
        rows = self._row(age0[:, None] + torch.arange(k, device=self.storage_device))
        envs = env[:, None].expand_as(rows)
        out = {key: storage[rows, envs].to(self.device) for key, storage in self.data.items()}
        # Include the terminal transition itself, exclude its post-reset successors.
        continuation = 1-out["done"].float()
        out["mask"] = torch.cat((torch.ones_like(continuation[:, :1]), continuation[:, :-1]), -1).cumprod(-1)
        return out

    def sample_starts(self, B, hist):
        if self.size < 1 or B < 1 or hist < 1:
            raise ValueError("need nonempty replay, positive batch and history")
        age = torch.randint(self.size, (B,), device=self.storage_device)
        env = torch.randint(self.N, (B,), device=self.storage_device)
        row = self._row(age)
        episode_age = self.data["ep_step"][row, env].long()
        back = torch.arange(hist-1, -1, -1, device=self.storage_device)[None]
        back = torch.minimum(torch.minimum(back, episode_age[:, None]), age[:, None])
        rows = self._row(age[:, None]-back)
        envs = env[:, None].expand_as(rows)
        keys = ("s", "a_prev", "xi", "f", "ep_step")
        start = {key: self.data[key][row, env].to(self.device) for key in keys}
        history = {key: self.data[key][rows, envs].to(self.device) for key in keys}
        return start, history
