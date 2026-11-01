# SPDX-License-Identifier: MIT
# MIT-licensed algorithm utility; see THIRD_PARTY_LICENSE.txt.
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

from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, T, N, device, *, state_dim=75, action_dim=31,
                 context_dim=0, extra_fields=None, storage_device=None,
                 validation_envs=0, failure_replay_fraction=0.0,
                 failure_lookback_steps=100, failure_gap_steps=10):
        if T < 1 or N < 1 or state_dim < 1 or action_dim < 1 or context_dim < 0:
            raise ValueError("invalid replay dimensions")
        if validation_envs < 0 or validation_envs >= N:
            raise ValueError("validation_envs must be nonnegative and leave a training environment")
        if not 0 <= failure_replay_fraction <= 1:
            raise ValueError("failure replay fraction must be in [0,1]")
        if not 0 <= failure_gap_steps <= failure_lookback_steps:
            raise ValueError("require 0 <= failure gap <= lookback")
        self.failure_replay_fraction = float(failure_replay_fraction)
        self.failure_lookback_steps = int(failure_lookback_steps)
        self.failure_gap_steps = int(failure_gap_steps)
        self.total_added = 0
        self._failure_events = deque()
        self._failure_cache = {}
        self.failure_replay_stats = {
            "physical_events": 0, "wm_priority_requested": 0, "wm_priority_selected": 0,
            "actor_priority_requested": 0, "actor_priority_proposals": 0,
            "actor_priority_accepted": 0, "actor_total_accepted": 0,
        }
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
        if self.failure_replay_fraction > 0:
            self.fields["term"] = ((), torch.bool)  # Physical failure, never timeout/done.
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
        if self.failure_replay_fraction > 0:
            # Reuse the row already copied into replay; no extra GPU transfer in
            # the production CPU-backed buffer. Holdout never feeds priority.
            terminal = self.data["term"][self.ptr, :self.N-self.validation_envs]
            ages = self.data["ep_step"][self.ptr]
            for env in terminal.nonzero(as_tuple=False).flatten().cpu().tolist():
                age = int(ages[env])
                if age >= self.failure_gap_steps:
                    self._failure_events.append((self.total_added, env, age))
                    self.failure_replay_stats["physical_events"] += 1
        self.total_added += 1
        oldest = self.total_added - min(self.size + 1, self.T)
        while self._failure_events and self._failure_events[0][0] < oldest:
            self._failure_events.popleft()
        self._failure_cache.clear()
        self.ptr = (self.ptr+1) % self.T
        self.size = min(self.size+1, self.T)

    def _row(self, age):
        return (self.ptr-self.size+age) % self.T

    def priority_candidates(self, k=1, hist=1):
        """Resident (absolute row sequence, env) pairs before real failure.

        Rebuilt at most once per replay generation and (k,hist), then reused
        across the four WM updates. Absolute chronology and episode-age tokens
        prevent stale ring slots or an intervening reset from creating anchors.
        Returned tensor is on replay storage_device; it contains no states.
        """
        key = (int(k), int(hist))
        if min(key) < 1:
            raise ValueError("positive window/history required")
        if key in self._failure_cache:
            return self._failure_cache[key]
        empty = torch.empty((0, 2), dtype=torch.long, device=self.storage_device)
        if not self.failure_replay_fraction or not self._failure_events:
            return empty
        events = torch.tensor(list(self._failure_events), dtype=torch.long, device=self.storage_device)
        offsets = torch.arange(self.failure_gap_steps, self.failure_lookback_steps + 1, device=self.storage_device)
        event = events[:, None].expand(-1, len(offsets), -1).reshape(-1, 3)
        distance = offsets[None].expand(len(events), -1).reshape(-1)
        seq, env, episode_age = event[:, 0] - distance, event[:, 1], event[:, 2] - distance
        oldest = self.total_added - self.size
        valid = ((seq >= oldest) & (seq + k <= self.total_added)
                 & (seq + k - 1 <= event[:, 0]) & (episode_age >= 0)
                 & (seq - torch.minimum(episode_age, torch.full_like(episode_age, hist-1)) >= oldest))
        seq, env, episode_age, event = seq[valid], env[valid], episode_age[valid], event[valid]
        if not len(seq):
            self._failure_cache[key] = empty
            return empty
        endpoint = event[:, 0] % self.T
        valid = (self.data["term"][endpoint, env]
                 & (self.data["ep_step"][endpoint, env].long() == event[:, 2]))
        steps = torch.arange(k, device=self.storage_device)
        rows = (seq[:, None] + steps) % self.T
        actual_age = self.data["ep_step"][rows, env[:, None]].long()
        valid &= (actual_age == episode_age[:, None] + steps).all(-1)
        if k > 1:
            valid &= ~self.data["done"][rows[:, :-1], env[:, None]].any(-1)
        back = torch.minimum(torch.arange(hist-1, -1, -1, device=self.storage_device)[None], episode_age[:, None])
        history_rows = (seq[:, None] - back) % self.T
        valid &= (self.data["ep_step"][history_rows, env[:, None]].long()
                  == episode_age[:, None] - back).all(-1)
        # Endpoint/start age agreement spans the actual episode, not ref phase.
        pairs = torch.stack((seq[valid], env[valid]), -1)
        self._failure_cache[key] = pairs
        return pairs

    def _priority_mix(self, ages, envs, *, kind, k=1, hist=1, generator=None,
                      priority_frame_ends=None, priority_horizon=0):
        """Replace random positions, never a priority prefix in the 2B actor pool."""
        flags = torch.zeros(len(ages), dtype=torch.bool, device=self.storage_device)
        requested = int(len(ages) * self.failure_replay_fraction)
        if not requested:
            return ages, envs, flags
        self.failure_replay_stats[kind + "_priority_requested"] += requested
        candidates = self.priority_candidates(k=k, hist=hist)
        if priority_frame_ends is not None and len(candidates):
            frames = self.data["f"][candidates[:, 0] % self.T, candidates[:, 1]]
            candidates = candidates[frames + priority_horizon <= priority_frame_ends[frames]]
        count = min(requested, len(candidates))
        if count:
            chosen = torch.randperm(len(candidates), device=self.storage_device, generator=generator)[:count]
            positions = torch.randperm(len(ages), device=self.storage_device, generator=generator)[:count]
            ages[positions] = candidates[chosen, 0] - (self.total_added - self.size)
            envs[positions] = candidates[chosen, 1]
            flags[positions] = True
        label = "wm_priority_selected" if kind == "wm" else "actor_priority_proposals"
        self.failure_replay_stats[label] += count
        return ages, envs, flags

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
        if self.failure_replay_fraction > 0 and split == "train":
            age0, env, _ = self._priority_mix(age0, env, kind="wm", k=k, generator=generator)
        rows = self._row(age0[:, None] + torch.arange(k, device=self.storage_device))
        envs = env[:, None].expand_as(rows)
        out = {key: storage[rows, envs].to(self.device) for key, storage in self.data.items()}
        # Include the terminal transition itself, exclude its post-reset successors.
        continuation = 1-out["done"].float()
        out["mask"] = torch.cat((torch.ones_like(continuation[:, :1]), continuation[:, :-1]), -1).cumprod(-1)
        return out

    def sample_starts(self, B, hist, *, priority_frame_ends=None, priority_horizon=0):
        if self.size < 1 or B < 1 or hist < 1:
            raise ValueError("need nonempty replay, positive batch and history")
        age = torch.randint(self.size, (B,), device=self.storage_device)
        env = torch.randint(self.N, (B,), device=self.storage_device)
        flags = None
        if self.failure_replay_fraction > 0:
            age, env, flags = self._priority_mix(age, env, kind="actor", hist=hist,
                priority_frame_ends=priority_frame_ends, priority_horizon=priority_horizon)
        row = self._row(age)
        episode_age = self.data["ep_step"][row, env].long()
        back = torch.arange(hist-1, -1, -1, device=self.storage_device)[None]
        back = torch.minimum(torch.minimum(back, episode_age[:, None]), age[:, None])
        rows = self._row(age[:, None]-back)
        envs = env[:, None].expand_as(rows)
        keys = ("s", "a_prev", "xi", "f", "ep_step")
        start = {key: self.data[key][row, env].to(self.device) for key in keys}
        history = {key: self.data[key][rows, envs].to(self.device) for key in keys}
        if flags is not None:
            start["_failure_priority"] = flags.to(self.device)
        return start, history
