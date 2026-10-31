"""Equal clip mass, with return-adaptive reset bins only WITHIN each clip."""
import torch


class ReturnSampler:
    def __init__(self, motion, gamma=.99, reward_scale=.1):
        self.motion, self.gamma = motion, gamma
        self.max_return = reward_scale / (1-gamma)
        self.returns = torch.zeros(motion.num_bins, device=motion.root_p.device)
        self.num_clips = len(motion.clip_end)
        self.clip_bin_count = torch.bincount(motion.bin_clip, minlength=self.num_clips)
        self.uniform = 1. / (self.num_clips * self.clip_bin_count[motion.bin_clip])
        self.past = self.uniform.clone()

    def _hard_probabilities(self):
        clip = self.motion.bin_clip
        logits = -25*self.returns/self.max_return
        peaks = logits.new_full((self.num_clips,), -torch.inf)
        peaks.scatter_reduce_(0, clip, logits, reduce="amax", include_self=True)
        weights = (logits - peaks[clip]).exp()
        total = weights.new_zeros(self.num_clips).scatter_add_(0, clip, weights)
        return weights / total[clip] / self.num_clips

    def probabilities(self):
        clip = self.motion.bin_clip
        total = self.past.new_zeros(self.num_clips).scatter_add_(0, clip, self.past)
        past = self.past / total[clip].clamp_min(torch.finfo(self.past.dtype).tiny) / self.num_clips
        return .5*self.uniform + .4*self._hard_probabilities() + .1*past

    def __call__(self, count):
        bins = torch.multinomial(self.probabilities(), count, replacement=True)
        start = self.motion.bin_start[bins]
        width = self.motion.bin_end[bins] - start
        frames = start + (torch.rand(count, device=start.device)*width).long()
        return frames, bins

    def for_clips(self, clip_ids):
        """Sample each requested clip using its current conditional bin weights.

        Input/output are matching one-dimensional tensors, in caller order.
        Fixed env_id % num_clips assignments can thus retain each physical
        environment's clip across early failures and asynchronous resets.
        """
        clip_ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.returns.device)
        if clip_ids.ndim != 1:
            raise ValueError("clip_ids must be a one-dimensional vector")
        if ((clip_ids < 0) | (clip_ids >= self.num_clips)).any():
            raise ValueError("requested clip id is out of range")
        probabilities = self.probabilities()
        bins = torch.empty_like(clip_ids)
        for clip in range(self.num_clips):
            positions = (clip_ids == clip).nonzero(as_tuple=False).flatten()
            if not positions.numel():
                continue
            candidates = (self.motion.bin_clip == clip).nonzero(as_tuple=False).flatten()
            selected = torch.multinomial(probabilities[candidates], positions.numel(), replacement=True)
            bins[positions] = candidates[selected]
        start = self.motion.bin_start[bins]
        width = self.motion.bin_end[bins] - start
        frames = start + (torch.rand(len(bins), device=start.device)*width).long()
        return frames, bins

    @torch.no_grad()
    def update(self, out):
        if "ep_ids" not in out:
            return
        ids, bins = out["ep_ids"], out["ep_start_bin"]
        length = out["ep_len"].float()
        G = out["ep_disc_ret"].clone()
        # Only a reset-priority estimate. Complete a truncated return by its
        # mean reward so short clips/end bins are not spuriously called hard.
        # This is NOT a timeout value added to physical rewards or critic data.
        continue_estimate = ~out["term"][ids]
        G += continue_estimate * self.gamma**length * (out["ep_ret"]/length) / (1-self.gamma)
        sums, counts = torch.zeros_like(self.returns), torch.zeros_like(self.returns)
        sums.scatter_add_(0, bins, G)
        counts.scatter_add_(0, bins, torch.ones_like(G))
        used = counts > 0
        self.returns[used] = .8*self.returns[used] + .2*sums[used]/counts[used]
        self.past.lerp_(self._hard_probabilities(), .01)
