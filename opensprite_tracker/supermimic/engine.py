"""SuperMimic equations 1--3, with no PPO or simulator differentiation.

Robot-independent loop adapted from supermimic_vibed; see THIRD_PARTY_LICENSE.txt.
"""

from __future__ import annotations

import torch
from torch import nn

from .state import unflatten_state
from .mdp import CONTROL_DT


def actor_return(rewards, deaths, tail_value, gamma):
    """The terminal transition earns r, then survival gates the next transition."""
    alive_next = torch.cumprod(1 - deaths, dim=0)
    alive_now = torch.cat((torch.ones_like(alive_next[:1]), alive_next[:-1]), 0)
    discount = gamma ** torch.arange(len(rewards), device=rewards.device)
    objective = (discount[:, None] * alive_now * rewards).sum(0)
    objective = objective + gamma ** len(rewards) * alive_next[-1] * tail_value
    return objective, alive_now, alive_next


@torch.no_grad()
def lambda_returns(rewards, deaths, values, gamma, lam):
    """Compute once before critic epochs; true death removes all bootstrapping."""
    target, targets = values[-1], []
    for t in range(len(rewards) - 1, -1, -1):
        target = rewards[t] + gamma * (1 - deaths[t]) * (
            (1 - lam) * values[t + 1] + lam * target
        )
        targets.append(target)
    return torch.stack(targets[::-1])


def set_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class Learner:
    def __init__(
        self,
        wm,
        policy,
        critic,
        mdp,
        motion,
        buffer,
        normalize_xi,
        args,
        transition_model=None,
    ):
        self.wm, self.pi, self.critic = wm, policy, critic
        self.mdp, self.motion, self.buffer = mdp, motion, buffer
        self.normalize_xi, self.args = normalize_xi, args
        self.transition_model = transition_model
        self.impact_ramp = torch.ones((), device=next(wm.parameters()).device)
        self.opt_transition = (
            torch.optim.Adam(transition_model.parameters(), lr=args.lr)
            if transition_model is not None
            else None
        )
        if getattr(mdp, "uses_transition_metrics", False) and transition_model is None:
            raise ValueError(
                "Full legacy rewards require supervised transition metrics, not fake endpoint peaks"
            )
        self.opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
        self.opt_pi = torch.optim.Adam(
            policy.parameters(), lr=args.lr, betas=(0.9, 0.95)
        )
        self.opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
        self.error_floor = torch.tensor(
            [0.003] * 3
            + [0.005] * 3
            + [0.02] * 3
            + [0.05] * 3
            + [0.003] * 31
            + [0.05] * 31
            + [0.02] * args.memory_dim,
            device=next(wm.parameters()).device,
        )
        self.wm_rollout = self._wm_rollout
        self.imag_step = self._imag_step
        self.reward_term = (
            self._legacy_reward_term
            if transition_model is not None
            else self._reward_term
        )
        self.critic_input = self._critic_input
        if args.compile:
            self.wm_rollout = torch.compile(self._wm_rollout)
            self.imag_step = torch.compile(self._imag_step)
            self.reward_term = torch.compile(self.reward_term)
            self.critic_input = torch.compile(self._critic_input)

    def _wm_rollout(self, flat, actions, xi):
        state = unflatten_state(flat[:, 0])
        predictions = []
        for j in range(self.args.wm_window):
            state = self.wm.step(state, actions[:, j], self.normalize_xi(xi[:, j]))
            predictions.append(state)
        return tuple(
            torch.stack([s[i] for s in predictions], 1) for i in range(len(state))
        )

    @staticmethod
    def _chunks(total, requested):
        size = requested if requested and requested > 0 else total
        return (slice(start, min(start + size, total)) for start in range(0, total, size))

    def wm_update(self):
        # One sampled logical batch and one Adam step, regardless of partition.
        batch = self.args.wm_batch
        window = self.buffer.sample_windows(batch, self.args.wm_window)
        scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        self.opt_wm.zero_grad(set_to_none=True)
        loss_sum = scale.new_zeros(())
        valid_error, valid_count, last_error, last_count = [scale.new_zeros(()) for _ in range(4)]
        for chunk in self._chunks(batch, getattr(self.args, "wm_microbatch", 0)):
            part = {key: value[chunk] for key, value in window.items()}
            prediction = self.wm_rollout(part["s"], part["a"], part["xi"])
            error = (
                (self.wm.state_error(prediction, unflatten_state(part["s1"])) / scale)
                .square().mean(-1)
            )
            mask = part["mask"]
            loss = (error * mask).sum() / batch
            loss.backward()
            loss_sum += loss.detach()
            valid_error += (error.detach() * mask).sum()
            valid_count += mask.sum()
            last_error += (error[:, -1].detach() * mask[:, -1]).sum()
            last_count += mask[:, -1].sum()
        norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite world-model gradient; no update applied")
        self.opt_wm.step()
        self.last_model_metrics = {
            "last_step_valid_fraction": last_count / batch,
            "error_per_valid_transition": valid_error / valid_count.clamp_min(1),
        }
        if self.transition_model is not None:
            # Small independent cost heads still see the complete logical batch;
            # their bootstrap membership and per-head valid-count denominator are
            # therefore drawn/computed once, not independently per microbatch.
            self._transition_update(window)
        return loss_sum / self.args.wm_window, last_error / last_count.clamp_min(1)

    def _transition_update(self, window):
        """Actual substep measurements supervise both independent small heads.

        The terminal physical transition is retained. No reward/action gradient
        is used to fit these heads and no force measurement enters the actor.
        """
        model = self.transition_model
        prediction = model(
            unflatten_state(window["s"]),
            unflatten_state(window["s1"]),
            window["a"],
            self.normalize_xi(window["xi"]),
        )
        loss = model.loss(prediction, window["transition_metrics"], window["mask"])
        self.opt_transition.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite transition-metrics gradient")
        self.opt_transition.step()
        self.last_model_metrics["transition_metrics_loss"] = loss.detach()
        self.last_model_metrics["transition_metrics_disagreement"] = (
            (prediction[0] - prediction[1]).detach().square().mean()
        )

    def _imag_step(self, state, history, previous_action, frame, xi, xi_n, goal):
        # Same command mapping as real collection/evaluation. Without this the
        # actor can optimize dynamics at actions never executed by the simulator.
        action = self.pi(history, goal).clamp(-20, 20)
        next_state = self.wm.step(state, action, xi_n)
        next_frame = frame + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, xi, next_frame))
        history = torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        return (
            next_state,
            action,
            history,
            next_frame,
            self.motion.goal(next_frame, self.args.goal_frames),
        )

    def _reward_term(self, state, frame, action, previous_action):
        return (
            self.mdp.reward(state, frame, action, previous_action),
            self.mdp.soft_termination(state, frame),
        )

    def _legacy_reward_term(
        self, before, state, frame, action, previous_action, xi, ep_step
    ):
        from .transition_metrics import decode_metrics

        prediction = self.transition_model(before, state, action, self.normalize_xi(xi))
        predicted = decode_metrics(
            prediction,
            before,
            state,
            xi,
            self.mdp.velocity_limits,
            self.mdp.kin.total_mass_kg * 9.81,
        )
        # Compute the expensive analytic FK once, not once per metrics head.
        metrics0 = {key: value[0] for key, value in predicted.items()}
        reward = self.mdp.reward(
            state,
            frame,
            action,
            previous_action,
            metrics=metrics0,
            xi=xi,
            ep_step=ep_step,
            impact_ramp=self.impact_ramp,
        )
        costs = [
            self.mdp.transition_cost_terms(
                {key: value[h] for key, value in predicted.items()},
                frame,
                xi,
                ep_step,
                impact_ramp=self.impact_ramp,
            )
            for h in range(len(prediction))
        ]
        # Larger physical force is not always a larger load-SHARE cost. Compare
        # the resulting weighted costs, never elementwise maxima of forces.
        for key in costs[0]:
            conservative = torch.stack([head[key] for head in costs]).amin(0)
            reward = reward + CONTROL_DT * (conservative - costs[0][key])
        return reward, self.mdp.soft_termination(state, frame)

    def _critic_input(self, state, frame, xi_n, goal):
        return torch.cat((self.mdp.critic_features(state, frame, xi_n), goal), -1)

    def sample_valid_starts(self):
        batch, starts, histories, remaining = (
            self.args.pi_batch,
            [],
            [],
            self.args.pi_batch,
        )
        # A single 45 s clip, sampled throughout its length. No clamped future
        # state masquerading as a continuing transition at the end of the clip.
        for _ in range(16):
            start, history = self.buffer.sample_starts(2 * batch, self.args.hist)
            end = self.motion.clip_end[self.motion.clip_id[start["f"]]]
            idx = (
                (start["f"] + self.args.horizon <= end)
                .nonzero()
                .squeeze(-1)[:remaining]
            )
            starts.append({k: v[idx] for k, v in start.items()})
            histories.append({k: v[idx] for k, v in history.items()})
            remaining -= len(idx)
            if not remaining:
                cat = lambda parts: {
                    k: torch.cat([p[k] for p in parts]) for k in parts[0]
                }
                return cat(starts), cat(histories)
        raise RuntimeError("Replay has no valid H-step motion starts")

    def _policy_rollout(self, start, history):
        """One BPTT partition; frozen model weights retain all state derivatives."""
        args = self.args
        state = initial_state = unflatten_state(start["s"])
        frame, xi, previous = start["f"], start["xi"], start["a_prev"]
        xi_n = self.normalize_xi(xi)
        with torch.no_grad():
            obs_history = self.mdp.add_noise(
                self.mdp.obs(unflatten_state(history["s"]), history["a_prev"],
                             history["xi"], history["f"])
            )
        goal = initial_goal = self.motion.goal(frame, args.goal_frames)
        states, actions, previous_actions, frames, goals, pre_states = [], [], [], [], [], []
        for _ in range(args.horizon):
            pre_states.append(state)
            previous_actions.append(previous)
            state, previous, obs_history, frame, goal = self.imag_step(
                state, obs_history, previous, frame, xi, xi_n, goal
            )
            states.append(state)
            actions.append(previous)
            frames.append(frame)
            goals.append(goal)
        flatten_time = lambda xs: torch.stack(xs).flatten(0, 1)
        all_states = tuple(flatten_time([s[i] for s in states]) for i in range(len(state)))
        if self.transition_model is None:
            rewards, deaths = self.reward_term(
                all_states, flatten_time(frames), flatten_time(actions), flatten_time(previous_actions)
            )
        else:
            all_before = tuple(flatten_time([s[i] for s in pre_states]) for i in range(len(state)))
            context = xi[None].expand(args.horizon, -1, -1).flatten(0, 1)
            age = (start["ep_step"][None]
                   + torch.arange(1, args.horizon + 1, device=xi.device)[:, None]).flatten()
            rewards, deaths = self.reward_term(
                all_before, all_states, flatten_time(frames), flatten_time(actions),
                flatten_time(previous_actions), context, age,
            )
        rewards, deaths = rewards.view(args.horizon, -1), deaths.view(args.horizon, -1)
        # clip_end is inclusive: arriving at its final frame earns that
        # transition's reward, then ends continuation and tail bootstrapping.
        # This boundary is independent of legitimate lying/physical survival.
        rollout_frames = torch.stack(frames)
        source_end = self.motion.clip_end[self.motion.clip_id[rollout_frames]]
        deaths = torch.where(rollout_frames >= source_end, torch.ones_like(deaths), deaths)
        tail = self.critic(self.critic_input(state, frame, xi_n, goal))
        objective, alive_now, alive_next = actor_return(rewards, deaths, tail, args.gamma)
        # Construct critic features from these SAME trajectories before freeing
        # this partition's graph. The full logical batch norm is updated later.
        with torch.no_grad():
            critic_states = tuple(
                torch.cat((initial_state[i][None], torch.stack([s[i].detach() for s in states])))
                .flatten(0, 1) for i in range(len(state))
            )
            critic_frames = torch.cat((start["f"][None], torch.stack(frames))).flatten()
            critic_goals = torch.cat((initial_goal[None], torch.stack(goals))).flatten(0, 1)
            context = xi_n[None].expand(args.horizon + 1, -1, -1).flatten(0, 1)
            features = self.critic_input(critic_states, critic_frames, context, critic_goals)
            features = features.view(args.horizon + 1, len(frame), -1)
        return objective, alive_now, alive_next, rewards, deaths, features, torch.stack(actions).detach().abs().amax()

    @staticmethod
    @torch.no_grad()
    def _update_full_norm(norm, features):
        """One logical-batch RunningNorm update, even with host critic storage.

        Compute one population mean/variance over every time and rollout row.
        Moving only sufficient statistics preserves the RunningNorm equations
        without uploading the entire ~0.8 GB critic dataset to the GPU.
        """
        if norm.count >= norm.max_count:
            return
        x = features.reshape(-1, norm.mean.numel()).to(dtype=norm.mean.dtype)
        if len(x) == 0:
            return
        variance, mean = torch.var_mean(x, dim=0, unbiased=False)
        mean, variance = mean.to(norm.mean.device), variance.to(norm.var.device)
        n = len(x)
        total = norm.count + n
        delta = mean - norm.mean
        new_var = (norm.var * norm.count + variance * n
                   + delta.square() * norm.count * n / total) / total
        norm.mean.add_(delta * n / total)
        norm.var.copy_(new_var)
        norm.count.copy_(total)

    def _critic_update(self, X, rewards, deaths):
        args = self.args
        device = next(self.critic.parameters()).device
        # Exactly one normalization update over all (H+1)*logical_batch rows,
        # followed by fixed pre-update values and lambda targets for all epochs.
        with torch.no_grad():
            self._update_full_norm(self.critic.norm, X)
            flat = X.flatten(0, 1)
            values = torch.empty(X.shape[:2], device=X.device, dtype=X.dtype)
            for chunk in self._chunks(len(flat), getattr(args, "critic_microbatch", 0)):
                values.view(-1)[chunk] = self.critic(flat[chunk].to(device)).to(X.device)
            targets = lambda_returns(rewards, deaths, values, args.gamma, args.lam)
        Xf, Yf = X[:-1].flatten(0, 1), targets.flatten()
        critic_loss = torch.zeros((), device=device)
        for _ in range(args.critic_epochs):
            # A global shuffle, never an independent shuffle per actor partition.
            indices = torch.randperm(len(Yf), device=device).to(X.device)
            for idx in indices.tensor_split(args.critic_minibatches):
                self.opt_v.zero_grad(set_to_none=True)
                critic_loss = torch.zeros((), device=device)
                for chunk in self._chunks(len(idx), getattr(args, "critic_microbatch", 0)):
                    part = idx[chunk]
                    loss = (self.critic(Xf[part].to(device)) - Yf[part].to(device)).square().sum() / len(idx)
                    loss.backward()
                    critic_loss += loss.detach()
                vnorm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if not torch.isfinite(vnorm):
                    raise RuntimeError("Nonfinite critic gradient")
                self.opt_v.step()
        return critic_loss

    def policy_update(self):
        args = self.args
        # Draw the complete logical batch ONCE. Each partition contributes its
        # sum divided by the SAME logical denominator before one optimizer step.
        start, history = self.sample_valid_starts()
        batch = args.pi_batch
        device = next(self.pi.parameters()).device
        storage = torch.device("cpu") if getattr(args, "critic_storage", "device") == "cpu" else device
        loss_sum, reward_sum, survival_sum, action_peak = [torch.zeros((), device=device) for _ in range(4)]
        critic_data = None
        self.opt_pi.zero_grad(set_to_none=True)
        frozen = [self.wm, self.critic] + ([self.transition_model] if self.transition_model is not None else [])
        for module in frozen:
            set_grad(module, False)
        try:
            for chunk in self._chunks(batch, getattr(args, "pi_microbatch", 0)):
                part_start = {key: value[chunk] for key, value in start.items()}
                part_history = {key: value[chunk] for key, value in history.items()}
                objective, alive_now, alive_next, rewards, deaths, features, peak = self._policy_rollout(part_start, part_history)
                loss = -objective.sum() / batch
                loss.backward()
                loss_sum += loss.detach()
                reward_sum += (alive_now * rewards).detach().sum() / (args.horizon * batch)
                survival_sum += alive_next[-1].detach().sum() / batch
                action_peak = torch.maximum(action_peak, peak)
                if critic_data is None:
                    X = torch.empty((args.horizon + 1, batch, features.shape[-1]), device=storage, dtype=features.dtype)
                    R = torch.empty((args.horizon, batch), device=storage, dtype=rewards.dtype)
                    D = torch.empty_like(R)
                    critic_data = X, R, D
                X, R, D = critic_data
                with torch.no_grad():
                    X[:, chunk].copy_(features.detach())
                    R[:, chunk].copy_(rewards.detach())
                    D[:, chunk].copy_(deaths.detach())
                # Do not retain the previous partition's autograd graph.
                del objective, alive_now, alive_next, rewards, deaths, features, loss
            norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite actor gradient; no update applied")
            self.opt_pi.step()
        finally:
            for module in frozen:
                set_grad(module, True)
        critic_loss = self._critic_update(*critic_data)
        return {
            "actor_loss": loss_sum,
            "critic_loss": critic_loss,
            "imagined_reward": reward_sum / self.mdp.reward_scale,
            "actor_grad_norm": norm.detach(),
            "imagined_survival": survival_sum,
            "action_peak": action_peak,
        }
