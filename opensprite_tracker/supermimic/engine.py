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

    def wm_update(self):
        window = self.buffer.sample_windows(self.args.wm_batch, self.args.wm_window)
        prediction = self.wm_rollout(window["s"], window["a"], window["xi"])
        scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        error = (
            (self.wm.state_error(prediction, unflatten_state(window["s1"])) / scale)
            .square()
            .mean(-1)
        )
        mask = window["mask"]
        loss = (error * mask).sum() / self.args.wm_batch
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite world-model gradient; no update applied")
        self.opt_wm.step()
        last = (error[:, -1].detach() * mask[:, -1]).sum() / mask[
            :, -1
        ].sum().clamp_min(1)
        self.last_model_metrics = {
            "last_step_valid_fraction": mask[:, -1].mean().detach(),
            "error_per_valid_transition": (error.detach() * mask).sum()
            / mask.sum().clamp_min(1),
        }
        if self.transition_model is not None:
            self._transition_update(window)
        return (loss.detach() / self.args.wm_window, last)

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

    def policy_update(self):
        args = self.args
        start, history = self.sample_valid_starts()
        state = initial_state = unflatten_state(start["s"])
        frame, xi, previous = start["f"], start["xi"], start["a_prev"]
        xi_n = self.normalize_xi(xi)
        with torch.no_grad():
            obs_history = self.mdp.add_noise(
                self.mdp.obs(
                    unflatten_state(history["s"]),
                    history["a_prev"],
                    history["xi"],
                    history["f"],
                )
            )
        goal = initial_goal = self.motion.goal(frame, args.goal_frames)
        set_grad(self.wm, False)
        set_grad(self.critic, False)
        if self.transition_model is not None:
            set_grad(self.transition_model, False)
        states, actions, previous_actions, frames, goals, pre_states = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
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
        all_states = tuple(
            flatten_time([s[i] for s in states]) for i in range(len(state))
        )
        if self.transition_model is None:
            rewards, deaths = self.reward_term(
                all_states,
                flatten_time(frames),
                flatten_time(actions),
                flatten_time(previous_actions),
            )
        else:
            all_before = tuple(
                flatten_time([s[i] for s in pre_states]) for i in range(len(state))
            )
            context = xi[None].expand(args.horizon, -1, -1).flatten(0, 1)
            age = (
                start["ep_step"][None]
                + torch.arange(1, args.horizon + 1, device=xi.device)[:, None]
            ).flatten()
            rewards, deaths = self.reward_term(
                all_before,
                all_states,
                flatten_time(frames),
                flatten_time(actions),
                flatten_time(previous_actions),
                context,
                age,
            )
        rewards, deaths = rewards.view(args.horizon, -1), deaths.view(args.horizon, -1)
        # Frozen critic WEIGHTS, but tail state's derivative stays in the graph.
        tail = self.critic(self.critic_input(state, frame, xi_n, goal))
        objective, alive_now, alive_next = actor_return(
            rewards, deaths, tail, args.gamma
        )
        loss = -objective.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite actor gradient; no update applied")
        self.opt_pi.step()
        set_grad(self.wm, True)
        set_grad(self.critic, True)
        if self.transition_model is not None:
            set_grad(self.transition_model, True)
        # Critic learns from the SAME imagined transitions. Targets are detached
        # once, not recomputed after each minibatch with newly updated weights.
        with torch.no_grad():
            critic_states = tuple(
                torch.cat(
                    (
                        initial_state[i][None],
                        torch.stack([s[i].detach() for s in states]),
                    )
                ).flatten(0, 1)
                for i in range(len(state))
            )
            critic_frames = torch.cat((start["f"][None], torch.stack(frames))).flatten()
            critic_goals = torch.cat((initial_goal[None], torch.stack(goals))).flatten(
                0, 1
            )
            context = xi_n[None].expand(args.horizon + 1, -1, -1).flatten(0, 1)
            X = self.critic_input(critic_states, critic_frames, context, critic_goals)
            X = X.view(args.horizon + 1, args.pi_batch, -1)
            self.critic.norm.update(X)
            values = self.critic(X)
            targets = lambda_returns(rewards, deaths, values, args.gamma, args.lam)
        Xf, Yf = X[:-1].flatten(0, 1), targets.flatten()
        critic_loss = torch.zeros((), device=X.device)
        for _ in range(args.critic_epochs):
            indices = torch.randperm(len(Yf), device=X.device)
            for idx in indices.tensor_split(args.critic_minibatches):
                critic_loss = (self.critic(Xf[idx]) - Yf[idx]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                critic_loss.backward()
                vnorm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if not torch.isfinite(vnorm):
                    raise RuntimeError("Nonfinite critic gradient")
                self.opt_v.step()
        return {
            "actor_loss": loss.detach(),
            "critic_loss": critic_loss.detach(),
            "imagined_reward": (alive_now * rewards).detach().mean()
            / self.mdp.reward_scale,
            "actor_grad_norm": norm.detach(),
            "imagined_survival": alive_next[-1].detach().mean(),
            "action_peak": torch.stack(actions).detach().abs().amax(),
        }
