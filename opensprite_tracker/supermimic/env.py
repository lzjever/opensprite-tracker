"""Flat X2 simulator adapter with explicit frame ownership and terminal snapshots.

simulate() never calls ManagerBasedRlEnv.step/reset or MotionCommand.compute.
step() labels a cloned physical transition BEFORE optional reset. Simulator
snapshots expose the stochastic delay state, not a deterministic RNG replay API.
"""

import torch

from .robot import (
    JOINT_NAMES,
    MEM_DIM,
    CONTEXT_DIM,
    CONTROL_DT,
    ACTION_SCALE,
    context_normalization,
    make_env_cfg,
)
from .rotations import quat_to_mat, mat_to_quat, so3_exp
from .state import flatten_state


class OpenSpriteTrackingEnv:
    mem_dim = memory_dim = MEM_DIM
    context_dim = xi_dim = CONTEXT_DIM

    def __init__(
        self,
        num_envs,
        motion,
        mdp=None,
        device="cuda:0",
        seed=42,
        eval_mode=False,
        max_episode_steps=500,
        reset_noise=True,
        randomize=True,
        auto_reset=None,
        gamma=0.99,
        full_body_collision=False,
        eval_clip=0,
    ):
        from mjlab.envs import ManagerBasedRlEnv

        self.device, self.N, self.num_envs = str(device), num_envs, num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.full_body_collision, self.eval_clip = full_body_collision, eval_clip
        self.reward_head = mdp.reward if mdp is not None else None
        self.eval_mode, self.reset_noise = eval_mode, reset_noise and not eval_mode
        self.auto_reset = not eval_mode if auto_reset is None else auto_reset
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.raw_env = ManagerBasedRlEnv(
            make_env_cfg(
                num_envs, seed, randomize, full_body_collision=full_body_collision
            ),
            device=self.device,
        )
        self.env = self.raw_env
        self.sim, self.scene = self.raw_env.sim, self.raw_env.scene
        self.robot = self.scene["robot"]
        self.joint_ids = self.robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
        self.actuator = self.robot.actuators[0]
        if len(self.robot.actuators) != 1:
            raise ValueError("Expected the baseline's single grouped X2 actuator")
        names = [n.rsplit("/", 1)[-1] for n in self.actuator.target_names]
        self.act_ids = [names.index(n) for n in JOINT_NAMES]
        action_names = [
            n.rsplit("/", 1)[-1]
            for n in self.raw_env.action_manager.get_term("joint_pos").target_names
        ]
        self.action_order = [JOINT_NAMES.index(n) for n in action_names]
        m = self.sim.mj_model
        joint_map = {m.joint(j).name.rsplit("/", 1)[-1]: j for j in range(m.njnt)}
        self.dof_ids = [int(m.jnt_dofadr[joint_map[n]]) for n in JOINT_NAMES]
        self.torso_id = next(
            i
            for i in range(m.nbody)
            if m.body(i).name.rsplit("/", 1)[-1] == "waist_yaw_link"
        )
        self.foot_geom_id = next(
            i
            for i in range(m.ngeom)
            if m.geom(i).name.rsplit("/", 1)[-1] == "left_foot_collision"
        )
        self.default_q = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.alpha = torch.full((31,), ACTION_SCALE, device=self.device)
        self.context_offset, self.context_scale = context_normalization(self.device)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 31, device=self.device)
        self.ep_ret = torch.zeros(num_envs, device=self.device)
        self.ep_disc_ret = torch.zeros_like(self.ep_ret)
        self.start_bin = torch.zeros_like(self.f)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.frame_sampler = (
            None  # optional callable(n) -> (frames,bins), owned by trainer
        )
        self.clip_sampler = None  # optional stratified sampler(clip_ids)
        # Mutable scalar tensor avoids recompiling the analytic reward at every
        # curriculum update. Evaluation leaves it at its final value of one.
        self.impact_ramp = torch.ones((), device=self.device)
        self.reset(torch.arange(num_envs, device=self.device))

    @property
    def encoder_bias(self):
        return self.robot.data.encoder_bias[:, self.joint_ids].clone()

    def _model_field(self, name, indices):
        field = getattr(self.sim.model, name)
        return field[:, indices].expand(self.N, *field[:, indices].shape[1:])

    @property
    def xi(self):
        a, ids = self.actuator, self.act_ids
        mu = self._model_field("geom_friction", [self.foot_geom_id])[:, 0, :1]
        com = (
            self._model_field("body_ipos", [self.torso_id])[:, 0]
            - self.sim.get_default_field("body_ipos")[self.torso_id]
        )
        return torch.cat(
            (
                mu,
                self.encoder_bias,
                com,
                a.stiffness[:, ids],
                a.damping[:, ids],
                a.force_limit[:, ids],
                self._model_field("dof_armature", self.dof_ids),
                self._model_field("dof_frictionloss", self.dof_ids),
                a.torque_time_constant[:, ids],
            ),
            -1,
        ).clone()

    def normalize_xi(self, xi):
        return (xi - self.context_offset) / self.context_scale

    context_normalizer = normalize_xi

    @property
    def normalized_xi(self):
        return self.normalize_xi(self.xi)

    def memory(self):
        a, ids = self.actuator, self.act_ids
        delay = a._delay_buffer
        queue = delay._buffer
        valid = queue.current_length > 0
        target = (
            queue[0][..., 0][:, ids]
            if queue.is_initialized
            else torch.zeros_like(self.a_prev)
        )
        previous = (target - self.default_q) / ACTION_SCALE
        previous = torch.where(valid[:, None], previous, torch.zeros_like(previous))
        effort = torch.where(
            a.filter_initialized[:, None],
            a.filtered_effort[:, ids] / a.force_limit[:, ids],
            torch.zeros_like(previous),
        )
        phase = ((delay._step_count + delay._phase_offsets) % 4).float() / 3.0
        return torch.cat(
            (
                effort,
                previous,
                delay.current_lags[:, None].float(),
                phase[:, None],
                a.filter_initialized[:, None].float(),
                valid[:, None].float(),
            ),
            -1,
        ).clone()

    def actuator_snapshot(self):
        """Physical memory for audits. Reading this never advances the delay RNG."""
        a, delay = self.actuator, self.actuator._delay_buffer
        queue = delay._buffer
        return {
            "filtered_effort": a.filtered_effort[:, self.act_ids].clone(),
            "filter_initialized": a.filter_initialized.clone(),
            "command_history": queue.buffer[:, :, self.act_ids].clone()
            if queue.is_initialized
            else None,
            "queue_length": queue.current_length.clone(),
            "lag": delay.current_lags.clone(),
            "delay_step": delay._step_count.clone(),
            "delay_phase": delay._phase_offsets.clone(),
        }

    def state(self):
        d = self.robot.data
        return (
            (d.root_link_pos_w - self.scene.env_origins).clone(),
            quat_to_mat(d.root_link_quat_w.clone()),
            d.root_link_lin_vel_w.clone(),
            d.root_link_ang_vel_w.clone(),
            d.joint_pos[:, self.joint_ids].clone(),
            d.joint_vel[:, self.joint_ids].clone(),
            self.memory(),
        )

    def snapshot(self, include_actuator=False):
        S = self.state()
        out = {
            "S": S,
            "s": flatten_state(S),
            "h": S[6],
            "xi": self.xi,
            "f": self.f.clone(),
            "a_prev": self.a_prev.clone(),
            "ep_step": self.ep_step.clone(),
        }
        if include_actuator:
            out["actuator"] = self.actuator_snapshot()
        return out

    prestep = snapshot

    @torch.no_grad()
    def simulate(self, a):
        """Physical transition only; returned s1 always precedes any reset/frame change.

        h is predicted by the WM. Exact analytic filter advancement requires the
        four intermediate q/qd and realized stochastic lag draws; endpoints alone
        are insufficient. New random draws remain transition noise, not actor input.
        """
        a = a.to(device=self.device, dtype=self.a_prev.dtype)
        if a.shape != (self.N, 31) or not torch.isfinite(a).all():
            raise ValueError("Expected finite OpenSprite actions of shape (N,31)")
        pre = self.prestep()
        raw = self.raw_env
        raw.action_manager.process_action(a[:, self.action_order])
        self.actuator.speed_peak.zero_()
        self.actuator.raw_peak.zero_()
        self.actuator.applied_peak.zero_()
        for _ in range(4):
            raw._sim_step_counter += 1
            raw.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=0.005)
        # Capture the actual four contact samples before any forward/reset.
        sensor = self.scene["support_foot_contact"].data
        history = sensor.force_history.clone()
        body_history = (
            self.scene["nonfoot_ground_contact"].data.force_history.clone()
            if self.full_body_collision
            else None
        )
        self.sim.forward()
        self.sim.sense()
        self.ep_step += 1
        raw.episode_length_buf[:] = self.ep_step
        raw.common_step_counter += 1
        self.a_prev.copy_(a)
        post = self.snapshot()
        end = self.motion.clip_end[self.motion.clip_id[pre["f"]]]
        f1 = torch.minimum(pre["f"] + 1, end)
        result = {
            "s": pre["s"],
            "s1": post["s"],
            "S": pre["S"],
            "S1": post["S"],
            "a": a.clone(),
            "a_prev": pre["a_prev"],
            "xi": pre["xi"],
            "f": pre["f"],
            "f1": f1,
            "ep_step": pre["ep_step"],
            "clip_end": pre["f"] + 1 >= end,
            "speed_peak": torch.maximum(
                self.actuator.speed_peak[:, self.act_ids], post["S"][5].abs()
            ).clone(),
            "contact_force_history": history,
            "contact_mean_n": history[..., 2].abs().mean(-1),
            "contact_peak_n": (-history[..., 2]).clamp_min(0).amax(-1),
            "raw_torque_peak": self.actuator.raw_peak[:, self.act_ids].clone(),
            "applied_torque_peak": self.actuator.applied_peak[:, self.act_ids].clone(),
        }
        if self.full_body_collision:
            result["body_contact_peak_n"] = (
                (-body_history[..., 2]).clamp_min(0).amax(dim=(-1, -2))[:, None].clone()
            )
        return result

    def advance_reference(self):
        end = self.motion.clip_end[self.motion.clip_id[self.f]]
        done = self.f + 1 >= end
        self.f.copy_(torch.minimum(self.f + 1, end))
        return done

    @torch.no_grad()
    def step(self, a):
        if self.mdp is None:
            raise RuntimeError(
                "step requires mdp; use simulate for unlabeled transitions"
            )
        out = self.simulate(a)
        if getattr(self.mdp, "uses_transition_metrics", False):
            out["r"], out["terms"] = self.reward_head(
                out["S1"],
                out["f1"],
                out["a"],
                out["a_prev"],
                terms=True,
                metrics=out,
                xi=out["xi"],
                ep_step=out["ep_step"] + 1,
                impact_ramp=self.impact_ramp,
            )
        else:
            out["r"], out["terms"] = self.reward_head(
                out["S1"], out["f1"], out["a"], out["a_prev"], terms=True
            )
        out["term"] = self.mdp.termination(out["S1"], out["f1"])
        out["timeout"] = out["clip_end"] | (self.ep_step >= self.max_episode_steps)
        out["done"] = out["term"] | out["timeout"]
        self.ep_ret += out["r"]
        self.ep_disc_ret += self.gamma ** out["ep_step"].float() * out["r"]
        self.advance_reference()
        ids = out["done"].nonzero(as_tuple=False).flatten()
        if len(ids):
            out["ep_ids"] = ids
            out["ep_len"] = self.ep_step[ids].clone()
            out["ep_ret"] = self.ep_ret[ids].clone()
            out["ep_disc_ret"] = self.ep_disc_ret[ids].clone()
            out["ep_start_bin"] = self.start_bin[ids].clone()
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
        if self.auto_reset and len(ids):
            self.reset(ids)
        return out

    @torch.no_grad()
    def reset(self, env_ids=None, frames=None):
        ids = (
            torch.arange(self.N, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(
                -1
            )
        )
        if not len(ids):
            return
        if frames is None:
            if self.clip_sampler is not None and not self.eval_mode:
                frames, _ = self.clip_sampler(ids % len(self.motion.clip_end))
            elif self.frame_sampler is not None and not self.eval_mode:
                frames, _ = self.frame_sampler(len(ids))
            else:
                if self.eval_mode:
                    frames = self.motion.clip_start[self.eval_clip].expand(len(ids))
                else:
                    # Initial fill also uses equal clip mass, not packed length.
                    clip = ids % len(self.motion.clip_end)
                    lo, hi = self.motion.clip_start[clip], self.motion.clip_end[clip]
                    frames = (
                        lo
                        + (torch.rand(len(ids), device=self.device) * (hi - lo)).long()
                    )
        frames = torch.as_tensor(frames, device=self.device, dtype=torch.long).expand(
            len(ids)
        )
        if ((frames < 0) | (frames >= self.motion.num_frames)).any():
            raise ValueError("Reset reference frame is out of range")
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise:
            sample = lambda half: (
                (2 * torch.rand(len(ids), len(half), device=self.device) - 1)
                * torch.tensor(half, device=self.device)
            )
            p += sample([0.05, 0.05, 0.01])
            R = so3_exp(sample([0.1, 0.1, 0.2])) @ R
            v += sample([0.5, 0.5, 0.2])
            w += sample([0.52, 0.52, 0.78])
            q += sample([0.1] * 31)
        if self.full_body_collision:
            kin = self.mdp.kin
            q = q.clamp(kin.q_lo, kin.q_hi)
            bottom = kin.ground_clearance(p, R, v, w, q, qd)
            p[..., 2] += (0.001 - bottom).clamp_min(0)
        else:
            limits = self.robot.data.soft_joint_pos_limits[ids][:, self.joint_ids]
            q = q.clamp(limits[..., 0], limits[..., 1])
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.raw_env.action_manager.reset(ids)
        self.robot.write_root_state_to_sim(
            torch.cat((p + self.scene.env_origins[ids], mat_to_quat(R), v, w), -1),
            env_ids=ids,
        )
        self.robot.write_joint_state_to_sim(
            q, qd, joint_ids=self.joint_ids, env_ids=ids
        )
        self.actuator.filtered_effort[ids] = 0
        # No write_data_to_sim here: that would advance ALL actuators/delay clocks
        # and pollute other envs during a partial reset.
        self.sim.forward()
        self.sim.sense()
        self.f[ids], self.ep_step[ids], self.a_prev[ids] = frames, 0, 0
        self.ep_ret[ids], self.ep_disc_ret[ids] = 0, 0
        self.start_bin[ids] = self.motion.frame_bin[frames]
        self.done_eval[ids], self.failed_eval[ids] = False, False
        self.raw_env.episode_length_buf[ids] = 0
        self.raw_env.observation_manager.reset(ids)

    def refresh(self):
        self.sim.forward()
        self.sim.sense()

    def close(self):
        self.raw_env.close()


TrackingSim = OpenSpriteTrackingEnv


def make_env(num_envs, motion, mdp=None, **kwargs):
    return OpenSpriteTrackingEnv(num_envs, motion, mdp, **kwargs)
