# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified to read Isaac Lab's native state, action, and termination interfaces.

"""Compose the reference tracking equations with a native Isaac Lab environment."""

import torch
from tensordict import TensorDict

from isaaclab_rl.model_based.replay_buffer import FieldSpec
from isaaclab_rl.model_based.representations import NUM_GOAL_FRAMES, create_representation
from isaaclab_rl.model_based.reset_sampling import ReturnSampler
from isaaclab_rl.model_based.rollout import MultiTaskWorldModelEnv
from isaaclab_rl.model_based.task_specs import build_actor_obs_td

from isaaclab_tasks.core.motion_tracking.mdp.model_based.goals import resolve_task_specs
from isaaclab_tasks.core.motion_tracking.mdp.model_based.observations import motion_end
from isaaclab_tasks.core.motion_tracking.mdp.model_based.sim_context import SimContext

from .rewards import tracking_reward


class MotionTrackingAdapter:
    """Physical collection and reference construction, preserving pre-step snapshot timing."""

    def __init__(self, env, cfg):
        self.env = env.unwrapped
        self._reset_env, self._step_env = env.reset, env.step
        self.cfg = cfg
        self.device, self.num_envs = self.env.device, self.env.num_envs
        self.command = self.env.command_manager.get_term("motion")
        self.sensor_noise = not self.command.cfg.evaluation
        self.motion = self.command.motion
        self.num_actions = self.env.action_manager.total_action_dim
        if cfg.history_length > 1 and cfg.actor_obs != "proprio":
            raise ValueError("history_length > 1 requires actor_obs='proprio'.")
        if cfg.buffer_size < max(cfg.wm.horizon, cfg.history_length):
            raise ValueError("Replay capacity must cover the model window and actor history.")
        if abs(self.env.step_dt - 0.02) > 1e-8:
            raise ValueError("Reference motion frames require a 20 ms control period.")
        expected = "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
        if self.command.cfg.sampling_mode != expected:
            raise ValueError("Resolve sampling_mode into the motion command before creating the environment.")
        self.obs, _ = self._reset_env()
        self.sim_context = SimContext(self.num_actions, self.device)
        self.representation = create_representation(
            cfg.representation,
            self.obs["full_state"].shape[-1],
            self.num_actions,
            self.env.step_dt,
            self.env.cfg.decimation,
            self.command.robot.data.default_joint_pos.torch[0].clone(),
            cfg.actor_obs,
        ).to(self.device)
        self.representation.bind(
            self.env.command_manager.get_term("motion").motion.fk,
            self.env.scene["robot"].data.soft_joint_pos_limits.torch[0],
            self.sim_context,
            tracking_reward,
        )
        self.num_goals = NUM_GOAL_FRAMES * (2 * self.num_actions + 9)
        self.obs_dim = (
            self.representation.rep_state_dim
            if cfg.actor_obs == "rep_state"
            else self.representation.proprio_obs_dim(self.num_actions)
        )
        self.critic_dim = self.representation.rep_state_dim + self.num_goals + self.sim_context.output_dim
        self.task_specs = resolve_task_specs(cfg.tasks, self.representation, self.num_goals, self.device)
        self.slot_mix = cfg.slot_mix
        self.active_tasks = [spec for spec in self.task_specs if cfg.slot_mix.get(spec.name, 0) > 0]
        if not self.active_tasks or self.active_tasks[0].name != "motion":
            raise ValueError("The reference motion encoder must have a nonempty anchor slot.")
        action = self.env.action_manager.get_term("joint_pos")
        self.action_scale = torch.as_tensor(action._scale, device=self.device).expand(self.num_envs, self.num_actions)[
            :1
        ]
        self.action_offset = torch.as_tensor(action._offset, device=self.device).expand(
            self.num_envs, self.num_actions
        )[:1]
        self.buffer_schema = {
            "state": FieldSpec(self.representation.full_state_dim),
            "action": FieldSpec(self.num_actions),
            "prev_action": FieldSpec(self.num_actions),
            "motion_time": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
            "sim_ctx": FieldSpec(self.sim_context.output_dim),
        }
        self.stores_actor_obs = cfg.actor_obs == "proprio" and cfg.history_length > 1
        if self.stores_actor_obs:
            self.buffer_schema["actor_obs"] = FieldSpec(self.obs_dim)
        self.max_step_reward = 5.0 * self.env.step_dt
        self.sampler = (
            ReturnSampler(
                cfg.return_sampler,
                self.command.bin_count,
                self.motion.time_step_total,
                self.num_envs,
                self.max_step_reward,
                self.device,
            )
            if cfg.sampling_mode == "return"
            else None
        )
        self.history = None
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_budgets = torch.ones(self.num_envs, device=self.device)
        self.last_episode_return = torch.zeros((), device=self.device)

    def reset(self) -> None:
        """Reset collection after the model initialization, as in the reference runner."""
        self.obs, _ = self._reset_env()
        self.episode_returns.zero_()
        self.episode_budgets.copy_(self._remaining_episode_steps())
        if self.cfg.history_length > 1:
            seed = self.obs["actor"]
            if self.sensor_noise:
                seed = self.representation.add_proprio_noise(seed)
            self.history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

    def _remaining_episode_steps(self) -> torch.Tensor:
        frames = self.command.time_steps
        clips = torch.searchsorted(self.motion.motion_ends, frames, right=True)
        return (self.motion.motion_ends[clips] - frames).clamp(max=self.env.max_episode_length)

    def create_rollout(self, dynamics, replay_buffer):
        return MultiTaskWorldModelEnv(
            dynamics=dynamics,
            representation=self.representation,
            motion=self.motion,
            num_actions=self.num_actions,
            max_step_reward=self.max_step_reward,
            anchor_pos_threshold=self.env.cfg.terminations.anchor_pos.params["threshold"],
            anchor_ori_threshold=self.env.cfg.terminations.anchor_ori.params["threshold"],
            batch_size=self.cfg.policy.batch_size,
            replay_buffer=replay_buffer,
            task_specs=self.active_tasks,
            history_length=self.cfg.history_length,
            termination_softness=self.cfg.termination_softness,
            device=self.device,
        )

    def actor_input(self) -> TensorDict:
        full = self.obs["full_state"]
        rep, aux = self.representation.represent(full), self.representation.extract_aux_state(full)
        sensor = self.representation.compute_actor_sensor(
            rep,
            self.sim_context.read(self.env),
            aux_state=aux,
            sim_actor_obs=self.obs["actor"] if self.cfg.actor_obs == "proprio" else None,
            noisy=self.cfg.actor_obs == "proprio" and self.sensor_noise,
        )
        goals = {
            spec.name: spec.extract_training_goal(self.motion, self.command.time_steps, aux)
            for spec in self.active_tasks
        }
        history = sensor.unsqueeze(1) if self.cfg.history_length == 1 else self.history
        return build_actor_obs_td(goals, history)

    def advance(self, actions):
        """Step physics and advance the history, repeating one noised frame on resets."""
        self.obs, rewards, terminated, truncated, extras = self._step_env(actions)
        if self.cfg.history_length > 1:
            sensor = self.obs["actor"]
            if self.sensor_noise:
                sensor = self.representation.add_proprio_noise(sensor)
            shifted = torch.cat((self.history[:, 1:], sensor.unsqueeze(1)), dim=1)
            self.history = torch.where((terminated | truncated).view(-1, 1, 1), sensor.unsqueeze(1), shifted)
        return rewards, terminated, truncated, extras

    def collect(self, actor, noise: float):
        """Store pre-step state/action/context and post-step episode boundary flags."""
        frames = self.command.time_steps.clone()
        inputs = self.actor_input()
        actions = actor(inputs)
        actions = actions + torch.randn_like(actions) * noise
        fields = {
            "state": self.obs["full_state"].clone(),
            "action": actions.detach().clone(),
            "prev_action": self.env.action_manager.action.clone(),
            "motion_time": frames,
            "sim_ctx": self.sim_context.read(self.env),
        }
        if self.stores_actor_obs:
            fields["actor_obs"] = self.obs["actor"].clone()
        # The command is advanced/reset inside env.step; remember whether this reference ends.
        clip_end = motion_end(self.env)
        rewards, terminated, truncated, _ = self.advance(actions)
        done = terminated | truncated
        self.episode_returns.add_(rewards)
        count = done.sum()
        completed_return = (self.episode_returns / (self.max_step_reward * self.episode_budgets) * done).sum()
        self.last_episode_return = torch.where(
            count > 0, completed_return / count.clamp_min(1), self.last_episode_return
        )
        self.episode_returns.masked_fill_(done, 0)
        self.episode_budgets = torch.where(done, self._remaining_episode_steps(), self.episode_budgets)
        fields["terminated"], fields["truncated"] = terminated & ~truncated, truncated
        if self.sampler is not None:
            weights = self.sampler.record(frames, rewards, terminated & ~truncated, clip_end, truncated & ~clip_end)
            if weights is not None:
                self.command.set_sampling_distribution(weights)
        logs = {"Train/reward": rewards.mean() / self.env.step_dt, "Train/termination_rate": terminated.float().mean()}
        logs["Train/normalized_episode_return"] = self.last_episode_return
        logs["Train/completed_episodes"] = count
        return TensorDict(fields, batch_size=[self.num_envs]), inputs, logs
