# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Physical transition collection, replay, and differentiable world-model rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from .representations import NUM_GOAL_FRAMES, TaskSpec, build_actor_obs_td, create_representation

if TYPE_CHECKING:
    from isaaclab_rl.model_based.models import Dynamics
    from isaaclab_rl.model_based.representations import StateRepresentation


@dataclass(frozen=True)
class FieldSpec:
    dim: int = 0  # 0 = scalar (no trailing axis); >0 = vector with trailing dim
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
        return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
    """Per-env circular buffer over a TensorDict, with env-stratified window sampling."""

    def __init__(
        self,
        schema: dict[str, FieldSpec],
        buffer_size: int,
        num_envs: int,
        device: str = "cpu",
    ):
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError("schema needs 'terminated' and 'truncated'")
        self.num_transitions = 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self._step = 0
        self._storage = TensorDict(
            {
                name: torch.zeros(spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device)
                for name, spec in schema.items()
            },
            batch_size=[num_envs, buffer_size],
        )

    def insert(self, td: TensorDict) -> None:
        if td.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}")
        if set(td.keys()) != set(self._storage.keys()):
            raise ValueError("schema mismatch")
        self._storage[:, self._step] = td
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def sample_causal(self, sequence_length: int, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample a window of ``L`` frames ENDING at a uniformly-drawn filled frame.

        The newest frame (index ``-1``) is the sampled "current" state — drawn like
        a 1-step ``sample``, so its distribution is independent of ``L``. Preceding
        slots hold the current frame's real in-episode predecessors; slots before
        the episode start (a stored ``terminated``/``truncated``) or before the
        oldest filled frame are clamped to the first in-episode frame, matching the
        real-sim history ring's reset padding.

        Returns ``(window, padded)``: the ``(B, L)`` window and a ``(B, L)`` bool
        mask that is ``True`` on the clamp-padded slots, so the caller can noise the
        oldest real frame once and share that realization across them.
        """
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        if n < 1:
            raise RuntimeError("sample_causal on empty buffer")
        base = self._step if n == T else 0
        env_idx = torch.randint(E, (B,), device=dev)
        current_age = torch.randint(n, (B,), device=dev)
        ages = current_age[:, None] + (torch.arange(L, device=dev) - (L - 1))  # (B, L), newest last
        ended = (self._storage["terminated"] | self._storage["truncated"])[
            env_idx[:, None], (base + ages.clamp(min=0)) % T
        ]
        # Reachable = contiguous in-episode suffix ending at the (always-valid) newest
        # frame; a stored done at an earlier slot ends the prior episode there.
        valid = (ages >= 0) & ~ended.bool()
        valid[:, -1] = True
        reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
        first_age = torch.where(reachable, ages, current_age[:, None]).amin(dim=-1, keepdim=True)
        slot = (base + torch.where(reachable, ages, first_age)) % T
        return self._storage[env_idx[:, None], slot], ~reachable

    def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
        # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
        base = self._step if n == T else 0
        ages = torch.arange(T, device=dev)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (base + ages) % T].long()
        # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
        c = torch.cat([ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1)
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        if not (total > 0).any():
            raise RuntimeError(f"no valid sequences L={L} n={n}")
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
        u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
        storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
        return self._storage[env_idx[:, None], storage_idx]


class MultiTaskWorldModelEnv:
    """Integrate learned dynamics and reference goals while retaining action gradients."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        motion,
        num_actions: int,
        max_step_reward: float,
        anchor_pos_threshold: float,
        anchor_ori_threshold: float,
        batch_size: int,
        replay_buffer: ReplayBuffer,
        task_specs: list[TaskSpec],
        history_length: int,
        termination_softness: float,
        device: str = "cpu",
    ) -> None:
        self.task_specs = task_specs
        assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
        self.device = device
        self.dynamics = dynamics
        self.representation = representation
        self.num_envs = batch_size
        self.replay_buffer = replay_buffer
        self.history_length = history_length

        self.max_step_reward = max_step_reward
        self._num_actions = num_actions
        self.motion = motion
        self._anchor_index = 0

        if termination_softness <= 0.0:
            raise ValueError(f"termination_softness must be positive, got {termination_softness}")
        self._termination_softness = termination_softness
        self._anchor_pos_threshold = anchor_pos_threshold
        self._anchor_ori_threshold = anchor_ori_threshold

    @property
    def num_actions(self) -> int:
        """Dimension of the action space for the high-level policy."""
        return self._num_actions

    def reset(self, batch: TensorDict | None = None) -> TensorDict:
        """Reset buffers to ``batch`` (a fresh ``sample_initial_states()`` by
        default) and refresh goals."""
        if batch is None:
            batch = self.sample_initial_states()
        self.aux_state, self.rep_state = self._unpack_initial_states(batch)
        self.joint_state = self.representation.joint_state(self.rep_state)
        # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
        # the reward path consumes these directly.
        self.last_actions = batch["prev_action"].detach()
        self.sim_ctx = batch["sim_ctx"]
        self._on_reset(batch)
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        next_aux_state, next_rep_state, next_joint_state, term_prob, rewards = self._wm_step(
            self.rep_state,
            self.aux_state,
            actions,
            self.last_actions,
            self.sim_ctx,
        )
        self.rep_state = next_rep_state
        self.aux_state = next_aux_state
        self.joint_state = next_joint_state
        self.last_actions = actions

        self._compute_commands()

        if self.history_length > 1:
            self._past_sensors = torch.cat([self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)], dim=1)
        self._latest_sensor = self._synth_sensor()

        return self.get_observations(), rewards, term_prob, {}

    def _synth_sensor(self) -> torch.Tensor:
        """Synthesize the actor's sensor from the current WM state."""
        return self.representation.compute_actor_sensor(
            self.rep_state,
            self.sim_ctx,
            aux_state=self.aux_state,
            joint_state=self.joint_state,
            last_action=self.last_actions,
            noisy=True,
        )

    def _sensor_history(self) -> torch.Tensor:
        """The ``(B, history_length, sensor_dim)`` window; newest frame last."""
        latest = self._latest_sensor.unsqueeze(1)
        if self.history_length == 1:
            return latest
        return torch.cat([self._past_sensors, latest], dim=1)

    def _critic_obs(self) -> torch.Tensor:
        """The privileged critic input — full WM state, not the actor's sensor obs."""
        return torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)

    def _wm_step(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bundled dynamics + termination + reward for one env step.

        Returns ``(next_aux_state, next_rep_state, next_joint_state, term_prob,
        reward)``. ``action`` / ``last_action`` are raw policy outputs; Dynamics
        applies the scale/offset transform internally. Termination and reward both
        score the post-step state against the cached reference body state, the way
        the simulator scores it before advancing the clip.
        """
        next_aux_state, next_rep_state = self.dynamics(rep_state, action, aux_state, sim_ctx)
        next_joint_state = self.representation.joint_state(next_rep_state)
        ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = self._ref_body_state
        term_prob = self._compute_term_prob(next_aux_state, ref_body_pos_w, ref_body_quat_w)
        reward = self.representation.compute_wm_reward_from_reference(
            next_rep_state,
            next_aux_state,
            next_joint_state,
            action,
            last_action,
            ref_body_pos_w,
            ref_body_quat_w,
            ref_body_lin_vel_w,
            ref_body_ang_vel_w,
        )
        return next_aux_state, next_rep_state, next_joint_state, term_prob, reward

    def _compute_term_prob(
        self,
        aux_state: torch.Tensor,
        ref_body_pos_w: torch.Tensor,
        ref_body_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Soft termination from the real threshold rules."""
        anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
        anchor_quat_w = ref_body_quat_w[:, self._anchor_index]
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        height_error = (anchor_pos_w[..., 2] - root_pos[..., 2]).abs()
        # Gravity's z in a body frame is 2(x^2 + y^2) - 1 for a unit quaternion.
        ref_gravity_z = 2 * (anchor_quat_w[..., 1:3] ** 2).sum(-1) - 1.0
        root_gravity_z = 2 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
        tilt_error = (ref_gravity_z - root_gravity_z).abs()
        normalized_error = torch.maximum(
            height_error / self._anchor_pos_threshold,
            tilt_error / self._anchor_ori_threshold,
        )
        return torch.sigmoid((normalized_error - 1.0) / self._termination_softness)

    def _on_reset(self, batch: TensorDict) -> None:
        """Seed motion times from the sampled batch and refresh goals."""
        self.motion_times = batch["motion_time"]
        self._update_goal()
        if self.history_length > 1:
            self._past_sensors = self._warmstart_past_sensors(batch["sensor_window"], batch["pad_mask"])
        self._latest_sensor = self._synth_sensor()

    def _compute_commands(self) -> None:
        """Advance the reference clip one frame (clamped at clip ends) and refresh."""
        next_motion_times = self.motion_times + 1
        # Membership in the sorted clip ends; torch.isin syncs the host on long lists.
        ends = self.motion.motion_ends
        pos = torch.searchsorted(ends, next_motion_times).clamp(max=ends.numel() - 1)
        at_end = ends[pos] == next_motion_times
        self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
        self._update_goal()

    def _update_goal(self) -> None:
        """Compute the reference body state at ``motion_times`` and rebuild goals."""
        self._ref_body_state = self.motion.body_state(self.motion_times)
        self._wm_goal_obs, self._actor_goal_obs = self.representation.compute_wm_goal(
            self.motion, self.motion_times, self.aux_state
        )
        self._build_task_goals()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def sample_initial_states(self) -> TensorDict:
        """Sample initial state snapshots from the replay buffer.

        With history_length > 1 the warm start draws the init state 1-step
        (H-independent) and keeps its real in-episode predecessors to seed the ring:
        ``sensor_window`` holds the H sensor frames ending at the init state, and
        ``pad_mask`` flags its clamp-padded slots.
        """
        if self.history_length == 1:
            return self.replay_buffer.sample(1, self.num_envs).squeeze(1)
        window, padded = self.replay_buffer.sample_causal(self.history_length, self.num_envs)
        batch = window[:, -1]
        batch["sensor_window"] = window["actor_obs"]
        batch["pad_mask"] = padded
        return batch

    def _warmstart_past_sensors(self, sensor_window: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Noise the sampled predecessor window; clamp-padded slots (before the
        episode start) share the oldest real frame's single noise realization, the
        way the real-sim ring's reset padding does."""
        noisy = self.representation.add_proprio_noise(sensor_window[:, :-1])
        pad = pad_mask[:, :-1]  # padding is a leading run (sample_causal)
        first_real = (~pad).int().argmax(dim=1)
        shared = noisy[torch.arange(self.num_envs, device=self.device), first_real]
        return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)

    def _unpack_initial_states(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (aux_state, rep_state) from a sampled batch."""
        full_states = batch["state"]
        aux_state = self.representation.extract_aux_state(full_states)
        rep_state = self.representation.represent(full_states)
        return aux_state.detach(), rep_state.detach()

    def _build_task_goals(self) -> None:
        # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
        anchor, *rest = self.task_specs
        self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
        for ts in rest:
            self._task_goal_obs[ts.name] = ts.extract_training_goal(
                self.motion, self.motion_times, self.aux_state
            ).clone()

    def get_observations(self) -> TensorDict:
        return build_actor_obs_td(self._task_goal_obs, self._sensor_history(), critic=self._critic_obs())


# Physics-context normalization: Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Retains BSD-3-Clause licensing; see the repository LICENSE.
class SimContext(torch.nn.Module):
    """Reference column order: foot friction, absolute body CoM, encoder bias.

    Normalization uses randomization range midpoint and half-width, as in the
    reference implementation. The CoM channels contain the actual local CoM,
    including its nominal offset, rather than just the sampled perturbation.
    """

    def __init__(self, num_joints: int, device: str):
        super().__init__()
        self.output_dim = 4 + num_joints
        self.register_buffer("mean", torch.tensor([0.75, 0, 0, 0] + [0] * num_joints, device=device))
        self.register_buffer("scale", torch.tensor([0.45, 0.025, 0.05, 0.05] + [0.01] * num_joints, device=device))

    def read(self, env) -> torch.Tensor:
        command = env.command_manager.get_term("motion")
        return (torch.cat((command.friction, command.com, command.encoder_bias), -1) - self.mean) / self.scale

    def find(self, name: str) -> slice | None:
        return {"foot_friction": slice(0, 1), "base_com": slice(1, 4), "encoder_bias": slice(4, self.output_dim)}.get(
            name
        )

    def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
        return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]


class PhysicalRollout:
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
            self.motion.fk,
            self.env.scene["robot"].data.soft_joint_pos_limits.torch[0],
            self.sim_context,
            self.command.reward_fn,
        )
        self.num_goals = NUM_GOAL_FRAMES * (2 * self.num_actions + 9)
        self.obs_dim = (
            self.representation.rep_state_dim
            if cfg.actor_obs == "rep_state"
            else self.representation.proprio_obs_dim(self.num_actions)
        )
        self.critic_dim = self.representation.rep_state_dim + self.num_goals + self.sim_context.output_dim
        self.task_specs = self.command.task_specs(cfg.tasks, self.representation, self.num_goals)
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
            self.command.create_sampler(cfg.return_sampler, self.max_step_reward)
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
        clip_end = self.command.at_clip_end
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
