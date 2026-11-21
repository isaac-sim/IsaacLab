# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Differentiable world-model rollout with history and per-task goal encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from isaaclab_rl.model_based.storage import ReplayBuffer

from .representations import TaskSpec, build_actor_obs_td

if TYPE_CHECKING:
    from isaaclab_rl.model_based.models import Dynamics
    from isaaclab_rl.model_based.representations import StateRepresentation


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
