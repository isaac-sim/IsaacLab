# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Callable
from contextlib import suppress
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from typing import Any

from .raw_interface import RawMPInterface


def _find_attr_across_wrappers(env, names, default=None):
    head = env
    while head is not None:
        for name in names:
            if hasattr(head, name):
                return getattr(head, name)
        if not hasattr(head, "env"):
            break
        head = head.env
    return default


def _ensure_batch(tensor: torch.Tensor, target: int, squeeze_last: bool = False) -> torch.Tensor:
    """Broadcast or slice a tensor to match the target batch size."""
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if squeeze_last and tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.shape[0] == target:
        return tensor
    if tensor.shape[0] == 1 and target > 1:
        reps = [target] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*reps)
    if target == 1 and tensor.shape[0] > 1:
        return tensor[:1]
    reps = [target] + [1] * (tensor.ndim - 1)
    return tensor[:1].repeat(*reps)


class BlackBoxWrapper(gym.ObservationWrapper):
    """Torch-only MP rollout wrapper."""

    def __init__(
        self,
        env: RawMPInterface,
        trajectory_generator,
        tracking_controller,
        duration: float,
        reward_aggregation: Callable[[torch.Tensor, int], torch.Tensor] = torch.sum,
        learn_sub_trajectories: bool = False,
        replanning_schedule: None | (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], bool]
        ) = None,
        max_planning_times: int = math.inf,
        condition_on_desired: bool = False,
        device: str | torch.device | None = None,
        verbose: int = 1,
        **kwargs,
    ):
        super().__init__(env)
        self.device = torch.device(device) if device is not None else torch.device(getattr(env, "device", "cpu"))
        self.verbose = verbose
        _ = kwargs  # ignore unused extras
        self.num_envs = getattr(env, "num_envs", None)

        self.duration = duration
        self.learn_sub_trajectories = learn_sub_trajectories
        self.do_replanning = replanning_schedule is not None
        self.replanning_schedule = replanning_schedule or (lambda *x: False)
        self.current_traj_steps = 0
        self.plan_steps = 0
        self.max_planning_times = max_planning_times
        # cache context mask for downstream wrappers
        self.context_mask = getattr(self.env, "context_mask", None)

        self.traj_gen = trajectory_generator
        self.tracking_controller = tracking_controller
        self.dt = _find_attr_across_wrappers(env, ["step_dt", "dt", "sim_dt"], default=0.02)
        self.traj_gen.set_duration(self.duration, self.dt)

        self.tau_bound = [-torch.inf, torch.inf]
        self.delay_bound = [-torch.inf, torch.inf]
        if hasattr(self.traj_gen.phase_gn, "tau_bound"):
            self.tau_bound = self.traj_gen.phase_gn.tau_bound
        if hasattr(self.traj_gen.phase_gn, "delay_bound"):
            self.delay_bound = self.traj_gen.phase_gn.delay_bound

        self.reward_aggregation = reward_aggregation
        self.return_context_observation = not (learn_sub_trajectories or self.do_replanning)
        self.traj_gen_action_space = self._get_traj_gen_action_space()
        self.action_space = self._get_action_space()
        # expose single_action_space for downstream wrappers (MP parameter space)
        self.single_action_space = self.traj_gen_action_space
        # build masked observation spaces (single + batched) from the underlying env's single space
        masked_single_obs_space = self._get_observation_space()
        self.single_observation_space = masked_single_obs_space
        self.observation_space = self._batch_observation_space(masked_single_obs_space)
        # propagate masked spaces to self and underlying env so downstream wrappers see correct shapes
        for target in (self.env, getattr(self.env, "env", None)):
            if target is None:
                continue
            try:
                target.single_observation_space = masked_single_obs_space
                target.observation_space = self.observation_space
            except Exception:
                pass
        self.render_kwargs: dict[str, Any] = {}

        self.condition_on_desired = condition_on_desired
        self.condition_pos = None
        self.condition_vel = None

    def observation(self, observation):
        obs = observation
        if isinstance(obs, dict):
            if "policy" in obs:
                obs = obs["policy"]
            elif len(obs) == 1:
                obs = next(iter(obs.values()))
        if self.return_context_observation and hasattr(self.env, "context_mask"):
            mask = getattr(self.env, "context_mask")
            if torch.is_tensor(mask):
                obs = obs[..., mask.to(obs.device)]
        # Return dict-form to align with manager-based observation spaces
        return {"policy": obs}

    def get_trajectory(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        batch_size = action.shape[0]
        if self.num_envs and batch_size == 1 and self.num_envs > 1:
            action = action.repeat(self.num_envs, 1)
            batch_size = action.shape[0]

        duration = None if self.learn_sub_trajectories else self.duration
        if self.learn_sub_trajectories:
            self.traj_gen.reset()

        self.traj_gen.set_add_dim([batch_size])
        clipped_params = torch.clamp(
            action,
            torch.as_tensor(self.traj_gen_action_space.low, device=self.device),
            torch.as_tensor(self.traj_gen_action_space.high, device=self.device),
        )
        self.traj_gen.set_params(clipped_params)

        init_time = torch.zeros(batch_size, device=self.device)
        if self.do_replanning:
            init_time = torch.full((batch_size,), self.current_traj_steps * self.dt, device=self.device)

        condition_pos = (
            self.condition_pos if self.condition_pos is not None else self.env.get_wrapper_attr("current_pos")
        )
        condition_vel = (
            self.condition_vel if self.condition_vel is not None else self.env.get_wrapper_attr("current_vel")
        )
        condition_pos = _ensure_batch(condition_pos, batch_size)
        condition_vel = _ensure_batch(condition_vel, batch_size)

        self.traj_gen.set_initial_conditions(init_time, condition_pos, condition_vel)
        self.traj_gen.set_duration(duration, self.dt)

        position = self.traj_gen.get_traj_pos()
        velocity = self.traj_gen.get_traj_vel()
        return position, velocity

    def _get_traj_gen_action_space(self):
        min_action_bounds, max_action_bounds = self.traj_gen.get_params_bounds()
        dtype = min_action_bounds.detach().cpu().numpy().dtype
        return spaces.Box(
            low=min_action_bounds.detach().cpu().numpy(),
            high=max_action_bounds.detach().cpu().numpy(),
            dtype=dtype,
        )

    def _get_observation_space(self):
        # prefer the underlying single_observation_space (may live deeper in the wrapper stack)
        obs_space = _find_attr_across_wrappers(self.env, ["single_observation_space"], None)
        if obs_space is None:
            obs_space = _find_attr_across_wrappers(self.env, ["observation_space"], None)
        if not self.return_context_observation or obs_space is None:
            return obs_space

        mask = getattr(self.env, "context_mask", None)

        def _mask_box(box_space: spaces.Box):
            if mask is None:
                return box_space
            mask_np = mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask) else mask
            return spaces.Box(
                low=box_space.low[..., mask_np],
                high=box_space.high[..., mask_np],
                dtype=box_space.dtype,
            )

        if isinstance(obs_space, spaces.Dict):
            # only mask the policy entry if present
            masked_spaces = dict(obs_space.spaces)
            if "policy" in masked_spaces and isinstance(masked_spaces["policy"], spaces.Box):
                masked_spaces["policy"] = _mask_box(masked_spaces["policy"])
            return spaces.Dict(masked_spaces)

        if isinstance(obs_space, spaces.Box):
            return spaces.Dict({"policy": _mask_box(obs_space)})

        return obs_space

    def _batch_observation_space(self, single_obs_space: spaces.Space) -> spaces.Space:
        """Batch a single-environment observation space if num_envs is available."""
        if single_obs_space is None:
            return single_obs_space
        if self.num_envs is not None:
            with suppress(Exception):
                return batch_space(single_obs_space, self.num_envs)
        return single_obs_space

    def _get_action_space(self):
        """Expose the MP parameter space as the action space by default."""
        base_space = getattr(self, "traj_gen_action_space", None)
        if base_space is None:
            base_space = self._get_traj_gen_action_space()
        if self.num_envs is not None:
            with suppress(Exception):
                return batch_space(base_space, self.num_envs)
        return base_space

    def _clamp_to_env_bounds(self, action: torch.Tensor) -> torch.Tensor:
        # Prefer explicit action bounds from wrapper if available
        bounds = getattr(self.env, "action_bounds", None)
        if callable(bounds):
            bounds = bounds()
        if bounds is not None:
            low, high = bounds
            low_t = torch.as_tensor(low, device=action.device).expand_as(action)
            high_t = torch.as_tensor(high, device=action.device).expand_as(action)
            return torch.clamp(action, low_t, high_t)

        if not isinstance(self.env.action_space, spaces.Box):
            return action
        low = torch.as_tensor(self.env.action_space.low, device=action.device)
        high = torch.as_tensor(self.env.action_space.high, device=action.device)
        finite_low = torch.isfinite(low)
        finite_high = torch.isfinite(high)
        if not finite_low.any() and not finite_high.any():
            return action
        clamped = action
        if finite_low.any():
            clamped = torch.max(clamped, low)
        if finite_high.any():
            clamped = torch.min(clamped, high)
        return clamped

    def step(self, action: torch.Tensor):
        if action.dim() == 1:
            action = action.unsqueeze(0)

        position, velocity = self.get_trajectory(action)
        position, velocity = self.env.set_episode_arguments(action, position, velocity)
        traj_is_valid, position, velocity = self.env.preprocessing_and_validity_callback(
            action, position, velocity, self.tau_bound, self.delay_bound
        )

        batch_size, traj_len = position.shape[0], position.shape[1]
        rewards = torch.zeros((batch_size, traj_len), device=self.device)
        last_info: dict[str, Any] = {}
        step_infos: dict[str, list[Any]] = {}
        terminated = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        truncated = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)

        if not traj_is_valid:
            return self.env.invalid_traj_callback(action, position, velocity, self.tau_bound, self.delay_bound)

        self.plan_steps += 1
        obs = None
        for t in range(traj_len):
            pos_t = position[:, t]
            vel_t = velocity[:, t]
            condition_pos = self.env.get_wrapper_attr("current_pos")
            condition_vel = self.env.get_wrapper_attr("current_vel")
            condition_pos = _ensure_batch(condition_pos, batch_size)
            condition_vel = _ensure_batch(condition_vel, batch_size)

            step_action = self.tracking_controller.get_action(pos_t, vel_t, condition_pos, condition_vel)
            c_action = self._clamp_to_env_bounds(step_action)

            step_result = self.env.step(c_action)
            if len(step_result) == 5:
                obs, c_reward, term, trunc, info = step_result
            else:
                obs, c_reward, term, trunc, info = (
                    step_result[0],
                    step_result[1],
                    step_result[2],
                    step_result[3],
                    step_result[-1],
                )

            # normalize reward shape to (batch_size,)
            c_reward = _ensure_batch(c_reward, batch_size, squeeze_last=True)
            rewards[:, t] = c_reward

            terminated = term if torch.is_tensor(term) else torch.as_tensor(term, device=self.device)
            truncated = trunc if torch.is_tensor(trunc) else torch.as_tensor(trunc, device=self.device)
            terminated = _ensure_batch(terminated, batch_size, squeeze_last=True)
            truncated = _ensure_batch(truncated, batch_size, squeeze_last=True)

            if isinstance(info, dict):
                last_info = info
                for k, v in info.items():
                    step_infos.setdefault(k, []).append(v)

            if self.render_kwargs:
                self.env.render(**self.render_kwargs)

            all_terminated = bool(terminated.all().item()) if torch.is_tensor(terminated) else bool(terminated)
            all_truncated = bool(truncated.all().item()) if torch.is_tensor(truncated) else bool(truncated)

            if (
                all_terminated
                or all_truncated
                or (
                    self.replanning_schedule(
                        self.env.get_wrapper_attr("current_pos"),
                        self.env.get_wrapper_attr("current_vel"),
                        obs if obs is not None else condition_pos,
                        c_action,
                        t + 1 + self.current_traj_steps,
                    )
                    and self.plan_steps < self.max_planning_times
                )
            ):
                if self.condition_on_desired:
                    self.condition_pos = pos_t
                    self.condition_vel = vel_t
                break

        self.current_traj_steps += t + 1
        if self.reward_aggregation:
            try:
                trajectory_return = self.reward_aggregation(rewards[:, : t + 1], dim=1)
            except TypeError:
                trajectory_return = self.reward_aggregation(rewards[:, : t + 1])
        else:
            trajectory_return = rewards[:, : t + 1].sum(dim=1)

        infos_out = dict(last_info) if last_info else {}
        if step_infos:
            infos_out["step_infos"] = step_infos
        infos_out["trajectory_length"] = t + 1

        return self.observation(obs), trajectory_return, terminated, truncated, infos_out

    def render(self, **kwargs):
        self.render_kwargs = kwargs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.current_traj_steps = 0
        self.plan_steps = 0
        self.traj_gen.reset()
        self.condition_pos = None
        self.condition_vel = None
        return super().reset(seed=seed, options=options)
