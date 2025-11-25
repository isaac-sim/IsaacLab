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
    """Walk the wrapper stack to fetch the first available attribute in `names`.

    Args:
        env: Wrapped environment or wrapper instance.
        names (list[str] | tuple[str, ...]): Attribute names to search for, in order.
        default (Any): Value to return when none of the names are present.

    Returns:
        Any: Attribute value from the first match, or `default` if none is found.

    Notes:
        This helper is used to discover observation spaces, devices, or timing fields
        without coupling to a specific wrapper ordering.
    """
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
    """Broadcast or slice a tensor to match the target batch size.

    Args:
        tensor (torch.Tensor): Input value to reshape or replicate.
        target (int): Desired batch dimension size.
        squeeze_last (bool): If `True`, squeeze a trailing singleton dim after aligning.

    Returns:
        torch.Tensor: Tensor with first dimension `target`; values are repeated when needed.

    Notes:
        This keeps batch alignment across MP rollout components. Scalars become shape
        `(1,)`, then repeat to `(target, ...)` when `target>1`. When `target==1` but
        the input batch is larger, only the first entry is used to avoid hidden loops.
    """
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
    """Roll out trajectory generators and controllers against a step-based env.

    The wrapper translates high-level MP parameters into per-step env actions while
    preserving Gym compatibility. It handles batching, clamping to env bounds, reward
    aggregation, and optional replanning. All tensors are expected to live on the
    provided `device` (falls back to `env.device` or CPU).
    """

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
        """Build an MP wrapper around a `RawMPInterface` environment.

        Args:
            env (RawMPInterface): Environment exposing MP hooks and observation mask.
            trajectory_generator: MP implementation with `set_params`, `get_traj_pos`,
                `get_traj_vel`, and `get_params_bounds`. Expects tensors on `device`.
            tracking_controller: Controller with `get_action(des_pos, des_vel, cur_pos, cur_vel)`
                producing step actions shaped `(batch, action_dim)`.
            duration (float): Rollout horizon in seconds when `learn_sub_trajectories` is
                `False`; otherwise the trajectory generator controls duration.
            reward_aggregation (Callable[[torch.Tensor, int], torch.Tensor]): Aggregation
                over per-step rewards; called with `dim=1` when it accepts that argument.
            learn_sub_trajectories (bool): If `True`, the trajectory generator can shorten
                trajectories and is reset every step.
            replanning_schedule (Callable | None): Predicate called as
                `fn(cur_pos, cur_vel, last_obs, last_action, step_idx)` each step. When
                `True`, rollout stops early and `step` returns with current cumulative reward.
            max_planning_times (int): Maximum number of planned segments per episode; caps
                replanning frequency.
            condition_on_desired (bool): If `True`, the next plan conditions on the desired
                state at the stop point rather than the measured `current_*`.
            device (str | torch.device | None): Device for all tensors; defaults to
                `env.device` or CPU.
            verbose (int): Reserved for debugging verbosity.
            **kwargs: Ignored extra arguments for compatibility with registries.

        Notes:
            - The wrapper mirrors `single_action_space` to downstream wrappers so managers
              see the MP parameter space while the env action space remains controller outputs.
            - Observation spaces are masked using `env.context_mask` when sub-trajectory
              learning and replanning are disabled to maintain Fancy Gym parity.
        """
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
        """Convert raw env observations into masked policy observations.

        Args:
            observation: Observation returned by the wrapped env (dict or tensor).

        Returns:
            dict: Dict with key `"policy"` containing the masked observation tensor.

        Notes:
            - If the underlying env returns a dict, `"policy"` is prioritized; otherwise
              the single value is used.
            - When `return_context_observation` is `True`, `context_mask` is applied on
              the last dimension to expose only MP-relevant fields.
        """
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
        """Generate desired position and velocity trajectories from MP parameters.

        Args:
            action (torch.Tensor): MP parameters shaped `(batch, param_dim)`; a 1D vector
                is automatically expanded to batch size 1. When `num_envs>1`, a single
                action is broadcast to all environments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(position, velocity)` each shaped
            `(batch, horizon, dof)` on `self.device`.

        Notes:
            - Parameters are clamped to `traj_gen_action_space` bounds before use.
            - Initial conditions pull `current_pos`/`current_vel` (or `condition_*` when
              `condition_on_desired` was set) and broadcast to the batch.
            - When `learn_sub_trajectories` is `True`, the trajectory generator reset lets
              it emit shorter horizons.
        """
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
        """Build the action space describing valid MP parameters for the trajectory generator."""
        min_action_bounds, max_action_bounds = self.traj_gen.get_params_bounds()
        dtype = min_action_bounds.detach().cpu().numpy().dtype
        return spaces.Box(
            low=min_action_bounds.detach().cpu().numpy(),
            high=max_action_bounds.detach().cpu().numpy(),
            dtype=dtype,
        )

    def _get_observation_space(self):
        """Create the masked single-environment observation space for policy consumption.

        Returns:
            gym.Space | None: Dict space with `"policy"` entry when masking succeeds,
            otherwise the original observation space from the wrapped env.

        Notes:
            - The function prefers `single_observation_space` (IsaacLab convention) and
              falls back to `observation_space`.
            - When masking is enabled, only the policy branch is masked; other dict fields
              remain untouched to preserve logging data for downstream consumers.
        """
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
        """Batch a single-environment observation space if `num_envs` is available.

        Args:
            single_obs_space (gym.Space): Observation space for one environment.

        Returns:
            gym.Space: Batched space via `gymnasium.vector.utils.batch_space` when
            `num_envs` is set; otherwise the input space.
        """
        if single_obs_space is None:
            return single_obs_space
        if self.num_envs is not None:
            with suppress(Exception):
                return batch_space(single_obs_space, self.num_envs)
        return single_obs_space

    def _get_action_space(self):
        """Expose the MP parameter space as the reported Gym action space.

        Returns:
            gym.Space: Batched `Box` matching MP parameters; falls back to the trajectory
            generator bounds when not initialized.

        Notes:
            Downstream RL agents see this as the action space, even though the real env
            receives controller outputs. The per-env batching keeps parity with Gym's
            vector env expectations.
        """
        base_space = getattr(self, "traj_gen_action_space", None)
        if base_space is None:
            base_space = self._get_traj_gen_action_space()
        if self.num_envs is not None:
            with suppress(Exception):
                return batch_space(base_space, self.num_envs)
        return base_space

    def _clamp_to_env_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Clamp controller outputs to the wrapped env bounds.

        Args:
            action (torch.Tensor): Controller action `(batch, action_dim)` on `self.device`.

        Returns:
            torch.Tensor: Action clipped to `action_bounds` or `env.action_space` limits.

        Notes:
            - Explicit `env.action_bounds` (callable or tuple) take precedence over
              `action_space` to allow custom safety limits.
            - When the env bounds are infinite on both sides, actions are returned unchanged.
        """
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
        """Roll out an MP trajectory, feed actions to the env, and aggregate results.

        Args:
            action (torch.Tensor): MP parameters `(batch, param_dim)`; 1D tensors are
                expanded. When only one set is provided and `num_envs>1`, it is broadcast
                to all envs.

        Returns:
            tuple: `(obs, reward, terminated, truncated, info)` where
            - `obs` is a dict `{"policy": masked_obs}` matching `observation_space`,
            - `reward` is aggregated with `reward_aggregation` over the executed steps,
            - `terminated`/`truncated` are broadcast to `(batch,)`,
            - `info` contains the last env info, optional `step_infos` (list per key), and
              `trajectory_length`.

        Flow:
            1. Convert MP parameters to desired trajectories with `get_trajectory`.
            2. Run `set_episode_arguments` then `preprocessing_and_validity_callback`.
               Invalid trajectories call `invalid_traj_callback` immediately.
            3. For each time index:
               - Query `current_pos/vel`, compute controller action, clamp to bounds.
               - Step the env; normalize rewards and done flags to batch.
               - Collect infos; render if requested.
               - Stop early if all done, truncated, or `replanning_schedule` fires
                 (capped by `max_planning_times`). When `condition_on_desired=True`,
                 store the desired state at the stop point for the next plan.
            4. Aggregate rewards with `reward_aggregation(rewards, dim=1)` when supported;
               otherwise call the function without `dim`.

        Error Handling:
            Relies on hooks in `RawMPInterface` to flag invalid trajectories; any exception
            from the env step bubbles up. Reward aggregation falls back to sum when
            `reward_aggregation` is `None`.
        """
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
        """Store render kwargs so `step` can forward them during rollout."""
        self.render_kwargs = kwargs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset cached trajectory state and delegate to the wrapped env."""
        self.current_traj_steps = 0
        self.plan_steps = 0
        self.traj_gen.reset()
        self.condition_pos = None
        self.condition_vel = None
        return super().reset(seed=seed, options=options)
