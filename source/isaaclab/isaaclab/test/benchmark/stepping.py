# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic random-action stepping helpers for benchmarks.

This module is intentionally lightweight: ``torch`` and ``numpy`` are
imported lazily inside each function so that importing this module has
no heavy-weight side effects.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .schema import MeanStd


def sample_random_actions(env) -> torch.Tensor | dict[str, torch.Tensor]:
    """Sample random actions for a single-agent or multi-agent environment.

    For multi-agent environments (those where ``env.unwrapped`` exposes an
    ``action_spaces`` attribute), one batch of actions is sampled per agent
    using that agent's action space.  For single-agent environments a uniform
    sample in [-1, 1] is returned.

    Args:
        env: A Gym-compatible environment wrapper.  ``env.unwrapped`` must
            expose ``num_envs`` and ``device``, plus either ``action_spaces``
            (multi-agent) or ``single_action_space`` (single-agent).

    Returns:
        A ``torch.Tensor`` of shape ``(num_envs, action_dim)`` for
        single-agent environments, or a ``dict`` mapping agent name to a
        tensor of the same shape for multi-agent environments.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    u = env.unwrapped

    if hasattr(u, "action_spaces"):
        # Multi-agent: sample each agent's action space independently.
        return {
            agent: torch.as_tensor(
                np.stack([space.sample() for _ in range(u.num_envs)]),
                dtype=torch.float32,
                device=u.device,
            )
            for agent, space in u.action_spaces.items()
        }
    else:
        # Single-agent: uniform random actions in [-1, 1].
        return 2.0 * torch.rand(u.num_envs, u.single_action_space.shape[0], device=u.device) - 1.0


def _find_cuda_devices(value) -> set[str]:
    """Collect CUDA device names from nested tensor-like values."""
    import torch  # noqa: PLC0415

    devices: set[str] = set()

    def collect(item) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                collect(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)
            return
        device = item if isinstance(item, (str, torch.device)) else getattr(item, "device", None)
        if device is None:
            return
        try:
            device = torch.device(device)
        except (RuntimeError, TypeError):
            return
        if device.type == "cuda":
            devices.add(str(device))

    collect(value)
    return devices


class EnvironmentStepTimingRecorder(AbstractContextManager):
    """Record host-return step time or an optional synchronized step breakdown.

    By default, this context manager records host wall time until ``env.step()``
    returns without forcing queued device work to complete. When synchronized
    breakdown is requested, it also intercepts ``SimulationContext.step()``,
    drains pending work before every measured boundary, and uses
    :class:`isaaclab.utils.timer.Timer` to synchronize completion.

    The synchronized mode serializes execution and perturbs throughput. Its
    arithmetic remainder is time outside simulation calls in that instrumented
    schedule. The remainder includes required task, manager, reset, wrapper,
    and synchronization work; it is not removable Isaac Lab overhead.

    Args:
        env: Environment interface whose ``step`` method is called by the workload.
        measure_synchronized_step_breakdown: Whether to collect the serialized
            synchronized simulation and outside-simulation breakdown.
        warmup_steps: Number of initial ``env.step()`` calls to exclude from the
            recorded timings. Drops one-time cold-start costs (e.g. CUDA graph
            capture and lazy kernel compilation) so the aggregate reflects
            steady-state stepping. Defaults to ``0`` (record every step).
    """

    def __init__(self, env, *, measure_synchronized_step_breakdown: bool = False, warmup_steps: int = 0):
        self._env = env
        self._measure_synchronized_step_breakdown = measure_synchronized_step_breakdown
        self._warmup_steps = warmup_steps
        self._simulation_context = env.unwrapped.sim if measure_synchronized_step_breakdown else None
        self._had_env_instance_step = "step" in vars(env)
        self._env_instance_step = vars(env).get("step")
        self._had_sim_instance_step = measure_synchronized_step_breakdown and "step" in vars(self._simulation_context)
        self._sim_instance_step = (
            vars(self._simulation_context).get("step") if measure_synchronized_step_breakdown else None
        )
        self._original_env_step = None
        self._original_sim_step = None
        self._simulation_total_time_s = 0.0
        self._simulation_step_calls = 0
        self._inside_environment_step = False
        self._environment_step_index = 0
        self.step_times_s: list[float] = []
        self.simulation_step_times_s: list[float] | None = [] if measure_synchronized_step_breakdown else None

    @property
    def simulation_step_calls(self) -> int | None:
        """Number of measured simulation-step calls."""
        return self._simulation_step_calls if self._measure_synchronized_step_breakdown else None

    def __enter__(self) -> EnvironmentStepTimingRecorder:
        """Install the recording wrappers and reset measurements."""
        if self._original_env_step is not None:
            raise RuntimeError("EnvironmentStepTimingRecorder is already active")

        self.step_times_s.clear()
        self._environment_step_index = 0
        self._original_env_step = self._env.step

        if self._measure_synchronized_step_breakdown:
            import torch  # noqa: PLC0415
            import warp as wp  # noqa: PLC0415

            from isaaclab.utils.timer import Timer  # noqa: PLC0415

            assert self.simulation_step_times_s is not None
            self.simulation_step_times_s.clear()
            self._inside_environment_step = False
            self._simulation_total_time_s = 0.0
            self._simulation_step_calls = 0
            assert self._simulation_context is not None
            self._original_sim_step = self._simulation_context.step
            environment_cuda_devices = _find_cuda_devices(getattr(self._env.unwrapped, "device", None))
            active_cuda_devices = environment_cuda_devices

            def synchronize_torch(devices: set[str]) -> None:
                for device in sorted(devices):
                    torch.cuda.synchronize(device)

            def timed_simulation_step(*args, **kwargs):
                if not self._inside_environment_step:
                    return self._original_sim_step(*args, **kwargs)
                synchronize_torch(active_cuda_devices)
                wp.synchronize()
                timer = Timer()
                timer.start()
                try:
                    return self._original_sim_step(*args, **kwargs)
                finally:
                    synchronize_torch(active_cuda_devices)
                    timer.stop()
                    self._simulation_total_time_s += timer.total_run_time
                    self._simulation_step_calls += 1

            self._simulation_context.step = timed_simulation_step

            def timed_environment_step(*args, **kwargs):
                nonlocal active_cuda_devices
                recording = self._environment_step_index >= self._warmup_steps
                self._environment_step_index += 1
                simulation_start_time_s = self._simulation_total_time_s
                simulation_start_calls = self._simulation_step_calls
                previous_cuda_devices = active_cuda_devices
                active_cuda_devices = environment_cuda_devices | _find_cuda_devices(args) | _find_cuda_devices(kwargs)
                synchronize_torch(active_cuda_devices)
                wp.synchronize()
                timer = Timer()
                timer.start()
                self._inside_environment_step = True
                try:
                    return self._original_env_step(*args, **kwargs)
                finally:
                    self._inside_environment_step = False
                    synchronize_torch(active_cuda_devices)
                    timer.stop()
                    active_cuda_devices = previous_cuda_devices
                    if recording:
                        self.step_times_s.append(timer.total_run_time)
                        self.simulation_step_times_s.append(self._simulation_total_time_s - simulation_start_time_s)
                    else:
                        # Discard warmup-step accounting so recorded calls and times stay consistent.
                        self._simulation_total_time_s = simulation_start_time_s
                        self._simulation_step_calls = simulation_start_calls

        else:

            def timed_environment_step(*args, **kwargs):
                recording = self._environment_step_index >= self._warmup_steps
                self._environment_step_index += 1
                start_time_ns = time.perf_counter_ns()
                try:
                    return self._original_env_step(*args, **kwargs)
                finally:
                    if recording:
                        self.step_times_s.append((time.perf_counter_ns() - start_time_ns) / 1e9)

        self._env.step = timed_environment_step
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Restore the original methods."""
        if self._original_env_step is not None:
            if self._had_env_instance_step:
                self._env.step = self._env_instance_step
            else:
                del self._env.step
            self._original_env_step = None
        if self._original_sim_step is not None:
            if self._had_sim_instance_step:
                self._simulation_context.step = self._sim_instance_step
            else:
                del self._simulation_context.step
            self._original_sim_step = None


def run_runtime_loop(env, num_frames: int, *, reset: bool = True) -> list[float]:
    """Step the environment ``num_frames`` times and record per-step wall times [s].

    Optionally calls ``env.reset()`` once before the loop, then on each frame
    samples random actions via :func:`sample_random_actions`, steps the
    environment, and records the elapsed wall-clock time for that step.

    Args:
        env: A Gym-compatible environment.
        num_frames: Number of environment steps to run.
        reset: Whether to reset the environment before stepping.

    Returns:
        A list of length ``num_frames`` containing per-step wall times [s].
    """
    if reset:
        env.reset()

    step_times: list[float] = []

    for _ in range(num_frames):
        actions = sample_random_actions(env)
        t0 = time.perf_counter_ns()
        env.step(actions)
        t1 = time.perf_counter_ns()
        step_times.append((t1 - t0) / 1e9)

    return step_times


def run_runtime_warmup(env, num_frames: int) -> list[float]:
    """Run exactly ``num_frames`` excluded warmup steps.

    Args:
        env: A Gym-compatible environment.
        num_frames: Requested number of warmup environment steps.

    Returns:
        Per-step wall times [s] for the requested excluded steps.
    """
    return run_runtime_loop(env, num_frames)


def _extract_success(extras) -> float | None:
    """Pull a scalar success value out of a step's ``extras`` mapping.

    Scans the ``"log"`` and ``"episode"`` sub-dicts of *extras* for the first
    key whose name contains ``"success"`` (case-insensitive) and returns its
    value as a ``float`` (calling ``.item()`` when the value is a tensor).
    Returns ``None`` when *extras* is not a dict (e.g. the per-env list of info
    dicts that Stable-Baselines3 vec envs return), since no single scannable
    mapping is available in that case.

    Args:
        extras: The per-step ``extras``/``info`` value returned by ``env.step``.

    Returns:
        The success value as a ``float``, or ``None`` when no success key is
        present.
    """
    if not isinstance(extras, dict):
        return None
    for sub_key in ("log", "episode"):
        sub = extras.get(sub_key)
        if not isinstance(sub, dict):
            continue
        for key, value in sub.items():
            if "success" in key.lower():
                return float(value.item()) if hasattr(value, "item") else float(value)
    return None


def run_play_loop(env, policy, num_frames: int) -> tuple[list[float], MeanStd | None, MeanStd | None, float | None]:
    """Roll out *policy* in *env* for *num_frames* steps and aggregate episode metrics.

    Resets the environment, then on each frame runs the policy under
    ``torch.inference_mode()`` and steps the environment, recording the
    per-step wall time [s].  Per-environment returns and lengths are accumulated
    and, whenever an environment signals ``done``, that episode's return,
    length, and (if present) success value are recorded and the environment's
    accumulators are reset.

    Both the four-tuple ``(obs, reward, dones, extras)`` and the Gym five-tuple
    ``(obs, reward, terminated, truncated, info)`` step signatures are accepted;
    for the latter ``dones`` is ``terminated | truncated`` and ``extras`` is
    ``info``.  Rewards and dones are coerced via ``torch.as_tensor`` so NumPy
    returns (e.g. from Stable-Baselines3) work as well.

    Args:
        env: A Gym-compatible environment whose ``unwrapped`` exposes
            ``num_envs`` and ``device``.
        policy: Callable mapping an observation batch to an action batch.
        num_frames: Number of environment steps to run.

    Returns:
        A tuple ``(step_times, reward, ep_length, success_rate)`` where
        ``step_times`` is the per-step wall times [s], ``reward`` and
        ``ep_length`` are :class:`~isaaclab.test.benchmark.schema.MeanStd`
        aggregates over completed episodes (or ``None`` if none completed), and
        ``success_rate`` is the mean of collected success values rounded to four
        decimals (or ``None`` if none were reported).
    """
    import torch  # noqa: PLC0415

    from isaaclab.test.benchmark.metrics import mean_std_peak  # noqa: PLC0415

    u = env.unwrapped
    num_envs = u.num_envs
    device = u.device

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    running_return = torch.zeros(num_envs, device=device)
    running_length = torch.zeros(num_envs, device=device)

    step_times: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    successes: list[float] = []

    for _ in range(num_frames):
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            actions = policy(obs)
        result = env.step(actions)
        t1 = time.perf_counter_ns()
        step_times.append((t1 - t0) / 1e9)

        if len(result) == 5:
            obs, reward, terminated, truncated, extras = result
            dones = torch.as_tensor(terminated, device=device) | torch.as_tensor(truncated, device=device)
        else:
            obs, reward, dones, extras = result
            dones = torch.as_tensor(dones, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)

        # Flatten reward/done to (num_envs,) — some wrappers (e.g. skrl) return shape (num_envs, 1).
        reward = reward.reshape(num_envs)
        dones = dones.reshape(num_envs)

        running_return += reward
        running_length += 1.0

        done_mask = dones.to(torch.bool)
        if bool(done_mask.any()):
            success_value = _extract_success(extras)
            for env_idx in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                episode_returns.append(float(running_return[env_idx].item()))
                episode_lengths.append(float(running_length[env_idx].item()))
                if success_value is not None:
                    successes.append(success_value)
                running_return[env_idx] = 0.0
                running_length[env_idx] = 0.0

    reward_agg = mean_std_peak(episode_returns) if episode_returns else None
    ep_length_agg = mean_std_peak(episode_lengths) if episode_lengths else None
    success_rate = round(sum(successes) / len(successes), 4) if successes else None

    return step_times, reward_agg, ep_length_agg, success_rate
