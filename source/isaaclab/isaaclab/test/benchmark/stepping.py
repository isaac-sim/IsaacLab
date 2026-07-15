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


def run_runtime_loop(env, num_frames: int) -> list[float]:
    """Step the environment ``num_frames`` times and record per-step wall times [s].

    Calls ``env.reset()`` once before the loop, then on each frame samples
    random actions via :func:`sample_random_actions`, steps the environment,
    and records the elapsed wall-clock time for that step.

    Args:
        env: A Gym-compatible environment.
        num_frames: Number of environment steps to run.

    Returns:
        A list of length ``num_frames`` containing per-step wall times [s].
    """
    env.reset()

    step_times: list[float] = []

    for _ in range(num_frames):
        actions = sample_random_actions(env)
        t0 = time.perf_counter_ns()
        env.step(actions)
        t1 = time.perf_counter_ns()
        step_times.append((t1 - t0) / 1e9)

    return step_times


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
