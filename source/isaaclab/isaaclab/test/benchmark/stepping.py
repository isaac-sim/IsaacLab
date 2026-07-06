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
