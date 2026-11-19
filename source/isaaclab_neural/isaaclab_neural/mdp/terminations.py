# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for tasks that run on the world-model backend."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..physics import WorldModelManager


def is_chunk_boundary(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the step that just ran completed a model chunk, as a boolean tensor of shape ``(num_envs,)``.

    Environments can only reset on these steps, and only then do new frames exist.
    """
    manager: WorldModelManager = env.sim.backend_manager
    return manager.chunk_boundary


def check_chunk_timing(env: ManagerBasedRLEnv) -> None:
    """Check that environment steps and time-outs fall on chunk boundaries.

    Raises:
        ValueError: If the decimation does not divide the chunk length, or an episode is not a whole number of chunks.
    """
    chunk = env.sim.backend_manager.spec.chunk_length
    decimation = env.cfg.decimation
    if chunk % decimation:
        raise ValueError(f"decimation={decimation} must divide the world model's chunk of {chunk} frames.")
    if env.max_episode_length * decimation % chunk:
        raise ValueError(
            f"An episode of {env.max_episode_length} steps is not a whole number of {chunk // decimation}-step chunks,"
            " so its time-out would fall inside a chunk; adjust episode_length_s."
        )


def at_chunk_boundary(term: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Make a termination function fire only on chunk boundaries.

    A termination in the middle of a chunk would reset the environment where it cannot be reset. The wrapped term is
    judged on the step that completes a chunk and ignored in between: a condition that stays true (a success or a
    failure state) is caught at the next boundary, at most one chunk late, and one that holds only briefly inside a
    chunk is missed. Use it on the function of a :class:`~isaaclab.managers.TerminationTermCfg`::

        done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})

    The first call checks the environment's timing with :func:`check_chunk_timing`.

    Args:
        term: A termination function ``term(env, **params)`` returning a boolean tensor of shape ``(num_envs,)``.
    """
    checked = set()

    @functools.wraps(term)
    def wrapped(env: ManagerBasedRLEnv, *args, **kwargs) -> torch.Tensor:
        if id(env) not in checked:
            check_chunk_timing(env)
            checked.add(id(env))
        return term(env, *args, **kwargs) & is_chunk_boundary(env)

    return wrapped
