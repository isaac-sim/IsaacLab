# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for tasks that run on the world-model backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..physics import NeuralManager


def is_chunk_boundary(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the step that just ran completed a model chunk, as a boolean tensor of shape ``(num_envs,)``.

    Environments can only reset on these steps, and only then do new frames exist.
    """
    manager: NeuralManager = env.sim.backend_manager
    return manager.chunk_boundary


def check_chunk_timing(env: ManagerBasedRLEnv) -> None:
    """Check that environment steps fall on chunk boundaries.

    Raises:
        ValueError: If the decimation does not divide the chunk length.
    """
    chunk = env.sim.backend_manager.spec.chunk_length
    decimation = env.cfg.decimation
    if chunk % decimation:
        raise ValueError(f"decimation={decimation} must divide the world model's chunk of {chunk} frames.")


def at_chunk_boundary(
    env: ManagerBasedRLEnv, term: Callable[..., torch.Tensor], term_params: dict[str, Any] | None = None
) -> torch.Tensor:
    """Evaluate a termination function and let it fire only on chunk boundaries.

    A termination in the middle of a chunk would reset the environment where it cannot be reset. The term is judged on
    the step that completes a chunk and ignored in between: a condition that stays true (a success, a failure or the
    time limit) is caught at the next boundary, at most one chunk late, and one that holds only briefly inside a chunk
    is missed. Use it as the function of a :class:`~isaaclab.managers.TerminationTermCfg`, with the condition as a
    parameter, so the configuration stays serializable::

        done = DoneTerm(func=at_chunk_boundary, params={"term": my_condition, "term_params": {...}})
        time_out = DoneTerm(func=at_chunk_boundary, params={"term": mdp.time_out}, time_out=True)

    With the time-out on boundaries, episodes can have any length, including the random first episode lengths some
    trainers use (rsl_rl's ``init_at_random_ep_len``): an episode ends at the first boundary after its time limit.

    Args:
        env: The environment.
        term: A termination function ``term(env, **term_params)`` returning a boolean tensor of shape ``(num_envs,)``.
        term_params: Keyword arguments of ``term``, passed as given. Defaults to none.

    Raises:
        ValueError: If the environment's timing does not fit the chunks, see :func:`check_chunk_timing`.
    """
    check_chunk_timing(env)
    return term(env, **(term_params or {})) & is_chunk_boundary(env)
