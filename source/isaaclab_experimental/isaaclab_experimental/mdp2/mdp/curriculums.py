# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.curriculums`.

A curriculum is a reset event that keeps its progress in persistent state, which other terms read through
:func:`~isaaclab_experimental.mdp2.term_state`, for example to scale observation noise or to move a bound.
"""

from __future__ import annotations

import torch
import warp as wp

from ..terms import State, Term


@wp.kernel
def _promote(
    mask: wp.array(dtype=wp.bool),
    succeeded: wp.array(dtype=float),
    difficulty: wp.array(dtype=float),
    total: wp.array(dtype=float),
    min_difficulty: float,
    max_difficulty: float,
    demotion: float,
):
    world = wp.tid()
    if not mask[world]:
        return

    step = wp.where(succeeded[world] > 0.5, 1.0, demotion)
    promoted = wp.clamp(difficulty[world] + step, min_difficulty, max_difficulty)
    # Difficulties are whole numbers, so their sum over the worlds is exact in any order.
    wp.atomic_add(total, 0, promoted - difficulty[world])
    difficulty[world] = promoted


@wp.kernel
def _set_progress(total: wp.array(dtype=float), progress: wp.array(dtype=float), scale: float, min_progress: float):
    world = wp.tid()
    mean = total[0] * scale
    if mean >= min_progress:
        progress[world] = mean


def _difficulty_scheduler(
    mask,
    succeeded,
    difficulty,
    progress,
    total,
    *,
    min_difficulty=0,
    max_difficulty=50,
    promotion_only=False,
    min_progress=0.1,
):
    num_worlds, device = difficulty.shape[0], difficulty.device
    wp.launch(
        _promote,
        dim=num_worlds,
        inputs=[mask, succeeded, difficulty, total, min_difficulty, max_difficulty, 0.0 if promotion_only else -1.0],
        device=device,
    )
    scale = 1.0 / (num_worlds * max(max_difficulty, 1))
    wp.launch(_set_progress, dim=num_worlds, inputs=[total, progress, scale, min_progress], device=device)


def _difficulty_scheduler_torch(
    mask,
    succeeded,
    difficulty,
    progress,
    *,
    min_difficulty=0,
    max_difficulty=50,
    promotion_only=False,
    min_progress=0.1,
):
    step = torch.where(succeeded > 0.5, 1.0, 0.0 if promotion_only else -1.0)
    promoted = (difficulty + step).clamp(min_difficulty, max_difficulty)
    difficulty.copy_(torch.where(mask, promoted, difficulty))

    mean = difficulty.mean() / max(max_difficulty, 1)
    progress.copy_(torch.where(mean >= min_progress, mean, progress))


difficulty_scheduler = Term(
    torch=_difficulty_scheduler_torch,
    warp=_difficulty_scheduler,
    state={
        "difficulty": State(persistent=True),
        "progress": State(persistent=True),
        "total": State(shape=(1,)),  # Sum of the difficulties, which the Warp kernels keep to take their mean
    },
)
"""Adaptive difficulty, like MDP1's ``DifficultyScheduler`` with ``initial_final_interpolate_fn``.

Each world's difficulty rises by one when the episode that ended ``succeeded`` (an input, such as a sticky success
quantity) and falls by one otherwise, unless ``promotion_only``, within ``[min_difficulty, max_difficulty]``.
Its ``progress`` state is the mean difficulty over ``max_difficulty``, once that reaches ``min_progress``: dependent
terms interpolate between their initial and final values by it, and keep their initial values until then. The same in
every world.
"""
