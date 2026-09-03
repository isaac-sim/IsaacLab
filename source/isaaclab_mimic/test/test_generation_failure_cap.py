# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``max_num_failures`` bound in the Mimic generation loop.

``env_loop`` is driven with a fake environment whose ``step`` consumes one scripted attempt outcome
per call and feeds the action queue back, so the loop's own termination logic runs unmodified
without a simulator, data generator, or dataset. A step-count fuse turns an unbounded loop into a
test failure instead of a hang.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import asyncio
from types import SimpleNamespace

import pytest
import torch

from isaaclab_mimic.datagen import generation

FUSE_STEPS = 40


class _Fuse(Exception):
    """Raised by the fake environment once the loop has run for longer than any test expects."""


def _run(
    outcomes,
    *,
    max_num_failures,
    generation_num_trials=3,
    generation_guarantee=True,
    num_envs=1,
    attempts_per_step=1,
):
    """Run ``env_loop`` over scripted attempt outcomes; return how it ended and the final counters.

    ``attempts_per_step`` is how many of the ``num_envs`` generators finish an attempt on the same
    step, which is what decides whether the bound is read before or after the extra attempts land.

    Returns:
        A tuple ``(how, num_success, num_failures, num_attempts)`` where ``how`` is ``"exited"`` if
        the loop returned on its own and ``"fuse"`` if it was still running after ``FUSE_STEPS``.
    """
    generation.num_success = generation.num_failures = generation.num_attempts = 0
    loop = asyncio.get_event_loop()
    action_queue: asyncio.Queue = asyncio.Queue()
    reset_queue: asyncio.Queue = asyncio.Queue()
    for env_id in range(num_envs):
        action_queue.put_nowait((env_id, torch.zeros(7)))
    scripted = iter(outcomes)
    steps = {"n": 0}

    def step(actions):
        steps["n"] += 1
        if steps["n"] > FUSE_STEPS:
            raise _Fuse(f"env_loop still running after {FUSE_STEPS} steps")
        for _ in range(attempts_per_step):
            if next(scripted):
                generation.num_success += 1
            else:
                generation.num_failures += 1
            generation.num_attempts += 1
        for env_id in range(num_envs):
            action_queue.put_nowait((env_id, torch.zeros(7)))

    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        action_space=SimpleNamespace(shape=(num_envs, 7)),
        step=step,
        reset=lambda env_ids=None: None,
        close=lambda: None,
        sim=SimpleNamespace(is_stopped=lambda: False),
        cfg=SimpleNamespace(
            datagen_config=SimpleNamespace(
                generation_guarantee=generation_guarantee,
                generation_num_trials=generation_num_trials,
                max_num_failures=max_num_failures,
            )
        ),
    )
    try:
        generation.env_loop(env, reset_queue, action_queue, None, loop)
        how = "exited"
    except _Fuse:
        how = "fuse"
    return how, generation.num_success, generation.num_failures, generation.num_attempts


def test_failure_cap_stops_a_run_that_never_succeeds():
    how, succ, fail, attempts = _run([False] * 100, max_num_failures=5)
    assert how == "exited"
    assert (succ, fail, attempts) == (0, 5, 5)


def test_no_cap_by_default_keeps_the_success_guarantee():
    """``None`` means what the guarantee always meant: keep going until enough demos succeed."""
    how, _, fail, _ = _run([False] * 100, max_num_failures=None)
    assert how == "fuse"
    assert fail == FUSE_STEPS


def test_enough_successes_still_end_the_run_first():
    outcomes = [False, True, False, True, True] + [False] * 50
    how, succ, fail, _ = _run(outcomes, max_num_failures=100, generation_num_trials=3)
    assert how == "exited"
    assert (succ, fail) == (3, 2)


def test_cap_reached_before_enough_successes_ends_the_run():
    outcomes = [False, True, False, True, False, False] + [True] * 50
    how, succ, fail, _ = _run(outcomes, max_num_failures=4, generation_num_trials=3)
    assert how == "exited"
    assert (succ, fail) == (2, 4)


@pytest.mark.parametrize("max_num_failures", [None, 3, 100])
def test_attempt_based_termination_is_unchanged(max_num_failures):
    """With the guarantee off the run stops on ``generation_num_trials`` attempts, cap or no cap.

    A cap of 3 against 10 requested attempts is the case that matters: the fixed-attempt contract
    says the run delivers the attempts it was asked for, so the cap must not end it at 3.
    """
    how, _, _, attempts = _run(
        [False] * 100, max_num_failures=max_num_failures, generation_num_trials=10, generation_guarantee=False
    )
    assert how == "exited"
    assert attempts == 10


def test_bound_is_exact_when_attempts_end_on_separate_steps():
    """Four environments, one attempt landing per step: the bound is read between every attempt."""
    how, _, fail, _ = _run([False] * 100, max_num_failures=5, num_envs=4, attempts_per_step=1)
    assert how == "exited"
    assert fail == 5


def test_attempts_ending_together_overshoot_the_bound_by_at_most_num_envs_minus_one():
    """Attempts that end on the step that crosses the bound are already complete and still count."""
    how, _, fail, _ = _run([False] * 100, max_num_failures=5, num_envs=4, attempts_per_step=4)
    assert how == "exited"
    assert 5 < fail <= 5 + 4 - 1
