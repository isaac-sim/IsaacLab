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


def _run(outcomes, *, max_num_failures, generation_num_trials=3, generation_guarantee=True):
    """Run ``env_loop`` over scripted attempt outcomes; return how it ended and the final counters.

    Returns:
        A tuple ``(how, num_success, num_failures, num_attempts)`` where ``how`` is ``"exited"`` if
        the loop returned on its own and ``"fuse"`` if it was still running after ``FUSE_STEPS``.
    """
    generation.num_success = generation.num_failures = generation.num_attempts = 0
    loop = asyncio.get_event_loop()
    action_queue: asyncio.Queue = asyncio.Queue()
    reset_queue: asyncio.Queue = asyncio.Queue()
    action_queue.put_nowait((0, torch.zeros(7)))
    scripted = iter(outcomes)
    steps = {"n": 0}

    def step(actions):
        steps["n"] += 1
        if steps["n"] > FUSE_STEPS:
            raise _Fuse(f"env_loop still running after {FUSE_STEPS} steps")
        if next(scripted):
            generation.num_success += 1
        else:
            generation.num_failures += 1
        generation.num_attempts += 1
        action_queue.put_nowait((0, torch.zeros(7)))

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        action_space=SimpleNamespace(shape=(1, 7)),
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
