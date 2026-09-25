# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU unit tests for event dispatch without a simulator."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import EventManager, EventTermCfg

pytestmark = pytest.mark.unit


def count_resets(env, env_ids):
    """Count how often each environment receives a reset event."""
    env.counts[env_ids] += 1


@pytest.mark.parametrize("selector", ["list", "tuple", "tensor", "none", "empty_list", "empty_tuple"])
@pytest.mark.parametrize("min_steps", [0, 5])
def test_reset_environment_selectors(selector, min_steps):
    """Sequence selectors obey the same per-environment trigger intervals as tensors."""
    env = SimpleNamespace(num_envs=3, device="cpu", sim=SimpleNamespace(is_playing=lambda: True), counts=torch.zeros(3))
    manager = EventManager(
        {"count": EventTermCfg(func=count_resets, mode="reset", min_step_count_between_reset=min_steps)}, env
    )
    # Trigger one environment first so that a later subset has mixed eligibility.
    manager.apply("reset", env_ids=torch.tensor([0]), global_env_step_count=0)
    env_ids = {
        "list": [0, 2],
        "tuple": (0, 2),
        "tensor": torch.tensor([0, 2]),
        "none": None,
        "empty_list": [],
        "empty_tuple": (),
    }[selector]
    selected = [] if selector.startswith("empty") else ([0, 1, 2] if selector == "none" else [0, 2])
    expected = [1.0, 0.0, 0.0]
    last_triggered = [0, None, None]
    for step in (1, 2, 5, 6):
        manager.apply("reset", env_ids=env_ids, global_env_step_count=step)
        for index in selected:
            if last_triggered[index] is None or step - last_triggered[index] >= min_steps:
                expected[index] += 1
                last_triggered[index] = step
        torch.testing.assert_close(env.counts, torch.tensor(expected))
