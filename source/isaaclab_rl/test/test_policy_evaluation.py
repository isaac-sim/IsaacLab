# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode evaluation excludes autoresets and resets recurrent policy state."""

from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.backends import play_rsl_rl

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_score_counts_each_first_episode_once():
    import torch

    step = 0
    resets = []
    termination = SimpleNamespace(
        active_terms=["failure", "timeout"],
        get_term=lambda name: (
            torch.tensor([step == 1, False, False]) if name == "failure" else torch.tensor([False, step == 2, False])
        ),
    )
    task = SimpleNamespace(num_envs=3, device="cpu", max_episode_length=3, step_dt=0.1, termination_manager=termination)

    class Env:
        unwrapped = task

        def reset(self):
            return {}, {}

        def step(self, actions):
            nonlocal step
            step += 1
            return {}, torch.tensor([1.0 if step == 1 else 100.0, 2.0, 3.0]), torch.tensor([True, step >= 2, False]), {}

    class Policy:
        def __call__(self, obs):
            return torch.zeros(3, 1)

        def reset(self, dones):
            resets.append(dones.clone())

    count, reward, seconds, endings = play_rsl_rl.score_policy(Env(), Policy())
    assert count == 2
    assert reward == pytest.approx(2.5)
    assert seconds == pytest.approx(0.15)
    assert endings == {"failure": 0.5, "timeout": 0.5}
    assert len(resets) == 3
