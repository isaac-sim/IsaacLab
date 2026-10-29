# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the replay loop stepping past the end of the recorded data.

``replay_episodes_loop`` used to call ``env.step`` once more after every environment had exhausted
its episodes, applying the untouched idle action. For a task-space (IK) task that idle action is
all zeros, i.e. a zero-norm quaternion, which crashed the run after a successful replay.

The loop runs against stub objects instead of a simulator.
"""

import pytest
import torch
from isaaclab_teleop.entrypoints.replay_demos import replay_episodes_loop

from isaaclab.utils.datasets import EpisodeData

pytestmark = pytest.mark.integration


class _SimStub:
    def is_running(self):
        return True

    def reset(self):
        pass

    def render(self):
        pass


class _EnvStub:
    """Records the actions passed to :meth:`step`."""

    device = "cpu"

    def __init__(self):
        self.sim = _SimStub()
        self.stepped_actions: list[torch.Tensor] = []

    def reset_to(self, state, env_ids, is_relative=True):
        pass

    def step(self, actions):
        self.stepped_actions.append(actions.clone())


class _DatasetFileHandlerStub:
    def __init__(self, actions: torch.Tensor):
        self._actions = actions

    def load_episode(self, episode_name, device):
        episode = EpisodeData()
        episode.data = {"initial_state": {}, "actions": list(self._actions)}
        return episode


def test_replay_loop_does_not_step_after_the_recorded_actions():
    """The loop steps exactly once per recorded action and never applies the idle action."""
    # absolute task-space actions: [pos_xyz, quat_xyzw, gripper]
    recorded_actions = torch.tensor(
        [
            [0.30, -0.10, 0.20, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.31, -0.10, 0.20, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.32, -0.10, 0.20, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    idle_action = torch.zeros(1, recorded_actions.shape[-1])
    env = _EnvStub()

    replayed_episode_count, _, _ = replay_episodes_loop(
        env,
        _DatasetFileHandlerStub(recorded_actions),
        episode_names=["demo_0"],
        episode_count=1,
        episode_indices_to_replay=[0],
        num_envs=1,
        success_term=None,
        state_validation_enabled=False,
        idle_action=idle_action,
        reset_sim_buffer_each_episode=False,
    )

    assert replayed_episode_count == 1
    assert len(env.stepped_actions) == len(recorded_actions)
    for stepped, recorded in zip(env.stepped_actions, recorded_actions):
        torch.testing.assert_close(stepped, recorded.unsqueeze(0), atol=1e-6, rtol=0.0)
