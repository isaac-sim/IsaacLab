# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the replay loop stepping past the end of the recorded data.

``replay_episodes_loop`` used to call ``env.step`` once more after every environment had exhausted
its episodes, applying the untouched idle action. For a task-space (IK) task that idle action is
all zeros, i.e. a zero-norm quaternion, which crashed the run after a successful replay.

``scripts/tools/replay_demos.py`` launches the simulator at import time, so the loop function is
extracted from the source and executed against stub objects instead.
"""

import ast
import contextlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumManager, RewardManager, TerminationManager
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

pytestmark = pytest.mark.integration

# This test lives at source/isaaclab/test/cli/test_replay_demos_loop_termination.py.
_REPLAY_DEMOS_PATH = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "replay_demos.py"


def _load_replay_episodes_loop():
    """Compile ``replay_episodes_loop`` from the script source."""
    source = _REPLAY_DEMOS_PATH.read_text()
    tree = ast.parse(source)
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "replay_episodes_loop")
    namespace = {
        "contextlib": contextlib,
        "torch": torch,
        "EpisodeData": EpisodeData,
        "HDF5DatasetFileHandler": HDF5DatasetFileHandler,
        "is_paused": False,
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(_REPLAY_DEMOS_PATH), "exec"), namespace)
    return namespace["replay_episodes_loop"]


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
    replay_episodes_loop = _load_replay_episodes_loop()

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


def test_replay_prepares_success_reward_tasks_without_training_managers(monkeypatch, tmp_path):
    """HDF5 replay starts with success-referencing rewards and retains manual validation."""
    dataset_path = str(tmp_path / "reach.hdf5")
    dataset = HDF5DatasetFileHandler()
    dataset.create(dataset_path, env_name="Isaac-Reach-Franka")
    episode = EpisodeData()
    episode.data = {"actions": torch.zeros(1, 7)}
    dataset.write_episode(episode)
    dataset.close()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_REPLAY_DEMOS_PATH), "--dataset_file", dataset_path, "--validate_success_rate", "--device", "cpu"],
    )
    replay = runpy.run_path(str(_REPLAY_DEMOS_PATH), run_name="replay_demos_test")
    main = replay["main"]
    replayed = []

    def check_replay_environment(cfg, dataset_handler, episode_count, selected_episodes, success):
        env = SimpleNamespace(num_envs=1, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))
        env.termination_manager = TerminationManager(cfg.terminations, env)
        success_reward = cfg.rewards.get("success") if isinstance(cfg.rewards, dict) else cfg.rewards.success
        if success_reward is not None:
            # Exercise the real missing-termination lookup before constructing
            # other reward terms, which would require a physics scene.
            mdp.is_terminated_term(success_reward, env)
        env.reward_manager = RewardManager(cfg.rewards, env)
        env.curriculum_manager = CurriculumManager(cfg.curriculum, env)
        assert not env.termination_manager.compute().any()
        assert env.reward_manager.compute(dt=1.0 / 30.0).eq(0.0).all()
        env.curriculum_manager.compute(torch.tensor([0]))
        assert success is not None
        replayed.append(success)

    monkeypatch.setitem(main.__globals__, "launch_simulation", lambda *_: contextlib.nullcontext())
    monkeypatch.setitem(main.__globals__, "replay_dataset", check_replay_environment)
    main()
    assert replayed
