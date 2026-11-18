# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations of the environments that run an MDP2 task on an interactive scene."""

from __future__ import annotations

from dataclasses import MISSING, replace

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_experimental.mdp2 import Backend, TaskCfg


@configclass
class MDP2EnvCfg:
    """Configuration for :class:`~isaaclab_experimental.envs.MDP2Env`."""

    sim: SimulationCfg = SimulationCfg()
    scene: InteractiveSceneCfg = MISSING
    # Actions, commands, observations, rewards, terminations, and events; a list splits the worlds between tasks
    task: TaskCfg | list[TaskCfg] = MISSING
    task_worlds: list[int] | None = None  # Number of worlds of each task of a list, in order; None for even shares
    decimation: int = MISSING  # Physics steps per control step
    backend: Backend = Backend.WARP  # How the task's term implementations run
    seed: int | None = None
    capture: bool = True  # Replay each step, physics included, as one CUDA graph when the MDP allows it


@configclass
class MDP2RLEnvCfg(MDP2EnvCfg):
    """Configuration for :class:`~isaaclab_experimental.envs.MDP2RLEnv`."""

    is_finite_horizon: bool = False  # Whether time-outs are terminal states rather than truncations
    # Whether step returns, in extras["final_obs"], the observations of the worlds that ended before their reset
    compute_final_obs: bool = False
    log_dir: str | None = None  # Set by the training scripts
    video_recorders: list = []  # Not supported yet; kept for the training scripts

    def play_mode(self) -> None:
        """Adjust the configuration for playing a trained policy, as MDP1 does: at most 50 worlds, and observations
        without noise. The play scripts call it; tasks may extend it."""
        if self.task_worlds is None:
            self.scene.num_envs = min(self.scene.num_envs, 50)
        tasks = [self.task] if isinstance(self.task, TaskCfg) else self.task
        tasks = [_without_observation_noise(task) for task in tasks]
        self.task = tasks[0] if isinstance(self.task, TaskCfg) else tasks


def _without_observation_noise(task: TaskCfg) -> TaskCfg:
    """Return a copy of a task whose observations have no noise; the uses it shares with others are not changed."""
    observations = {
        group: {name: use if use is None else replace(use, noise=None) for name, use in uses.items()}
        for group, uses in task.observations.items()
    }
    return replace(task, observations=observations)
