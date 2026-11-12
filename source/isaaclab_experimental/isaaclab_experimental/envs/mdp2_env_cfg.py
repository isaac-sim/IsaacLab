# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations of the environments that run an MDP2 task on an interactive scene."""

from __future__ import annotations

from dataclasses import MISSING

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
    log_dir: str | None = None  # Set by the training scripts
    video_recorders: list = []  # Not supported yet; kept for the training scripts
