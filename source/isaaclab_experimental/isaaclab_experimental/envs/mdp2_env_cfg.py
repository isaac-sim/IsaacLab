# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations of the environments that run an MDP2 task on an interactive scene."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import MISSING, dataclass, replace
from typing import Any

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_experimental.mdp2 import Backend, TaskCfg


class StagePoint(enum.Enum):
    """Where a :class:`Stage` runs in a control step, which applies the actions, steps the physics, evaluates the MDP,
    resets the worlds that ended, and observes them."""

    BEFORE_PHYSICS = enum.auto()  # Before the actions are applied and the physics steps
    AFTER_PHYSICS = enum.auto()  # After the physics steps, before the MDP is evaluated
    BEFORE_RESET = enum.auto()  # After the MDP is evaluated, before the worlds that ended are reset
    AFTER_RESET = enum.auto()  # After the reset events and their commit, before the reset worlds are observed
    AFTER_OBSERVE = enum.auto()  # At the end of the step


@dataclass(frozen=True)
class Stage:
    """User work at a point of every control step, such as kernels that update a scene field the MDP reads.

    ``fn(env)`` may read and write anything, through the environment, its scene, and its MDP state. A capturable stage
    runs inside the step's CUDA graph: it is called once, at capture, and its launches replay every step, so it must
    not read values on the host or draw from Torch's random generators, and Python branches take the path they took
    at capture. Check it with
    ``MDP2EnvCfg.verify_capture``. A stage that is not capturable runs on the host between the graphs. Stages at
    :attr:`StagePoint.AFTER_RESET` also run in :meth:`~isaaclab_experimental.envs.MDP2Env.reset`.
    """

    fn: Callable[[Any], None]
    at: StagePoint
    capturable: bool = True


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
    stages: list[Stage] = []  # User work at points of every step, in order
    # Debug: before building, step the task this many times uncaptured and captured, and raise where they differ,
    # naming the value; see verify_capture. 0 to skip.
    verify_capture: int = 0


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
