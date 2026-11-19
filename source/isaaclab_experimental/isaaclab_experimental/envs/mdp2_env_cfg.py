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


class Step(enum.Enum):
    """Built-in work of a control step, in :attr:`MDP2EnvCfg.step`."""

    PHYSICS = enum.auto()  # Apply the actions and step the physics ``decimation`` times
    EVALUATE = enum.auto()  # Evaluate the MDP: commands, quantities, terminations, rewards, observations
    RESET = enum.auto()  # Reset the worlds whose episodes ended, by :attr:`MDP2EnvCfg.reset_step`


class ResetStep(enum.Enum):
    """Built-in work of a reset, in :attr:`MDP2EnvCfg.reset_step`."""

    EVENTS = enum.auto()  # Reset the assets and sensors, and run the reset events, which write scene fields in place
    COMMIT = enum.auto()  # Pass the writes of the reset events to the simulation
    OBSERVE = enum.auto()  # Observe the reset worlds


@dataclass(frozen=True)
class Stage:
    """User work in a control step or a reset, at its place in :attr:`MDP2EnvCfg.step` or
    :attr:`MDP2EnvCfg.reset_step`, such as kernels that update a scene field the MDP reads.

    ``fn(env)`` may read and write anything, through the environment, its scene, and its MDP state. A capturable stage
    runs inside the step's CUDA graph: it is called once, at capture, and its launches replay every step, so it must
    not read values on the host or draw from Torch's random generators, and Python branches take the path they took
    at capture. Check it with ``MDP2EnvCfg.verify_capture``. A stage that is not capturable runs on the host between
    the graphs. After each stage, the simulation applies the model and state changes it made through the assets.
    """

    fn: Callable[[Any], None]
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
    # The work of a control step, in order: the built-in steps, each once and in this order, and stages anywhere
    step: list[Step | Stage] = [Step.PHYSICS, Step.EVALUATE, Step.RESET]
    # The work of resetting worlds, in a step's Step.RESET and in reset(): as for step
    reset_step: list[ResetStep | Stage] = [ResetStep.EVENTS, ResetStep.COMMIT, ResetStep.OBSERVE]
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
