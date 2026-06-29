# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .manager_based_env_cfg import ManagerBasedEnvCfg


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    # ui settings
    ui_window_class_type: type | str | None = "isaaclab.envs.ui.manager_based_rl_env_window:ManagerBasedRLEnvWindow"

    # general settings
    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    compute_final_obs: bool = False
    """Whether to capture the terminal observation before a Same-Step autoreset and expose it.

    Under Same-Step autoreset (see :attr:`~isaaclab.envs.ManagerBasedRLEnv.metadata`), an environment
    that terminates is reset within the same :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` call, so the
    returned observation belongs to the *new* episode. When this flag is True, the observation is
    computed once more *before* the reset and stored under ``extras["final_obs"]``, so wrappers can
    report it as the true terminal observation for value bootstrapping.

    Defaults to False, which preserves the previous behavior: no terminal observation is captured,
    ``extras["final_obs"]`` is not populated, and the extra observation computation is skipped.

    Note:
        Currently consumed by the :class:`~isaaclab_rl.sb3.Sb3VecEnvWrapper` wrapper.
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # environment settings
    rewards: object = MISSING
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: object = MISSING
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """
