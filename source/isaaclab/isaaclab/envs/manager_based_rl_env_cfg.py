# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from ..utils import configclass
from .manager_based_env_cfg import ManagerBasedEnvCfg


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    # ui settings
    class_type: type | str = "{DIR}.manager_based_rl_env:ManagerBasedRLEnv"
    """The environment class constructed from this configuration."""

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
    """Whether to preview terminal observations before Same-Step autoreset. Defaults to False.

    When any environment terminates, times out or is manually reset, capture every observation
    group once before reset, command updates and interval events in ``extras["final_obs"]``.
    The tensors live on ``env.device`` and own their storage. Only rows for environments that
    reset are valid for bootstrapping; they use the command from before this step's command update.
    The key is absent on steps without resets.

    Preview reads include the current sample in delay and observation history without committing
    it. Custom stateful observation terms should check ``env.is_observation_preview``; stateful
    tensor processors should override their ``preview`` method. Observation noise is evaluated as
    in ordinary observations. Extra reads from active recorder terms remain outside this guarantee.
    Warp environments do not support terminal capture and warn when this flag is enabled.
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

    def play_mode(self):
        """Adjust the configuration for interactive playback and policy inference.

        Play scripts call this method after the configuration is fully initialized (i.e. after
        :meth:`__post_init__`) unless the user requests the training configuration as-is.
        The base implementation applies defaults that are useful for most tasks:

        * caps the number of environments at 50 to keep the scene lightweight, and
        * disables observation corruption for all observation groups.

        Override this method in a task configuration to customize playback behavior. Call
        ``super().play_mode()`` to keep the shared defaults.
        """
        # make a smaller scene for play
        self.scene.num_envs = min(self.scene.num_envs, 50)
        # disable observation corruption for all observation groups
        for group_name in getattr(self.observations, "__dataclass_fields__", {}):
            group = getattr(self.observations, group_name)
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = False
