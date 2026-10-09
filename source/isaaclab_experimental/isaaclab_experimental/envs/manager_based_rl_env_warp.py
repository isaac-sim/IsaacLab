# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental manager-based RL environment (Warp entry point).

This module provides an experimental fork of the stable manager-based RL environment
so it can diverge (Warp-first / graph-friendly) without inheriting from the stable
`isaaclab.envs.ManagerBasedRLEnv` implementation.
"""

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumManager

from isaaclab_experimental.managers import CommandManager, RewardManager, TerminationManager

from .manager_based_env_warp import ManagerBasedEnvWarp

logger = logging.getLogger(__name__)


class ManagerBasedRLEnvWarp(ManagerBasedEnvWarp, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
        # "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # Adapt the cfg for the warp managers (Newton physics check, SceneEntityCfg
        # promotion, MDP twin swap). Idempotent: a warp-native cfg passes through
        # unchanged, and a stable-derived cfg (``--frontend=warp`` or a registered
        # warp task variant subclassing a stable cfg) is adapted in place.
        from isaaclab_experimental.envs.frontend import WarpFrontend

        WarpFrontend.adapt_cfg(cfg)

        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        # Warp array is the source of truth; torch view is zero-copy for += and indexed assignment.
        self._episode_length_buf_wp = wp.zeros(cfg.scene.num_envs, dtype=wp.int64, device=cfg.sim.device)
        self._episode_length_buf = wp.to_torch(self._episode_length_buf_wp)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed
        # of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")

        logger.info("Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Episode length buffer (torch view of the underlying warp array)."""
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Copy into the existing buffer to preserve the warp array linkage."""
        self._episode_length_buf[:] = value

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager = CommandManager(self.cfg.commands, self)
        logger.info(f"Command Manager: {self.command_manager}")

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        logger.info(f"Termination Manager: {self.termination_manager}")
        # -- reward manager (experimental fork; Warp-compatible rewards)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        logger.info(f"Reward Manager: {self.reward_manager}")
        # -- curriculum manager (stable implementation)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        logger.info(f"Curriculum Manager: {self.curriculum_manager}")

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Wire manager terms into live plots for all active visualizer backends."""
        managers = {
            "action_manager": self.action_manager,
            "observation_manager": self.observation_manager,
            "command_manager": self.command_manager,
            "termination_manager": self.termination_manager,
            "reward_manager": self.reward_manager,
            "curriculum_manager": self.curriculum_manager,
        }
        for viz in self.sim.visualizers:
            viz.add_live_plots(managers)
        self.manager_visualizers = {
            name: mlv for v in self.sim.visualizers for name, mlv in getattr(v, "kit_manager_visualizers", {}).items()
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.is_rendering

        # physics-owned decimation covers all substeps in one call
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(self.cfg.decimation // steps_per_call):
            self._sim_step_counter += steps_per_call
            # set actions into buffers
            self.action_manager.apply_action()
            # scene writes cross the actuator and asset host boundaries, so they stay eager
            self.scene.write_data_to_sim()

            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt * steps_per_call)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # -- post-processing: check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.observation_manager.compute(return_cloned_output=False)
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        # the reset pipeline advances curriculum and logging state even for an empty mask, so one
        # host synchronization per step decides whether it runs
        if self.reset_buf.any().item():
            reset_mask = self.termination_manager.dones_wp
            # the stable recorder manager hooks take indices
            reset_env_ids = (
                wp.to_torch(reset_mask).nonzero(as_tuple=False).squeeze(-1)
                if self.recorder_manager.active_terms
                else None
            )
            # capture the terminal observation before reset and expose it for Same-Step autoreset.
            # the reset overwrites the persistent observation buffers, so keep a copy
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self.observation_manager.compute()
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_mask(reset_mask)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.

        .. note::
            ``render_mode="rgb_array"`` is no longer supported.  Use
            :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` on
            ``env_cfg.video_recorders`` instead.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.has_rtx_sensors and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "rgb_array":
            import warnings

            warnings.warn(
                "render_mode='rgb_array' is deprecated and will be removed in a future release. "
                "Use VideoRecorderCfg on env_cfg.video_recorders to capture frames instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return None
        if self.render_mode == "human" or self.render_mode is None:
            return None
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_mask(self, env_mask: wp.array):
        """Reset the environments selected by a boolean mask.

        Args:
            env_mask: Boolean mask of the environments to reset, of shape ``(num_envs,)``.
        """
        # recorded reset stages read the environment-owned mask
        if env_mask is not self.reset_mask_wp:
            wp.copy(self.reset_mask_wp, env_mask)
        env_mask = self.reset_mask_wp
        # the stable curriculum and recorder managers reset by index
        env_ids = (
            wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            if self.curriculum_manager.active_terms or self.recorder_manager.active_terms
            else None
        )

        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)

        # reset the internal buffers of the scene elements
        self.scene.reset(env_mask=env_mask)

        if "reset" in self.event_manager.available_modes:
            self._global_env_step_count_wp.fill_(self._sim_step_counter // self.cfg.decimation)
            self.event_manager.apply(
                mode="reset", env_mask_wp=env_mask, global_env_step_count=self._global_env_step_count_wp
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        # -- observation manager + action + reward managers
        obs_info = self.observation_manager.reset(env_mask=env_mask)
        action_info = self.action_manager.reset(env_mask=env_mask)
        reward_info = self.reward_manager.reset(env_mask=env_mask)

        # -- curriculum manager
        curriculum_info = self.curriculum_manager.reset(env_ids=env_ids)

        # -- command + event + termination managers
        command_info = self.command_manager.reset(env_mask=env_mask)
        event_info = self.event_manager.reset(env_mask=env_mask)
        termination_info = self.termination_manager.reset(env_mask=env_mask)

        # -- recorder manager
        recorder_info = self.recorder_manager.reset(env_ids=env_ids)

        # reset the episode length buffer
        self.episode_length_buf.masked_fill_(wp.to_torch(env_mask), 0)

        # aggregate logging info
        log: dict[str, Any] = {}
        for info in (
            obs_info,
            action_info,
            reward_info,
            curriculum_info,
            command_info,
            event_info,
            termination_info,
            recorder_info,
        ):
            log.update(info)
        # the managers log persistent buffers that the next reset overwrites, and RL libraries keep the log of
        # every step until they average it, so hand out copies like the stable managers' fresh tensors
        self.extras["log"] = {key: value.clone() if torch.is_tensor(value) else value for key, value in log.items()}
