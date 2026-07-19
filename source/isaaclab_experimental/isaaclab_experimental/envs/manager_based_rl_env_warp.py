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

import math
import os
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.timer import Timer

from isaaclab_experimental.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab_experimental.utils.torch_utils import clone_obs_buffer
from isaaclab_experimental.utils.warp import any_env_set, increment_all_int64, zero_masked_int64

from .manager_based_env_warp import ManagerBasedEnvWarp

DEBUG_TIMERS = os.environ.get("DEBUG_TIMERS", "0") == "1"
"""Enable outer step() timer. Set DEBUG_TIMERS=1 env var to enable."""

DEBUG_TIMER_STEP = os.environ.get("DEBUG_TIMER_STEP", "0") == "1"
"""Enable step sub-phase timers. Set DEBUG_TIMER_STEP=1 env var to enable."""

DEBUG_TIMER_RESET = os.environ.get("DEBUG_TIMER_RESET", "0") == "1"
"""Enable reset sub-phase timers. Set DEBUG_TIMER_RESET=1 env var to enable."""


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

        print("[INFO]: Completed setting up the environment...")

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
        # -- Warp-first command manager
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- Warp-first curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step_warp_termination_compute(self) -> None:
        """Captured stage: compute terminations (env-step frequency)."""
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

    @Timer(name="env_step", msg="Step took:", enable=DEBUG_TIMER_STEP, time_unit="us")
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
        # NOTE: keep a persistent action input buffer for graph pointer stability.
        # IMPORTANT: Do NOT re-wrap/replace the `wp.array` used by captured graphs each step.
        # Instead, copy the latest actions into the persistent buffer.
        with Timer(name="action_preprocess", msg="Action preprocessing took:", enable=DEBUG_TIMER_STEP, time_unit="us"):
            action_device = action.to(device=self.device, dtype=torch.float32).contiguous()
            wp.copy(self._action_in_wp, wp.from_torch(action_device, dtype=wp.float32))

        self._warp_graph_cache.call(
            "ActionManager_process_action",
            self.action_manager.process_action,
            action=self._action_in_wp,
            timer=DEBUG_TIMER_STEP,
        )

        if self._has_recorders:
            self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.is_rendering

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._warp_graph_cache.call(
                "ActionManager_apply_action",
                self.action_manager.apply_action,
                timer=DEBUG_TIMER_STEP,
            )
            with Timer(
                name="Scene_write_data_to_sim",
                msg="Scene write took:",
                enable=DEBUG_TIMER_STEP,
                time_unit="us",
            ):
                self.scene.write_data_to_sim()

            # simulate
            with Timer(name="simulate", msg="Newton simulation step took:", enable=DEBUG_TIMER_STEP, time_unit="us"):
                self.sim.step(render=False)
            if self._has_recorders:
                self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            with Timer(
                name="scene.update",
                msg="Scene.update took:",
                enable=DEBUG_TIMER_STEP,
                time_unit="us",
            ):
                self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self._warp_launch.launch(
            increment_all_int64,
            dim=self.num_envs,
            inputs=[self._episode_length_buf_wp, 1],
            site=("manager_based_rl_env", "episode_length_increment"),
        )
        self.common_step_counter += 1  # total step (common for all envs)

        # -- post-processing (termination + reward) as independently configurable stages
        self._warp_graph_cache.call(
            "TerminationManager_compute",
            self.step_warp_termination_compute,
            timer=DEBUG_TIMER_STEP,
        )
        self.reward_buf = self._warp_graph_cache.call(
            "RewardManager_compute",
            self.reward_manager.compute,
            dt=self.step_dt,
            timer=DEBUG_TIMER_STEP,
        )

        if self._has_recorders:
            # update observations for recording if needed
            self._warp_graph_cache.call(
                "ObservationManager_compute_no_history",
                self.observation_manager.compute,
                return_cloned_output=False,
                timer=DEBUG_TIMER_STEP,
            )
            self.recorder_manager.record_post_step()

        self._reset_terminated_envs()

        # -- update command
        self._warp_graph_cache.call(
            "CommandManager_compute",
            self.command_manager.compute,
            dt=self.step_dt,
            timer=DEBUG_TIMER_STEP,
        )

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self._warp_graph_cache.call(
                "EventManager_apply_interval",
                self.event_manager.apply,
                mode="interval",
                dt=self.step_dt,
                timer=DEBUG_TIMER_STEP,
            )

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self._warp_graph_cache.call(
            "ObservationManager_compute_update_history",
            self.observation_manager.compute,
            update_history=True,
            return_cloned_output=False,
            output=clone_obs_buffer,
            timer=DEBUG_TIMER_STEP,
        )
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

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
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # rendering requires a GUI or offscreen rendering (mirrors the stable env)
            if not (self.sim.has_gui or self.sim.has_offscreen_render):
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode does not support"
                    " rendering. Please set the simulation render mode to 'PARTIAL_RENDERING' or"
                    " 'FULL_RENDERING'. If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            self.invalidate_wp_graphs()
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

    def _reset_mask(
        self,
        *,
        env_mask: wp.array(dtype=wp.bool),
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset Warp-owned RL state for selected environments.

        Args:
            env_mask: Boolean Warp mask selecting environments to reset.
            env_ids: Compact environment IDs matching :paramref:`env_mask`, when a
                host consumer (e.g. the recorder boundary) already materialized them.
        """
        if env_mask is not self.reset_mask_wp:
            wp.copy(self.reset_mask_wp, env_mask)
        env_mask = self.reset_mask_wp

        # Legacy curriculum terms consume compact IDs. Materialize them at most
        # once per reset and share the recorder boundary's IDs when available.
        curriculum_kwargs: dict[str, Any] = {"env_mask": env_mask}
        if self.curriculum_manager.requires_host_ids:
            if env_ids is None:
                env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            curriculum_kwargs["env_ids"] = env_ids

        self._warp_graph_cache.call(
            "CurriculumManager_compute",
            self.curriculum_manager.compute,
            timer=DEBUG_TIMER_RESET,
            **curriculum_kwargs,
        )

        with Timer(name="Scene_reset", msg="Scene reset took:", enable=DEBUG_TIMER_RESET, time_unit="us"):
            self.scene.reset(env_mask=env_mask)

        if "reset" in self.event_manager.available_modes:
            self._global_env_step_count_wp.fill_(self._sim_step_counter // self.cfg.decimation)
            self._warp_graph_cache.call(
                "EventManager_apply_reset",
                self.event_manager.apply,
                mode="reset",
                env_mask_wp=env_mask,
                global_env_step_count=self._global_env_step_count_wp,
                timer=DEBUG_TIMER_RESET,
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        # -- observation manager + action + reward managers
        obs_info = self._warp_graph_cache.call(
            "ObservationManager_reset",
            self.observation_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )
        action_info = self._warp_graph_cache.call(
            "ActionManager_reset",
            self.action_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )
        reward_info = self._warp_graph_cache.call(
            "RewardManager_reset",
            self.reward_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )
        curriculum_info = self._warp_graph_cache.call(
            "CurriculumManager_reset",
            self.curriculum_manager.reset,
            timer=DEBUG_TIMER_RESET,
            **curriculum_kwargs,
        )

        # -- command + event + termination managers
        command_info = self._warp_graph_cache.call(
            "CommandManager_reset",
            self.command_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )
        event_info = self._warp_graph_cache.call(
            "EventManager_reset",
            self.event_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )
        termination_info = self._warp_graph_cache.call(
            "TerminationManager_reset",
            self.termination_manager.reset,
            env_mask=env_mask,
            timer=DEBUG_TIMER_RESET,
        )

        # reset the episode length buffer
        self._warp_launch.launch(
            zero_masked_int64,
            dim=self.num_envs,
            inputs=[env_mask, self._episode_length_buf_wp],
            site=("manager_based_rl_env", "episode_length_reset", env_mask),
        )

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
        ):
            log.update(info)
        self.extras["log"] = log

    def _reset_terminated_envs(self) -> None:
        """Reset terminated environments using the canonical Warp mask."""
        reset_mask = self.termination_manager.dones_wp
        # Keep the mask as the canonical selection, but use one host predicate
        # to avoid dispatching the complete reset pipeline when it is empty.
        if not any_env_set(self.reset_buf):
            return

        # Same-step autoreset exposes terminal observations before any selected
        # environment is reset, matching the stable manager-based environment.
        if self.cfg.compute_final_obs:
            self.extras["final_obs"] = self.observation_manager.compute()

        recorder_env_ids = None
        if self._has_recorders:
            with Timer(
                name="reset_selection_host",
                msg="Recorder reset selection took:",
                enable=DEBUG_TIMER_STEP,
                time_unit="us",
            ):
                recorder_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.recorder_manager.record_pre_reset(recorder_env_ids)

        with Timer(
            name="reset_mask",
            msg="Reset mask took:",
            enable=DEBUG_TIMER_STEP,
            time_unit="us",
        ):
            self._reset_mask(env_mask=reset_mask, env_ids=recorder_env_ids)

        if self._has_recorders:
            self.extras["log"].update(self.recorder_manager.reset(recorder_env_ids))
            self.recorder_manager.record_post_reset(recorder_env_ids)
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
