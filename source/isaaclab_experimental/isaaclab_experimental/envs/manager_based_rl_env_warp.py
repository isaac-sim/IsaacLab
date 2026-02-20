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
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.timer import Timer

from isaaclab_experimental.utils.torch_utils import clone_obs_buffer

from .manager_based_env_warp import ManagerBasedEnvWarp, ManagerCallMode

# Controls per-section timing inside `step()`.
TIMER_ENABLED_STEP = False

# Controls per-section timing inside `_reset_idx`.
TIMER_ENABLED_RESET_IDX = False


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
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # The persistent reset mask needed for warp capture
        # The intended use is to copy into this mask whenever capture is needed
        # TODO: termination manager provides the same mask, investigate whether this can be replaced.
        self.reset_mask_wp = wp.zeros(cfg.scene.num_envs, dtype=wp.bool, device=cfg.sim.device)

        # Persistent action input buffer to keep pointer stable for captured graphs.
        self._action_in_wp: wp.array = wp.zeros(
            (self.num_envs, self.action_manager.total_action_dim), dtype=wp.float32, device=self.device
        )

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed
        # of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

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
        self.command_manager = self._manager_call_switch.resolve_manager_class("CommandManager")(
            self.cfg.commands, self
        )
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = self._manager_call_switch.resolve_manager_class("TerminationManager")(
            self.cfg.terminations, self
        )
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager (experimental fork; Warp-compatible rewards)
        self.reward_manager = self._manager_call_switch.resolve_manager_class("RewardManager")(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = self._manager_call_switch.resolve_manager_class("CurriculumManager")(
            self.cfg.curriculum, self
        )
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

    def invalidate_wp_graphs(self) -> None:
        """Invalidate all cached Warp graphs.

        Call this if the captured launch topology changes (e.g. different term list, shapes, etc.).
        """
        self._manager_call_switch.invalidate_graphs()

    def step_warp_termination_compute(self) -> None:
        """Captured stage: compute terminations (env-step frequency)."""
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

    @Timer(name="env_step", msg="Step took:", enable=True, format="us")
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
        with Timer(name="action_preprocess", msg="Action preprocessing took:", enable=TIMER_ENABLED_STEP, format="us"):
            # NOTE: keep a persistent action input buffer for graph pointer stability.
            # IMPORTANT: Do NOT re-wrap/replace the `wp.array` used by captured graphs each step.
            # Instead, copy the latest actions into the persistent buffer.
            assert self._action_in_wp is not None
            action_device = action.to(self.device)
            wp.copy(self._action_in_wp, wp.from_torch(action_device, dtype=wp.float32))

        with Timer(
            name="action_manager.process_action",
            msg="ActionManager.process_action took:",
            enable=TIMER_ENABLED_STEP,
            format="us",
        ):
            self._manager_call_switch.call_stage(
                stage="ActionManager_process_action",
                stable_calls=[{"fn": self.action_manager.process_action, "args": (action_device,)}],
                warp_calls=[{"fn": self.action_manager.process_action, "kwargs": {"action": self._action_in_wp}}],
            )

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = bool(self.sim.settings.get("/isaaclab/visualizer")) or self.sim.settings.get(
            "/isaaclab/render/rtx_sensors"
        )

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            with Timer(
                name="action_manager.apply_action",
                msg="ActionManager.apply_action took:",
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self._manager_call_switch.call_stage(
                    stage="ActionManager_apply_action",
                    stable_calls=[{"fn": self.action_manager.apply_action}],
                    warp_calls=[{"fn": self.action_manager.apply_action}],
                )
            with Timer(
                name="scene.write_data_to_sim",
                msg="Scene.write_data_to_sim took:",
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self._manager_call_switch.call_stage(
                    stage="Scene_write_data_to_sim",
                    stable_calls=[{"fn": self.scene.write_data_to_sim}],
                    warp_calls=[{"fn": self.scene.write_data_to_sim}],
                )

            # simulate
            with Timer(name="simulate", msg="Newton simulation step took:", enable=TIMER_ENABLED_STEP, format="us"):
                self.sim.step(render=False)
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
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # -- post-processing (termination + reward) as independently configurable stages
        with Timer(name="post_processing", msg="Termination+Reward took:", enable=TIMER_ENABLED_STEP, format="us"):
            self._manager_call_switch.call_stage(
                stage="TerminationManager_compute",
                stable_calls=[{"fn": self.step_warp_termination_compute}],
                warp_calls=[{"fn": self.step_warp_termination_compute}],
            )
            reward_out = self._manager_call_switch.call_stage(
                stage="RewardManager_compute",
                stable_calls=[{"fn": self.reward_manager.compute, "kwargs": {"dt": float(self.step_dt)}}],
                warp_calls=[{"fn": self.reward_manager.compute, "kwargs": {"dt": float(self.step_dt)}}],
            )
            reward_mode = self._manager_call_switch.get_mode_for_manager("RewardManager")
            if reward_mode == ManagerCallMode.WARP_CAPTURED:
                self.reward_buf = self.reward_manager._reward_tensor_view
            else:
                self.reward_buf = reward_out

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            with Timer(
                name="observation_manager.compute",
                msg="ObservationManager.compute took:",
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self._manager_call_switch.call_stage(
                    stage="ObservationManager_compute_no_history",
                    stable_calls=[{"fn": self.observation_manager.compute}],
                    warp_calls=[{"fn": self.observation_manager.compute, "kwargs": {"return_cloned_output": False}}],
                )
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        # NOTE: Interim path (intentional).
        # We still compact `reset_buf` into `env_ids` here because several reset-time managers/recorders
        # are still `env_ids`-based. Do NOT remove/replace this until mask-based reset is end-to-end.
        with Timer(
            name="reset_selection",
            msg="Reset selection took:",
            enable=TIMER_ENABLED_STEP,
            format="us",
        ):
            # Keep the reset-mask handoff fully in Warp when experimental termination buffers exist.
            # Stable termination manager path exposes torch-only dones/reset buffers.
            termination_manager_mode = self._manager_call_switch.get_mode_for_manager("TerminationManager")
            if termination_manager_mode == ManagerCallMode.STABLE:
                # copy still needed as mask will be used if manager is set to mode > 0
                wp.copy(self.reset_mask_wp, wp.from_torch(self.reset_buf, dtype=wp.bool))
            else:
                wp.copy(self.reset_mask_wp, self.termination_manager.dones_wp)
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            with Timer(
                name="reset_idx",
                msg="Reset idx took:",
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self._reset_idx(env_ids=reset_env_ids, env_mask=self.reset_mask_wp)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.settings.get("/isaaclab/render/rtx_sensors") and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        with Timer(
            name="command_manager.compute",
            msg="CommandManager.compute took:",
            enable=TIMER_ENABLED_STEP,
            format="us",
        ):
            self._manager_call_switch.call_stage(
                stage="CommandManager_compute",
                stable_calls=[{"fn": self.command_manager.compute, "kwargs": {"dt": float(self.step_dt)}}],
                warp_calls=[{"fn": self.command_manager.compute, "kwargs": {"dt": float(self.step_dt)}}],
            )

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            with Timer(
                name="event_manager.apply_interval",
                msg="EventManager.apply (interval) took:",
                enable=TIMER_ENABLED_STEP,
                format="us",
            ):
                self._manager_call_switch.call_stage(
                    stage="EventManager_apply_interval",
                    stable_calls=[
                        {"fn": self.event_manager.apply, "kwargs": {"mode": "interval", "dt": float(self.step_dt)}}
                    ],
                    warp_calls=[
                        {"fn": self.event_manager.apply, "kwargs": {"mode": "interval", "dt": float(self.step_dt)}}
                    ],
                )

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        with Timer(
            name="observation_manager.compute_update_history",
            msg="ObservationManager.compute (update_history) took:",
            enable=TIMER_ENABLED_STEP,
            format="us",
        ):
            obs_buf = self._manager_call_switch.call_stage(
                stage="ObservationManager_compute_update_history",
                stable_calls=[{"fn": self.observation_manager.compute, "kwargs": {"update_history": True}}],
                warp_calls=[
                    {
                        "fn": self.observation_manager.compute,
                        "kwargs": {"update_history": True, "return_cloned_output": False},
                    }
                ],
            )
            obs_mode = self._manager_call_switch.get_mode_for_manager("ObservationManager")
            if obs_mode == ManagerCallMode.STABLE:
                self.obs_buf = obs_buf
            elif obs_mode == ManagerCallMode.WARP_NOT_CAPTURED:
                self.obs_buf = clone_obs_buffer(obs_buf)
            elif obs_mode == ManagerCallMode.WARP_CAPTURED:
                self.obs_buf = clone_obs_buffer(self.observation_manager._obs_buffer)
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
        if not self.sim.settings.get("/isaaclab/render/rtx_sensors") and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            has_gui = bool(self.sim.get_setting("/isaaclab/has_gui"))
            offscreen_render = bool(self.sim.get_setting("/isaaclab/render/offscreen"))
            if not (has_gui or offscreen_render):
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

    def _reset_idx(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
        *,
        env_mask: wp.array | None = None,
    ):
        """Reset environments based on specified indices.

        IMPORTANT:
            This function always uses the **TerminationManager-produced Warp env mask** (`self.reset_buf`) to select
            which envs to reset. The ids/mask conversion is performed in `step()` before calling this function.

            In other words:
            - If `env_mask` is provided, it **must** be `self.reset_buf` (Warp bool mask)
            - If `env_mask` is not provided, this function will populate `self.reset_buf` from `env_ids`
            - When `env_mask` is provided, `env_ids` **must** correspond to the same mask

        Args:
            env_ids: Environment indices to reset.
            env_mask: Warp boolean env mask selecting envs to reset. Must be `self.reset_buf`.
                If None, uses and populates `self.reset_buf` from `env_ids`.
        """
        if env_mask is None:
            # Base `reset()` / `reset_to()` call-path provides only `env_ids`.
            # Populate the stable TerminationManager-owned mask (`self.reset_buf`) from ids.
            env_mask = self.reset_mask_wp
            # Use the centralized env-id/mask resolution from the base Warp env, then copy into the
            # stable TerminationManager-owned buffer (`self.reset_buf`) used by captured graphs.
            resolved_mask = self.resolve_env_mask(env_ids=env_ids)
            wp.copy(env_mask, resolved_mask)

        if not isinstance(env_mask, wp.array):
            raise TypeError(f"env_mask must be a wp.array (got {type(env_mask)}).")

        # update the curriculum for environments that need a reset
        with Timer(
            name="curriculum_manager.compute_reset",
            msg="CurriculumManager.compute (reset) took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        with Timer(
            name="scene.reset",
            msg="Scene.reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            self._manager_call_switch.call_stage(
                stage="Scene_reset",
                stable_calls=[{"fn": self.scene.reset, "args": (env_ids,)}],
                warp_calls=[{"fn": self.scene.reset, "kwargs": {"env_mask": env_mask}}],
            )

        if "reset" in self.event_manager.available_modes:
            with Timer(
                name="event_manager.prepare_reset",
                msg="EventManager.prepare (reset) took:",
                enable=TIMER_ENABLED_RESET_IDX,
                format="us",
            ):
                self._global_env_step_count_wp.fill_(self._sim_step_counter // self.cfg.decimation)
            with Timer(
                name="event_manager.apply_reset",
                msg="EventManager.apply (reset) took:",
                enable=TIMER_ENABLED_RESET_IDX,
                format="us",
            ):
                self._manager_call_switch.call_stage(
                    stage="EventManager_apply_reset",
                    stable_calls=[
                        {
                            "fn": self.event_manager.apply,
                            "kwargs": {
                                "mode": "reset",
                                "env_ids": env_ids,
                                "global_env_step_count": int(self._sim_step_counter // self.cfg.decimation),
                            },
                        }
                    ],
                    warp_calls=[
                        {
                            "fn": self.event_manager.apply,
                            "kwargs": {
                                "mode": "reset",
                                "env_mask_wp": env_mask,
                                "global_env_step_count": self._global_env_step_count_wp,
                            },
                        }
                    ],
                )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        # -- observation manager + action + reward managers (per-manager configurable stage mode)
        with Timer(
            name="obs_action_reward_reset",
            msg="Observation+Action+Reward reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            obs_info = self._manager_call_switch.call_stage(
                stage="ObservationManager_reset",
                stable_calls=[{"fn": self.observation_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.observation_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            action_info = self._manager_call_switch.call_stage(
                stage="ActionManager_reset",
                stable_calls=[{"fn": self.action_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.action_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            reward_info = self._manager_call_switch.call_stage(
                stage="RewardManager_reset",
                stable_calls=[{"fn": self.reward_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.reward_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            if self._manager_call_switch.get_mode_for_manager("ObservationManager") == ManagerCallMode.WARP_CAPTURED:
                obs_info = {}
            if self._manager_call_switch.get_mode_for_manager("ActionManager") == ManagerCallMode.WARP_CAPTURED:
                action_info = {}
            if self._manager_call_switch.get_mode_for_manager("RewardManager") == ManagerCallMode.WARP_CAPTURED:
                reward_info = self.reward_manager._reset_extras

        # -- curriculum manager
        with Timer(
            name="curriculum_manager.reset",
            msg="CurriculumManager.reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            curriculum_info = self.curriculum_manager.reset(env_ids=env_ids)

        # -- command + event + termination managers (per-manager configurable stage mode)
        with Timer(
            name="command_event_termination_manager.reset",
            msg="Command+Event+TerminationManager.reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            command_info = self._manager_call_switch.call_stage(
                stage="CommandManager_reset",
                stable_calls=[{"fn": self.command_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.command_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            event_info = self._manager_call_switch.call_stage(
                stage="EventManager_reset",
                stable_calls=[{"fn": self.event_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.event_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            termination_info = self._manager_call_switch.call_stage(
                stage="TerminationManager_reset",
                stable_calls=[{"fn": self.termination_manager.reset, "kwargs": {"env_ids": env_ids}}],
                warp_calls=[{"fn": self.termination_manager.reset, "kwargs": {"env_mask": env_mask}}],
            )
            if self._manager_call_switch.get_mode_for_manager("CommandManager") == ManagerCallMode.WARP_CAPTURED:
                command_info = self.command_manager.reset_extras
            if self._manager_call_switch.get_mode_for_manager("EventManager") == ManagerCallMode.WARP_CAPTURED:
                event_info = {}
            if self._manager_call_switch.get_mode_for_manager("TerminationManager") == ManagerCallMode.WARP_CAPTURED:
                termination_info = self.termination_manager.episode_termination_extras

        # -- recorder manager
        with Timer(
            name="recorder_manager.reset",
            msg="RecorderManager.reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            recorder_info = self.recorder_manager.reset(env_ids=env_ids)

        # reset the episode length buffer
        with Timer(
            name="episode_length_buf.reset",
            msg="Episode length buffer reset took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
            self.episode_length_buf[env_ids] = 0

        # aggregate logging info at the end (order-sensitive; avoid dict mutation inside captured stages)
        with Timer(
            name="extras_log.aggregate",
            msg="extras['log'] aggregation took:",
            enable=TIMER_ENABLED_RESET_IDX,
            format="us",
        ):
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
            self.extras["log"] = log
