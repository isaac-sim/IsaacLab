# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from ..managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from ..utils import env_ids_from_mask, env_mask_from_ids
from ..utils.step_graph import StepGraph, StepOp
from ..utils.warp import ProxyArray
from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

if TYPE_CHECKING:
    from ..physics import PhysicsManager

logger = logging.getLogger(__name__)


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
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
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        *,
        physics_manager: PhysicsManager | None = None,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
            physics_manager: Physics manager for the simulation context the environment creates. Defaults to the
                manager selected by ``cfg.sim.physics``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.reset_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.bool)
        # stable mask of the environments reset within a step, shared by every reset consumer
        self._step_reset_mask = ProxyArray(wp.zeros(cfg.scene.num_envs, dtype=wp.bool, device=cfg.sim.device))
        # environment steps taken, on the device so that captured reset events read the current count
        self._env_step_count = torch.zeros((), dtype=torch.long, device=cfg.sim.device)
        # captured step (see ``ManagerBasedRLEnvCfg.capture_step``)
        self._step_graph: StepGraph | None = None
        self._step_graph_physics_ops: list[StepOp] | None = None
        self._step_action: torch.Tensor | None = None
        self._step_graph_rebuilds: list[int] = []
        self._step_graph_warm: StepGraph | None = None
        # operations found to synchronize the device; they run eagerly from then on
        self._step_sync_ops: set[str] = set()
        self._step_capture_blocker: str | None = None if cfg.capture_step else "disabled by capture_step"

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg, physics_manager=physics_manager)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the
        #    produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
        logger.info("Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def step_graph(self) -> StepGraph | None:
        """The operations :meth:`step` runs, grouped into CUDA graphs, or None while stepping eagerly.

        See :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.capture_step`.
        """
        return self._step_graph

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
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        logger.info(f"Command Manager: {self.command_manager}")

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        logger.info(f"Termination Manager: {self.termination_manager}")
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        logger.info(f"Reward Manager: {self.reward_manager}")
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        logger.info(f"Curriculum Manager: {self.curriculum_manager}")

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Wire manager terms and episode metrics into live plots for all active visualizer backends.

        Calls :meth:`~isaaclab.visualizers.BaseVisualizer.add_live_plots` on every visualizer
        registered with the simulation context.  In addition to manager terms, registers
        ``episode/mean_reward`` and ``episode/episode_length`` as direct scalar sources so
        that top-level training metrics are always visible alongside per-term breakdowns.
        For the Kit backend this also populates :attr:`manager_visualizers` with
        :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` instances so that
        :class:`~isaaclab.envs.ui.BaseEnvWindow` can build the omni.ui panels.
        """
        managers = {
            "action_manager": self.action_manager,
            "observation_manager": self.observation_manager,
            "command_manager": self.command_manager,
            "termination_manager": self.termination_manager,
            "reward_manager": self.reward_manager,
            "curriculum_manager": self.curriculum_manager,
        }
        scalars = {
            "episode": {
                "mean_reward": lambda: (
                    float(getattr(self, "reward_buf", None).mean())
                    if getattr(self, "reward_buf", None) is not None
                    else 0.0
                ),
                "episode_length": lambda: float(self.episode_length_buf.float().mean()),
            }
        }
        for viz in self.sim.visualizers:
            viz.add_live_plots(managers, scalars=scalars)
        # Populate manager_visualizers for the Kit window (BaseEnvWindow reads this attribute).
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

        Rendering can be controlled per-step via :attr:`render_enabled`.

        When ``render_enabled`` is False:

        - The Kit app loop (``app.update()``) is **skipped**, which also disables
          camera/RTX sensor rendering and GUI viewport updates.  Kit bundles these
          operations together, so they cannot be separated.
        - Standalone visualizers (Newton, Rerun, Viser) **continue to update**
          normally because their ``step()`` methods are independent of the Kit
          app loop.
        - Post-reset re-renders for RTX sensors are also skipped.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        step_graph = self._prepare_step_graph(action)
        if step_graph is not None:
            return self._launch_step_graph(step_graph, action)

        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: evaluated once per step rather than once per physics substep
        is_rendering = self.sim.is_rendering

        # physics-owned decimation covers all substeps in one call
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(self.cfg.decimation // steps_per_call):
            self._sim_step_counter += steps_per_call
            if not self._physics_handles_actions:
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render_enabled=False skips Kit (camera/GUI); standalone visualizers still update
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt * steps_per_call)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self._env_step_count += 1
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        # note: the reset runs every step on the reset mask, so the step never waits on the device
        if self.cfg.compute_final_obs:
            # expose the terminal observation before reset for Same-Step autoreset
            self.extras["final_obs"] = self.observation_manager.compute()
        self._reset_terminated_envs(self.reset_buf)
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            if self.reset_buf.any():
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # -- handle episode reset requested from visualizer UI controls
        if self.sim.consume_reset_request():
            # Only reset envs not already reset this step to avoid redundant resets.
            manual_reset = ~self.reset_buf
            # mark as terminated so RL wrappers observe the episode boundary
            self.reset_terminated |= manual_reset
            self._reset_terminated_envs(manual_reset)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.active_terms:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- advance video recorders (after render and resets, before final obs)
        for recorder in self.video_recorders:
            recorder.step()
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

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
        # if we have rtx sensors, we do not need to render again since step already rendered
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
            # Drop the observation/action space objects. gymnasium's wrapper chain keeps
            # the env referenced past close, so without this they leak — and for image
            # observations each space holds a large gym.spaces.Box bounds array.
            self.single_observation_space = None
            self.single_action_space = None
            self.observation_space = None
            self.action_space = None
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

    def _reset_terminated_envs(self, reset: torch.Tensor):
        """Reset the environments selected by ``env_mask`` within :meth:`step`.

        Subclasses that override :meth:`_reset_idx` keep receiving indices, which waits on the device.

        Args:
            reset: Boolean flags of the environments to reset. Shape is (num_envs,).
        """
        env_mask = self._step_reset_mask
        env_mask.torch.copy_(reset)
        recording = len(self.recorder_manager.active_terms) > 0
        if recording or type(self)._reset_idx is not ManagerBasedRLEnv._reset_idx:
            env_ids = env_ids_from_mask(env_mask)
            if len(env_ids) == 0:
                return
            self.recorder_manager.record_pre_reset(env_ids)
            self._reset_idx(env_ids)
            self.recorder_manager.record_post_reset(env_ids)
        else:
            self._reset_mask(env_mask)

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        """Reset environments based on specified indices.

        Args:
            env_ids: A slice or environment indices on the environment device.
        """
        self._reset_mask(env_mask_from_ids(env_ids, self.num_envs, self.device))

    def _reset_mask(self, env_mask: ProxyArray):
        """Reset the environments selected by a boolean mask.

        Args:
            env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        """
        for op in self._reset_mask_ops(env_mask):
            op.fn()

    def _reset_mask_ops(self, env_mask: ProxyArray) -> list[StepOp]:
        """Return the masked reset as operations, flagged by whether a CUDA graph can record them.

        Args:
            env_mask: Boolean mask of the environments to reset, a stable buffer when the operations are recorded.
        """
        # note: This is order-sensitive! Certain things need be reset before others.
        managers = (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
            self.recorder_manager,
        )
        # update the curriculum for environments that need a reset
        ops = [
            StepOp(
                partial(self.curriculum_manager.compute, env_mask=env_mask),
                self.curriculum_manager.supports_graph_capture,
                "curriculum",
            )
        ]
        # reset the internal buffers of the scene elements; entities whose reset cannot be recorded run eagerly
        entities = self.scene.entities
        recordable = [entity for entity in entities if entity.reset_supports_graph_capture]
        ops.append(StepOp(partial(self.scene.reset, env_mask=env_mask.warp, entities=recordable), True, "scene.reset"))
        for entity in entities:
            if not entity.reset_supports_graph_capture:
                reset = partial(self.scene.reset, env_mask=env_mask.warp, entities=[entity])
                ops.append(StepOp(reset, False, f"scene.reset.{type(entity).__name__}"))
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.active_terms:
            reset_events = partial(
                self.event_manager.apply, mode="reset", env_mask=env_mask, global_env_step_count=self._env_step_count
            )
            ops.append(StepOp(reset_events, self.event_manager.supports_graph_capture, "events.reset"))
        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        reset_managers = partial(self._reset_managers, env_mask, managers)
        ops.append(StepOp(reset_managers, all(m.supports_graph_capture for m in managers), "managers.reset"))
        ops.append(StepOp(partial(self.episode_length_buf.masked_fill_, env_mask.torch, 0), True, "episode_length"))
        return ops

    """
    Operations - Captured step.
    """

    def invalidate_step_graph(self) -> None:
        """Record the captured step again before the next step.

        Call it after changing host state that captured work reads as a constant, such as a manager term's
        configuration. :meth:`~isaaclab.managers.ManagerBase` setters such as ``set_term_cfg`` call it already.
        """
        if self._step_graph is not None:
            self._step_graph = None
            self._step_graph_rebuilds.append(self.common_step_counter)

    def _prepare_step_graph(self, action: torch.Tensor) -> StepGraph | None:
        """Return the step as a graph of operations, or None to step eagerly.

        The graph is rebuilt when the physics step's operations change (e.g. new step callbacks) or when
        :meth:`invalidate_step_graph` was called.
        """
        if self._step_capture_blocker is not None:
            return None
        if self.sim.is_rendering or self.video_recorders:
            # rendering and video run on the host on their own cadence
            return None
        physics_ops = self.sim.physics_manager.step_ops()
        if physics_ops is None:
            self._step_capture_blocker = "the physics backend exposes no step operations"
            return None
        if self._step_graph is not None and physics_ops is self._step_graph_physics_ops:
            return self._step_graph
        if self._step_graph is not None:
            self._step_graph_rebuilds.append(self.common_step_counter)
        # Rebuilding every few steps costs more than it saves: some term keeps changing captured constants.
        recent = [step for step in self._step_graph_rebuilds if self.common_step_counter - step < 50]
        if len(recent) > 3:
            return self._block_step_capture("the captured step is invalidated every few steps")
        blocker = self._step_graph_blocker()
        if blocker is not None:
            return self._block_step_capture(blocker)
        if self._step_action is None or self._step_action.shape != action.shape:
            self._step_action = torch.zeros(action.shape, dtype=torch.float32, device=self.device)
        self._step_graph_physics_ops = physics_ops
        self._step_graph = self._step_graph_from(self._step_ops(physics_ops))
        return self._step_graph

    def _step_graph_from(self, ops: list[StepOp]) -> StepGraph:
        """Group operations into a step graph, keeping operations known to synchronize the device eager."""
        sync = self._step_sync_ops
        return StepGraph([StepOp(op.fn, False, op.name) if op.name in sync and op.graphable else op for op in ops])

    def _step_graph_blocker(self) -> str | None:
        """Return why the step cannot run as a captured graph, or None."""
        if self.recorder_manager.active_terms:
            return "recorder terms are active"
        if type(self)._reset_idx is not ManagerBasedRLEnv._reset_idx:
            return f"{type(self).__name__} overrides _reset_idx, which resets by index"
        if not self.scene.cfg.lazy_sensor_update:
            return "the scene updates sensors eagerly (lazy_sensor_update is False)"
        if self.has_rtx_sensors:
            return "camera sensors render on the host"
        for name, sensor in self.scene.sensors.items():
            if not sensor.supports_graph_capture:
                return f"sensor '{name}' ({type(sensor).__name__}) cannot be recorded"
        return None

    def _block_step_capture(self, reason: str) -> None:
        logger.warning("Stepping %s eagerly: %s.", type(self).__name__, reason)
        self._step_capture_blocker = reason
        self._step_graph = None

    def _step_ops(self, physics_ops: list[StepOp]) -> list[StepOp]:
        """Return one environment step as operations, flagged by whether a CUDA graph can record them."""
        actions = self.action_manager
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        process = StepOp(partial(actions.process_action, self._step_action), actions.supports_graph_capture, "actions")
        # Physics may open with eager host work (e.g. model-change notification) that actions do not depend on;
        # processing actions after it lets them share a graph with the physics step.
        leading = next((i for i, op in enumerate(physics_ops) if op.graphable), len(physics_ops))
        ops = list(physics_ops[:leading])
        for _ in range(self.cfg.decimation // steps_per_call):
            if process is not None:
                ops.append(process)
                process = None
            elif leading:
                ops.extend(physics_ops[:leading])
            if not self._physics_handles_actions:
                ops.append(StepOp(actions.apply_action, actions.supports_graph_capture, "actions.apply"))
                writes = all(asset.supports_graph_capture for asset in self.scene.assets.values())
                ops.append(StepOp(self.scene.write_data_to_sim, writes, "scene.write"))
            ops.extend(physics_ops[leading:])
            ops.append(StepOp(partial(self.scene.update, dt=self.physics_dt * steps_per_call), True, "scene.update"))
        ops.append(StepOp(self._advance_episode, True, "episode"))
        terminations, rewards, observations = self.termination_manager, self.reward_manager, self.observation_manager
        ops.append(StepOp(self._compute_terminations, terminations.supports_graph_capture, "terminations"))
        ops.append(StepOp(self._compute_rewards, rewards.supports_graph_capture, "rewards"))
        if self.cfg.compute_final_obs:
            ops.append(StepOp(self._compute_final_obs, observations.supports_graph_capture, "final_obs"))
        ops.append(StepOp(self._copy_reset_mask, True, "reset_mask"))
        ops.extend(self._reset_mask_ops(self._step_reset_mask))
        commands = self.command_manager
        ops.append(StepOp(partial(commands.compute, dt=self.step_dt), commands.supports_graph_capture, "commands"))
        if "interval" in self.event_manager.active_terms:
            interval = partial(self.event_manager.apply, mode="interval", dt=self.step_dt)
            ops.append(StepOp(interval, self.event_manager.supports_graph_capture, "events.interval"))
        ops.append(StepOp(self._compute_observations, observations.supports_graph_capture, "observations"))
        return ops

    def _launch_step_graph(self, graph: StepGraph, action: torch.Tensor) -> VecEnvStepReturn:
        """Run one environment step from the step graph, recording its graphs on the first run."""
        self._step_action.copy_(action)
        if not graph.captured and self._step_graph_warm is graph:
            # The previous step ran the operations eagerly (lazy allocations); record them now and replay, so that
            # every returned buffer holds executed work.
            try:
                graph.capture(self._capture_graph)
            except Exception as error:  # noqa: BLE001
                self._block_step_capture(f"recording the step failed ({error})")
        replayed = graph.captured
        if replayed:
            graph.launch()
        else:
            graph = self._launch_watching_syncs(graph)
        self._step_graph_warm = graph
        # host bookkeeping that the replayed operations skip
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(self.cfg.decimation // steps_per_call):
            self._sim_step_counter += steps_per_call
            self.sim.finish_physics_step()
            if replayed:
                self.scene.notify_update_replayed(dt=self.physics_dt * steps_per_call)
        self.common_step_counter += 1
        # -- handle episode reset requested from visualizer UI controls
        if self.sim.consume_reset_request():
            manual_reset = ~self.reset_buf
            self.reset_terminated |= manual_reset
            self._reset_terminated_envs(manual_reset)
            self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _launch_watching_syncs(self, graph: StepGraph) -> StepGraph:
        """Run the operations eagerly and return the graph with operations that synchronized the device made eager.

        A synchronizing operation (e.g. a term that selects environments by index) would fail inside a capture, so it
        becomes an eager island instead of preventing the capture.
        """
        if "cuda" not in str(self.device):
            graph.launch()
            return graph
        synced: set[str] = set()
        previous_mode = torch.cuda.get_sync_debug_mode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                for op in graph.ops:
                    count = len(caught)
                    op.fn()
                    if op.graphable and any("synchroniz" in str(w.message) for w in caught[count:]):
                        synced.add(op.name)
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
        if not synced:
            return graph
        logger.info("Running %s eagerly in the captured step: they synchronize the device.", sorted(synced))
        self._step_sync_ops |= synced
        self._step_graph = self._step_graph_from(graph.ops)
        return self._step_graph

    def _capture_graph(self, fn) -> Any:
        graph = self.sim.physics_manager.capture_graph(fn)
        if graph is None:
            raise RuntimeError("the physics backend does not capture graphs")
        return graph

    def _advance_episode(self) -> None:
        self.episode_length_buf += 1
        self._env_step_count += 1

    def _compute_terminations(self) -> None:
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

    def _compute_rewards(self) -> None:
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    def _compute_final_obs(self) -> None:
        self.extras["final_obs"] = self.observation_manager.compute()

    def _copy_reset_mask(self) -> None:
        self._step_reset_mask.torch.copy_(self.reset_buf)

    def _compute_observations(self) -> None:
        self.obs_buf = self.observation_manager.compute(update_history=True)
