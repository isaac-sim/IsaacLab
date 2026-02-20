# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental manager-based base environment.

This is a local copy of :class:`isaaclab.envs.ManagerBasedEnv` placed under
``isaaclab_experimental`` so we can evolve the manager-based workflow for Warp-first
pipelines without depending on (or subclassing) the stable env implementation.

Behavior is intended to match the stable environment initially.
"""

# import builtins
import contextlib
import importlib
import json
import logging
import warnings
from collections.abc import Sequence
from copy import deepcopy
from enum import IntEnum
from typing import Any

import torch
import warp as wp

from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.envs.utils.io_descriptors import export_articulations_data, export_scene_data
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.timer import Timer

# import logger
logger = logging.getLogger(__name__)


@wp.kernel
def initialize_rng_state(
    # input
    seed: wp.int32,
    # output
    state: wp.array(dtype=wp.uint32),
):
    env_id = wp.tid()
    state[env_id] = wp.rand_init(seed, wp.int32(env_id))


@wp.kernel
def _generate_env_mask_from_ids_int32(
    mask: wp.array(dtype=wp.bool),
    env_ids: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    mask[env_ids[i]] = True


class ManagerCallMode(IntEnum):
    """Execution mode for manager stage calls."""

    STABLE = 0
    WARP_NOT_CAPTURED = 1
    WARP_CAPTURED = 2


class ManagerCallSwitch:
    """Per-manager call switch for stable/warp/captured execution."""

    DEFAULT_CONFIG: dict[str, int] = {"default": 2}
    DEFAULT_KEY = "default"
    MANAGER_NAMES: tuple[str, ...] = (
        "ActionManager",
        "ObservationManager",
        "EventManager",
        "RecorderManager",
        "CommandManager",
        "TerminationManager",
        "RewardManager",
        "CurriculumManager",
    )

    def __init__(self, cfg_source: str | None = None):
        self._wp_graphs: dict[str, Any] = {}
        self._cfg = self._load_cfg(cfg_source)
        print("[INFO] ManagerCallSwitch configuration:")
        print(f"  - {self.DEFAULT_KEY}: {self._cfg[self.DEFAULT_KEY]}")
        for manager_name in self.MANAGER_NAMES:
            print(f"  - {manager_name}: {int(self.get_mode_for_manager(manager_name))}")

    def invalidate_graphs(self) -> None:
        """Invalidate cached capture graphs."""
        self._wp_graphs.clear()

    def call_stage(
        self,
        *,
        stage: str,
        stable_calls: Sequence[dict[str, Any]],
        warp_calls: Sequence[dict[str, Any]],
    ) -> Any:
        """Run the stage according to configured mode."""
        manager_name = self._manager_name_from_stage(stage)
        mode = self.get_mode_for_manager(manager_name)
        if mode == ManagerCallMode.STABLE:
            return self._run_calls(stable_calls)
        if mode == ManagerCallMode.WARP_NOT_CAPTURED:
            return self._run_calls(warp_calls)
        self._wp_capture_or_launch(stage=stage, calls=warp_calls)
        return None

    def _manager_name_from_stage(self, stage: str) -> str:
        if "_" not in stage:
            raise ValueError(f"Invalid stage '{stage}'. Expected '{{manager_name}}_{{function_name}}'.")
        return stage.split("_", 1)[0]

    def get_mode_for_manager(self, manager_name: str) -> ManagerCallMode:
        default_key = next(iter(self.DEFAULT_CONFIG))
        mode_value = self._cfg.get(manager_name, self._cfg[default_key])
        return ManagerCallMode(mode_value)

    def resolve_manager_class(self, manager_name: str) -> type:
        module_name = (
            "isaaclab.managers"
            if self.get_mode_for_manager(manager_name) == ManagerCallMode.STABLE
            else "isaaclab_experimental.managers"
        )
        module = importlib.import_module(module_name)
        if not hasattr(module, manager_name):
            raise AttributeError(f"Manager '{manager_name}' not found in module '{module_name}'.")
        return getattr(module, manager_name)

    def _run_calls(self, calls: Sequence[dict[str, Any]]) -> Any:
        result = None
        for spec in calls:
            fn = spec["fn"]
            fn_args = spec.get("args", ())
            fn_kwargs = spec.get("kwargs", {})
            result = fn(*fn_args, **fn_kwargs)
        return result

    def _wp_capture_or_launch(self, stage: str, calls: Sequence[dict[str, Any]]) -> None:
        """Capture Warp CUDA graph for a stage on first call, then replay."""
        graph = self._wp_graphs.get(stage)
        if graph is None:
            with wp.ScopedCapture() as capture:
                for spec in calls:
                    fn = spec["fn"]
                    fn_args = spec.get("args", ())
                    fn_kwargs = spec.get("kwargs", {})
                    fn(*fn_args, **fn_kwargs)
            graph = capture.graph
            self._wp_graphs[stage] = graph
        wp.capture_launch(graph)

    def _load_cfg(self, cfg_source: str | None) -> dict[str, int]:
        if cfg_source is not None and not isinstance(cfg_source, str):
            raise TypeError(f"cfg_source must be a string or None, got: {type(cfg_source)}")
        if cfg_source is None or cfg_source.strip() == "":
            return dict(self.DEFAULT_CONFIG)

        parsed = json.loads(cfg_source)
        if not isinstance(parsed, dict):
            raise TypeError("manager_call_config must decode to a dict.")

        cfg = dict(parsed)
        if self.DEFAULT_KEY not in cfg:
            cfg[self.DEFAULT_KEY] = self.DEFAULT_CONFIG[self.DEFAULT_KEY]

        # validation
        for manager_name, mode_value in cfg.items():
            if not isinstance(mode_value, int):
                raise TypeError(
                    f"manager_call_config value for '{manager_name}' must be int (0/1/2), got: {type(mode_value)}"
                )
            try:
                ManagerCallMode(mode_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid manager_call_config value for '{manager_name}': {mode_value}. Expected 0/1/2."
                ) from exc
        return cfg


class ManagerBasedEnvWarp:
    """The base environment for the manager-based workflow (experimental fork).

    The implementation mirrors :class:`isaaclab.envs.ManagerBasedEnv` to provide
    an isolated base class for experimental Warp-based workflows.
    """

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False
        # temporary debug runtime config for manager source/call switching.
        cfg_source: str | None = getattr(self.cfg, "manager_call_config", None)
        # if cfg_source is None:
        #     try:
        #         import __main__

        #         args_cli = getattr(__main__, "args_cli", None)
        #         cfg_source = getattr(args_cli, "manager_call_config", None)
        #     except Exception:
        #         cfg_source = None
        self._manager_call_switch = ManagerCallSwitch(cfg_source)
        self._apply_manager_term_cfg_profile()

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            # if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            #     raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # allocate dictionary to store metrics
        self.extras = {}

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # set the stage context for scene creation steps which use the stage
            with use_stage(self.sim.get_initial_stage()):
                self.scene = InteractiveScene(self.cfg.scene)
                # attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        # Shared per-env Warp RNG state (accessible to all managers/terms via `env`).
        # This is a single stream per env (no lookup) and is initialized once when `num_envs` is known.
        self.rng_state_wp = wp.zeros((self.num_envs,), dtype=wp.uint32, device=self.device)
        seed_val = int(self.cfg.seed) if self.cfg.seed is not None else -1
        wp.launch(
            kernel=initialize_rng_state,
            dim=self.num_envs,
            inputs=[seed_val, self.rng_state_wp],
            device=self.device,
        )

        # TODO(jichuanh): this is problematic as warp capture requires stable pointers,
        #                 using different masks for different managers/terms will cause problems.
        # Pre-allocated env masks (shared across managers/terms via `env`).
        self.ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=self.device)
        self.ENV_MASK = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)

        # Persistent scalar buffer for global env step count (stable pointer for capture).
        self._global_env_step_count_wp = wp.zeros((1,), dtype=wp.int32, device=self.device)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = self._manager_call_switch.resolve_manager_class("EventManager")(self.cfg.events, self)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        # if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
        with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
            # since the reset can trigger callbacks which use the stage,
            # we need to set the stage context here
            with use_stage(self.sim.get_initial_stage()):
                self.sim.reset()
            # update scene to pre populate data buffers for assets and sensors.
            # this is needed for the observation manager to get valid tensors for initialization.
            # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy
            # buffers would be reset.
            self.scene.update(dt=self.physics_dt)

        # TODO(jichuanh): This is a temporary solution for event_manager only, but it should be general for all managers
        # Resolve SceneEntityCfg-dependent term params once before any captured event paths.
        if (not self.event_manager._is_scene_entities_resolved) and self.sim.is_playing():
            self.event_manager._resolve_terms_callback(None)

        # add timeline event to load managers
        self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # initialize observation buffers
        self.obs_buf = {}

        # export IO descriptors if requested
        if self.cfg.export_io_descriptors:
            self.export_IO_descriptors()

        # show deprecation message for rerender_on_reset
        if self.cfg.rerender_on_reset:
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] ManagerBasedEnvCfg.rerender_on_reset is deprecated. Use"
                " ManagerBasedEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1

    def __del__(self):
        """Cleanup for the environment."""
        # Suppress errors during Python shutdown to avoid noisy tracebacks
        # Note: contextlib may be None during interpreter shutdown
        if contextlib is not None:
            with contextlib.suppress(ImportError, AttributeError, TypeError):
                self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    def resolve_env_mask(
        self,
        *,
        env_ids: Sequence[int] | slice | wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> wp.array:
        """Resolve environment ids/mask into a Warp boolean mask of shape ``(num_envs,)``.

        Notes:
        - Uses pre-allocated masks (`ALL_ENV_MASK`, `ENV_MASK`) to avoid allocations.
        - Not thread-safe / re-entrant (intended for the manager-based execution model).
        """
        # --- Normalize mask (direct mask inputs) ---
        # If an explicit mask is provided, normalize and return it.
        if env_mask is not None:
            if isinstance(env_mask, wp.array):
                return env_mask
            if isinstance(env_mask, torch.Tensor):
                if env_mask.dtype != torch.bool:
                    env_mask = env_mask.to(dtype=torch.bool)
                if str(env_mask.device) != self.device:
                    env_mask = env_mask.to(self.device)
                return wp.from_torch(env_mask, dtype=wp.bool)
            raise TypeError(f"Unsupported env_mask type: {type(env_mask)}")

        # Fast path: all envs.
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self.ALL_ENV_MASK

        # --- Prepare id list (normalize env_ids into indices) ---
        # Normalize slice ids into explicit indices.
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            env_ids = list(range(start, stop, step))
        # Normalize python sequences into a concrete list early (keeps control-flow linear).
        elif not isinstance(env_ids, (torch.Tensor, wp.array)):
            env_ids = list(env_ids)

        # --- Resolve mask (ids -> ENV_MASK) ---
        # Populate scratch mask.
        self.ENV_MASK.fill_(False)

        # ids provided as torch tensor
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return self.ENV_MASK
            if str(env_ids.device) != self.device:
                env_ids = env_ids.to(self.device)
            if env_ids.dtype != torch.int32:
                env_ids = env_ids.to(dtype=torch.int32)
            if not env_ids.is_contiguous():
                env_ids = env_ids.contiguous()
            ids_wp = wp.from_torch(env_ids, dtype=wp.int32)
            wp.launch(
                kernel=_generate_env_mask_from_ids_int32,
                dim=ids_wp.shape[0],
                inputs=[self.ENV_MASK, ids_wp],
                device=self.device,
            )
            return self.ENV_MASK

        # ids provided as Warp array
        if isinstance(env_ids, wp.array):
            if env_ids.dtype == wp.int32:
                if env_ids.shape[0] == 0:
                    return self.ENV_MASK
                wp.launch(
                    kernel=_generate_env_mask_from_ids_int32,
                    dim=env_ids.shape[0],
                    inputs=[self.ENV_MASK, env_ids],
                    device=self.device,
                )
                return self.ENV_MASK
            raise TypeError(
                f"Unsupported wp.array dtype for env_ids: {env_ids.dtype}. Expected wp.int32 indices or wp.bool mask."
            )

        # ids provided as python sequence (already normalized to list above)
        if len(env_ids) == 0:
            return self.ENV_MASK
        ids_wp = wp.array(env_ids, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_generate_env_mask_from_ids_int32,
            dim=ids_wp.shape[0],
            inputs=[self.ENV_MASK, ids_wp],
            device=self.device,
        )
        return self.ENV_MASK

    @property
    def get_IO_descriptors(self):
        """Get the IO descriptors for the environment.

        Returns:
            A dictionary with keys as the group names and values as the IO descriptors.
        """
        return {
            "observations": self.observation_manager.get_IO_descriptors,
            "actions": self.action_manager.get_IO_descriptors,
            "articulations": export_articulations_data(self),
            "scene": export_scene_data(self),
        }

    def export_IO_descriptors(self, output_dir: str | None = None):
        """Export the IO descriptors for the environment.

        Args:
            output_dir: The directory to export the IO descriptors to.
        """
        import os

        import yaml

        IO_descriptors = self.get_IO_descriptors

        if output_dir is None:
            if self.cfg.log_dir is not None:
                output_dir = os.path.join(self.cfg.log_dir, "io_descriptors")
            else:
                raise ValueError(
                    "Output directory is not set. Please set the log directory using the `log_dir`"
                    " configuration or provide an explicit output_dir parameter."
                )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "IO_descriptors.yaml"), "w") as f:
            print(f"[INFO]: Exporting IO descriptors to {os.path.join(output_dir, 'IO_descriptors.yaml')}")
            yaml.safe_dump(IO_descriptors, f)

    """
    Operations - Setup.
    """

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = self._manager_call_switch.resolve_manager_class("RecorderManager")(
            self.cfg.recorders, self
        )
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = self._manager_call_switch.resolve_manager_class("ActionManager")(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = self._manager_call_switch.resolve_manager_class("ObservationManager")(
            self.cfg.observations, self
        )
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnvWarp and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }

    """
    Operations - MDP.
    """

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            used_seed = self.seed(seed)
            # keep cfg seed in sync for downstream users
            self.cfg.seed = used_seed
            # re-initialize per-env Warp RNG state without reallocating (stable pointer for capture)
            wp.launch(
                kernel=initialize_rng_state,
                dim=self.num_envs,
                inputs=[int(used_seed), self.rng_state_wp],
                device=self.device,
            )

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is
        decimated at a lower time-step. This is to ensure that the simulation is stable. These two
        time-steps can be configured independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step)
        parameters. Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """
        # process actions
        action_device = action.to(self.device)
        if action_device.dtype != torch.float32:
            action_device = action_device.float()
        if not action_device.is_contiguous():
            action_device = action_device.contiguous()
        action_wp = wp.from_torch(action_device, dtype=wp.float32)
        self.action_manager.process_action(action_wp)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_step()

        # return observations and extras
        return self.obs_buf, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return configure_seed(seed)

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene

            # self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """

    def _resolve_stable_cfg_counterpart(self) -> ManagerBasedEnvCfg | None:
        """Resolve a stable task config counterpart for the current experimental task config.

        The lookup follows a module-name mirror convention:
        ``isaaclab_tasks_experimental...`` -> ``isaaclab_tasks...`` with the same config class name.
        """
        cfg_cls = self.cfg.__class__
        cfg_module_name = cfg_cls.__module__
        if "isaaclab_tasks_experimental" not in cfg_module_name:
            return None

        stable_module_name = cfg_module_name.replace("isaaclab_tasks_experimental", "isaaclab_tasks", 1)
        try:
            stable_module = importlib.import_module(stable_module_name)
        except Exception as exc:
            logger.warning(
                "Failed to import stable task cfg module '%s' for manager_call_config stable mode: %s",
                stable_module_name,
                exc,
            )
            return None

        stable_cfg_cls = getattr(stable_module, cfg_cls.__name__, None)
        if stable_cfg_cls is None:
            logger.warning(
                "Stable task cfg class '%s' not found in module '%s'.",
                cfg_cls.__name__,
                stable_module_name,
            )
            return None

        try:
            return stable_cfg_cls()
        except Exception as exc:
            logger.warning(
                "Failed to instantiate stable task cfg '%s.%s': %s",
                stable_module_name,
                cfg_cls.__name__,
                exc,
            )
            return None

    def _apply_manager_term_cfg_profile(self) -> None:
        """Align term configs with manager modes for stable manager selections.

        When a manager is configured as STABLE (0), swap its corresponding config subtree
        from the stable task counterpart to keep manager-term type/signature compatibility.
        """
        manager_to_cfg_attr = {
            "ActionManager": "actions",
            "ObservationManager": "observations",
            "EventManager": "events",
            "RecorderManager": "recorders",
            "CommandManager": "commands",
            "TerminationManager": "terminations",
            "RewardManager": "rewards",
            "CurriculumManager": "curriculum",
        }

        stable_manager_names = [
            manager_name
            for manager_name in manager_to_cfg_attr
            if self._manager_call_switch.get_mode_for_manager(manager_name) == ManagerCallMode.STABLE
        ]
        if not stable_manager_names:
            return

        stable_cfg = self._resolve_stable_cfg_counterpart()
        if stable_cfg is None:
            logger.warning(
                "Stable managers requested (%s), but no stable cfg counterpart could be resolved."
                " Keeping experimental term configs.",
                ", ".join(stable_manager_names),
            )
            return

        replaced_items: list[str] = []
        for manager_name, cfg_attr in manager_to_cfg_attr.items():
            if self._manager_call_switch.get_mode_for_manager(manager_name) != ManagerCallMode.STABLE:
                continue
            if not hasattr(self.cfg, cfg_attr) or not hasattr(stable_cfg, cfg_attr):
                continue
            setattr(self.cfg, cfg_attr, deepcopy(getattr(stable_cfg, cfg_attr)))
            replaced_items.append(f"{manager_name} -> cfg.{cfg_attr}")

        if replaced_items:
            print("[INFO] Applied stable term config profile for managers:")
            for item in replaced_items:
                print(f"  - {item}")

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self._global_env_step_count_wp.fill_(env_step_count)
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=self._global_env_step_count_wp
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        env_mask = self.resolve_env_mask(env_ids=env_ids)
        # -- observation manager
        info = self.observation_manager.reset(env_mask=env_mask)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_mask=env_mask)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_mask=env_mask)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)
