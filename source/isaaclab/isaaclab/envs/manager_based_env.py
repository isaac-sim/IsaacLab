# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.timer import Timer

from .common import VecEnvObs
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ViewportCameraController
from .utils.io_descriptors import export_articulations_data, export_scene_data

# import logger
logger = logging.getLogger(__name__)


class ManagerBasedEnv:
    """The base environment encapsulates the simulation scene and the environment managers for
    the manager-based workflow.

    While a simulation scene or world comprises of different components such as the robots, objects,
    and sensors (cameras, lidars, etc.), the environment is a higher level abstraction
    that provides an interface for interacting with the simulation. The environment is comprised of
    the following components:

    * **Scene**: The scene manager that creates and manages the virtual world in which the robot operates.
      This includes defining the robot, static and dynamic objects, sensors, etc.
    * **Observation Manager**: The observation manager that generates observations from the current simulation
      state and the data gathered from the sensors. These observations may include privileged information
      that is not available to the robot in the real world. Additionally, user-defined terms can be added
      to process the observations and generate custom observations. For example, using a network to embed
      high-dimensional observations into a lower-dimensional space.
    * **Action Manager**: The action manager that processes the raw actions sent to the environment and
      converts them to low-level commands that are sent to the simulation. It can be configured to accept
      raw actions at different levels of abstraction. For example, in case of a robotic arm, the raw actions
      can be joint torques, joint positions, or end-effector poses. Similarly for a mobile base, it can be
      the joint torques, or the desired velocity of the floating base.
    * **Event Manager**: The event manager orchestrates operations triggered based on simulation events.
      This includes resetting the scene to a default state, applying random pushes to the robot at different intervals
      of time, or randomizing properties such as mass and friction coefficients. This is useful for training
      and evaluating the robot in a variety of scenarios.
    * **Recorder Manager**: The recorder manager that handles recording data produced during different steps
      in the simulation. This includes recording in the beginning and end of a reset and a step. The recorded data
      is distinguished per episode, per environment and can be exported through a dataset file handler to a file.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
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
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
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
            with use_stage(self.sim.stage):
                self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.has_gui:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                # since the reset can trigger callbacks which use the stage,
                # we need to set the stage context here
                with use_stage(self.sim.stage):
                    self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments
                # so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
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
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
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
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.wait_for_textures and self.has_rtx_sensors:
            # Wait for assets to finish loading (PhysX-specific)
            if hasattr(self.sim.physics_manager, "assets_loading"):
                while self.sim.physics_manager.assets_loading():
                    self.sim.render()

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
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

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
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
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
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step).
        Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: uses cached property to avoid settings lookup every step
        is_rendering = self.sim.is_rendering

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
            # Stop simulation first to allow physics to clean up properly
            self.sim.stop()

            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene

            # clear callbacks and instance
            self.sim.clear_instance()

            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """

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
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)
