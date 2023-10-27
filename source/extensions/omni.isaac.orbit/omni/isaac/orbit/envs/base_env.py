# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins

import omni.isaac.core.utils.torch as torch_utils

from omni.isaac.orbit.managers import ActionManager, ObservationManager, RandomizationManager
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.timer import Timer

from .base_env_cfg import BaseEnvCfg


class BaseEnv:
    """The base environment encapsulates the simulation scene and the environment managers.

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
    * **Randomization Manager**: The randomization manager that randomizes different elements in the scene.
      This includes resetting the scene to a default state or randomize the scene at different intervals
      of time. The randomization manager can be configured to randomize different elements of the scene
      such as the masses of objects, friction coefficients, or apply random pushes to the robot.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`BaseEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`BaseEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
    """

    def __init__(self, cfg: BaseEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.substeps}")
        print(f"\tEnvironment step-size : {self.step_dt}")
        print(f"\tPhysics GPU pipeline  : {self.cfg.sim.use_gpu_pipeline}")
        print(f"\tPhysics GPU simulation: {self.cfg.sim.physx.use_gpu}")

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            with Timer("[INFO]: Time taken for simulation reset"):
                self.sim.reset()
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="Orbit")
        else:
            # if no window, then we don't need to store the window
            self._window = None

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

    """
    Operations - Setup.
    """

    def load_managers(self):
        """Load the managers for the environment.

        Note:
            This must happen after the simulator is reset, i.e. after the first call to :meth:`self.sim.reset`.
        """
        # prepare the managers
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- randomization manager
        self.randomization_manager = RandomizationManager(self.cfg.randomization, self)
        print("[INFO] Randomization Manager: ", self.randomization_manager)

    """
    Operations - MDP.
    """

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        import omni.replicator.core as rep

        rep.set_global_seed(seed)
        return torch_utils.set_seed(seed)

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True
