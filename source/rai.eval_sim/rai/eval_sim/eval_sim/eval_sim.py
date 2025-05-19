# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import os
import signal
from abc import abstractmethod
from collections import deque
from time import perf_counter, sleep

import isaacsim.core.utils.stage as stage_utils
import rclpy
import torch
from isaaclab.envs import (
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from rai.eval_sim.eval_sim import EvalSimCfg
from rai.eval_sim.ros_manager import RosManager, RosManagerCfg
from rai.eval_sim.utils import (
    USER_EVAL_SIM_CFG_PATH,
    find_subclasses_of_base,
    log_error,
    log_info,
    log_warn,
    update_env_cfg,
    zero_actions,
)
from rclpy.node import Node
from std_srvs.srv import Empty


class EvalSim:
    def __init__(self, cfg: EvalSimCfg) -> None:
        """Base class for EvalSim.

        This class provides the base functionality for EvalSim. It is subclassed by EvalSimGUI for the extension-based EvalSim
        and EvalSimStandalone for the standalone EvalSim (non-extension).

        Attributes:
            cfg: The configuration object.
            env_cfg: The Isaac Lab environment configuration object.
            env: The Isaac Lab environment object.
            ros_manager_cfg: The RosManager configuration object.
            ros_manager: The RosManager object for publishing and subscribing to ROS topics.
            wallclock_dt_buffer: A deque to store the wallclock time deltas.
            t_prev: The previous wallclock time.
            action_queue: A list of actions to be applied. Only used if control_delay > 0.
            video_recorder: The video recorder object for recording the viewport.

        """
        self.cfg = cfg

        log_info(f"Starting EvalSim with config: \n{cfg}")

        self.env_cfg = None
        self.env = None
        self.ros_manager_cfg = None
        self.ros_manager = None

        self.load_default_configs()
        self.setup_settings()

        # initialize runtime variables
        self.wallclock_dt_buffer = deque(maxlen=cfg.wallclock_dt_buffer_size)
        self.t_prev = perf_counter()

        # action delay queue
        self.action_queue: list[torch.Tensor] = []

        # records video from viewport
        self.video_recorder = None

        # used to skip the first subscription after enabling ROS
        self._prev_ros_enable = False
        self._prev_control_delay = cfg.control_delay

        # if True, random actions are used. if False, zero actions are used
        self._random_actions = False

        # create shared ros node
        rclpy.init()
        self.start_ros_node()

        # if True all articulations in the scene will become fixed based. This can be used to fix floating base robots
        # and release them in the middle of simulation.
        self._fix_robot_base = False

        # register signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.sigint_handler)

        # create service to reload the simulation
        # NOTE: we use Empty service type because we currently don't have a way of reporting success of the reload,
        # so we don't have a response message
        self._reload_service = self._ros_node.create_service(
            Empty, f"{self._ros_node_name}/reload", self._reload_callback
        )

    """
    Core Functions
    """

    def step(self):
        # deployment specific step
        self.step_deployment()

        # get the wallclock time of the last simulation decimation step
        sim_wallclock_dt = self.wallclock_dt_buffer[-1]
        if self.cfg.sync_to_real_time and sim_wallclock_dt < self.env.step_dt:
            # sleep simulation loop if running faster than real time
            sleep(self.env.step_dt - sim_wallclock_dt)

    @abstractmethod
    def step_deployment(self):
        """This method is implemented in the subclasses to define the deployment specific step."""

    def reset(self):
        # reset environment
        if self.env is not None:
            self.env.reset()

        # reset timeline variables
        self.reset_timeline_variables()

    def close(self):
        """Close environment, ros manager, and the ros node."""
        self.close_env_and_ros_manager()

        # close ros node
        self.stop_ros_node()
        rclpy.try_shutdown()

    def close_env_and_ros_manager(self):
        """Close active EvalSim components (environment and ros manager)."""

        # close environment and simulation context
        if self.env is not None:
            try:
                self.env.close()
                self.env = None
            except Exception as e:
                log_error(f"Environment couldn't be closed. Exception: {e}")

        # close ros manager
        if self.ros_manager is not None:
            try:
                self.ros_manager.close()
                self.ros_manager = None
            except Exception as e:
                log_error(f"RosManager couldn't be shutdown. Exception: {e}")

    def clear(self):
        """Closes the environment, and ros manager, creates a new scene and resets the simulator"""
        # close environment and ros_manager
        self.close_env_and_ros_manager()

        # create a new stage to clear the scene
        stage_utils.create_new_stage()

        # reset timeline variables
        self.reset_timeline_variables()

    def pause(self):
        """Pauses the simulation."""
        if self.env is not None and self.env.sim is not None:
            self.env.sim.pause()

    def play(self):
        if self.env.sim is not None:
            self.env.sim.play()

    def stop(self) -> None:
        """Stops the simulation.

        This function handles the cleanup of the simulation context and the stage.
        """
        # check if simulation context exists
        if self.env.sim is not None:
            # stop the simulation
            if not self.env.sim.is_stopped():
                self.env.sim.stop()
            # clear TODO: check if this is necessary here
            self.env.sim.clear()
            # clear all callbacks
            self.env.sim.clear_all_callbacks()
            # clear simulation
            self.env.sim.clear_instance()
            # collect garbage
            gc.collect()

        # close environment
        self.env.close()

        # close ros manager
        self.ros_manager.close()

    def start_ros_node(self):
        """Creates an instance of the ROS node eval_sim."""
        self._ros_node_name = "eval_sim"
        self._ros_node = Node(self._ros_node_name)

    def stop_ros_node(self):
        """Destroys ROS node and its publishers and subscribers."""
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None

    """
    Properties
    """

    @property
    def ros_enabled(self) -> bool:
        return self._ros_enabled

    @property
    def physics_dt(self):
        """Ensures always using the environments physics dt."""
        if self.env is None:
            raise ValueError("Environment must exist for accessing physics_dt.")

        return self.env.physics_dt

    """
    Configuration and Loading
    """

    def load(self):
        """Load the environment and ROS manager based on the current configuration."""
        if self._ros_node is None:
            self.start_ros_node()

        self.load_env()
        self.load_ros_manager()

    def load_env(self):
        """Load the Isaac Lab environment based on the current configuration."""
        if isinstance(self.env_cfg, ManagerBasedRLEnvCfg):
            self.env = ManagerBasedRLEnv(self.env_cfg)
        elif isinstance(self.env_cfg, ManagerBasedEnvCfg):
            self.env = ManagerBasedEnv(self.env_cfg)
        else:
            raise ValueError(
                "Current set environment configuration is not of type ManagerBasedRLEnvCfg or ManagerBasedEnvCfg."
                f" Current type: {type(self.env_cfg)}. If you are using EvalSimStandalone, make sure to set a valid"
                f" environment configuration in the user settings file ({USER_EVAL_SIM_CFG_PATH}) as this can't be set"
                " in GUI."
            )
        # log wallclock at env load
        self.log_wallclock_time()
        # prefill action buffer with zero_actions
        self.action_queue = [zero_actions(self.env) for _ in range(self.cfg.control_delay)]

        # need to update user settings to reflect the loaded environment
        if self.cfg.auto_save:
            self.cfg.to_yaml()

    def load_ros_manager(self):
        """Load the ROS manager based on the current configuration."""
        if self.env is None:
            raise ValueError("Environment must exist to load a ros manager.")

        # load ros manager
        if self.ros_manager_cfg is not None:
            self.ros_manager = RosManager(cfg=self.ros_manager_cfg, env=self.env, node=self._ros_node)

            # enable / disable ros based on user settings
            if self.cfg.enable_ros:
                self.enable_ros()
            else:
                self.disable_ros()
        else:
            log_warn("RosManager is not loaded, due to non-existent configuration.")

        # need to update user settings to reflect the loaded ros manager
        if self.cfg.auto_save:
            self.cfg.to_yaml()

    @abstractmethod
    def reload(self):
        """Reload EvalSim configs and the currently loaded environment."""

    def _reload_callback(self, request, response):
        """Callback for the reload service."""
        self.reload()

        return response

    def set_env_cfg(self, env_cfg_str: str):
        """Construct the environment configuration based on the given string.

        Args:
            env_cfg_str: The environment configuration to set in string form. Should be a class that inherits from
                ManagerBasedEnvCfg or ManagerBasedRLEnvCfg and contained within self.env_configs.
                For example: "AnymalDEnvCfg".

        Raises:
            ValueError: If the environment configuration is not found in the loaded configurations.
        """
        try:
            env_cfg_class = self.env_configs[env_cfg_str]
        except KeyError:
            raise ValueError(
                f"Environment configuration {env_cfg_str} not found in loaded configurations."
                f" Available configurations: {self.env_configs.keys()}"
            )

        # construct the config object
        self.env_cfg: ManagerBasedEnvCfg | ManagerBasedRLEnvCfg = env_cfg_class()

        # update configuration for use with EvalSim
        update_env_cfg(self.env_cfg)

        # log loaded environment
        log_info(f"EnvCfg set to {env_cfg_str}")

        # update self.cfg
        self.cfg.env_cfg = env_cfg_str

    def set_ros_manager_cfg(self, ros_manager_cfg_str: str):
        """Construct the ROS manager configuration based on the given string.

        Args:
            ros_manager_cfg_str: The ROS manager configuration to set in string form.
                Should be a class that inherits from RosManagerCfg and contained within self.ros_manager_configs.
                For example: "AnymalDRosManagerCfg".

        Raises:
            ValueError: If the ROS manager configuration is not found in the loaded configurations.
        """
        try:
            ros_manager_cfg_class = self.ros_manager_configs[ros_manager_cfg_str]
        except KeyError:
            raise ValueError(
                f"RosManager configuration {ros_manager_cfg_str} not found in loaded configurations."
                f" Available configurations: {self.ros_manager_configs.keys()}"
            )

        # construct the config object
        self.ros_manager_cfg: RosManagerCfg = ros_manager_cfg_class()

        # log loaded environment
        log_info(f"RosManagerCfg set to {ros_manager_cfg_str}")

        # update self.cfg
        self.cfg.ros_manager_cfg = ros_manager_cfg_str

    def load_default_configs(self):
        """Load the default environment and ROS manager configurations.

        These are loaded from the packages specified in the configuration in the search_pkgs_for_cfgs list.

        Raises:
            AssertionError: If there are duplicate class names found in the loaded configurations.
        """
        self.env_configs = {}
        self.ros_manager_configs = {}

        # Extract all environment and ros manager configurations from the specified packages
        env_cfg_pkgs = find_subclasses_of_base(
            packages=self.cfg.search_pkgs_for_cfgs,
            base_class=ManagerBasedEnvCfg,
            ignore_classes=[ManagerBasedRLEnvCfg],
        )

        ros_manager_cfg_pkgs = find_subclasses_of_base(packages=self.cfg.search_pkgs_for_cfgs, base_class=RosManagerCfg)

        # Store each configuration in the respective dictionary
        # We use the module name as the key e.g. "rai.eval_sim.tasks.anymal_env_cfg.AnymalDEnvCfg"
        # (rather than "AnymalDEnvCfg" to avoid conflicts in case of same class name in different packages)
        # and store the class object for later use as the value
        for module_name, class_cfg in env_cfg_pkgs.items():
            assert module_name not in self.env_configs, f"Duplicate class name found: {module_name}"
            self.env_configs[module_name] = class_cfg

        for module_name, class_cfg in ros_manager_cfg_pkgs.items():
            assert module_name not in self.ros_manager_configs, f"Duplicate class name found: {module_name}"
            self.ros_manager_configs[module_name] = class_cfg

    def setup_settings(self):
        """Set settings based on self.cfg."""
        env_cfg = self.cfg.env_cfg
        ros_manager_cfg = self.cfg.ros_manager_cfg

        # set configurations only if they are not empty strings in user settings
        if env_cfg != "":
            try:
                self.set_env_cfg(env_cfg)
            except ValueError as e:
                log_warn(
                    f"EnvCfg defined in user settings file doesn't contain a ManagerBasedEnvCfg class. Exception {e}"
                )
        if ros_manager_cfg != "":
            try:
                self.set_ros_manager_cfg(ros_manager_cfg)
            except ValueError as e:
                log_warn(
                    f"RosManagerCfg defined in user settings file doesn't contain a RosManagerCfg class. Exception {e}"
                )

        self._ros_enabled = self.cfg.enable_ros

    """
    Sigint Handler
    """

    def sigint_handler(self, signum, frame):
        """Custom SIGINT handling callback.

        Runs when the user presses Ctrl+C to exit the program.
        """
        self.close()
        # SIGINT is currently being caught by the simulator, so we need to raise another SIGTERM to
        # forcibly exit the program
        os.kill(os.getpid(), signal.SIGTERM)

    """
    Profiling
    """

    def log_wallclock_time(self):
        """Stores the wallclock time deltas between decimations in a deque for profiling purposes."""
        # append dt for computing simulation control speeds
        t_curr = perf_counter()
        dt = t_curr - self.t_prev
        self.wallclock_dt_buffer.append(dt)
        self.dt_avg_ms = 1000 * sum(self.wallclock_dt_buffer) / len(self.wallclock_dt_buffer)
        self.t_prev = t_curr

    def get_simulation_time_profile(self):
        """Returns the simulation time profile.

        Returns:
            The time per physics step, time per step in Hz, and the simulation speed.

        """
        # update simulation parameters if buffer is full
        if len(self.wallclock_dt_buffer) == self.wallclock_dt_buffer.maxlen:
            physics_dt = self.dt_avg_ms / self.env.cfg.decimation
            time_per_step = f"{physics_dt:.3f} ms"
            time_per_step_hz = f"({1000.0 / physics_dt:.0f} Hz)"
            sim_speed = f"{100 * (1000 * self.env.step_dt) / self.dt_avg_ms:.2f}%" if self.dt_avg_ms > 0 else "N/A"
        else:
            time_per_step = "N/A"
            time_per_step_hz = "N/A"
            sim_speed = "N/A"
        return time_per_step, time_per_step_hz, sim_speed

    """
    Other
    """

    def set_fix_robot_base(self, enable: bool = False):
        """Sets the fixed based functionality for a robot.

        Args:
            enable: The flag to fix articulation base frames in place, if True."""
        self._fix_robot_base = enable

    def reset_timeline_variables(self):
        """Resets the timeline variables."""
        if self.env is not None:
            # Hacky, to modify privater time variables
            # but can't find another way to reset the timeline variables currently
            self.env.sim._current_time = 0
            self.env.sim._current_steps = 0

        # reset wallclock
        self.wallclock_dt_buffer.clear()
        self.log_wallclock_time()

    def set_physics_dt(self, dt: float):
        """Updates the physics_dt of the environment.

        Args:
            dt: The desired physics dt.

        Raises:
            ValueError: If the environment does not exist.
        """
        if self.env is None:
            raise ValueError("Environment must exist for setting physics_dt.")

        # update environment variable
        self.env.cfg.sim.dt = dt

        # update simulation
        self.env.sim.set_simulation_dt(physics_dt=dt)

        # inform user and reset clock
        log_info(f"Set physics_dt to {self.physics_dt:.4f}s")
        self.reset_timeline_variables()

    def apply_delay_to_action_queue(self):
        """Applies the control delay to the action queue.

        Raises:
            ValueError: If the delay specified in config is negative.
        """
        if self.cfg.control_delay < 0:
            raise ValueError("Delay must be positive")

        diff_delay = self.cfg.control_delay - len(self.action_queue)
        if diff_delay > 0:
            for _ in range(diff_delay):
                if not self.action_queue:
                    self.action_queue.append(zero_actions(self.env))
                else:
                    self.action_queue.append(self.action_queue[-1])
        elif diff_delay < 0:
            for _ in range(-diff_delay):
                if self.action_queue:
                    self.action_queue.pop()
        else:
            pass

    def time_delay_actions(self):
        """Adds a delayed action to the action queue.

        NOTE: This is currently not functional and needs to be updated to handle
        the delayed actions correctly.
        """
        # transition: skip first subscription after ros enabling
        if not self._prev_ros_enable:  # noqa
            self.action_queue.append(self.action_queue[-1])
        # transition: skip first subscription after increasing delay skip subscription
        elif self._prev_control_delay < self.cfg.control_delay:
            self.action_queue.append(self.action_queue[-1])
        # transition: decreasing delay
        elif self._prev_control_delay > self.cfg.control_delay:
            self.action_queue.append(self.ros_manager.subscribe())
        # non zero delay constant value
        else:
            self.action_queue.append(self.ros_manager.subscribe())

    """
    Video Recording
    """

    def start_recording(self):
        """Starts the video recording."""
        if self.video_recorder is not None:
            self.video_recorder.start_recording()
        else:
            log_warn("Video recorder is not loaded. Can not start / stop recording.")

    def stop_recording(self):
        """Stops the video recording."""
        if self.video_recorder is not None:
            self.video_recorder.stop_recording()
        else:
            log_warn("Video recorder is not loaded. Can not start / stop recording.")

    """
    Other
    """

    def set_debug_vis(self, debug_vis: bool) -> None:
        """This function is used to toggle the debug visualization of the simulator.

        Args:
            debug_vis: The flag to toggle the debug visualization.
        """
        if self.env:
            # visualize sensors
            for sensor_name, sensor in self.env.scene.sensors.items():
                log_warn(f"Setting {sensor_name} to {debug_vis}")
                sensor.set_debug_vis(debug_vis)

            # visualize commands
            command_manager = getattr(self.env, "command_manager", None)
            if command_manager is not None:
                command_manager.set_debug_vis(debug_vis)
        else:
            log_warn("Toggling debug visualization ineffective. Environment has not been loaded.")

    def enable_ros(self) -> None:
        """Enables the ROS manager."""
        assert self.ros_manager is not None, "RosManager must exist to enable ROS."
        self._ros_enabled = True
        log_info("ROS is enabled")
        if self.ros_manager.cfg.lockstep_timeout is None:
            log_warn(
                "RosManager.lockstep_timeout is not set. Simulation will run in lockstep with incoming control."
                " ROS Manager will wait indefinitely for action messages from a controller."
            )
        self.ros_manager.publish_static()

    def disable_ros(self) -> None:
        """Disables the ROS manager."""
        self._ros_enabled = False
        log_info("ROS is disabled.")

    def set_random_actions(self, case: bool) -> None:
        """Updates the random actions flag.

        Args:
            case: The flag to set the random actions.

        """
        self._random_actions = bool(case)
