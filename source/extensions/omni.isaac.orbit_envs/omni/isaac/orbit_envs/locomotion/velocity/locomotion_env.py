# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List

import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.orbit.command_generators import *  # noqa: F401, F403
from omni.isaac.orbit.command_generators import CommandGeneratorBase
from omni.isaac.orbit.managers import (
    ActionManager,
    CurriculumManager,
    ObservationManager,
    RandomizationManager,
    RewardManager,
    TerminationManager,
)
from omni.isaac.orbit.robots.legged_robot import LeggedRobot
from omni.isaac.orbit.terrains import TerrainImporter

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .locomotion_cfg import LocomotionEnvCfg


class LocomotionEnv(IsaacEnv):
    """Environment for a legged robot."""

    def __init__(self, cfg: LocomotionEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg

        # create classes
        self.robot = LeggedRobot(cfg=self.cfg.robot)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()
        # setup randomization in environment
        # self._setup_randomization()

        # prepare the managers
        # -- command manager
        self._command_manager: CommandGeneratorBase = eval(self.cfg.commands.class_name)(
            self.cfg.commands, self
        )  # noqa: F405
        print("[INFO] Command Manager: ", self._command_manager)

        # -- action manager
        self._action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self._action_manager)
        self.num_actions = self._action_manager.total_action_dim
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.action_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_actions,))

        # -- observation manager
        self._observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self._observation_manager)
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))

        # -- reward manager
        self._reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # -- termination manager
        self._termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self._termination_manager)

        # -- curriculum manager
        self._curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self._curriculum_manager)

        # -- randomization manager
        self._randomization_manager = RandomizationManager(self.cfg.randomization, self)
        print("[INFO] Randomization Manager: ", self._randomization_manager)
        self._randomization_manager.randomize(mode="startup")

        print("[INFO]: Completed setting up the environment...")

        # extend UI elements
        # we need to this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self._orbit_window is not None:
            self._extend_ui()

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        """Design the scene for the environment."""

        # lights
        prim_utils.create_prim(
            "/World/distantLight",
            "DistantLight",
            translation=(0.0, 0.0, 500.0),
            attributes={"intensity": 3000.0, "color": (0.75, 0.75, 0.75)},
        )

        # terrain
        self.cfg.terrain.prim_path = "/World/ground"
        # check if curriculum is enabled
        if self.cfg.terrain.terrain_type == "generator":
            if hasattr(self.cfg.curriculum, "terrain_levels") and self.cfg.curriculum.terrain_levels is not None:
                self.cfg.terrain.terrain_generator.curriculum = True
            else:
                self.cfg.terrain.terrain_generator.curriculum = False
        # self.cfg.terrain.max_init_terrain_level = None
        self.terrain_importer = TerrainImporter(self.cfg.terrain, self.num_envs, device=self.device)

        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")
        return ["/World/ground"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        """Reset environments based on specified indices.

        Calls the following functions on reset:
        - :func:`_reset_robot_state`: Reset the root state and DOF state of the robot.
        - :func:`_resample_commands`: Resample the goal/command for the task. E.x.: desired velocity command.
        - :func:`_sim_randomization`: Randomizes simulation properties using replicator. E.x.: friction, body mass.

        Addition to above, the function fills up episode information into extras and resets buffers.

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        # -- curriculum manager: fills the current state of curriculum
        self._curriculum_manager.compute(env_ids=env_ids)
        # randomize the MDP
        # -- apply reset randomization
        self._randomization_manager.randomize(env_ids=env_ids, mode="reset")
        # -- robot buffers
        self.robot.reset_buffers(env_ids)

        # -- sensors
        for sensor in self.sensors.values():
            sensor.reset_buffers(env_ids)
        # -- Reward logging
        # fill extras with episode information
        self.extras["log"] = dict()
        # -- rewards manager: fills the sums for terminated episodes
        self.extras["log"].update(self._reward_manager.log_info(env_ids))
        # -- command manager
        self.extras["log"].update(self._command_manager.log_info(env_ids))
        # -- curriculum manager
        self.extras["log"].update(self._curriculum_manager.log_info(env_ids))
        self._command_manager.reset(env_ids)
        # -- randomization manager
        self.extras["log"].update(self._randomization_manager.log_info(env_ids))
        # -- termination manager
        self.extras["log"].update(self._termination_manager.log_info(env_ids))
        # -- reset history
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1  # should not be needed anymore since it's already 1

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        # clip actions and move to env device
        self.actions = actions.clone().to(device=self.device)
        # scaled actions
        self._action_manager.process_actions(self.actions)
        # perform physics stepping
        for i in range(self.cfg.control.decimation):
            # set actions into buffers
            self._action_manager.apply_actions()
            self.robot.write_commands_to_sim()
            # simulate
            need_render = self.enable_render and (i == 0)
            self.sim.step(render=need_render)
            # refresh root states
            self.robot.refresh_sim_data()
            # -- update sensor buffers
            for sensor in self.sensors.values():
                sensor.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- update robot buffers
        self.robot.update_buffers(dt=self.dt)

        # compute MDP signals
        # check terminations
        self.reset_buf = self._termination_manager.compute().to(torch.long)
        # reward computation
        self.reward_buf = self._reward_manager.compute()
        # reset envs that terminated
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- store history
        self.previous_actions[:] = self.actions
        # -- update command
        self._command_manager.compute()
        # -- step interval randomizations
        self._randomization_manager.randomize(mode="interval")

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        if self.cfg.env.send_time_outs:
            self.extras["time_outs"] = self._termination_manager.time_outs

    def _get_observations(self) -> VecEnvObs:
        return self._observation_manager.compute()

    def _debug_vis(self):
        self.robot.debug_vis()
        self._command_manager.debug_vis()
        # sensors
        for sensor in self.sensors.values():
            sensor.debug_vis()

    """
    Internal functions.
    """

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        # define action space
        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- history

    """
    Build UI with individual toggling.
    """

    def _extend_ui(self):
        # need to import here to wait for the GUI extension to be loaded
        import omni.isaac.ui.ui_utils as ui_utils
        import omni.ui as ui

        with self._orbit_window_elements["main_frame"]:
            with self._orbit_window_elements["main_vstack"]:
                # create collapsable frame for debug visualization
                self._orbit_window_elements["debug_frame"] = ui.CollapsableFrame(
                    title="Visualization Elements",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=ui_utils.get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._orbit_window_elements["debug_frame"]:
                    # create stack for controls
                    self._orbit_window_elements["debug_vstack"] = ui.VStack(spacing=5, height=0)
                    with self._orbit_window_elements["debug_vstack"]:
                        # create debug visualization checkbox
                        debug_vis_checkbox = {
                            "label": "Command Generator",
                            "type": "checkbox",
                            "default_val": self._command_manager.cfg.debug_vis,
                            "tooltip": "Toggle command visualization",
                            "on_clicked_fn": self._toggle_command_vis_flag,
                        }
                        _ = ui_utils.cb_builder(**debug_vis_checkbox)
                        # create debug visualization checkbox
                        debug_vis_checkbox = {
                            "label": "Robot",
                            "type": "checkbox",
                            "default_val": self.robot.cfg.debug_vis,
                            "tooltip": "Toggle robot frames visualization",
                            "on_clicked_fn": self._toggle_robot_vis_flag,
                        }
                        _ = ui_utils.cb_builder(**debug_vis_checkbox)
                        # create debug visualization checkbox
                        if "height_scanner" in self.sensors:
                            debug_vis_checkbox = {
                                "label": "Height Scanner",
                                "type": "checkbox",
                                "default_val": self.cfg.sensors.height_scanner.debug_vis,
                                "tooltip": "Toggle height scanner visualization",
                                "on_clicked_fn": self._toggle_height_scanner_vis_flag,
                            }
                            _ = ui_utils.cb_builder(**debug_vis_checkbox)
                        # create debug visualization checkbox
                        if "contact_forces" in self.sensors:
                            debug_vis_checkbox = {
                                "label": "Contact Sensor",
                                "type": "checkbox",
                                "default_val": self.sensors["contact_forces"].cfg.debug_vis,
                                "tooltip": "Toggle contact sensor visualization",
                                "on_clicked_fn": self._toggle_contact_sensor_vis_flag,
                            }
                            _ = ui_utils.cb_builder(**debug_vis_checkbox)

    def _toggle_command_vis_flag(self, value: bool):
        """Toggle command debug visualization flag."""
        self._command_manager.cfg.debug_vis = value

    def _toggle_robot_vis_flag(self, value: bool):
        """Toggle robot debug visualization flag."""
        self.robot.cfg.debug_vis = value

    def _toggle_height_scanner_vis_flag(self, value: bool):
        """Toggle sensor debug visualization flag."""
        self.sensors["height_scanner"].cfg.debug_vis = value
        if self.sensors["height_scanner"].ray_visualizer is not None:
            self.sensors["height_scanner"].ray_visualizer.set_visibility(value)

    def _toggle_contact_sensor_vis_flag(self, value: bool):
        """Toggle sensor debug visualization flag."""
        self.sensors["contact_forces"].cfg.debug_vis = value
        if self.sensors["contact_forces"].contact_visualizer is not None:
            self.sensors["contact_forces"].contact_visualizer.set_visibility(value)
