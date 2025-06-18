# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import torch
from pathlib import Path

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.gr1t2_locomanipulation_robot_cfg import (  # isort: skip
    GR1T2_LOCOMANIPULATION_ROBOT_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.pink_controller_cfg import (  # isort: skip
    GR1T2_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp


##
# Scene definition
##
@configclass
class FixedBaseUpperBodyIKGR1T2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed base upper body IK GR1T2 environment."""

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = GR1T2_LOCOMANIPULATION_ROBOT_CFG

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Post initialization."""
        # Set the robot to fixed base
        self.robot.init_state.pos = (0, 0, 0.93)
        self.robot.init_state.rot = (0.7071, 0, 0, 0.7071)

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = GR1T2_UPPER_BODY_IK_ACTION_CFG

    def __post_init__(self):
        """Post initialization."""
        # Rotation by -90 degrees around Y-axis: (cos(90/2), 0, sin(90/2), 0) = (0.7071, 0, -0.7071, 0)
        self.upper_body_ik.controller.hand_rotational_offset = (0.7071, 0.0, -0.7071, 0.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    policy = None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# MDP settings
##


@configclass
class FixedBaseUpperBodyIKGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 fixed base upper body IK environment."""

    # Scene settings
    scene: FixedBaseUpperBodyIKGR1T2SceneCfg = FixedBaseUpperBodyIKGR1T2SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200
        self.sim.render_interval = 2

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, tempfile.gettempdir(), force_conversion=True
        )

        ControllerUtils.change_revolute_to_fixed(temp_urdf_output_path, self.actions.upper_body_ik.ik_urdf_fixed_joint_names)

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
