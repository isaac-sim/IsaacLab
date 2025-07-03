# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import torch
from pathlib import Path

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation import mdp

from source.isaaclab_tasks.isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.g1_locomanipulation_robot_cfg import (  # isort: skip
    G1_LOCOMANIPULATION_ROBOT_CFG,
)
from source.isaaclab_tasks.isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)



##
# Scene definition
##
@configclass
class FixedBaseUpperBodyIKG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed base upper body IK environment with G1 robot.

    This configuration sets up the G1 humanoid robot with fixed pelvis and legs,
    allowing only arm manipulation while the base remains stationary. The robot is
    controlled using upper body IK.
    """

    # Unitree G1 Humanoid robot - fixed base configuration
    robot: ArticulationCfg = G1_LOCOMANIPULATION_ROBOT_CFG

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
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG


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
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fixed base upper body IK environment.

    This environment is designed for manipulation tasks where the G1 humanoid robot
    has a fixed pelvis and legs, allowing only arm and hand movements for manipulation. The robot is
    controlled using upper body IK.
    """

    # Scene settings
    scene: FixedBaseUpperBodyIKG1SceneCfg = FixedBaseUpperBodyIKG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, tempfile.gettempdir(), force_conversion=True
        )

        # Convert revolute joints to fixed joints for pelvis and legs
        ControllerUtils.change_revolute_to_fixed_regex(
            temp_urdf_output_path, self.actions.upper_body_ik.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path
