# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import torch
from pathlib import Path

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.gr1t2_locomanipulation_robot_cfg import (  # isort: skip
    GR1T2_LOCOMANIPULATION_ROBOT_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.pink_controller_cfg import (  # isort: skip
    GR1T2_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


##
# Scene definition
##
@configclass
class FixedBaseUpperBodyIKGR1T2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed base upper body IK GR1T2 environment."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.30, 1.0413-0.3], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.35,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

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
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_left_eef_pos, params={"link_name": "left_hand_pitch_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_left_eef_quat, params={"link_name": "left_hand_pitch_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_right_eef_pos, params={"link_name": "right_hand_pitch_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_right_eef_quat, params={"link_name": "right_hand_pitch_link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_hand_state, params={"hand_joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(func=manip_mdp.get_head_state, params={"head_joint_names": []})

        object = ObsTerm(func=manip_mdp.object_obs, params={"left_eef_link_name": "left_hand_pitch_link", "right_eef_link_name": "right_hand_pitch_link"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    success = DoneTerm(func=manip_mdp.task_done_pick_place, params={"relevant_link_name": "right_hand_roll_link"})


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
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120
        self.sim.render_interval = 2

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, tempfile.gettempdir(), force_conversion=True
        )

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
