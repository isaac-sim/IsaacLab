# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_instance_randomize_env_cfg import (
    StackInstanceRandomizeEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cubes_in_focus = EventTerm(
        func=franka_stack_events.randomize_rigid_objects_in_focus,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
            "out_focus_state": torch.tensor([10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
        },
    )


@configclass
class FrankaCubeStackInstanceRandomizeEnvCfg(StackInstanceRandomizeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Reduce the number of environments due to camera resources
        self.scene.num_envs = 2

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube to be a collection of rigid objects
        cube_1_config_dict = {
            "blue_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_1_Blue",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
            "red_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_1_Red",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0403], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
        }

        cube_2_config_dict = {
            "red_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_2_Red",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
            "yellow_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_2_Yellow",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0403], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
        }

        cube_3_config_dict = {
            "yellow_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_3_Yellow",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
            "green_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_2_Green",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0403], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
        }

        self.scene.cube_1 = RigidObjectCollectionCfg(rigid_objects=cube_1_config_dict)
        self.scene.cube_2 = RigidObjectCollectionCfg(rigid_objects=cube_2_config_dict)
        self.scene.cube_3 = RigidObjectCollectionCfg(rigid_objects=cube_3_config_dict)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
