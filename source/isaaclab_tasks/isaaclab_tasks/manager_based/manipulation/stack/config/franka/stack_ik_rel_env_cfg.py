# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import mdp

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def _build_franka_stack_rel_pipeline():
    """Build a IsaacTeleop retargeting pipeline for Franka cube stacking (relative mode).

    Creates an Se3RelRetargeter for right-hand relative pose tracking and a GripperRetargeter
    for right-hand gripper control, flattened into a single action tensor via
    TensorReorderer.

    Returns:
        OutputCombiner with a single "action" output containing the flattened
        7D action tensor: [delta_pos_x, delta_pos_y, delta_pos_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper].
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to hand poses so that
    # downstream retargeters receive data in the simulation world frame.
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # SE3 Relative Pose Retargeter (right hand)
    # Note: Se3RelRetargeter outputs 6D delta (3D position + 3D rotation)
    se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=True,
        use_wrist_rotation=False,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # Gripper Retargeter (right hand)
    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # TensorReorderer to flatten into a single action vector
    # Se3RelRetargeter outputs a 6D NDArray (delta_pos xyz + delta_rot xyz)
    # GripperRetargeter outputs a single float (gripper command)
    ee_pose_elements = ["delta_pos_x", "delta_pos_y", "delta_pos_z", "delta_rot_x", "delta_rot_y", "delta_rot_z"]
    gripper_elements = ["gripper_value"]

    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            "gripper_command": gripper_elements,
        },
        output_order=ee_pose_elements + gripper_elements,
        name="action_reorderer",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper_command": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class FrankaCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # IsaacTeleop-based teleoperation pipeline
        if _TELEOP_AVAILABLE:
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=_build_franka_stack_rel_pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )


@configclass
class FrankaCubeStackRedGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={"cube_1_cfg": SceneEntityCfg("cube_2"), "cube_2_cfg": SceneEntityCfg("cube_3"), "cube_3_cfg": None},
        )


@configclass
class FrankaCubeStackRedGreenBlueEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_2"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_1"),
            },
        )


@configclass
class FrankaCubeStackBlueGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={"cube_1_cfg": SceneEntityCfg("cube_1"), "cube_2_cfg": SceneEntityCfg("cube_3"), "cube_3_cfg": None},
        )


@configclass
class FrankaCubeStackBlueGreenRedEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_2"),
            },
        )
