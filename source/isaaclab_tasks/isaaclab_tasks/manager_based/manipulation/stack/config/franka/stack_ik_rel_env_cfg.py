# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def _build_franka_se3_rel_gripper_pipeline(hand_side="right"):
    """Build an IsaacTeleop Se3Rel + Gripper pipeline for Franka manipulator teleoperation.

    Creates a Se3RelRetargeter for end-effector delta pose tracking and
    a GripperRetargeter for pinch-based gripper control from hand tracking data.
    All outputs are flattened into a single 7D action tensor via TensorReorderer.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, PassthroughInput
    from isaacteleop.retargeting_engine.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = PassthroughInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(PassthroughInput.VALUE))

    hand_key = HandsSource.LEFT if hand_side == "left" else HandsSource.RIGHT

    # SE3 Relative Pose Retargeter
    se3_cfg = Se3RetargeterConfig(
        input_device=hand_key,
        zero_out_xy_rotation=True,
        use_wrist_rotation=False,
        use_wrist_position=True,
        delta_pos_scale_factor=10.0,
        delta_rot_scale_factor=10.0,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({hand_key: transformed_hands.output(hand_key)})

    # Gripper Retargeter (pinch-based)
    gripper_cfg = GripperRetargeterConfig(hand_side=hand_side)
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    controller_key = ControllersSource.LEFT if hand_side == "left" else ControllersSource.RIGHT
    connected_gripper = gripper.connect(
        {
            f"hand_{hand_side}": hands.output(hand_key),
            f"controller_{hand_side}": controllers.output(controller_key),
        }
    )

    # TensorReorderer: flatten into a 7D action tensor [delta_pose(6), gripper(1)]
    ee_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
    gripper_elements = ["gripper_cmd"]

    reorderer = TensorReorderer(
        input_config={
            "ee_delta": ee_elements,
            "gripper": gripper_elements,
        },
        output_order=ee_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_delta": "array",
            "gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_delta": connected_se3.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline


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
        pipeline = _build_franka_se3_rel_gripper_pipeline(hand_side="right")
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
