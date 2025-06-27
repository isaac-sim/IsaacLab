# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

from . import dev_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# wp.init()
# ## state machine config 
# class PickSmState:
#     """States for the pick state machine."""
#     REST = wp.constant(0)
#     APPR = wp.constant(1)
#     GRASP = wp.constant(2)
#     LIFT = wp.constant(3)
#     APP_GOAL = wp.constant(4)

# class PickSmWaitTime:
#     """Additional wait times (in s) for states for before switching."""

#     REST = wp.constant(0.5)
#     APPR = wp.constant(0.5)
#     GRASP = wp.constant(0.5)
#     LIFT = wp.constant(0.5)
#     APPR_GOAL = wp.constant(0.5)

# class GripperState:
#     """States for the gripper."""

#     OPEN = wp.constant(1.0)
#     CLOSE = wp.constant(-1.0)

# @wp.kernal
# def infer_state(
#     dt: wp.array(dtype=float),
#     sm_state: wp.array(dtype=int),
#     sm_wait_time: wp.array(dtype=float),
#     ee_pose: wp.array(dtype=wp.transform),
#     object_pose: wp.array(dtype=wp.transform),
#     des_object_pose: wp.array(dtype=wp.transform),
#     des_ee_pose: wp.array(dtype=wp.transform),
#     gripper_state: wp.array(dtype=float),
#     offset: wp.array(dtype=wp.transform),
# ):
#     tid = wp.tid()
#     state = sm_state[tid]
#     # decide next state
#     if state == PickSmState.REST:
#         des_ee_pose[tid] = ee_pose[tid]
#         gripper_state[tid] = GripperState.OPEN
#         # wait for a while
#         if sm_wait_time[tid] >= PickSmWaitTime.REST:
#             # move to next state and reset wait time
#             sm_state[tid] = PickSmState.APPR
#             sm_wait_time[tid] = 0.0
#     elif state == PickSmState.APPR:
#         des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
#         gripper_state[tid] = GripperState.OPEN
#         # TODO: error between current and desired ee pose below threshold

#         # wait for a while
#         if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
#             # move to next state and reset wait time
#             sm_state[tid] = PickSmState.APPROACH_OBJECT
#             sm_wait_time[tid] = 0.0
#     elif state == PickSmState.APPROACH_OBJECT:
#         des_ee_pose[tid] = object_pose[tid]
#         gripper_state[tid] = GripperState.OPEN
#         # TODO: error between current and desired ee pose below threshold
#         # wait for a while
#         if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
#             # move to next state and reset wait time
#             sm_state[tid] = PickSmState.GRASP_OBJECT
#             sm_wait_time[tid] = 0.0
#     elif state == PickSmState.GRASP_OBJECT:
#         des_ee_pose[tid] = object_pose[tid]
#         gripper_state[tid] = GripperState.CLOSE
#         # wait for a while
#         if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
#             # move to next state and reset wait time
#             sm_state[tid] = PickSmState.LIFT_OBJECT
#             sm_wait_time[tid] = 0.0
#     elif state == PickSmState.LIFT_OBJECT:
#         des_ee_pose[tid] = des_object_pose[tid]
#         gripper_state[tid] = GripperState.CLOSE
#         # TODO: error between current and desired ee pose below threshold
#         # wait for a while
#         if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
#             # move to next state and reset wait time
#             sm_state[tid] = PickSmState.LIFT_OBJECT
#             sm_wait_time[tid] = 0.0
#     # increment wait time
#     sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # replace with relative position controller 
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
       


@configclass
class FrankaCubeEnvCfg_PLAY(FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
