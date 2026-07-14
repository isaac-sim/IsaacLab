# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip


@configclass
class SO101CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SO-101 as robot, yawed 90 deg to face the cube workspace along world +X, with a
        # manipulation-ready home pose: gripper open ~10 cm above the workspace, fingers
        # pointing down, clear of joint-limit singularities. Starting from the asset's all-zero
        # state instead was smoke-tested and fails: the arm reaches the cube sideways, the
        # resulting side pinch cannot survive lifting, and no lift is learned within 2x the
        # iterations the home pose needs for 30% success
        self.scene.robot = SO101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.03008),
                rot=(0.0, 0.0, 0.70710678, 0.70710678),
                joint_pos={
                    "shoulder_pan": 0.093,
                    "shoulder_lift": -0.352,
                    "elbow_flex": 0.282,
                    "wrist_flex": 1.353,
                    "wrist_roll": 0.101,
                    "gripper": 1.745,
                },
            ),
        )
        # Cap joint speeds at the real STS3215 servo's no-load speed (the asset's default
        # limit lets the lightly damped arm whip and bat the cube off the table), and soften
        # the jaw drive so it cannot crush the lightweight cube
        self.scene.robot.actuators["arm"].velocity_limit_sim = 5.0
        self.scene.robot.actuators["gripper"].effort_limit_sim = 0.4
        self.scene.robot.actuators["gripper"].velocity_limit_sim = 5.0

        # Set actions for the specific robot type (SO-101). The action scale is smaller than
        # the Franka variant's: the SO-101 workspace is compact and larger position-target
        # steps drive violent contacts with the cube
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.25,
            use_default_offset=True,
        )
        # Continuous jaw control instead of the Franka-style binary gripper: the jaw sweeps
        # 1.745 rad in ~18 control steps, so a bang-bang command re-decided from a noisy
        # policy output every step chatters and never completes a close during exploration.
        # The offset centers the neutral action at a half-open pocket admitting the cube
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            offset=0.8,
            use_default_offset=False,
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper"
        # Shrink the goal workspace to the SO-101's compact reach. The base is yawed +90 deg,
        # so robot-root -Y points along world +X (toward the cube workspace)
        self.commands.object_pose.ranges.pos_x = (-0.10, 0.10)
        self.commands.object_pose.ranges.pos_y = (-0.28, -0.16)
        self.commands.object_pose.ranges.pos_z = (0.10, 0.20)

        # Set Cube as object: a light 3 cm cube sized to the SO-101 jaw span and effort
        # limits (the SO-101 gripper is much smaller than the Franka hand). The low
        # depenetration velocity keeps contacts from ejecting it ballistically
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.23, 0, 0.0155], rot=[0, 0, 0, 1]),
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=10.0,
                    max_linear_velocity=2.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            ),
        )
        # Randomize the cube around its nominal pose within the reachable workspace
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.08, 0.08),
            "z": (0.0, 0.0),
        }
        # Tighten and strengthen the reaching kernel: the single-jaw gripper needs the grasp
        # center within ~1 cm of the cube, where the Franka-tuned term has little gradient
        self.rewards.reaching_object.params["std"] = 0.04
        self.rewards.reaching_object.weight = 2.0

        # Reward a true pinch grasp (jaw stalled on the cube with the grasp center at the
        # object) and gate the lifting and goal-tracking rewards on it. Without the gate
        # those rewards pay equally for batting or scooping the cube, and the policy
        # converges to that instead of gripping (ablation-verified: 400-iteration smoke
        # reaches 30% success gated vs 2% ungated with zero genuine pinches)
        grasp_params = {
            "stall_position_range": (0.15, 0.8),
            "max_object_distance": 0.025,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        }
        self.rewards.grasping_object = RewTerm(func=mdp.object_grasped, params=dict(grasp_params), weight=7.5)
        self.rewards.lifting_object.params["grasp_params"] = dict(grasp_params)
        self.rewards.object_goal_tracking.params["grasp_params"] = dict(grasp_params)
        self.rewards.object_goal_tracking_fine_grained.params["grasp_params"] = dict(grasp_params)
        # Pay the lift reward as soon as the cube leaves the table (rests at z=0.015): the
        # Franka-tuned 0.04 threshold leaves a 2.5 cm zero-gradient gap between the learned
        # pinch and the first lift signal, and the policy parks in the pinch local optimum
        for term in ("lifting_object", "object_goal_tracking", "object_goal_tracking_fine_grained"):
            getattr(self.rewards, term).params["minimal_height"] = 0.02
        # ... and make attempting the lift strictly dominate parking in the pinch, so the
        # occasional drop during exploration does not tip the expected value back to parking
        self.rewards.lifting_object.weight = 30.0

        # Soften the action-smoothness curriculum: the Franka schedule ramps the penalties
        # to -1e-1, which outweighs the exploration needed to discover lifting the held cube
        self.curriculum.action_rate.params["weight"] = -1e-2
        self.curriculum.joint_vel.params["weight"] = -1e-2

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    # grasp center between the fixed finger and the moving jaw; the authored
                    # gripper frame lies on the fixed fingertip instead
                    offset=OffsetCfg(
                        pos=(0.01184, -0.00256, -0.09682),
                    ),
                ),
            ],
        )


@configclass
class SO101CubeLiftEnvCfg_PLAY(SO101CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
