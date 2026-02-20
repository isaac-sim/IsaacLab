# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.deploy.mdp.events as gear_assembly_events
from isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots import UR10e_CFG, UR10e_ROBOTIQ_GRIPPER_CFG, UR10e_ROBOTIQ_2F_85_CFG  # isort: skip

import os as _os

MENAGERIE_2F85_MJCF = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "assets_mjcf", "robotiq_2f85", "2f85.xml"
)

##
# Robotiq 2F-85: Load Menagerie MJCF model into Newton builder
##


def _custom_instantiate_builder_with_gripper():
    """Override Newton's default builder instantiation to compose the gripper between
    the robot arm and the rest of the scene.

    Uses the per-env world building pattern (begin_world/add_builder/end_world) to
    support multi-env training. Each env gets its own proto builder with MJCF gripper
    attached, then added as an isolated world. This mirrors the canonical pattern in
    ``cloner_utils.newton_replicate``.
    """
    from pxr import UsdGeom

    import warp as wp

    from newton import ModelBuilder, solvers

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    up_axis = UsdGeom.GetStageUpAxis(stage)

    mount_xform = wp.transform(
        p=(0.0, 0.0, 0.0),
        q=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),
    )

    builder = ModelBuilder(up_axis=up_axis)

    # Discover env paths
    envs_prim = stage.GetPrimAtPath("/World/envs")
    env_paths = sorted([c.GetPath().pathString for c in envs_prim.GetChildren()])
    num_envs = len(env_paths)
    print(f"[INFO] Building {num_envs} worlds with MJCF gripper (per-env pattern)")

    # Build each env as a separate world
    for env_idx, env_path in enumerate(env_paths):
        # Create proto builder for this env
        proto = ModelBuilder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(proto)  # CRITICAL for MJCF

        # Load robot arm
        robot_path = f"{env_path}/Robot"
        proto.add_usd(stage, root_path=robot_path)

        # Find wrist_3_link and attach MJCF gripper
        ee_body_idx = -1
        for idx in range(len(proto.body_key) - 1, -1, -1):
            if proto.body_key[idx].endswith("wrist_3_link"):
                ee_body_idx = idx
                break
        if ee_body_idx == -1:
            raise RuntimeError(f"'wrist_3_link' not found in {robot_path}")

        proto.add_mjcf(
            source=MENAGERIE_2F85_MJCF,
            parent_body=ee_body_idx,
            floating=False,
            xform=mount_xform,
            enable_self_collisions=False,
        )

        # Load remaining scene objects for this env (gears, base, stand)
        proto.add_usd(stage, root_path=env_path, ignore_paths=[robot_path])

        # Load shared scene objects (ground, light) into each proto so they
        # belong to a proper world — global shapes (world=-1) cause geom count
        # mismatches in SolverMuJoCo._convert_to_mjc
        proto.add_usd(stage, ignore_paths=["/World/envs"])

        # Apply SDF config and gravity compensation on each proto BEFORE
        # adding to builder — shapes added after begin_world/end_world don't
        # inherit the correct world assignment.
        pre_sdf = proto.shape_count
        NewtonManager._apply_sdf_config(proto)
        post_sdf = proto.shape_count
        NewtonManager._apply_gravity_compensation(proto, stage)
        print(f"[DEBUG] env_{env_idx} proto shapes: {pre_sdf} -> {post_sdf} (SDF added {post_sdf - pre_sdf})")

        # Add proto as an isolated world
        builder.begin_world()
        builder.add_builder(proto)
        builder.end_world()
        print(f"[DEBUG] builder total shapes after env_{env_idx}: {builder.shape_count}, worlds: {builder.world_count}")

        if env_idx == 0:
            print(f"[INFO] env_0 proto: Bodies={len(proto.body_key)}, Joints={len(proto.joint_key)}")
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = num_envs

    print(f"[INFO] Final builder: Bodies={len(builder.body_key)}, Joints={len(builder.joint_key)}, Worlds={num_envs}")


##
# Gripper-specific helper functions
##


def set_finger_joint_pos_robotiq_2f140(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-140 gripper.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices
        finger_joint_position: Target position for finger joints
    """
    for idx in reset_ind_joint_pos:
        # For 2F-140 gripper (8 joints expected)
        if len(finger_joints) < 8:
            raise ValueError(f"2F-140 gripper requires at least 8 finger joints, got {len(finger_joints)}")

        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position

        # outer finger joints set to 0
        joint_pos[idx, finger_joints[2]] = 0
        joint_pos[idx, finger_joints[3]] = 0

        # inner finger joints: multiply by -1
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position

        joint_pos[idx, finger_joints[6]] = finger_joint_position
        joint_pos[idx, finger_joints[7]] = finger_joint_position


def set_finger_joint_pos_robotiq_2f85(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-85 gripper (Menagerie model).

    Sets all 8 joints to kinematically consistent positions based on the driver
    angle. The constrained joint positions were determined by running the MJCF
    model to equilibrium in standalone MuJoCo. For the grasp range (0.4-0.6):
      - coupler ≈ 0
      - spring_link ≈ driver
      - follower ≈ -0.963 * driver

    Menagerie 2F-85 joint order (8 joints):
      [0] right_driver_joint       range=[0, 0.8]      (drive)
      [1] right_coupler_joint      range=[-1.57, 0]    (constrained)
      [2] right_spring_link_joint  range=[-0.297, 0.8]  (constrained)
      [3] right_follower_joint     range=[-0.87, 0.87]  (constrained)
      [4] left_driver_joint        range=[0, 0.8]      (coupled via tendon)
      [5] left_coupler_joint       range=[-1.57, 0]    (constrained)
      [6] left_spring_link_joint   range=[-0.297, 0.8]  (constrained)
      [7] left_follower_joint      range=[-0.87, 0.87]  (constrained)

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices
        finger_joint_position: Target position for driver joints [0, 0.8]
    """
    if len(finger_joints) < 8:
        raise ValueError(f"Menagerie 2F-85 gripper requires at least 8 finger joints, got {len(finger_joints)}")

    theta = finger_joint_position
    coupler_pos = 0.0
    spring_link_pos = theta
    follower_pos = -0.963 * theta

    for idx in reset_ind_joint_pos:
        # Right side
        joint_pos[idx, finger_joints[0]] = theta            # right_driver_joint
        joint_pos[idx, finger_joints[1]] = coupler_pos      # right_coupler_joint
        joint_pos[idx, finger_joints[2]] = spring_link_pos  # right_spring_link_joint
        joint_pos[idx, finger_joints[3]] = follower_pos     # right_follower_joint
        # Left side (symmetric)
        joint_pos[idx, finger_joints[4]] = theta            # left_driver_joint
        joint_pos[idx, finger_joints[5]] = coupler_pos      # left_coupler_joint
        joint_pos[idx, finger_joints[6]] = spring_link_pos  # left_spring_link_joint
        joint_pos[idx, finger_joints[7]] = follower_pos     # left_follower_joint


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events.

    Note: ``randomize_rigid_body_material`` events are disabled because they cause
    NaN values on Newton (known issue, see locomotion configs). Re-enable after
    basic training is verified to work.
    """

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]
    #         ),  # only the arm joints are randomized
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
    #         "friction_distribution_params": (0.3, 0.7),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    # NOTE: Material randomization disabled on Newton - causes NaNs.
    # Uncomment once Newton material DR is stable.
    # small_gear_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    #
    # medium_gear_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("factory_gear_medium", body_names=".*"),
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    #
    # large_gear_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("factory_gear_large", body_names=".*"),
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    #
    # gear_base_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("factory_gear_base", body_names=".*"),
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    #
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*finger"),
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={"gear_types": ["gear_small"]},  # ["gear_small", "gear_medium", "gear_large"]},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_gears_and_base_pose = EventTerm(
    #     func=gear_assembly_events.randomize_gears_and_base_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.1, 0.1],
    #             "y": [-0.25, 0.25],
    #             "z": [-0.1, 0.1],
    #             "roll": [-math.pi / 90, math.pi / 90],  # 2 degree
    #             "pitch": [-math.pi / 90, math.pi / 90],  # 2 degree
    #             "yaw": [-math.pi / 6, math.pi / 6],  # 30 degree
    #         },
    #         "gear_pos_range": {
    #             "x": [-0.02, 0.02],
    #             "y": [-0.02, 0.02],
    #             "z": [0.0575, 0.0775],  # 0.045 + 0.0225
    #         },
    #         "velocity_range": {},
    #     },
    # )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [0.0, 0.0]},  # {"x": [-0.0, 0.0], "y": [-0.005, 0.005], "z": [-0.003, 0.003]},
        },
    )


@configclass
class UR10eGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    """Base configuration for UR10e Gear Assembly Environment.

    This class contains common setup shared across different gripper configurations.
    Subclasses should configure gripper-specific parameters.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Robot-specific parameters
        self.end_effector_body_name = "wrist_3_link"
        self.num_arm_joints = 6
        # XYZW grasp rotation offset
        # Converted from WXYZ [0.0, sqrt(2)/2, sqrt(2)/2, 0.0]
        # -> XYZW [sqrt(2)/2, sqrt(2)/2, 0.0, 0.0]
        # self.grasp_rot_offset = [
        #     math.sqrt(2) / 2,
        #     math.sqrt(2) / 2,
        #     0.0,
        #     0.0,
        # ]

        self.grasp_rot_offset = [0.0, 1.0, 0.0, 0.0]
        self.gripper_joint_setter_func = None  # Set in subclass

        # Gear orientation termination thresholds (in degrees)
        # self.gear_orientation_roll_threshold_deg = 15.0
        # self.gear_orientation_pitch_threshold_deg = 15.0
        # self.gear_orientation_yaw_threshold_deg = 180.0

        # Common observation configuration
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # override events
        self.events = EventCfg()

        # Update termination thresholds from config
        # self.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] = (
        #     self.gear_orientation_roll_threshold_deg
        # )
        # self.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] = (
        #     self.gear_orientation_pitch_threshold_deg
        # )
        # self.terminations.gear_orientation_exceeded.params["yaw_threshold_deg"] = (
        #     self.gear_orientation_yaw_threshold_deg
        # )

        # override command generator body
        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )


@configclass
class UR10e2F140GearAssemblyEnvCfg(UR10eGearAssemblyEnvCfg):
    """Configuration for UR10e with Robotiq 2F-140 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10e with 2F-140 gripper
        # NOTE: joint_drive_props ensures UsdPhysics.DriveAPI is applied to ALL joints
        # (including gripper joints that lack it in the source USD). Without this, the
        # Newton/MuJoCo backend won't create actuators for those joints.
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_GRIPPER_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                # NOTE: Non-zero gains ensure Newton marks these joints as position-driven
                # (not effort-only) during USD import, so MuJoCo actuators are exported.
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                    drive_type="force",
                    stiffness=10,
                    damping=1.0e-6,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 2.7228,
                    "shoulder_lift_joint": -8.3962e-01,
                    "elbow_joint": 1.3684,
                    "wrist_1_joint": -2.1048,
                    "wrist_2_joint": -1.5691,
                    "wrist_3_joint": -1.9896,
                },
                pos=(0.0, 0.0, 0.0),
                # XYZW identity quaternion
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # 2F-140 gripper actuator configuration
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=5,
            friction=0.0,
            armature=0.0,
        )

        # Set gripper-specific joint setter function
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f140

        # gear offsets and grasp positions for the 2F-140 gripper
        self.gear_offsets_grasp = {
            "gear_small": [0.0, self.gear_offsets["gear_small"][0], -0.26],
            "gear_medium": [0.0, self.gear_offsets["gear_medium"][0], -0.26],
            "gear_large": [0.0, self.gear_offsets["gear_large"][0], -0.26],
        }

        # Grasp widths for 2F-140 gripper
        self.hand_grasp_width = {"gear_small": 0.64, "gear_medium": 0.54, "gear_large": 0.51}

        # Close widths for 2F-140 gripper
        self.hand_close_width = {"gear_small": 0.69, "gear_medium": 0.59, "gear_large": 0.56}

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Populate termination term parameters
        # self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        # self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        # self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        # self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        # self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class UR10e2F85GearAssemblyEnvCfg(UR10eGearAssemblyEnvCfg):
    """Configuration for UR10e with Robotiq 2F-85 gripper (Menagerie MJCF model).

    The gripper is loaded from MuJoCo Menagerie via ``builder.add_mjcf()`` in a
    Newton builder callback. The MJCF parser automatically imports the proper
    4-bar linkage with connect constraints, fixed tendon, damping, armature,
    and springs. No manual constraint injection needed.
    """

    def __post_init__(self):
        super().__post_init__()

        # Override Newton's default builder instantiation to split USD loading
        # so we can compose the MJCF gripper between arm and scene objects.
        # This replaces the default instantiate_builder_from_stage() method.
        NewtonManager.instantiate_builder_from_stage = staticmethod(_custom_instantiate_builder_with_gripper)

        # Use base UR10e WITHOUT gripper variant -- the gripper comes from MJCF
        self.scene.robot = UR10e_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                    drive_type="force",
                    stiffness=10,
                    damping=1.0e-6,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 2.7228,
                    "shoulder_lift_joint": -8.3962e-01,
                    "elbow_joint": 1.3684,
                    "wrist_1_joint": -2.1048,
                    "wrist_2_joint": -1.5691,
                    "wrist_3_joint": -1.9896,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        # Gripper actuator is imported from MJCF (tendon-targeted, CTRL_DIRECT mode).
        # No IsaacLab ImplicitActuatorCfg needed for the gripper.
        # The arm actuators (shoulder, elbow, wrist) come from UR10e_CFG.
        #
        # Gripper is NOT part of the RL action space. The tendon ctrl is set to a
        # constant close command in set_robot_to_grasp_pose (reset event) and the
        # MuJoCo ctrl buffer holds that value between steps.

        # Set gripper-specific joint setter function
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f85

        # gear offsets and grasp positions for the 2F-85 gripper
        # Offset is [x, y, z] in the combined frame (gear_quat * grasp_rot_offset).
        # The combined rotation flips the X direction, so negate the shaft offset.
        self.gear_offsets_grasp = {
            "gear_small": [-self.gear_offsets["gear_small"][0], 0.0, -0.20],
            "gear_medium": [-self.gear_offsets["gear_medium"][0], 0.0, -0.20],
            "gear_large": [-self.gear_offsets["gear_large"][0], 0.0, -0.20],
        }

        # Initial driver joint positions for grasp/close (range [0, 0.8]).
        # hand_grasp_width: open enough to surround the gear before closing.
        # hand_close_width: initial joint state for close (tendon actuator takes over).
        self.hand_grasp_width = {"gear_small": 0.58, "gear_medium": 0.48, "gear_large": 0.44}
        self.hand_close_width = {"gear_small": 0.8, "gear_medium": 0.8, "gear_large": 0.8}

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Populate termination term parameters
        self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class UR10e2F140GearAssemblyEnvCfg_PLAY(UR10e2F140GearAssemblyEnvCfg):
    """Play configuration for UR10e with Robotiq 2F-140 gripper."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1  # 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10e2F85GearAssemblyEnvCfg_PLAY(UR10e2F85GearAssemblyEnvCfg):
    """Play configuration for UR10e with Robotiq 2F-85 gripper."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1  # 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
