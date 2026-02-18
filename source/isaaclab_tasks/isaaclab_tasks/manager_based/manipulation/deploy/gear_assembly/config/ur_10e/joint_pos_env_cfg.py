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
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG, UR10e_ROBOTIQ_2F_85_CFG  # isort: skip


##
# Robotiq 2F-85 four-bar linkage: mimic joint equality constraints for Newton/MuJoCo
##


def _add_robotiq_2f85_mimic_joints():
    """Add MuJoCo equality constraints that replicate PhysxMimicJointAPI for the Robotiq 2F-85.

    The Robotiq 2F-85 USD defines mimic joints (PhysxMimicJointAPI) that couple passive
    gripper joints to the drive joint (finger_joint). Newton's USD importer does not
    parse PhysxMimicJointAPI, so we manually inject the equivalent MuJoCo joint-equality
    constraints into the builder before finalization.

    We also register eq_solref/eq_solimp custom attributes so the constraints can be
    made stiff enough to simulate a rigid mechanical linkage.

    Gearing from Robotiq_2F_85_phyisics_mimic.usda:
      - right_outer_knuckle_joint:        gearing = -1  (rotZ, ref: finger_joint)
      - left_inner_finger_joint:          gearing = +1  (rotX, ref: finger_joint)
      - left_inner_finger_knuckle_joint:  gearing = +1  (rotX, ref: finger_joint)
      - right_inner_finger_joint:         gearing = -1  (rotX, ref: finger_joint)
      - right_inner_finger_knuckle_joint: gearing = +1  (rotX, ref: finger_joint)

    The polycoef [a, b, 0, 0, 0] means: joint2 = a + b * joint1
    """
    import newton
    import warp as wp

    vec5 = wp.types.vector(length=5, dtype=wp.float32)

    builder = NewtonManager._builder
    if builder is None:
        return

    # Register eq_solref and eq_solimp custom attributes so we can set stiff
    # constraint parameters. These are normally registered by the MuJoCo solver,
    # but we need them available during builder phase.
    if not builder.has_custom_attribute("mujoco:eq_solref"):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="eq_solref",
                frequency=newton.Model.AttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
            )
        )
    if not builder.has_custom_attribute("mujoco:eq_solimp"):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="eq_solimp",
                frequency=newton.Model.AttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
            )
        )

    # Build joint name -> index lookup
    joint_name_to_idx = {}
    for idx, key in enumerate(builder.joint_key):
        # joint keys are full prim paths; extract the short name
        short_name = key.rsplit("/", 1)[-1] if "/" in key else key
        joint_name_to_idx[short_name] = idx

    drive_joint = "finger_joint"
    if drive_joint not in joint_name_to_idx:
        print(f"[WARN] Cannot add mimic joints: '{drive_joint}' not found in builder")
        return

    drive_idx = joint_name_to_idx[drive_joint]

    # Mimic joint definitions: (joint_name, mjcf_gearing, suppress_actuator)
    #
    # The USD uses PhysxMimicJointAPI with gearing in USD axis space (all joints axis="Z").
    # Newton's USD-to-MJCF export can flip axes, so we must convert gearing to MJCF space:
    #   mjcf_gearing = usd_gearing * (drive_axis_sign / passive_axis_sign)
    # where drive (finger_joint) has MJCF axis = -1.
    #
    #  Joint                          | MJCF axis | USD gear | MJCF gear
    #  right_outer_knuckle_joint      |    +1     |   -1     |   +1
    #  left_inner_finger_joint        |    -1     |   +1     |   +1
    #  left_inner_finger_knuckle_joint|    +1     |   +1     |   +1
    #  right_inner_finger_joint       |    -1     |   -1     |   -1
    #  right_inner_finger_knuckle_joint|   -1     |   +1     |   +1
    #
    # suppress_actuator=True for truly passive linkage joints (knuckles);
    # False for inner finger joints that keep their own actuators for grasping force.
    # Gearing values taken directly from Robotiq_2F_85_phyisics_mimic.usda.
    # Knuckle joints (passive linkage) have actuators suppressed.
    # Inner finger joints keep actuators for grasping force + constraint compliance.
    mimic_joints = [
        ("right_outer_knuckle_joint", 1.0, True),
        ("left_inner_finger_joint", 1.0, True),
        ("left_inner_finger_knuckle_joint", 1.0, True),
        ("right_inner_finger_joint", -1.0, True),
        ("right_inner_finger_knuckle_joint", 1.0, True),
    ]

    for passive_name, gearing, suppress_actuator in mimic_joints:
        if passive_name not in joint_name_to_idx:
            print(f"[WARN] Mimic joint '{passive_name}' not found in builder, skipping")
            continue

        passive_idx = joint_name_to_idx[passive_name]
        # MuJoCo convention: joint1 = constrained, joint2 = reference
        # polycoef: joint1 = polycoef[0] + polycoef[1]*joint2 + ...
        polycoef = [0.0, gearing, 0.0, 0.0, 0.0]

        builder.add_equality_constraint_joint(
            joint1=passive_idx,
            joint2=drive_idx,
            polycoef=polycoef,
            key=f"mimic_{passive_name}",
            enabled=True,
            # Stiff constraint for rigid mechanical linkage.
            # Direct mode solref: (-stiffness, -damping)
            custom_attributes={
                "mujoco:eq_solref": wp.vec2(0.02, 1.0),
                "mujoco:eq_solimp": vec5(0.99, 0.999, 0.00001, 0.5, 2.0),
            },
        )

        if suppress_actuator:
            # Set actuator mode to NONE so the MuJoCo exporter does not create
            # a <general> actuator for this purely passive joint.
            # Also set effort_limit to a large value so actfrcrange doesn't become
            # (0,0) which MuJoCo rejects.
            dof_start = builder.joint_qd_start[passive_idx]
            dof_dim = builder.joint_dof_dim[passive_idx]
            for d in range(dof_dim[0] + dof_dim[1]):
                builder.joint_act_mode[dof_start + d] = 0  # ActuatorMode.NONE
                builder.joint_effort_limit[dof_start + d] = 1e9

        print(f"[INFO] Added mimic constraint: {drive_joint}[{drive_idx}] -> {passive_name}[{passive_idx}], gearing={gearing}, suppress_actuator={suppress_actuator}")


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
    """Set finger joint positions for Robotiq 2F-85 gripper.

    Sets the drive joint and manually applies mimic gearing to all constrained
    joints.  MuJoCo equality constraints are only resolved during ``solver.step()``,
    **not** during ``forward_kinematics()``, so we must set all joint positions
    explicitly during reset to keep the kinematic state consistent.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices
        finger_joint_position: Target position for finger joints
    """
    for idx in reset_ind_joint_pos:
        # For 2F-85 gripper (6 joints expected)
        # Joint order (DOF 6-11):
        #   [0] finger_joint                     axis=-1  range=[0, 0.82]   (drive)
        #   [1] left_inner_finger_joint          axis=-1  range=[-pi, pi]   (gearing +1)
        #   [2] left_inner_finger_knuckle_joint  axis=+1  range=[-pi, pi]   (gearing +1)
        #   [3] right_outer_knuckle_joint        axis=+1  range=[0, 0.82]   (gearing +1)
        #   [4] right_inner_finger_joint         axis=-1  range=[-pi, pi]   (gearing -1)
        #   [5] right_inner_finger_knuckle_joint axis=-1  range=[-pi, pi]   (gearing +1)
        #
        # Mimic gearing from _add_robotiq_2f85_mimic_joints():
        #   passive_joint = gearing * drive_joint
        if len(finger_joints) < 6:
            raise ValueError(f"2F-85 gripper requires at least 6 finger joints, got {len(finger_joints)}")

        v = finger_joint_position
        joint_pos[idx, finger_joints[0]] = v     # finger_joint (drive)
        joint_pos[idx, finger_joints[1]] = v     # left_inner_finger_joint        (gearing +1)
        joint_pos[idx, finger_joints[2]] = v     # left_inner_finger_knuckle_joint (gearing +1)
        joint_pos[idx, finger_joints[3]] = v     # right_outer_knuckle_joint       (gearing +1)
        joint_pos[idx, finger_joints[4]] = -v    # right_inner_finger_joint        (gearing -1)
        joint_pos[idx, finger_joints[5]] = v     # right_inner_finger_knuckle_joint (gearing +1)


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
        self.grasp_rot_offset = [
            math.sqrt(2) / 2,
            math.sqrt(2) / 2,
            0.0,
            0.0,
        ]
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
                    disable_gravity=True,
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
    """Configuration for UR10e with Robotiq 2F-85 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Register Newton builder callback to inject MuJoCo equality constraints
        # for the Robotiq 2F-85 four-bar linkage (replaces PhysxMimicJointAPI).
        # This runs after builder.add_usd(stage) but before builder.finalize().
        NewtonManager.add_on_init_callback(_add_robotiq_2f85_mimic_joints)

        # switch robot to ur10e with 2F-85 gripper
        # NOTE: joint_drive_props ensures UsdPhysics.DriveAPI is applied to ALL joints
        # (including gripper joints that lack it in the source USD). Without this, the
        # Newton/MuJoCo backend won't create actuators for those joints.
        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_2F_85_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
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
            actuators=UR10e_ROBOTIQ_2F_85_CFG.actuators.copy(),
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

        # 2F-85 gripper actuator configuration
        # All coupled joints handled by stiff equality constraints -- no actuators needed.
        self.scene.robot.actuators.pop("gripper_finger", None)
        self.scene.robot.actuators.pop("gripper_passive", None)
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=5.0,
            friction=0.0,
            armature=0.0,
        )
        # Passive linkage joints are handled by MuJoCo equality constraints
        # (injected via _add_robotiq_2f85_mimic_joints callback above).
        # No actuator needed -- remove the inherited one.
        self.scene.robot.actuators.pop("gripper_passive", None)

        # Set gripper-specific joint setter function
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f85

        # gear offsets and grasp positions for the 2F-85 gripper
        self.gear_offsets_grasp = {
            "gear_small": [0.0, self.gear_offsets["gear_small"][0], -0.19],
            "gear_medium": [0.0, self.gear_offsets["gear_medium"][0], -0.19],
            "gear_large": [0.0, self.gear_offsets["gear_large"][0], -0.19],
        }

        # Grasp widths for 2F-85 gripper
        self.hand_grasp_width = {"gear_small": 0.60, "gear_medium": 0.50, "gear_large": 0.46}

        # Close widths for 2F-85 gripper
        self.hand_close_width = {"gear_small": 0.57, "gear_medium": 0.48, "gear_large": 0.44}

        # # Populate event term parameters
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
