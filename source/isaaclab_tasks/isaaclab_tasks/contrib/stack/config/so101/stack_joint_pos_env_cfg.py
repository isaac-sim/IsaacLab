# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events
from isaaclab_tasks.contrib.stack.stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip

# Default arm + gripper joint pose [rad].
# NOTE: kept mid-range (elbow/wrist bent) to avoid the boundary singularity of a fully
# extended 5-DOF arm. Tune in-sim so the gripper starts above the cube workspace pointing
# down for top-down grasps.
_SO101_STACK_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# Gripper jaw extremes [rad]. The SO-101 jaw is a single revolute joint; USD limits are
# roughly [-10 deg, 100 deg]. ``open`` ~= jaw fully open, ``close`` ~= jaw shut.
# Confirm sign/magnitude against the spawned articulation.
SO101_GRIPPER_OPEN = 1.745
SO101_GRIPPER_CLOSE = 0.0

# Base seat pose authoring the SO-101 root transform explicitly (it replaces the asset's baked-in
# default xform, not stacks on it). Load-bearing constraints:
# - ``_SO101_MOUNT_Z = -0.03008`` [m]: downward base correction that plants the column foot on the
#   tabletop (table-top surface is at world z~=0). Adjust only the root z; do NOT x-shift the base
#   (the cubes live at world x in [0.18, 0.30] and an x-shift would break reachability).
# - ``_SO101_BASE_SEAT_ROT`` [quat (x, y, z, w)]: 90 deg about +Z, matching the table's
#   ``rot=[0, 0, 0.707, 0.707]`` so the arm faces the cube workspace.
_SO101_MOUNT_Z = -0.03008
_SO101_BASE_SEAT_POS = (0.0, 0.0, _SO101_MOUNT_Z)
_SO101_BASE_SEAT_ROT = (0.0, 0.0, 0.70710678, 0.70710678)


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_so101_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            # Workspace shrunk to the SO-101's ~0.3 m reach (Franka used x in [0.4, 0.6]); verify
            # the box is fully reachable from the robot base.
            "pose_range": {"x": (0.15, 0.30), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.06,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class SO101CubeStackEnvCfg(StackEnvCfg):
    """Configuration for the SO-101 Cube Stack Environment (joint-position control)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set SO-101 as robot. Seat the base on the table-top and face it toward the cube
        # workspace (see ``_SO101_MOUNT_Z`` / ``_SO101_BASE_SEAT_ROT``).
        self.scene.robot = SO101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=_SO101_BASE_SEAT_POS,
                rot=_SO101_BASE_SEAT_ROT,
                joint_pos=_SO101_STACK_INIT_JOINT_POS,
            ),
        )

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Add semantics to robot
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for the specific robot type (so101)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": SO101_GRIPPER_OPEN},
            close_command_expr={"gripper": SO101_GRIPPER_CLOSE},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["gripper"]
        self.gripper_open_val = SO101_GRIPPER_OPEN
        self.gripper_threshold = 0.2

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically (within the SO-101 workspace)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.18, 0.0, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.24, 0.06, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.27, -0.07, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # SO-101 link chain (verified against the so101_new_calib USD):
        # base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper -> moving_jaw_so101_v1.
        # ``gripper`` is the static wrist/gripper body (the IK target); ``moving_jaw_so101_v1``
        # is the actuated jaw.
        # The ``end_effector`` offset should be set to the actual grasp point (between the jaws);
        # it is left at the gripper-link origin for now.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
                    name="tool_jaw",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )
