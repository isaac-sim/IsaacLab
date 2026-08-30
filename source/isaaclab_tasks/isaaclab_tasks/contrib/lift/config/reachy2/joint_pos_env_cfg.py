# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube lift environment configuration for Pollen Robotics Reachy 2 (right arm).

Scene layout: Reachy 2 stands on the ground plane (z=-1.05 in the lift scene
frame) facing a desk-height table (top at z=-0.30, i.e. 0.75 m in the robot
base frame). URDF forward kinematics: shoulder at z=1.00, hanging palm at
z=0.46 in the base frame, so the tabletop sits in the natural manipulation
zone between hip and shoulder — matching Reachy 2's intended desk workspace.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

from . import mdp as reachy2_mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import REACHY2_CFG  # isort: skip


@configclass
class Reachy2CubeLiftEnvCfg(LiftEnvCfg):
    """Reachy 2 right-arm cube lift environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ── Robot ───────────────────────────────────────────────────────────
        # Reachy 2 stands on the ground plane (z=-1.05); table top ends up at
        # ~1.05 m in the robot base frame — within the arm's natural workspace.
        self.scene.robot = REACHY2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, -1.05),
                joint_pos={
                    # Neck neutral
                    "neck_.*": 0.0,
                    # Right arm — ready pose hovering over the table edge.
                    # FK: grip point lands ~(0.55, -0.27, 0.80) in the base
                    # frame, ~5 cm above the tabletop and ~0.22 m from the
                    # cube spawn — inside the reaching reward's gradient.
                    "r_shoulder_pitch": -0.85,
                    "r_shoulder_roll": -0.1,
                    "r_elbow_yaw": 0.0,
                    "r_elbow_pitch": -0.65,
                    "r_wrist_.*": 0.0,
                    # Left arm — tucked away
                    "l_shoulder_pitch": 0.0,
                    "l_shoulder_roll": 0.5,
                    "l_elbow_yaw": 0.0,
                    "l_elbow_pitch": -1.0,
                    "l_wrist_.*": 0.0,
                    # Right gripper open — mimic-consistent joint values
                    # (r_hand_finger: 0 = closed, 2.27 = fully open;
                    # proximal = 0.554 - 0.4689*finger, distal = -proximal)
                    "r_hand_finger": 1.5,
                    "r_hand_finger_proximal": -0.149,
                    "r_hand_finger_proximal_mimic": -0.149,
                    "r_hand_finger_distal": 0.149,
                    "r_hand_finger_distal_mimic": 0.149,
                    # Left gripper closed/neutral
                    "l_hand_finger.*": 0.0,
                },
            ),
        )

        # ── Simple desk-sized table block instead of the SeattleLab table ────
        # URDF FK (zero pose): shoulder at z=1.00, hanging palm at z=0.46 in
        # the base frame — a 0.75 m desk puts the tabletop in the natural
        # manipulation zone. Block: 0.7 x 1.1 m surface, sitting on the ground
        # plane (z=-1.05), top surface at z=-0.30 world / 0.75 m base frame.
        # Near edge at x=0.30 — clears the robot's 0.25 m base footprint.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, -0.675)),
            spawn=sim_utils.CuboidCfg(
                size=(0.7, 1.1, 0.75),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.42, 0.30), metallic=0.0
                ),
            ),
        )

        # ── Actions ──────────────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw",
                         "r_elbow_pitch", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"],
            scale=0.5,
            use_default_offset=True,
        )
        # Binary open/close commanding ALL right-finger joints with values
        # consistent with the URDF mimic equations (proximal/distal follow
        # r_hand_finger with multiplier -/+0.4689, offset +/-0.554). The
        # URDF's mimic tags import as NewtonMimicAPI, which the PhysX backend
        # ignores — so the physical finger joints must be driven directly.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["r_hand_finger", "r_hand_finger_proximal", "r_hand_finger_proximal_mimic",
                         "r_hand_finger_distal", "r_hand_finger_distal_mimic"],
            open_command_expr={
                "r_hand_finger": 1.8,
                "r_hand_finger_proximal.*": -0.29,
                "r_hand_finger_distal.*": 0.29,
            },
            close_command_expr={
                "r_hand_finger": 0.05,
                "r_hand_finger_proximal.*": 0.53,
                "r_hand_finger_distal.*": -0.51,
            },
        )

        # ── Goal command — object target pose in robot base frame ───────────
        self.commands.object_pose.body_name = "r_hand_palm_link"
        # Table top is at 0.75 m in the robot base frame, so base-frame z of
        # 0.85–1.05 corresponds to 0.10–0.30 m above the table.
        self.commands.object_pose.ranges.pos_x = (0.30, 0.48)
        self.commands.object_pose.ranges.pos_y = (-0.35, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.85, 1.05)

        # ── Object — DexCube on the table, biased toward the right arm ──────
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # Cube rests on the desk-height table top (z=-0.30): center settles
            # at -0.268 (half-extent 0.032 for the 0.8-scaled DexCube).
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.38, -0.15, -0.245], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        # Keep the randomized spawn on the table and within right-arm reach
        # (~0.33–0.43 m forward of the base, biased toward the right side).
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.1, 0.15),
            "z": (0.0, 0.0),
        }

        # ── Reward shaping ───────────────────────────────────────────────────
        # Wider reaching kernel than the Franka default (0.1): Reachy's ready
        # pose starts ~0.22 m from the cube, and std=0.2 keeps a usable
        # gradient over the robot's larger approach distances.
        self.rewards.reaching_object.params["std"] = 0.2
        # Penalize knocking/dropping the cube off the table. Without this the
        # optimal policy is a toss-reset loop: swat the cube up (brief lifting
        # reward), let it fall (free reset), repeat. -100 * dt = -2 per drop.
        self.rewards.drop_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": "object_dropping"},
        )
        # Grasp shaping: pay for closing the gripper *near* the cube. Random
        # exploration almost never chains hover -> orient -> close -> lift on
        # its own; this term bridges the gap between the reaching and lifting
        # rewards so grasp attempts are discovered reliably.
        self.rewards.grasp_object = RewTerm(
            func=reachy2_mdp.grasp_object,
            weight=5.0,
            params={"std": 0.05},
        )
        # Double the lifting payoff (base: 15) — successful grasps are rare
        # events early on and must leave a strong learning signal.
        self.rewards.lifting_object.weight = 30.0
        # Strengthen the precise-placement bonus (base: 5). With coarse goal
        # tracking well-paid at 10-15 cm, a weak fine kernel lets the policy
        # plateau just outside the 5 cm success threshold — training success
        # peaked at ~750 iters and decayed as the policy settled there.
        self.rewards.object_goal_tracking_fine_grained.weight = 15.0

        # ── Curriculum ───────────────────────────────────────────────────────
        # The base curriculum ramps action penalties to -0.1, which for this
        # exploration-hard task makes "freeze near the cube" optimal. Keep the
        # late-training penalties an order of magnitude milder.
        self.curriculum.action_rate.params["weight"] = -0.01
        self.curriculum.joint_vel.params["weight"] = -0.01

        # ── Height-dependent MDP terms, shifted for the desk-height table ───
        # Cube rests with center at z=-0.268; count it "lifted" ~2 cm up.
        self.rewards.lifting_object.params["minimal_height"] = -0.25
        self.rewards.object_goal_tracking.params["minimal_height"] = -0.25
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = -0.25
        # Terminate once the cube falls below the table top.
        self.terminations.object_dropping.params["minimum_height"] = -0.40

        # ── End-effector frame — palm of the right hand ──────────────────────
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # Note: the URDF importer nests bodies as parent→child prim chains under
        # Robot/Geometry/world/. Prim path patterns are matched per path segment
        # (".*" spans a single level), so the full kinematic chain is spelled out.
        _r_palm_prim = (
            "{ENV_REGEX_NS}/Robot/Geometry/world/base_link/back_bar/torso"
            "/r_shoulder_base_link/r_shoulder_dummy_link1/r_shoulder_dummy_link2"
            "/r_shoulder_first_link/r_shoulder_ball_link/r_shoulder_fix_link"
            "/r_elbow_arm_link/r_elbow_base_link/r_elbow_dummy_link1"
            "/r_elbow_dummy_link2/r_elbow_first_link/r_elbow_ball_link"
            "/r_elbow_fix_link/r_elbow_forearm_link/r_wrist_base_link"
            "/r_wrist_dummy_link1/r_wrist_dummy_link2/r_wrist_link"
            "/r_wrist_ball_link/r_wrist_out_link/r_hand_palm_link"
        )
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/world/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=_r_palm_prim,
                    name="end_effector",
                    # Grasp center: finger joints attach 0.07 m along palm +z,
                    # grip midpoint sits ~0.12 m from the palm link origin.
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.12],
                    ),
                ),
            ],
        )


@configclass
class Reachy2CubeLiftEnvCfg_PLAY(Reachy2CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
