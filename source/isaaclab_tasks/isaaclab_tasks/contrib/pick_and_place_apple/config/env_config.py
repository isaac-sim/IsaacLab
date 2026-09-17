# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 + Sharpa Wave env for the pick-and-place apple task (PhysX, rigid apple)."""

import math
import os

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.rlinf_assets import NUREC_ASSET_ROOT, PROP_ASSET_ROOT

from .. import mdp
from .camera_config import FRONT_CAMERA_CFG, LEFT_WRIST_CAMERA_CFG, RIGHT_WRIST_CAMERA_CFG
from .metadata import (
    ACTION_DIM,
    H2_ACTION_JOINT_ORDER,
    POLICY_58_ORDER,
)
from .robot_config import H2RobotPresets, h2_body_joint_offsets

# pnp_apple_sim_new_bg_v1 episode_000027 frame 0, replayed from
# apple_pick_and_place_05142000_filtered episode_000043, in POLICY_58_ORDER
# (left_arm, right_arm, left_hand, right_hand).
_FRAME0_POLICY_58_POS: tuple[float, ...] = (
    0.2754,
    0.0215,
    -0.0674,
    -0.0774,
    0.0518,
    -0.1481,
    -0.0260,
    0.1734,
    -0.0544,
    0.0165,
    -0.0457,
    0.0006,
    0.0136,
    0.1644,
    0.2834,
    0.0498,
    -0.4345,
    0.2344,
    0.7783,
    -0.1513,
    -0.0731,
    0.0373,
    0.1479,
    -0.1540,
    0.0019,
    0.0358,
    0.1647,
    -0.1514,
    0.1047,
    0.0385,
    0.1432,
    0.1048,
    -0.1552,
    0.1278,
    0.0348,
    0.0924,
    -0.0452,
    0.0498,
    0.3536,
    0.0689,
    0.0869,
    -0.1104,
    0.1215,
    0.0687,
    0.1541,
    -0.1028,
    0.1457,
    0.0810,
    0.0946,
    -0.1501,
    0.1255,
    0.0361,
    0.0890,
    0.1171,
    -0.1064,
    0.0177,
    0.0611,
    0.0884,
)
assert len(_FRAME0_POLICY_58_POS) == ACTION_DIM

# Arm + Sharpa-hand start pose for the pnp_apple task, keyed by joint name.
CUSTOM_JOINT_POS: dict[str, float] = dict(zip(POLICY_58_ORDER, _FRAME0_POLICY_58_POS, strict=True))
CUSTOM_JOINT_POS["head_pitch_joint"] = 0.6
assert "head_pitch_joint" not in POLICY_58_ORDER, (
    "head must stay outside POLICY_58_ORDER so policy cannot lift the head"
)


# Task-specific start pose.
INIT_POS: tuple[float, float, float] = (-0.95, 0.0, 1.05)
INIT_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

TABLE_USD = f"{PROP_ASSET_ROOT}/Assets/Table256/Table256_cloth.usd"
APPLE_USD = f"{PROP_ASSET_ROOT}/Assets/Apple033/Apple033.usd"
PLATE_USD = f"{PROP_ASSET_ROOT}/Assets/SimReady_Furniture/plate_large/plate_large_rigid.usd"
BACKGROUND_USD = f"{NUREC_ASSET_ROOT}/IMG_6246_nurec_aligned_scaled.usdz"


# Reset-pose randomization half-ranges. PNP_APPLE_XY_RANGE (metres) drives the x
# half-range and the y magnitude; PNP_APPLE_YAW_RANGE_DEG drives yaw. The defaults
# reproduce the ranges the task's demonstrations were recorded with.
XY_RANGE = float(os.environ.get("PNP_APPLE_XY_RANGE", "0.015"))
YAW_RANGE = math.radians(float(os.environ.get("PNP_APPLE_YAW_RANGE_DEG", "0")))


def _randomize_asset_pose(
    asset_name: str,
    y_range: tuple[float, float],
) -> EventTermCfg:
    return EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-XY_RANGE, XY_RANGE),
                "y": y_range,
                "yaw": (-YAW_RANGE, YAW_RANGE),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(asset_name),
        },
    )


@configclass
class PnpAppleSceneCfg(InteractiveSceneCfg):
    """H2 scene with front, left-wrist, and right-wrist cameras."""

    robot = H2RobotPresets.h2_sharpa_base_fix(
        init_pos=INIT_POS,
        init_rot=INIT_ROT,
        custom_joint_pos=CUSTOM_JOINT_POS,
    )

    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/background",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BACKGROUND_USD,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.35, -0.45, 2.0),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    # The enclosing room keeps the rendered surroundings identical to the AGX Orin
    # packing task instead of an open ground plane behind the capture.
    room_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_left",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 0.05, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 2.0, 1.5)),
    )
    room_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_right",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 0.05, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -2.0, 1.5)),
    )
    room_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_front",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 4.0, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 1.5)),
    )
    room_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_back",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 4.0, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.5, 0.0, 1.5)),
    )
    room_ceiling = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 0.0, 3.0)),
    )

    room_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_floor",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.21), roughness=0.9, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.02)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=(1.0, 1.0, 1.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.42, 0, 0.565),
            rot=(0, 0, 0.70710678, 0.70710678),
        ),
    )

    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=APPLE_USD,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.95,
                restitution=0.0,
            ),
            # 13/20 of the original (1.5885, 1.5885, 1.4274).
            scale=(1.1357775, 1.1357775, 1.120591),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.48, 0.09, 1.081),
            rot=(0, 0, 0, 1),
        ),
    )

    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_USD,
            scale=(0.9, 0.9, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, -0.265, 1.055),
            rot=(0, 0, 0, 1),
        ),
    )

    # The room encloses the scene, so every light is local to the env. An infinite
    # dome or distant light would be blocked by the ceiling, and a copy per env
    # would light every other env as well.
    overhead_strip = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/overhead_strip",
        spawn=sim_utils.CylinderLightCfg(
            color=(1.0, 0.96, 0.90),
            intensity=36000.0,
            length=1.1,
            radius=0.025,
            treat_as_line=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.92, 0.0, 2.35),
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    table_bounce = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_bounce",
        spawn=sim_utils.DiskLightCfg(
            color=(1.0, 0.96, 0.92),
            intensity=60.0,
            radius=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 1.06),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )

    front_camera = FRONT_CAMERA_CFG.replace(height=240, width=320)
    left_wrist_camera = LEFT_WRIST_CAMERA_CFG.replace(height=240, width=320)
    right_wrist_camera = RIGHT_WRIST_CAMERA_CFG.replace(height=240, width=320)


@configclass
class ActionsCfg:
    """RL/Eval joint control with PhysX gravity feed-forward on both arms."""

    joint_pos = mdp.JointPositionActionCfg(
        class_type=mdp.H2GravityCompensatedJointPositionAction,
        asset_name="robot",
        joint_names=H2_ACTION_JOINT_ORDER,
        scale=1.0,
        use_default_offset=False,
        offset=h2_body_joint_offsets(CUSTOM_JOINT_POS),
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Joint state and three camera observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_joint_states)
        robot_policy_joint_pos = ObsTerm(func=mdp.get_robot_policy_joint_positions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class EventCfg:
    """Reset the scene, then randomize the apple in XY."""

    # Also latch PD targets (esp. head_pitch=0.6). Controllers that only
    # command a subset of joints otherwise leave head target at 0 and IdealPD
    # lifts the camera into the wall.
    reset_scene = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    reset_task_stage = EventTermCfg(
        func=mdp.reset_task_stage,
        mode="reset",
        params={"apple_cfg": SceneEntityCfg("apple"), "print_log": True},
    )

    # Must follow ``reset_scene`` so the default reset does not overwrite it.
    randomize_apple_xy = _randomize_asset_pose(
        "apple",
        y_range=(-XY_RANGE, XY_RANGE),
    )


@configclass
class TerminationsCfg:
    """Timeout, stage success, and apple-drop failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,
        params={"success_stage": 4, "print_log": False},
    )


@configclass
class RewardsCfg:
    """Sparse stage rewards: left lift, right catch, place, then release."""

    left_grasp_lift = RewTerm(
        func=mdp.left_grasp_lift_reward,
        weight=1.0,
        params={
            "apple_cfg": SceneEntityCfg("apple"),
            "plate_cfg": SceneEntityCfg("plate"),
            "robot_cfg": SceneEntityCfg("robot"),
            "lift_z": 0.10,
            "grasp_dist": 0.26,
            "release_dist": 0.08,
            "place_grasp_dist": 0.30,
            # Broad placement band relative to the plate root. XY proximity
            # and right-hand control remain required to reject unrelated drops.
            "xy_radius": 0.10,
            "z_above": 0.025,
            "z_window": 0.075,
            "hold_steps": 1,
            "print_log": False,
        },
    )
    handover_to_right = RewTerm(
        func=mdp.handover_to_right_reward,
        weight=1.0,
        params={"print_log": False},
    )
    place_on_plate = RewTerm(
        func=mdp.place_on_plate_reward,
        weight=1.0,
        params={"print_log": False},
    )
    release_on_plate = RewTerm(
        func=mdp.release_on_plate_reward,
        weight=1.0,
        params={"print_log": False},
    )


@configclass
class PnpAppleEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H2 + Sharpa Wave pick-and-place apple env (PhysX)."""

    scene: PnpAppleSceneCfg = PnpAppleSceneCfg(
        num_envs=1,
        # Prevent replicated background rooms from intersecting.
        env_spacing=10.0,
        # Envs carry per-env assets, so clone prims rather than replicate physics.
        replicate_physics=False,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.6, -0.2, 2.4),
        lookat=(-0.45, 0.0, 1.22),
        cam_prim_path="/OmniverseKit_Persp",
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_max_num_partitions = 32
        # One render per environment step; the policy reads a camera only once per step.
        self.sim.render_interval = 4
        # SimulationCfg.render only exists on IsaacLab builds that ship RenderCfg;
        # it is absent from the pinned checkout here, where the bare attribute
        # access raised AttributeError before the environment was ever built.
        if hasattr(self.sim, "render"):
            self.sim.render.antialiasing_mode = "DLAA"
