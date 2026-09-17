# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 + Sharpa Wave scene for packing an AGX Orin into a protective box."""

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

# Reuse the proven bimanual, camera-facing H2 teleop pose until a pack-task
# recording supplies a task-specific frame-zero pose.
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

# Arm + Sharpa-hand start pose for the pack_agx task, keyed by joint name.
CUSTOM_JOINT_POS: dict[str, float] = dict(zip(POLICY_58_ORDER, _FRAME0_POLICY_58_POS, strict=True))
CUSTOM_JOINT_POS["head_pitch_joint"] = 0.6
assert "head_pitch_joint" not in POLICY_58_ORDER, (
    "head must stay outside POLICY_58_ORDER so policy cannot lift the head"
)


# Task-specific start pose.
INIT_POS: tuple[float, float, float] = (-0.95, 0.0, 1.05)
INIT_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

TABLE_USD = f"{PROP_ASSET_ROOT}/Assets/Table256/Table256.usd"
AGX_ORIN_USD = f"{PROP_ASSET_ROOT}/Assets/MiniPc001/MiniPc001.usd"
PROTECTIVE_BOX_USD = f"{PROP_ASSET_ROOT}/Assets/ProtectiveBox001/ProtectiveBox001.usd"
BACKGROUND_USD = f"{NUREC_ASSET_ROOT}/IMG_6246_nurec_aligned_scaled.usdz"


# Reset-pose randomization half-ranges. PACK_AGX_XY_RANGE (metres) drives the x
# half-range and the y magnitude; PACK_AGX_YAW_RANGE_DEG drives yaw. The defaults
# reproduce the ranges the MimicGen dataset was generated with. The OSMO
# workflows set both from `--set xy_range=... yaw_range_deg=...`.
XY_RANGE = float(os.environ.get("PACK_AGX_XY_RANGE", "0.05"))
YAW_RANGE = math.radians(float(os.environ.get("PACK_AGX_YAW_RANGE_DEG", "10")))


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
class PackAgxOrinSceneCfg(InteractiveSceneCfg):
    """Independent H2 scene for packing an AGX Orin."""

    robot = H2RobotPresets.h2_sharpa_base_fix(
        init_pos=INIT_POS,
        init_rot=INIT_ROT,
        custom_joint_pos=CUSTOM_JOINT_POS,
    )

    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/background",
        spawn=sim_utils.UsdFileCfg(usd_path=BACKGROUND_USD),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.35, -0.45, 2.0),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=(1.07, 0.92, 1.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                roughness=0.65,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.435, 0.05, 0.5088),
            rot=(0, 0, 0.70710678, 0.70710678),
        ),
    )

    agx_orin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AgxOrin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=AGX_ORIN_USD,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.58, -0.24, 0.97),
            rot=(0, 0, 0.017099942, 0.99985379),
        ),
    )

    protective_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProtectiveBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PROTECTIVE_BOX_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.53, -0.03, 0.995),
            rot=(0, 0, 0.23344537, 0.97236992),
        ),
    )

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
            pos=(-0.55, 0.05, 0.98),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )

    front_camera = FRONT_CAMERA_CFG.replace(height=240, width=320)
    left_wrist_camera = LEFT_WRIST_CAMERA_CFG.replace(height=240, width=320)
    right_wrist_camera = RIGHT_WRIST_CAMERA_CFG.replace(height=240, width=320)


@configclass
class ActionsCfg:
    """RL joint control with PhysX gravity feed-forward on both arms."""

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
    """Robot state and three camera observations."""

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
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "normalize": False,
                "response_gamma": (0.4545, 0.4545, 0.4545),
                "response_gain": (0.879, 0.824, 0.848),
            },
        )
        left_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "response_gamma": (0.4545, 0.4545, 0.4545),
                "response_gain": (0.732, 0.745, 0.735),
            },
        )
        right_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "response_gamma": (0.4545, 0.4545, 0.4545),
                "response_gain": (0.815, 0.814, 0.826),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class EventCfg:
    """Reset the scene, then randomize the AGX Orin on the tabletop."""

    # Patches on materials
    align_table_material = EventTermCfg(
        func=mdp.align_table_material,
        mode="startup",
        params={"diffuse": 0.75, "emissive": 0.0},
    )
    align_agx_material = EventTermCfg(
        func=mdp.align_prop_material,
        mode="reset",
        params={
            "asset": "AgxOrin",
            "metallic": 0.15,
            "roughness": 0.55,
            "albedo_brightness": 3.4,
        },
    )
    align_box_material = EventTermCfg(
        func=mdp.align_prop_material,
        mode="reset",
        params={
            "asset": "ProtectiveBox",
            "metallic": 0.05,
            "roughness": 0.85,
            "albedo_brightness": 2.6,
        },
    )

    reset_scene = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    randomize_agx_orin_xy = _randomize_asset_pose(
        "agx_orin",
        y_range=(-XY_RANGE, 0.0),
    )
    randomize_box_xy = _randomize_asset_pose(
        "protective_box",
        y_range=(0.0, XY_RANGE),
    )

    reset_task_stage = EventTermCfg(
        func=mdp.reset_task_stage,
        mode="reset",
        params={
            "agx_orin_cfg": SceneEntityCfg("agx_orin"),
            "print_log": False,
        },
    )


@configclass
class TerminationsCfg:
    """Timeout and stage-based completion for RL post-training."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,
        params={"success_stage": 4},
    )


@configclass
class RewardsCfg:
    """Sparse stage rewards plus stage-specific positive progress shaping."""

    lift_agx = RewTerm(
        func=mdp.lift_agx_reward,
        weight=1.0,
        params={
            "agx_orin_cfg": SceneEntityCfg("agx_orin"),
            "protective_box_cfg": SceneEntityCfg("protective_box"),
            "lift_z": 0.01,
            "align_target_z_offset": 0.10,
            "align_xy": 0.025,  # 3D distance tolerance
            "rotation_tolerance": 0.2,
            "lift_hold_steps": 5,
            "align_hold_steps": 5,
            "seat_hold_steps": 5,
            "print_log": False,
        },
    )
    align_agx = RewTerm(
        func=mdp.align_agx_reward,
        weight=1.0,
        params={"distance_offset": 0.25},
    )
    seat_agx = RewTerm(
        func=mdp.seat_agx_reward,
        weight=1.0,
        params={"target_offset": 0.25},
    )


@configclass
class PackAgxOrinEnvCfg(ManagerBasedRLEnvCfg):
    """Independent joint-control environment for the AGX Orin packing task."""

    scene: PackAgxOrinSceneCfg = PackAgxOrinSceneCfg(
        num_envs=1,
        env_spacing=10.0,
        replicate_physics=False,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.2, -0.2, 2.2),
        lookat=(-0.40, 0.0, 1.10),
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
        self.episode_length_s = 15.0
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
