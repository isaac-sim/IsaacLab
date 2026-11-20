# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 + Sharpa Wave scene for packing an AGX Orin into a protective box."""

import math
import os

from isaaclab_isaacsim.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.tensor_processors import TensorProcessorCfg

from isaaclab_tasks.contrib.h2_sharpa.mdp import apply_camera_response, phase_reward
from isaaclab_tasks.contrib.h2_sharpa.tabletop_env_cfg import (
    LEFT_WRIST_CAMERA_CFG,
    RIGHT_WRIST_CAMERA_CFG,
    ActionsCfg,
    H2TabletopSceneCfg,
    PolicyStateCfg,
)

from isaaclab_assets.sensors import H2_HEAD_CAMERA_CFG

from .mdp import task_success

TABLE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Lightwheel/Table256/Table256.usd"
AGX_ORIN_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Lightwheel/MiniPc001/MiniPc001.usd"
PROTECTIVE_BOX_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Lightwheel/ProtectiveBox001/ProtectiveBox001.usd"


# Reset-pose randomization half-ranges. PACK_AGX_XY_RANGE (metres) drives the x
# half-range and the y magnitude; PACK_AGX_YAW_RANGE_DEG drives yaw. The defaults
# reproduce the ranges the MimicGen dataset was generated with. The OSMO
# workflows set both from `--set xy_range=... yaw_range_deg=...`.
XY_RANGE = float(os.environ.get("PACK_AGX_XY_RANGE", "0.05"))
YAW_RANGE = math.radians(float(os.environ.get("PACK_AGX_YAW_RANGE_DEG", "10")))


@configclass
class PackAgxOrinSceneCfg(H2TabletopSceneCfg):
    """Independent H2 scene for packing an AGX Orin."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=(1.07, 0.92, 1.12),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.75),
                roughness=0.8,
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
            visual_material_path=None,
            visual_material=sim_utils.PbrMdlCfg(
                diffuse_color_constant=None,
                metallic_constant=0.15,
                reflection_roughness_constant=0.55,
                albedo_brightness=3.4,
                metallic_texture_influence=0.0,
                reflection_roughness_texture_influence=0.0,
            ),
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
            visual_material_path=None,
            visual_material=sim_utils.PbrMdlCfg(
                diffuse_color_constant=None,
                metallic_constant=0.05,
                reflection_roughness_constant=0.85,
                albedo_brightness=2.6,
                metallic_texture_influence=0.0,
                reflection_roughness_texture_influence=0.0,
            ),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
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

    front_camera = H2_HEAD_CAMERA_CFG.replace(height=240, width=320)
    left_wrist_camera = LEFT_WRIST_CAMERA_CFG.replace(height=240, width=320)
    right_wrist_camera = RIGHT_WRIST_CAMERA_CFG.replace(height=240, width=320)


@configclass
class ObservationsCfg:
    """Robot state and three camera observations."""

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "normalize": False},
            tensor_processors=[
                TensorProcessorCfg(
                    func=apply_camera_response,
                    params={"response_gamma": (0.4545, 0.4545, 0.4545), "response_gain": (0.879, 0.824, 0.848)},
                )
            ],
        )
        left_wrist_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "normalize": False},
            tensor_processors=[
                TensorProcessorCfg(
                    func=apply_camera_response,
                    params={"response_gamma": (0.4545, 0.4545, 0.4545), "response_gain": (0.732, 0.745, 0.735)},
                )
            ],
        )
        right_wrist_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "normalize": False},
            tensor_processors=[
                TensorProcessorCfg(
                    func=apply_camera_response,
                    params={"response_gamma": (0.4545, 0.4545, 0.4545), "response_gain": (0.815, 0.814, 0.826)},
                )
            ],
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyStateCfg = PolicyStateCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class EventCfg:
    """Reset the scene, then randomize the AGX Orin on the tabletop."""

    reset_scene = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    randomize_agx_orin_xy = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("agx_orin"),
            "pose_range": {"x": (-XY_RANGE, XY_RANGE), "y": (-XY_RANGE, 0.0), "yaw": (-YAW_RANGE, YAW_RANGE)},
            "velocity_range": {},
        },
    )
    randomize_box_xy = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("protective_box"),
            "pose_range": {"x": (-XY_RANGE, XY_RANGE), "y": (0.0, XY_RANGE), "yaw": (-YAW_RANGE, YAW_RANGE)},
            "velocity_range": {},
        },
    )


@configclass
class TerminationsCfg:
    """Timeout and phase-based completion for RL post-training."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=task_success,
        time_out=False,
        params={
            "agx_orin_cfg": SceneEntityCfg("agx_orin"),
            "robot_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link", joint_names="right_thumb_MCP_FE"),
            "protective_box_cfg": SceneEntityCfg("protective_box"),
            "lift_z": 0.01,
            "align_target_z_offset": 0.10,
            "align_xy": 0.025,  # 3D distance tolerance
            "rotation_tolerance": 0.2,
            "lift_hold_steps": 5,
            "align_hold_steps": 5,
            "seat_hold_steps": 5,
        },
    )


@configclass
class RewardsCfg:
    """Sparse phase rewards: lift, align, seat, then release."""

    lift_agx = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 0})
    align_agx = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 1})
    seat_agx = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 2})
    release_agx = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 3})


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

    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=4, physics=PhysxCfg(gpu_max_num_partitions=32))
    decimation = 4
    episode_length_s = 15.0
    # Return camera observations from the reset episode, not the previous one.
    num_rerenders_on_reset = 2
