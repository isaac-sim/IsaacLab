# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 + Sharpa Wave env for the pick-and-place apple task (PhysX, rigid apple)."""

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

TABLE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Lightwheel/Table256/Table256_cloth.usd"
APPLE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Apple/Apple033.usd"
PLATE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Plate001/plate001.usd"


# Reset-pose randomization half-ranges. PNP_APPLE_XY_RANGE (metres) drives the x
# half-range and the y magnitude; PNP_APPLE_YAW_RANGE_DEG drives yaw. The defaults
# reproduce the ranges the task's demonstrations were recorded with.
XY_RANGE = float(os.environ.get("PNP_APPLE_XY_RANGE", "0.015"))
YAW_RANGE = math.radians(float(os.environ.get("PNP_APPLE_YAW_RANGE_DEG", "0")))


@configclass
class PickAndPlaceAppleSceneCfg(H2TabletopSceneCfg):
    """H2 scene with front, left-wrist, and right-wrist cameras."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=(1.0, 1.0, 1.20),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
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
            mass_props=sim_utils.MassCfg(mass=0.05),
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
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.2),
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

    front_camera = H2_HEAD_CAMERA_CFG.replace(height=240, width=320)
    left_wrist_camera = LEFT_WRIST_CAMERA_CFG.replace(height=240, width=320)
    right_wrist_camera = RIGHT_WRIST_CAMERA_CFG.replace(height=240, width=320)


@configclass
class ObservationsCfg:
    """Joint state and three camera observations."""

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "normalize": False},
            tensor_processors=[TensorProcessorCfg(func=apply_camera_response)],
        )
        left_wrist_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "normalize": False},
            tensor_processors=[TensorProcessorCfg(func=apply_camera_response)],
        )
        right_wrist_camera = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "normalize": False},
            tensor_processors=[TensorProcessorCfg(func=apply_camera_response)],
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyStateCfg = PolicyStateCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class EventCfg:
    """Reset the scene, then randomize the apple in XY."""

    # Also latch PD targets (esp. head_pitch=0.6). Controllers that only
    # command a subset of joints otherwise leave head target at 0 and IdealPD
    # lifts the camera into the wall.
    reset_scene = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # Must follow ``reset_scene`` so the default reset does not overwrite it.
    randomize_apple_xy = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("apple"),
            "pose_range": {"x": (-XY_RANGE, XY_RANGE), "y": (-XY_RANGE, XY_RANGE), "yaw": (-YAW_RANGE, YAW_RANGE)},
            "velocity_range": {},
        },
    )


@configclass
class TerminationsCfg:
    """Timeout, phase success, and apple-drop failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=task_success,
        time_out=False,
        params={
            "apple_cfg": SceneEntityCfg("apple"),
            "plate_cfg": SceneEntityCfg("plate"),
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                joint_names=["right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_AA", "right_thumb_IP"],
                preserve_order=True,
            ),
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
        },
    )


@configclass
class RewardsCfg:
    """Sparse phase rewards: left lift, right catch, place, then release."""

    left_grasp_lift = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 0})
    handover_to_right = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 1})
    place_on_plate = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 2})
    release_on_plate = RewTerm(func=phase_reward, weight=1.0, params={"from_phase": 3})


@configclass
class PickAndPlaceAppleEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H2 + Sharpa Wave pick-and-place apple env (PhysX)."""

    scene: PickAndPlaceAppleSceneCfg = PickAndPlaceAppleSceneCfg(
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

    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=4, physics=PhysxCfg(gpu_max_num_partitions=32))
    decimation = 4
    episode_length_s = 20.0
    # Return camera observations from the reset episode, not the previous one.
    num_rerenders_on_reset = 2
