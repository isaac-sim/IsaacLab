# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration shared by the VLA benchmark tasks."""

import math
from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from . import VLA_BENCHMARKS_ASSETS_DIR
from .config.franka.robot_cfg import FRANKA_ROBOTIQ_CFG

GRIPPER_BODY = "robot/Gripper/Robotiq_2F_85/left_inner_finger"
"""Gripper link, relative to the environment prim, whose contacts decide whether an object is held."""


def _camera(prim_path: str, focal_length: float, pos: tuple, rot: tuple, follow_link: bool = False) -> CameraCfg:
    """Return a 448 x 252 RGB camera with the RoboLab lens (the 1280 x 720 view, downsampled)."""
    return CameraCfg(
        prim_path=prim_path,
        width=448,
        height=252,
        data_types=["rgb"],
        update_period=0.0,
        update_latest_camera_pose=follow_link,
        # The renderer that draws the objects selected as mesh variants.
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.01, 1.0e6),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


@configclass
class VLABenchmarkSceneCfg(InteractiveSceneCfg):
    """Robot, robot table, ground, lights, cameras and contact sensor of the VLA benchmark tasks.

    Tasks add their rigid objects under the environment prim, the furniture first. The contact sensor
    senses every rigid object and the gripper link against each other.
    """

    robot = FRANKA_ROBOTIQ_CFG
    table_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/franka_table/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
    )
    # One ground below the furniture of all tasks; the RoboLab scenes hide theirs.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.697)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            texture_file=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/home_office.exr",
            texture_format="latlong",
            visible_in_primary_ray=True,
        ),
    )
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )
    over_shoulder_left_camera = _camera(
        "{ENV_REGEX_NS}/over_shoulder_left_camera", 2.1, (0.05, 0.57, 0.66), (-0.195, 0.399, 0.805, -0.393)
    )
    # The wrist camera rides on the gripper, so its pose follows the simulated link.
    wrist_cam = _camera(
        "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        2.8,
        (0.011, -0.031, -0.074),
        (0.57, 0.576, -0.409, -0.42),
        follow_link=True,
    )
    contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/([^/]+|{GRIPPER_BODY})",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/([^/]+|{GRIPPER_BODY})"],
        update_period=0.0,
    )


@configclass
class ActionsCfg:
    """Arm joint position targets and a binary gripper command (negative closes)."""

    arm = isaaclab_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True, use_default_offset=False
    )
    gripper = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": math.pi / 4},
    )


@configclass
class ObservationsCfg:
    """Camera images and arm and gripper positions for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        over_shoulder_left_camera = ObsTerm(
            func=isaaclab_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("over_shoulder_left_camera"), "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=isaaclab_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "normalize": False}
        )
        arm_joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"], preserve_order=True)},
        )
        # Gripper opening from 0 (open) to 1 (closed).
        gripper_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
            clip=(0.0, math.pi / 4),
            scale=4.0 / math.pi,
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy = PolicyCfg()


@configclass
class EventsCfg:
    """Reset the robot and the objects to their start states."""

    reset = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class TerminationsCfg:
    """Time-out; tasks add their success term."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)


@configclass
class VLABenchmarkEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration of a VLA benchmark task on Newton with MuJoCo Warp and Newton contacts."""

    scene: VLABenchmarkSceneCfg = VLABenchmarkSceneCfg(num_envs=1, env_spacing=10.0, lazy_sensor_update=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards = {}
    decimation = 8
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=8,
        # Newton preset of the Franka stack task, with sweep-and-prune broad phase.
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=4096,
                nconmax=4096,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            num_substeps=2,
            # Thousands of small hull meshes: cuBQL takes about 0.1 s to build each one's BVH, LBVH a few ms.
            bvh_constructor_geometry="lbvh",
        ),
    )

    instruction: str = MISSING
    """Language instruction of the task."""
