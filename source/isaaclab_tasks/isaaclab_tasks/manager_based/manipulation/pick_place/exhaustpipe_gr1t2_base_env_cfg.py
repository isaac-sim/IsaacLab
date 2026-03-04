# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import tempfile
from dataclasses import MISSING

import torch

try:
    from isaaclab_teleop import XrCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg

# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Exhaust Pipe Base Scene."""

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    blue_exhaust_pipe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueExhaustPipe",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04904, 0.31, 1.2590], rot=[0, 1.0, 0.0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd",
            scale=(0.5, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    blue_sorting_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueSortingBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.16605, 0.39, 0.98634], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd",
            scale=(1.0, 1.7, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    black_sorting_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlackSortingBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40132, 0.39, 0.98634], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/black_sorting_bin.usd",
            scale=(1.0, 1.7, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": -0.10933163,
                "left_shoulder_roll_joint": 0.43292055,
                "left_shoulder_yaw_joint": -0.15983289,
                "left_elbow_pitch_joint": -1.48233023,
                "left_wrist_yaw_joint": 0.2359135,
                "left_wrist_roll_joint": 0.26184522,
                "left_wrist_pitch_joint": 0.00830735,
                # right hand
                "R_index_intermediate_joint": 0.0,
                "R_index_proximal_joint": 0.0,
                "R_middle_intermediate_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_pinky_intermediate_joint": 0.0,
                "R_pinky_proximal_joint": 0.0,
                "R_ring_intermediate_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_thumb_distal_joint": 0.0,
                "R_thumb_proximal_pitch_joint": 0.0,
                "R_thumb_proximal_yaw_joint": -1.57,
                # left hand
                "L_index_intermediate_joint": 0.0,
                "L_index_proximal_joint": 0.0,
                "L_middle_intermediate_joint": 0.0,
                "L_middle_proximal_joint": 0.0,
                "L_pinky_intermediate_joint": 0.0,
                "L_pinky_proximal_joint": 0.0,
                "L_ring_intermediate_joint": 0.0,
                "L_ring_proximal_joint": 0.0,
                "L_thumb_distal_joint": 0.0,
                "L_thumb_proximal_pitch_joint": 0.0,
                "L_thumb_proximal_yaw_joint": -1.57,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Set table view camera
    robot_pov_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12, 1.85418), rot=(0.0, 0.98502, 0.0, -0.17246), convention="ros"),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    gr1_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        robot_pov_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    blue_exhaust_pipe_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("blue_exhaust_pipe")},
    )

    success = DoneTerm(func=mdp.task_done_exhaust_pipe)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_blue_exhaust_pipe = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("blue_exhaust_pipe"),
        },
    )


@configclass
class ExhaustPipeGR1T2BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3),
    #                 right arm quat (4), left/right hand joint pos (22)]
    idle_action = torch.tensor(
        [
            [
                -0.2909,
                0.2778,
                1.1247,
                0.5747,
                -0.4160,
                0.4699,
                0.5253,
                0.22878,
                0.2536,
                1.0953,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = 2

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering

        # List of image observations in policy observations
        self.image_obs_list = ["robot_pov_cam"]

        if _TELEOP_AVAILABLE:
            self.xr = XrCfg(
                anchor_pos=(0.0, 0.0, 0.0),
                anchor_rot=(0.0, 0.0, 0.0, 1.0),
            )
