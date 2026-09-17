# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
from dataclasses import field
from typing import Any

from isaaclab_teleop import XrCfg

from isaaclab.utils import REQUIRED, replace_config

# Marker consumed by ``env_test_utils._is_teleop_env`` to bucket teleop
# environments in the test suite.
_TELEOP_AVAILABLE = True

from dataclasses import dataclass

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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip


##
# Scene definition
##
@dataclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Exhaust Pipe Base Scene."""

    # Table
    table: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
                scale=(1.0, 1.0, 1.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )
    )

    blue_exhaust_pipe: Any = field(
        default_factory=lambda: RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BlueExhaustPipe",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04904, 0.31, 1.2590], rot=[0, 1.0, 0.0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd",
                scale=(0.5, 0.5, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )
    )

    blue_sorting_bin: Any = field(
        default_factory=lambda: RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BlueSortingBin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.16605, 0.39, 0.98634], rot=[0.0, 0.0, 0.0, 1.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd",
                scale=(1.0, 1.7, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )
    )

    black_sorting_bin: Any = field(
        default_factory=lambda: RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BlackSortingBin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40132, 0.39, 0.98634], rot=[0.0, 0.0, 0.0, 1.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/black_sorting_bin.usd",
                scale=(1.0, 1.7, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = field(
        default_factory=lambda: replace_config(
            GR1T2_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
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
    )

    # Set table view camera
    robot_pov_cam: Any = field(
        default_factory=lambda: CameraCfg(
            prim_path="{ENV_REGEX_NS}/RobotPOVCam",
            update_period=0.0,
            height=160,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12, 1.85418), rot=(0.98502, 0.0, 0.0, -0.17246), convention="ros"),
        )
    )

    # Ground plane
    ground: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=GroundPlaneCfg(),
        )
    )

    # Lights
    light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    )


##
# MDP settings
##
@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    gr1_action: ActionTermCfg = REQUIRED


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action))
        robot_joint_pos: Any = field(
            default_factory=lambda: ObsTerm(
                func=base_mdp.joint_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
        )

        left_eef_pos: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        )
        left_eef_quat: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        )
        right_eef_pos: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        )
        right_eef_quat: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})
        )

        hand_joint_state: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        )
        head_joint_state: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.get_robot_joint_state,
                params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
            )
        )

        robot_pov_cam: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))

    blue_exhaust_pipe_dropped: Any = field(
        default_factory=lambda: DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("blue_exhaust_pipe")},
        )
    )

    success: Any = field(default_factory=lambda: DoneTerm(func=mdp.task_done_exhaust_pipe))


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_all: Any = field(default_factory=lambda: EventTerm(func=mdp.reset_scene_to_default, mode="reset"))

    reset_blue_exhaust_pipe: Any = field(
        default_factory=lambda: EventTerm(
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
    )


@dataclass
class ExhaustPipeGR1T2BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = field(
        default_factory=lambda: ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    )
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    # MDP settings
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: Any = field(default_factory=EventCfg)

    # Unused managers
    commands: Any = None
    rewards: Any = None
    curriculum: Any = None

    # Temporary directory for URDF files
    temp_urdf_dir: Any = field(default_factory=tempfile.gettempdir)

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3),
    #                 right arm quat (4), left/right hand joint pos (22)]
    idle_action: Any = field(
        default_factory=lambda: [
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
        set_isaac_rtx_global_settings(self.scene.robot_pov_cam.renderer_cfg, antialiasing_mode="DLAA")

        # List of image observations in policy observations
        self.image_obs_list = ["robot_pov_cam"]

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
