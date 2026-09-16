# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
We modified parts of the environment, such as the target's position and orientation,
as well as certain object properties, to better suit the smaller robot.
"""

from dataclasses import MISSING, dataclass

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import config_field, copy_config, replace_config
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from typing import Any

FRAME_MARKER_SMALL_CFG = copy_config(FRAME_MARKER_CFG)
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

from isaaclab_tasks.core.cabinet import mdp

##
# Scene definition
##


@dataclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = config_field(MISSING)
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = config_field(MISSING)

    cabinet: Any = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                activate_contact_sensors=False,
                scale=(0.75, 0.75, 0.75),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.7, 0, 0.3),
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos={
                    "door_left_joint": 0.0,
                    "door_right_joint": 0.0,
                    "drawer_bottom_joint": 0.0,
                    "drawer_top_joint": 0.0,
                },
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                    joint_effort_limit=87.0,
                    joint_velocity_limit=100.0,
                    stiffness=10.0,
                    damping=1.0,
                ),
                "doors": ImplicitActuatorCfg(
                    joint_names_expr=["door_left_joint", "door_right_joint"],
                    joint_effort_limit=87.0,
                    joint_velocity_limit=100.0,
                    stiffness=10.0,
                    damping=2.5,
                ),
            },
        )
    )

    # Frame definitions for the cabinet.
    cabinet_frame: Any = config_field(
        FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
            debug_vis=True,
            visualizer_cfg=replace_config(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",
                    name="drawer_handle_bottom",
                    offset=OffsetCfg(
                        pos=(0.222, 0.0, 0.005),
                        rot=(0.5, -0.5, -0.5, 0.5),  # align with end-effector frame
                    ),
                ),
            ],
        )
    )

    # plane
    plane: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(),
            spawn=sim_utils.GroundPlaneCfg(),
            collision_group=-1,
        )
    )

    # lights
    light: Any = config_field(
        AssetBaseCfg(
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

    arm_action: mdp.JointPositionActionCfg = config_field(MISSING)
    gripper_action: mdp.BinaryJointPositionActionCfg = config_field(MISSING)


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = config_field(ObsTerm(func=mdp.joint_vel_rel))
        cabinet_joint_pos: Any = config_field(
            ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
            )
        )
        cabinet_joint_vel: Any = config_field(
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
            )
        )
        rel_ee_drawer_distance: Any = config_field(ObsTerm(func=mdp.rel_ee_drawer_distance))

        actions: Any = config_field(ObsTerm(func=mdp.last_action))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 1.25),
                "dynamic_friction_range": (0.8, 1.25),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
    )

    cabinet_physics_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_bottom"),
                "static_friction_range": (2.25, 2.5),
                "dynamic_friction_range": (2.0, 2.25),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
    )

    reset_all: Any = config_field(EventTerm(func=mdp.reset_scene_to_default, mode="reset"))

    reset_robot_joints: Any = config_field(
        EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle: Any = config_field(RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2}))
    align_ee_handle: Any = config_field(RewTerm(func=mdp.align_ee_handle, weight=0.5))

    # 2. Grasp the handle
    approach_gripper_handle: Any = config_field(
        RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    )
    align_grasp_around_handle: Any = config_field(RewTerm(func=mdp.align_grasp_around_handle, weight=0.125))
    grasp_handle: Any = config_field(
        RewTerm(
            func=mdp.grasp_handle,
            weight=0.5,
            params={
                "threshold": 0.03,
                "open_joint_pos": MISSING,
                "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            },
        )
    )

    # 3. Open the drawer
    open_drawer_bonus: Any = config_field(
        RewTerm(
            func=mdp.open_drawer_bonus,
            weight=7.5,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"]),
                "success_threshold": 0.30,
            },
        )
    )
    multi_stage_open_drawer: Any = config_field(
        RewTerm(
            func=mdp.multi_stage_open_drawer,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
        )
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2: Any = config_field(RewTerm(func=mdp.action_rate_l2, weight=-1e-2))
    joint_vel: Any = config_field(RewTerm(func=mdp.joint_vel_l2, weight=-0.0001))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))


##
# Environment configuration
##


@dataclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = config_field(CabinetSceneCfg(num_envs=4096, env_spacing=2.0))
    # Basic settings
    observations: ObservationsCfg = config_field(ObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    # MDP settings
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventCfg = config_field(EventCfg())

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5))
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
