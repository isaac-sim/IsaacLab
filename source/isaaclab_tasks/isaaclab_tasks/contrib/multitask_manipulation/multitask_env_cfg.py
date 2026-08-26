# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous OpenArm-lift, Franka-cabinet, and UR10-reach environment configuration."""

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import FRAME_MARKER_CFG, SPHERE_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as add_scene
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg
from isaaclab_tasks.core.cabinet.cabinet_env_cfg import LIGHT_CFG, PLANE_CFG
from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
from isaaclab_tasks.core.reach.config.ur_10.joint_pos_env_cfg import UR10ReachEnvCfg

from . import mdp
from .selection_utils import SceneEntitySelectionCfg

_LIFT_ROBOT = "robot"
_LIFT_OBJECT = "object"
_CABINET_ROBOT = "robot_1"
_CABINET = "cabinet"
_REACH_ROBOT = "robot_2"


def _selection(name: str, **kwargs) -> SceneEntitySelectionCfg:
    """Create a selection-aware scene entity configuration."""
    return SceneEntitySelectionCfg(name, **kwargs)


def _task_asset_cfgs() -> tuple[SceneEntitySelectionCfg, ...]:
    """Create task-defining entity selections in policy encoding order."""
    return tuple(_selection(name) for name in (_LIFT_ROBOT, _CABINET_ROBOT, _REACH_ROBOT))


def _lift_robot_cfg() -> SceneEntitySelectionCfg:
    """Create the lift OpenArm TCP selection."""
    return _selection(_LIFT_ROBOT, body_names="openarm_ee_tcp")


def _lift_joint_cfg() -> SceneEntitySelectionCfg:
    """Create the actuated OpenArm joint selection."""
    return _selection(_LIFT_ROBOT, joint_names=["openarm_joint.*", "openarm_finger_joint.*"])


def _lift_object_cfg() -> SceneEntitySelectionCfg:
    """Create the lift object selection."""
    return _selection(_LIFT_OBJECT)


def _cabinet_robot_cfg() -> SceneEntitySelectionCfg:
    """Create ordered cabinet Franka hand, fingertip, and finger-joint selections."""
    return _selection(
        _CABINET_ROBOT,
        body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        joint_names="panda_finger_.*",
        preserve_order=True,
    )


def _cabinet_cfg() -> SceneEntitySelectionCfg:
    """Create the cabinet handle and drawer-joint selection."""
    return _selection(_CABINET, body_names="drawer_handle_top", joint_names="drawer_top_joint")


def _reach_robot_cfg() -> SceneEntitySelectionCfg:
    """Create the UR10 end-effector selection."""
    return _selection(_REACH_ROBOT, body_names="ee_link")


def _frame_marker_cfg(prim_path: str) -> VisualizationMarkersCfg:
    """Create a compact frame marker at a task-unique path."""
    cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    return cfg


def _sphere_marker_cfg(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Create a compact sphere marker at a task-unique path."""
    cfg = SPHERE_MARKER_CFG.replace(prim_path=prim_path)
    cfg.markers["sphere"].radius = 0.025
    cfg.markers["sphere"].visual_material.diffuse_color = color
    return cfg


def _skip_global_asset(asset_cfg: AssetBaseCfg) -> bool:
    """Return whether an asset is outside the replicated environment namespace."""
    return not asset_cfg.prim_path.startswith("{ENV_REGEX_NS}/")


def _make_scene_cfg() -> InteractiveSceneCfg:
    """Compose three task layouts from their homogeneous source scenes."""
    scene = OpenArmCubeLiftEnvCfg().scene
    scene.table = None
    add_scene(scene, FrankaCabinetEnvCfg().scene, asset_skip=_skip_global_asset)
    reach_scene = UR10ReachEnvCfg().scene
    reach_scene.table = None
    add_scene(scene, reach_scene, asset_skip=_skip_global_asset)
    scene.num_envs = 4096
    scene.env_spacing = 3.0
    scene.replicate_physics = True
    scene.plane = PLANE_CFG.copy()
    scene.light = LIGHT_CFG.copy()
    return scene


@configclass
class ActionsCfg:
    """Task-headed action specification with a fixed global dimension of 22."""

    lift_arm_action = mdp.SelectedJointPositionActionCfg(
        asset_name=_LIFT_ROBOT,
        joint_names=["openarm_joint.*"],
        scale=0.5,
    )
    lift_gripper_action = mdp.SelectedBinaryJointPositionActionCfg(
        asset_name=_LIFT_ROBOT,
        joint_names=["openarm_finger_joint.*"],
        open_command=0.044,
        close_command=0.0,
    )
    cabinet_arm_action = mdp.SelectedJointPositionActionCfg(
        asset_name=_CABINET_ROBOT,
        joint_names=["panda_joint.*"],
        scale=1.0,
        joint_limit_margin=0.02,
    )
    cabinet_gripper_action = mdp.SelectedBinaryJointPositionActionCfg(
        asset_name=_CABINET_ROBOT,
        joint_names=["panda_finger.*"],
        open_command=0.04,
        close_command=0.0,
    )
    reach_action = mdp.SelectedJointPositionActionCfg(
        asset_name=_REACH_ROBOT,
        joint_names=[".*"],
        scale=0.5,
    )


@configclass
class CommandsCfg:
    """Goal commands for lift and reach environments."""

    lift_pose = mdp.SelectedUniformPoseCommandCfg(
        reference_cfg=_selection(_LIFT_ROBOT),
        tracked_cfg=_lift_object_cfg(),
        resampling_time_range=(5.0, 5.0),
        goal_pose_visualizer_cfg=_sphere_marker_cfg("/Visuals/Command/lift_pose/goal", (0.0, 1.0, 0.0)),
        current_pose_visualizer_cfg=_sphere_marker_cfg("/Visuals/Command/lift_pose/current", (0.0, 0.0, 1.0)),
        ranges=mdp.SelectedUniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.4),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(0.0, 0.0),
        ),
    )
    reach_pose = mdp.SelectedUniformPoseCommandCfg(
        reference_cfg=_selection(_REACH_ROBOT),
        tracked_cfg=_reach_robot_cfg(),
        resampling_time_range=(4.0, 4.0),
        goal_pose_visualizer_cfg=_frame_marker_cfg("/Visuals/Command/reach_pose/goal"),
        current_pose_visualizer_cfg=_frame_marker_cfg("/Visuals/Command/reach_pose/current"),
        ranges=mdp.SelectedUniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg:
    """Fixed-width observations with zero-filled inactive task blocks."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        task_id = ObsTerm(func=mdp.task_encoding, params={"task_asset_cfgs": _task_asset_cfgs()})

        lift_joint_pos = ObsTerm(
            func=mdp.selected_joint_pos_rel,
            params={"asset_cfg": _lift_joint_cfg()},
        )
        lift_joint_vel = ObsTerm(
            func=mdp.selected_joint_vel_rel,
            params={"asset_cfg": _lift_joint_cfg()},
        )
        lift_object_position = ObsTerm(
            func=mdp.lift_object_position_b,
            params={"robot_cfg": _lift_robot_cfg(), "object_cfg": _lift_object_cfg()},
        )
        lift_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "lift_pose"})

        cabinet_joint_pos = ObsTerm(
            func=mdp.selected_joint_pos_rel,
            params={"asset_cfg": _selection(_CABINET_ROBOT, joint_names=".*")},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.selected_joint_vel_rel,
            params={"asset_cfg": _selection(_CABINET_ROBOT, joint_names=".*")},
        )
        cabinet_drawer = ObsTerm(func=mdp.cabinet_drawer_state, params={"cabinet_cfg": _cabinet_cfg()})
        cabinet_ee_handle = ObsTerm(
            func=mdp.cabinet_ee_to_handle,
            params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg()},
        )

        reach_joint_pos = ObsTerm(
            func=mdp.selected_joint_pos_rel,
            params={"asset_cfg": _selection(_REACH_ROBOT, joint_names=".*")},
        )
        reach_joint_vel = ObsTerm(
            func=mdp.selected_joint_vel_rel,
            params={"asset_cfg": _selection(_REACH_ROBOT, joint_names=".*")},
        )
        reach_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "reach_pose"})

        actions = ObsTerm(func=base_mdp.last_action, clip=(-5.0, 5.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Selection-aware reset events."""

    reset_scene = EventTerm(
        func=mdp.reset_multitask_scene,
        mode="reset",
        params={
            "root_asset_cfgs": tuple(
                _selection(name) for name in (_LIFT_ROBOT, _CABINET_ROBOT, _CABINET, _REACH_ROBOT)
            ),
            "lift_robot_cfg": _selection(_LIFT_ROBOT),
            "lift_object_cfg": _lift_object_cfg(),
            "cabinet_robot_cfg": _selection(_CABINET_ROBOT),
            "cabinet_cfg": _selection(_CABINET),
            "reach_robot_cfg": _selection(_REACH_ROBOT),
        },
    )


@configclass
class RewardsCfg:
    """Task rewards adapted from the three homogeneous tasks."""

    lift_approach = RewTerm(
        func=mdp.lift_ee_object_distance,
        weight=0.11,
        params={"robot_cfg": _lift_robot_cfg(), "object_cfg": _lift_object_cfg(), "std": 0.1},
    )
    lift_height = RewTerm(
        func=mdp.lift_object_height,
        weight=1.5,
        params={"object_cfg": _lift_object_cfg(), "minimum_height": 0.04},
    )
    lift_goal = RewTerm(
        func=mdp.LiftGoalTracking,
        weight=1.6,
        params={
            "robot_cfg": _lift_robot_cfg(),
            "object_cfg": _lift_object_cfg(),
            "command_name": "lift_pose",
            "std": 0.3,
            "minimum_height": 0.04,
            "success_threshold": 0.05,
        },
    )
    lift_goal_fine = RewTerm(
        func=mdp.lift_goal_tracking,
        weight=0.5,
        params={
            "robot_cfg": _lift_robot_cfg(),
            "object_cfg": _lift_object_cfg(),
            "command_name": "lift_pose",
            "std": 0.05,
            "minimum_height": 0.04,
        },
    )
    lift_action_rate = RewTerm(
        func=mdp.selected_action_rate_l2,
        weight=-0.00001,
        params={
            "task_asset_cfg": _selection(_LIFT_ROBOT),
            "action_term_names": ("lift_arm_action", "lift_gripper_action"),
        },
    )
    lift_joint_vel = RewTerm(
        func=mdp.selected_joint_vel_l2,
        weight=-0.00001,
        params={"asset_cfg": _lift_joint_cfg()},
    )

    cabinet_approach = RewTerm(
        func=mdp.cabinet_approach_ee_handle,
        weight=2.0,
        params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg(), "threshold": 0.2},
    )
    cabinet_align = RewTerm(
        func=mdp.cabinet_align_ee_handle,
        weight=0.5,
        params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg()},
    )
    cabinet_approach_gripper = RewTerm(
        func=mdp.cabinet_approach_gripper,
        weight=5.0,
        params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg(), "offset": 0.04},
    )
    cabinet_align_grasp = RewTerm(
        func=mdp.cabinet_align_grasp,
        weight=0.125,
        params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg()},
    )
    cabinet_grasp = RewTerm(
        func=mdp.cabinet_grasp_handle,
        weight=0.5,
        params={
            "robot_cfg": _cabinet_robot_cfg(),
            "cabinet_cfg": _cabinet_cfg(),
            "threshold": 0.03,
            "open_joint_pos": 0.04,
        },
    )
    cabinet_open = RewTerm(
        func=mdp.CabinetOpenDrawerBonus,
        weight=7.5,
        params={
            "robot_cfg": _cabinet_robot_cfg(),
            "cabinet_cfg": _cabinet_cfg(),
            "success_threshold": 0.3,
        },
    )
    cabinet_stages = RewTerm(
        func=mdp.cabinet_multi_stage_open,
        weight=1.0,
        params={"robot_cfg": _cabinet_robot_cfg(), "cabinet_cfg": _cabinet_cfg()},
    )
    cabinet_action_rate = RewTerm(
        func=mdp.selected_action_rate_l2,
        weight=-0.01,
        params={
            "task_asset_cfg": _selection(_CABINET_ROBOT),
            "action_term_names": ("cabinet_arm_action", "cabinet_gripper_action"),
        },
    )
    cabinet_joint_vel = RewTerm(
        func=mdp.selected_joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": _selection(_CABINET_ROBOT), "max_velocity": 50.0},
    )

    reach_position = RewTerm(
        func=mdp.reach_position_error,
        weight=-0.2,
        params={"robot_cfg": _reach_robot_cfg(), "command_name": "reach_pose"},
    )
    reach_orientation = RewTerm(
        func=mdp.reach_orientation_error,
        weight=-0.1,
        params={"robot_cfg": _reach_robot_cfg(), "command_name": "reach_pose"},
    )
    reach_success = RewTerm(
        func=base_mdp.is_terminated_term,
        weight=10.0,
        params={"term_keys": ["reach_success"]},
    )
    reach_action_rate = RewTerm(
        func=mdp.selected_action_rate_l2,
        weight=-0.0001,
        params={"task_asset_cfg": _selection(_REACH_ROBOT), "action_term_names": ("reach_action",)},
    )
    reach_action_l2 = RewTerm(
        func=mdp.selected_action_l2,
        weight=-0.005,
        params={"task_asset_cfg": _selection(_REACH_ROBOT), "action_term_names": ("reach_action",)},
    )
    reach_joint_vel = RewTerm(
        func=mdp.selected_joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": _selection(_REACH_ROBOT)},
    )


@configclass
class TerminationsCfg:
    """Task-specific success, failure, and timeout conditions."""

    reach_success = DoneTerm(
        func=mdp.reach_success,
        params={
            "robot_cfg": _reach_robot_cfg(),
            "command_name": "reach_pose",
            "position_threshold": 0.05,
            "orientation_threshold": 0.2,
        },
    )
    lift_object_dropped = DoneTerm(
        func=mdp.lift_object_dropped,
        params={"object_cfg": _lift_object_cfg(), "minimum_height": -0.05},
    )
    cabinet_state_invalid = DoneTerm(
        func=mdp.articulation_state_invalid,
        params={
            "asset_cfg": _selection(_CABINET_ROBOT, joint_names=".*"),
            "max_joint_velocity": 50.0,
            "joint_position_margin": 0.1,
        },
    )
    time_out = DoneTerm(
        func=mdp.task_time_out,
        time_out=True,
        params={"task_asset_cfgs": _task_asset_cfgs(), "episode_lengths_s": (6.0, 8.0, 12.0)},
    )


@configclass
class CurriculumCfg:
    """OpenArm lift penalty curriculum."""

    lift_action_rate = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={
            "term_name": "lift_action_rate",
            "weight": -0.01,
            "num_steps": 10000,
        },
    )
    lift_joint_vel = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={
            "term_name": "lift_joint_vel",
            "weight": -0.01,
            "num_steps": 10000,
        },
    )


@configclass
class MultitaskManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based heterogeneous manipulation training environment."""

    scene: InteractiveSceneCfg = _make_scene_cfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Configure a common PhysX control clock for all three tasks."""
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**26,
        )
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-3.0, 3.0, 2.5), lookat=(0.0, 0.0, 0.5))

    def validate_config(self) -> None:
        """Validate the minimum batch size required to instantiate all task views."""
        if self.scene.num_envs < 3:
            raise ValueError("IsaacContrib-Multitask-Manipulation requires at least three environments, one per task.")

    def play_mode(self):
        """Enable short episodes and task command markers during policy playback."""
        super().play_mode()
        self.terminations.time_out.params["episode_lengths_s"] = (3.0, 4.0, 6.0)
        self.commands.lift_pose.debug_vis = True
        self.commands.reach_pose.debug_vis = True
