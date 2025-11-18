# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.box_pushing import mdp
from isaaclab_tasks.manager_based.box_pushing.mdp.commands.pose_command_min_dist_cfg import (
    UniformPoseWithMinDistCommandCfg,
)

ENV_DT = 0.01  # 100Hz / decimation

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.0, 1.0),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    def __init__(self, r2r) -> None:

        ranges = (
            UniformPoseWithMinDistCommandCfg.Ranges(
                pos_x=(0.3, 0.6),
                pos_y=(-0.45, 0.45),
                pos_z=(0.007, 0.007),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0, 2 * torch.pi),
            )
            if r2r
            else UniformPoseWithMinDistCommandCfg.Ranges(
                pos_x=(0.4, 0.4),
                pos_y=(-0.3, -0.3),
                pos_z=(-0.01, -0.01),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(torch.pi, torch.pi),
            )
        )

        self.object_pose = UniformPoseWithMinDistCommandCfg(
            asset_name="robot",
            min_dist=0.3,
            body_name=MISSING,  # will be set by agent env cfg
            box_name="object",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=ranges,
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_effort: mdp.JointEffortActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_abs)
        joint_vel = ObsTerm(func=mdp.joint_vel_abs)
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        target_object_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    def __init__(
        self,
        r2r,
        use_ik_reset: bool = True,
        use_cached_ik: bool = True,
        pose_range: dict[str, tuple[float, float]] | None = None,
    ) -> None:

        self.reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

        pose_range = pose_range or {
            "x": (-0.15, 0.15),
            "y": (-0.45, 0.45),
            "z": (0.0, 0.0),
            "yaw": (0.0, 2 * torch.pi),
        }

        if r2r:
            self.reset_box_pose = EventTerm(
                func=mdp.sample_box_poses,
                mode="reset",
                params={"pose_range": pose_range, "asset_cfg": SceneEntityCfg("object", body_names="Object")},
            )

        if r2r and use_ik_reset:
            if use_cached_ik:
                self.reset_object_position = EventTerm(
                    func=mdp.reset_robot_cfg_with_cached_IK,
                    mode="reset",
                    params={"pose_range": pose_range, "asset_cfg": SceneEntityCfg("object", body_names="Object")},
                )
            else:
                self.reset_object_position = EventTerm(
                    func=mdp.reset_robot_cfg_with_IK,
                    mode="reset",
                )


@configclass
class DenseRewardCfg:
    """Reward terms for the MDP."""

    object_ee_distance = RewTerm(func=mdp.object_ee_distance, weight=-2.0 / (ENV_DT * 2))

    object_goal_position_distance = RewTerm(
        func=mdp.object_goal_position_distance,
        params={"command_name": "object_pose"},
        weight=-3.5 / (ENV_DT * 2),
    )

    object_goal_orientation_distance = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"command_name": "object_pose"},
        weight=-1.0 / torch.pi / (ENV_DT * 2),
    )

    energy_cost = RewTerm(func=mdp.action_scaled_l2, weight=-5e-4 / (ENV_DT * 2))

    joint_position_limit = RewTerm(func=mdp.joint_pos_limits_bp, weight=-1.0 / (ENV_DT * 2))

    joint_velocity_limit = RewTerm(
        func=mdp.joint_vel_limits_bp,
        params={"soft_ratio": 1.0},
        weight=-1.0 / (ENV_DT * 2),
    )

    rod_inclined_angle = RewTerm(func=mdp.rod_inclined_angle, weight=-1.0 / (ENV_DT * 2))


@configclass
class TemporalSparseRewardCfg:
    """Reward terms for the MDP."""

    object_ee_distance = RewTerm(func=mdp.object_ee_distance, weight=-2.0 / (ENV_DT * 2))

    object_goal_position_distance = RewTerm(
        func=mdp.object_goal_position_distance,
        params={"end_ep": True, "end_ep_weight": 100.0, "command_name": "object_pose"},
        weight=-3.5 / (ENV_DT * 2),
    )

    object_goal_orientation_distance = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"end_ep": True, "end_ep_weight": 100.0, "command_name": "object_pose"},
        weight=-1.0 / torch.pi / (ENV_DT * 2),
    )

    energy_cost = RewTerm(func=mdp.action_scaled_l2, weight=-0.02 / (ENV_DT * 2))

    joint_position_limit = RewTerm(func=mdp.joint_pos_limits_bp, weight=-1.0 / (ENV_DT * 2))

    joint_velocity_limit = RewTerm(
        func=mdp.joint_vel_limits_bp,
        params={"soft_ratio": 1.0},
        weight=-1.0 / (ENV_DT * 2),
    )

    rod_inclined_angle = RewTerm(func=mdp.rod_inclined_angle, weight=-1.0 / (ENV_DT * 2))

    end_ep_vel = RewTerm(func=mdp.end_ep_vel, weight=-50.0 / (ENV_DT * 2))


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.is_success,
        params={
            "command_name": "object_pose",
            "limit_pose_dist": 0.05,
            "limit_or_dist": 0.5,
        },
    )


##
# Environment configuration
##


@configclass
class BoxPushingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Value setting the task to random to random.
    r2r = True

    # Toggle whether IK-based resets are used (expensive but diverse).
    use_ik_reset: bool = True
    use_cached_ik: bool = True

    ik_cache_num_samples: int = 2048

    ik_grid_precision = 100

    pose_sampling_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.15, 0.15),
            "y": (-0.45, 0.45),
            "z": (0.0, 0.0),
            "yaw": (0.0, 2 * torch.pi),
        }
    )

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg | None = None
    # MDP settings
    # rewards: will be populated by agent env cfg
    rewards = MISSING
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg | None = None

    def __post_init__(self):
        """Post initialization."""

        # simulation settings
        self.sim.dt = ENV_DT

        # general settings
        max_steps = 200
        self.decimation = 2
        self.episode_length_s = max_steps * self.sim.dt

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 50 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # lazily construct commands/events since they depend on the configuration
        if self.commands is None:
            self.commands = CommandsCfg(self.r2r)
        if self.events is None:
            self.events = EventCfg(
                self.r2r,
                self.use_ik_reset,
                self.use_cached_ik,
                pose_range=self.pose_sampling_range,
            )
