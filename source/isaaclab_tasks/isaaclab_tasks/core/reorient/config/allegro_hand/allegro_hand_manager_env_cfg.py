# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Allegro Hand Direct reorientation task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    ROBOT_CFG,
    AllegroHandTaskCfgBase,
    ObjectCfg,
)
from isaaclab_tasks.core.reorient.reorient_task_base import (
    ALLEGRO_ACTUATED_JOINT_NAMES,
    ALLEGRO_FINGERTIP_BODY_NAMES,
    ReorientTerminationsCfg,
    reorient_goal_command,
    reorient_joint_action,
    reorient_reset_event,
    reorient_reward_term,
)
from isaaclab_tasks.utils import PresetCfg


@configclass
class AllegroCubeSceneCfg(PresetCfg):
    """Backend-specific scene cloning settings matching the Direct task."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        """Allegro scene shared by the backend alternatives."""

        num_envs = 8192
        env_spacing = 0.75
        replicate_physics = True

        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = ROBOT_CFG
        object: ObjectCfg = OBJECT_CFG
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )

    physx = SceneCfg(clone_in_fabric=True)
    newton_mjwarp = SceneCfg(clone_in_fabric=False)
    ovphysx = physx
    default = physx

    def set_num_envs(self, num_envs: int) -> None:
        """Set the environment count on every backend alternative."""
        for scene in (self.physx, self.newton_mjwarp, self.ovphysx, self.default):
            scene.num_envs = num_envs


@configclass
class CommandsCfg:
    """The reorient goal command with the Allegro marker and tolerance."""

    object_pose = reorient_goal_command(orientation_success_threshold=0.2, goal_pose_visualizer_cfg=GOAL_OBJECT_CFG)


@configclass
class ActionsCfg:
    """Sixteen actuated Allegro Hand joints in Direct order."""

    joint_pos = reorient_joint_action(ALLEGRO_ACTUATED_JOINT_NAMES)


@configclass
class ObservationsCfg:
    """Full 124-dimensional state observation in Direct order."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
        )
        object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
        fingertip_pos = ObsTerm(
            func=mdp.fingertip_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        fingertip_quat = ObsTerm(
            func=mdp.fingertip_quat,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        fingertip_vel = ObsTerm(
            func=mdp.fingertip_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset distributions matching the Direct task, plus opt-in domain randomization.

    The domain-randomization terms reproduce the legacy manager recipe. They are
    startup-mode terms and are dropped by default (see
    :attr:`AllegroCubeEnvCfg.enable_domain_randomization`): the Direct task has no
    domain randomization, and the validated benchmark thresholds were calibrated
    without it. Enabling them requires retraining.
    """

    # -- opt-in domain randomization (legacy manager recipe parameters)
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.3, 3.0),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.4, 1.6),
            "operation": "scale",
        },
    )

    reset_state = reorient_reset_event()


@configclass
class RewardsCfg:
    """The reorient reward with the Allegro success tolerance."""

    reorient = reorient_reward_term(success_tolerance=0.2)


@configclass
class AllegroCubeEnvCfg(AllegroHandTaskCfgBase, ManagerBasedRLEnvCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    scene: AllegroCubeSceneCfg = AllegroCubeSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # the shared termination section from isaaclab_tasks.core.reorient.reorient_task_base
    terminations: ReorientTerminationsCfg = ReorientTerminationsCfg()
    events: EventCfg = EventCfg()

    enable_domain_randomization: bool = False
    """Enable the legacy startup domain-randomization terms.

    Disabled by default: the validated reference training runs and the benchmark
    thresholds were produced without domain randomization, so enabling it
    requires retraining and threshold recalibration.
    """

    _DOMAIN_RANDOMIZATION_TERMS = (
        "robot_physics_material",
        "robot_scale_mass",
        "robot_joint_stiffness_and_damping",
        "object_physics_material",
        "object_scale_mass",
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)
        if not self.enable_domain_randomization:
            for term_name in self._DOMAIN_RANDOMIZATION_TERMS:
                setattr(self.events, term_name, None)


@configclass
class AllegroCubeEnvCfg_PLAY(AllegroCubeEnvCfg):
    """Reduced, deterministic play configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.set_num_envs(50)
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
