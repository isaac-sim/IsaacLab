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
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    ROBOT_CFG,
    ObjectCfg,
    PhysicsCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


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
    default = newton_mjwarp

    def set_num_envs(self, num_envs: int) -> None:
        """Set the environment count on every backend alternative."""
        for scene in (self.physx, self.newton_mjwarp, self.ovphysx, self.default):
            scene.num_envs = num_envs


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.2,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Sixteen actuated Allegro Hand joints in Direct order."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=ALLEGRO_ACTUATED_JOINT_NAMES,
        alpha=1.0,
        rescale_to_limits=True,
    )


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

    reset_state = EventTerm(
        func=mdp.reset_reorient_state,
        mode="reset",
        params={
            "position_noise": 0.01,
            "joint_position_noise": 0.2,
            "joint_velocity_noise": 0.0,
            "action_name": "joint_pos",
        },
    )


@configclass
class RewardsCfg:
    """Direct-compatible reward and success accounting."""

    reorient = RewTerm(
        func=mdp.ReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": -10.0,
            "rotation_scale": 1.0,
            "rotation_epsilon": 0.1,
            "action_penalty_scale": -0.0002,
            "success_tolerance": 0.2,
            "success_bonus": 250.0,
            "fall_distance": 0.24,
            "fall_penalty": 0.0,
            "averaging_factor": 0.1,
            "success_count_threshold": 1,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions matching the Direct task."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class AllegroCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    scene: AllegroCubeSceneCfg = AllegroCubeSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
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
        # simulation — mirrors the Direct cfg (guarded by the value-parity test)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        self.sim.physics = PhysicsCfg()
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
