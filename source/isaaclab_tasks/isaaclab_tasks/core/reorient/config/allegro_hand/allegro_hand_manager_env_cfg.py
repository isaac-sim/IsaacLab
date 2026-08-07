# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Allegro Hand Direct reorientation task."""

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
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
)

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@configclass
class AllegroHandManagerSceneCfg(InteractiveSceneCfg):
    """Shared reorientation scene with the Allegro hand and a ground plane."""

    num_envs = 8192
    env_spacing = 0.75

    robot: ArticulationCfg = ALLEGRO_HAND_ROBOT_CFG
    object: RigidObjectCfg = CUBE_CFG
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    dome_light = None


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=0.2,
        make_quat_unique=False,
        fixed_marker_pos=(-0.2, -0.45, 0.68),
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
        # -- robot
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        # -- object
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
        # -- command
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
        # -- robot fingertips
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
    """Shared randomization terms with the Direct task's reset distribution.

    Gated by :attr:`AllegroHandManagerEnvCfg.enable_domain_randomization`.
    """

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
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
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
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
        mode="reset",
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
    """Shared reward terms tuned to the Direct task's scales."""

    track_orientation_inv_l2 = RewTerm(
        func=mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"},
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )
    track_pos_l2 = RewTerm(
        func=mdp.track_pos_l2,
        weight=-10.0,
        params={"command_name": "object_pose", "object_cfg": SceneEntityCfg("object")},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0002)


@configclass
class TerminationsCfg:
    """Shared terminations reduced to the Direct task's fall condition."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class AllegroHandManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    scene: AllegroHandManagerSceneCfg = AllegroHandManagerSceneCfg()
    decimation = 4
    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    enable_domain_randomization: bool = False
    """Apply the domain-randomization event terms.

    Off by default so the task matches its Direct counterpart, which randomizes nothing beyond
    the reset distributions. ``__post_init__`` reads it while building the configuration, before
    Hydra applies command-line overrides, so ``env.enable_domain_randomization=true`` has no
    effect -- set it on the configuration. Changing it requires retraining.
    """

    def __post_init__(self):
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0))
        if not self.enable_domain_randomization:
            self.events.robot_physics_material = None
            self.events.robot_scale_mass = None
            self.events.robot_joint_stiffness_and_damping = None
            self.events.object_physics_material = None
            self.events.object_scale_mass = None
