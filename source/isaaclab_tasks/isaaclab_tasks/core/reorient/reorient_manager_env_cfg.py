# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-agnostic manager-based reorientation task.

Every term the Shadow and Allegro tasks share lives here. A hand supplies its
fingertip bodies, its actuated joints, its scene, and its control rate; the term
lists, scales and weights are common.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.utils import PresetCfg


@configclass
class ReorientSceneBaseCfg(InteractiveSceneCfg):
    """Shared reorientation scene. A hand supplies its robot and its object."""

    num_envs = 8192
    env_spacing = 0.75

    robot: PresetCfg | ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    """In-hand goal pose, re-drawn on success."""

    object_pose = mdp.ReorientCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=MISSING,
        make_quat_unique=False,
        fixed_marker_pos=(-0.2, -0.45, 0.68),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Joint-position targets for the hand's actuated joints."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=MISSING,
        alpha=1.0,
        rescale_to_limits=True,
    )


@configclass
class ReorientRobotObsCfg(ObsGroup):
    """What the hand can measure about itself.

    The fingertip terms are the only hand-specific entries; the env cfg fills their
    ``asset_cfg`` from :attr:`ReorientManagerEnvBaseCfg.fingertip_body_names`.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": SceneEntityCfg("robot")})
    fingertip_pose = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": MISSING})
    fingertip_vel = ObsTerm(func=mdp.fingertip_vel, params={"asset_cfg": MISSING})

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ReorientObjectObsCfg(ObsGroup):
    """The object's state and its goal.

    A camera task replaces this half with learned features, so it is kept separate
    from :class:`ReorientRobotObsCfg` rather than nulled out per task.
    """

    object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )
    object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("object")})
    goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    goal_quat_diff = ObsTerm(
        func=mdp.goal_quat_diff,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
    )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ReorientFullStateObsCfg(ReorientRobotObsCfg, ReorientObjectObsCfg):
    """The full state, before the action terms."""


@configclass
class ObservationsCfg:
    """Full-state observation in Direct order."""

    @configclass
    class PolicyCfg(ReorientFullStateObsCfg):
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms tuned to the Direct task's scales."""

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
    """Episode budget, and the Direct task's fall condition."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={"threshold": 0.24, "command_name": "object_pose", "object_cfg": SceneEntityCfg("object")},
    )


@configclass
class ReorientManagerEnvBaseCfg(ManagerBasedRLEnvCfg):
    """Manager-based reorientation with Direct-compatible semantics.

    A hand subclass supplies :attr:`fingertip_body_names`, :attr:`actuated_joint_names`,
    its ``scene``, its ``events`` and its ``decimation``.
    """

    fingertip_body_names: list[str] = MISSING
    """Fingertip bodies the observation terms read."""
    actuated_joint_names: list[str] = MISSING
    """Joints the action term drives."""
    goal_orientation_threshold: float = MISSING
    """Orientation error below which a goal counts as reached [rad]."""
    goal_marker_cfg: VisualizationMarkersCfg = MISSING
    """Marker spawned at the goal pose."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    def __post_init__(self):
        for group in vars(self.observations).values():
            for name in ("fingertip_pose", "fingertip_vel"):
                term = getattr(group, name, None)
                if term is not None:
                    # a fresh entity per term: the manager fills in body_ids on the
                    # instance it resolves, so a shared one goes inconsistent
                    term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=self.fingertip_body_names)
        self.actions.joint_pos.joint_names = self.actuated_joint_names
        self.commands.object_pose.orientation_success_threshold = self.goal_orientation_threshold
        self.commands.object_pose.goal_pose_visualizer_cfg = self.goal_marker_cfg
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0))
