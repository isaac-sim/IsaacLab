# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the OpenAI Shadow Hand reorientation variants (FF and LSTM).

The observation, action-noise, and episode conventions follow OpenAI et al., "Learning
Dexterous In-Hand Manipulation" (https://arxiv.org/abs/1808.00177).
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    PhysicsCfg,
    ShadowHandManagerEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    FullStateObsCfg,
    ShadowHandManagerSceneCfg,
)

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES


@configclass
class CommandsCfg:
    """OpenAI goal command with its wider success tolerance."""

    object_pose = mdp.ReorientCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=0.4,
        make_quat_unique=False,
        fixed_marker_pos=(-0.2, -0.45, 0.68),
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=0.3,
        rescale_to_limits=True,
        noise_model=OPENAI_ACTION_NOISE_CFG,
    )


@configclass
class ObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.openai_policy_observation,
            params={
                "command_name": "object_pose",
                "action_name": "joint_pos",
                "noise_model": OPENAI_OBSERVATION_NOISE_CFG,
                "robot_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(FullStateObsCfg):
        fingertip_wrench = ObsTerm(
            func=mdp.fingertip_wrench,
            scale=10.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False
                )
            },
        )
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ShadowHandOpenAIManagerSceneCfg(ShadowHandManagerSceneCfg):
    """Shadow Hand scene with fingertip joint-wrench sensing."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RewardsCfg:
    """Shared reward terms tuned to the Direct OpenAI variant's scales."""

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
    object_away_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={"term_keys": "object_out_of_reach"},
    )


@configclass
class TerminationsCfg:
    """Shared terminations with the OpenAI streak cap and success-extended timer.

    The Direct variant reports both the streak cap and the elapsed-time limit as
    truncations, so both carry ``time_out=True``.
    """

    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success,
        time_out=True,
        params={"num_success": 50, "command_name": "object_pose"},
    )
    time_out = DoneTerm(
        func=mdp.reorient_timeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "success_tolerance": 0.4,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager counterpart shared by the OpenAI FF and LSTM variants.

    Standalone rather than a subclass of :class:`ShadowHandManagerEnvCfg`:
    every section differs from the state task, so this block is the complete
    recipe.
    """

    scene: ShadowHandOpenAIManagerSceneCfg = ShadowHandOpenAIManagerSceneCfg()
    decimation = 3
    episode_length_s = 8.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: ShadowHandManagerEventCfg = ShadowHandManagerEventCfg()

    enable_domain_randomization: bool = True
    """Apply the domain-randomization event terms.

    On by default: unlike the other reorientation tasks, the OpenAI Direct environment
    randomizes as well, so the two workflows only match with these enabled. ``__post_init__``
    reads it while building the configuration, before Hydra applies command-line overrides, so
    ``env.enable_domain_randomization=false`` has no effect -- set it on the configuration.
    Changing it requires retraining.
    """

    def __post_init__(self):
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0))
        if not self.enable_domain_randomization:
            self.events.robot_joint_stiffness_and_damping = None
            self.events.object_scale_mass = None
            self.events.reset_gravity = None
            self.events.robot_tendon_properties = None
            self.events.robot_physics_material = None
            self.events.object_physics_material = None
