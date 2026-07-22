# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the OpenAI Shadow Hand reorientation variants (FF and LSTM)."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    SHADOW_ACTUATED_JOINT_NAMES,
    SHADOW_FINGERTIP_BODY_NAMES,
    NewtonEventCfg,
    PhysicsCfg,
    PhysxEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    FullStateWithoutActionCfg,
    _ShadowHandManagerSceneCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.utils import PresetCfg


@configclass
class OpenAICommandsCfg:
    """OpenAI goal command with its wider success tolerance."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.4,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class OpenAIActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=0.3,
        rescale_to_limits=True,
        noise_model=OPENAI_ACTION_NOISE_CFG,
    )


@configclass
class OpenAIObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.OpenAIPolicyObservation,
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
    class CriticCfg(FullStateWithoutActionCfg):
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
class ShadowHandOpenAIManagerSceneCfg(PresetCfg):
    """Backend-specific OpenAI scene alternatives."""

    @configclass
    class SceneCfg(_ShadowHandManagerSceneCfg):
        """Shadow Hand scene with fingertip joint-wrench sensing."""

        joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")

    physx = SceneCfg(clone_in_fabric=True)
    newton_mjwarp = SceneCfg(clone_in_fabric=False)
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


_OPENAI_RESET_PARAMS = {
    "position_noise": 0.01,
    "joint_position_noise": 0.2,
    "joint_velocity_noise": 0.0,
    "action_name": "joint_pos",
}


@configclass
class OpenAIPhysxEventCfg(PhysxEventCfg):
    """PhysX OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAINewtonEventCfg(NewtonEventCfg):
    """Newton OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAIEventCfg(PresetCfg):
    """Backend-specific OpenAI event alternatives."""

    physx = OpenAIPhysxEventCfg()
    newton_mjwarp = OpenAINewtonEventCfg()
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class OpenAIRewardsCfg:
    """Direct-compatible OpenAI reward and success accounting."""

    reorient = RewTerm(
        func=mdp.ReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": -10.0,
            "rotation_scale": 1.0,
            "rotation_epsilon": 0.1,
            "action_penalty_scale": -0.0002,
            "success_tolerance": 0.4,
            "success_bonus": 250.0,
            "fall_distance": 0.24,
            "fall_penalty": -50.0,
            "averaging_factor": 0.1,
            "success_count_threshold": 1,
            "action_name": "joint_pos",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class OpenAITerminationsCfg:
    """Direct-compatible OpenAI termination conditions."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(
        func=mdp.ReorientTimeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "reward_name": "reorient",
            "success_tolerance": 0.4,
            "max_successes": 50,
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
    observations: OpenAIObservationsCfg = OpenAIObservationsCfg()
    actions: OpenAIActionsCfg = OpenAIActionsCfg()
    commands: OpenAICommandsCfg = OpenAICommandsCfg()
    rewards: OpenAIRewardsCfg = OpenAIRewardsCfg()
    terminations: OpenAITerminationsCfg = OpenAITerminationsCfg()
    events: OpenAIEventCfg = OpenAIEventCfg()

    def __post_init__(self):
        self.decimation = 3
        self.episode_length_s = 8.0
        # simulation — mirrors the Direct cfg (guarded by the value-parity test)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        self.sim.physics = PhysicsCfg()
        self.viewer.eye = (2.0, 2.0, 2.0)
