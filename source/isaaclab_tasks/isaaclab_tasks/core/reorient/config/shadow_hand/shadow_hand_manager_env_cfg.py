# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the state-based Shadow Hand reorientation task."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
    ShadowHandManagerEventPresetCfg,
    ShadowHandRobotCfg,
)
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import (
    ReorientFullStateObsCfg,
    ReorientManagerEnvBaseCfg,
    ReorientSceneBaseCfg,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES

##
# Default: full-state actor.
##


@configclass
class ShadowHandManagerSceneCfg(ReorientSceneBaseCfg):
    """The shared scene, holding the Shadow hand and its in-hand cube."""

    robot: PresetCfg = ShadowHandRobotCfg()
    object: RigidObjectCfg = CUBE_CFG


@configclass
class ShadowHandManagerEnvCfg(ReorientManagerEnvBaseCfg):
    """Manager-based state Shadow Hand task with Direct-compatible semantics."""

    fingertip_body_names = SHADOW_FINGERTIP_BODY_NAMES
    actuated_joint_names = SHADOW_ACTUATED_JOINT_NAMES
    goal_orientation_threshold = 0.1
    goal_marker_cfg = GOAL_OBJECT_CFG
    decimation = 2

    scene: ShadowHandManagerSceneCfg = ShadowHandManagerSceneCfg()
    # ``presets=randomized`` adds the domain-randomization terms
    events: ShadowHandManagerEventPresetCfg = ShadowHandManagerEventPresetCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = PhysicsCfg()


##
# ``presets=asymmetric``: reduced actor, privileged critic.
##


@configclass
class ShadowHandAsymmetricSceneCfg(ShadowHandManagerSceneCfg):
    """Scene with the fingertip joint-wrench sensing the privileged critic reads."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ShadowHandAsymmetricObservationsCfg:
    """A reduced actor observation paired with a privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Only what a physical hand can measure."""

        fingertip_pos = ObsTerm(
            func=mdp.fingertip_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES)},
        )
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
            },
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ReorientFullStateObsCfg):
        """The full state, plus contact sensing only the simulator can supply."""

        fingertip_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=10.0,
            params={"sensor_cfg": SceneEntityCfg("joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ShadowHandAsymmetricEnvCfg(ShadowHandManagerEnvCfg):
    """The state task with a reduced actor and a privileged critic."""

    scene: ShadowHandAsymmetricSceneCfg = ShadowHandAsymmetricSceneCfg()
    observations: ShadowHandAsymmetricObservationsCfg = ShadowHandAsymmetricObservationsCfg()


@configclass
class ShadowHandManagerEnvPresetCfg(PresetCfg):
    """``presets=asymmetric`` swaps in the reduced actor and its privileged critic."""

    asymmetric = ShadowHandAsymmetricEnvCfg()
    default = ShadowHandManagerEnvCfg()
