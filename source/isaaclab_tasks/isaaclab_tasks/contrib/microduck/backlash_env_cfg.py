# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat and rough walking with output-side encoders and passive gearbox play."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets import MICRODUCK_BACKLASH_CFG

from . import mdp
from .flat_env_cfg import MICRODUCK_HEAD_JOINT_NAMES, MICRODUCK_JOINT_NAMES, MicroDuckVelocityFlatEnvCfg
from .rough_env_cfg import MicroDuckVelocityRoughEnvCfg


def _configure_backlash(cfg: MicroDuckVelocityFlatEnvCfg) -> None:
    """Keep the walking recipe while measuring the joints through gearbox play."""
    cfg.scene.robot = MICRODUCK_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    play_cfg = SceneEntityCfg(
        "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES], preserve_order=True
    )
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.joint_pos.params["backlash_cfg"] = play_cfg
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.joint_vel.func = mdp.joint_vel_rel_backlash
        group.joint_vel.params["backlash_cfg"] = play_cfg
    head_play_cfg = SceneEntityCfg(
        "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_HEAD_JOINT_NAMES], preserve_order=True
    )
    cfg.rewards.head_pose_tracking.params["backlash_cfg"] = head_play_cfg
    cfg.rewards.head_pose_bias.params["backlash_cfg"] = head_play_cfg
    # Play hinges normally rest against their stops; only servo soft limits incur a penalty.
    cfg.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True
    )


@configclass
class MicroDuckVelocityBacklashFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Flat velocity walking with 14 servo joints and 14 passive play hinges."""

    def __post_init__(self):
        super().__post_init__()
        _configure_backlash(self)
        # Gearbox limit contacts exhaust the flat task's 10-iteration budget at training scale.
        self.sim.physics.solver_cfg.iterations = 100


@configclass
class MicroDuckVelocityBacklashRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """Rough velocity walking with 14 servo joints and 14 passive play hinges."""

    def __post_init__(self):
        super().__post_init__()
        _configure_backlash(self)
