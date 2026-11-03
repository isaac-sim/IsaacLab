# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI entry-point factories exposing gearbox play as a domain preset."""

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from isaaclab_tasks.contrib.microduck.assets import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG,
    MICRODUCK_BACKLASH_CFG,
    MICRODUCK_ROLLERS_BACKLASH_CFG,
)

from .backlash import apply_backlash_preset, backlash_agent_preset
from .ballkick.agents.rsl_rl_ppo_cfg import MicroDuckBallKickPPORunnerCfg
from .ballkick.ballkick_env_cfg import MicroDuckBallKickFlatEnvCfg
from .groundpick.agents.rsl_rl_ppo_cfg import MicroDuckGroundPickPPORunnerCfg
from .groundpick.groundpick_env_cfg import MicroDuckGroundPickFlatEnvCfg
from .pollen_rough_env_cfg import (
    MicroDuckGroundPickRoughEnvCfg,
    MicroDuckSitStandRoughEnvCfg,
    MicroDuckStandUpRoughEnvCfg,
    MicroDuckVelocityRoughPresetEnvCfg,
)
from .rollercrouch.agents.rsl_rl_ppo_cfg import MicroDuckRollerCrouchPPORunnerCfg
from .rollercrouch.rollercrouch_env_cfg import MicroDuckRollerCrouchFlatEnvCfg
from .rollerslope.agents.rsl_rl_ppo_cfg import MicroDuckRollerSlopePPORunnerCfg
from .rollerslope.rollerslope_env_cfg import MicroDuckRollerSlopeFlatEnvCfg
from .sitstand.agents.rsl_rl_ppo_cfg import MicroDuckSitStandPPORunnerCfg
from .sitstand.sitstand_env_cfg import MicroDuckSitStandFlatEnvCfg
from .standup.agents.rsl_rl_ppo_cfg import MicroDuckStandUpPPORunnerCfg
from .standup.standup_env_cfg import MicroDuckStandUpFlatEnvCfg
from .terrain import MicroDuckTerrainImporter, configure_foot_terrain_sensing
from .velocity.agents.rsl_rl_ppo_cfg import (
    MicroDuckPPORunnerCfg,
    MicroDuckRollersPPORunnerCfg,
    MicroDuckSwizzlePPORunnerCfg,
)
from .velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from .velocity.rollers_env_cfg import MicroDuckVelocityRollersFlatEnvCfg
from .velocity.swizzle_env_cfg import MicroDuckVelocitySwizzleEnvCfg


def velocity_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct Velocity-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckVelocityFlatEnvCfg(), MICRODUCK_BACKLASH_CFG)


def velocity_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the Velocity-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckPPORunnerCfg()
    return backlash_agent_preset(cfg)


def velocity_rough_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct Velocity-Rough-MicroDuck with its optional gearbox-play preset."""
    cfg = MicroDuckVelocityRoughPresetEnvCfg()
    cfg.scene.terrain.class_type = MicroDuckTerrainImporter
    configure_foot_terrain_sensing(cfg)
    return apply_backlash_preset(cfg, MICRODUCK_BACKLASH_CFG)


def velocity_rough_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the Velocity-Rough-MicroDuck training experiments distinct."""
    cfg = MicroDuckPPORunnerCfg()
    return backlash_agent_preset(cfg)


def velocity_rollers_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct Velocity-Flat-MicroDuck-Rollers with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckVelocityRollersFlatEnvCfg(), MICRODUCK_ROLLERS_BACKLASH_CFG)


def velocity_rollers_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the Velocity-Flat-MicroDuck-Rollers training experiments distinct."""
    cfg = MicroDuckRollersPPORunnerCfg()
    return backlash_agent_preset(cfg)


def velocity_swizzle_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct Velocity-Swizzle-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckVelocitySwizzleEnvCfg(), MICRODUCK_ROLLERS_BACKLASH_CFG)


def velocity_swizzle_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the Velocity-Swizzle-MicroDuck training experiments distinct."""
    cfg = MicroDuckSwizzlePPORunnerCfg()
    return backlash_agent_preset(cfg)


def standup_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct StandUp-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckStandUpFlatEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def standup_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the StandUp-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckStandUpPPORunnerCfg()
    return backlash_agent_preset(cfg)


def standup_rough_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct StandUp-Rough-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckStandUpRoughEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def standup_rough_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the StandUp-Rough-MicroDuck training experiments distinct."""
    cfg = MicroDuckStandUpPPORunnerCfg()
    cfg.experiment_name += "_rough"
    return backlash_agent_preset(cfg)


def sitstand_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct SitStand-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckSitStandFlatEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def sitstand_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the SitStand-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckSitStandPPORunnerCfg()
    return backlash_agent_preset(cfg)


def sitstand_rough_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct SitStand-Rough-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckSitStandRoughEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def sitstand_rough_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the SitStand-Rough-MicroDuck training experiments distinct."""
    cfg = MicroDuckSitStandPPORunnerCfg()
    cfg.experiment_name += "_rough"
    return backlash_agent_preset(cfg)


def groundpick_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct GroundPick-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckGroundPickFlatEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def groundpick_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the GroundPick-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckGroundPickPPORunnerCfg()
    return backlash_agent_preset(cfg)


def groundpick_rough_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct GroundPick-Rough-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckGroundPickRoughEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def groundpick_rough_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the GroundPick-Rough-MicroDuck training experiments distinct."""
    cfg = MicroDuckGroundPickPPORunnerCfg()
    cfg.experiment_name += "_rough"
    return backlash_agent_preset(cfg)


def ballkick_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct BallKick-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckBallKickFlatEnvCfg(), MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)


def ballkick_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the BallKick-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckBallKickPPORunnerCfg()
    return backlash_agent_preset(cfg)


def rollercrouch_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct RollerCrouch-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckRollerCrouchFlatEnvCfg(), MICRODUCK_ROLLERS_BACKLASH_CFG)


def rollercrouch_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the RollerCrouch-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckRollerCrouchPPORunnerCfg()
    return backlash_agent_preset(cfg)


def rollerslope_flat_env_cfg() -> ManagerBasedRLEnvCfg:
    """Construct RollerSlope-Flat-MicroDuck with its optional gearbox-play preset."""
    return apply_backlash_preset(MicroDuckRollerSlopeFlatEnvCfg(), MICRODUCK_ROLLERS_BACKLASH_CFG)


def rollerslope_flat_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the RollerSlope-Flat-MicroDuck training experiments distinct."""
    cfg = MicroDuckRollerSlopePPORunnerCfg()
    return backlash_agent_preset(cfg)
