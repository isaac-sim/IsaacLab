# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned bimanual cable-routing environments."""

import gymnasium as gym

from . import agents

_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:CableRoutingPPORunnerCfg"


def _register_environment(task_id: str, env_cfg: str) -> None:
    """Register one cable-routing configuration with the shared RSL-RL agent."""
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cable_routing_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
        },
    )


_register_environment("IsaacContrib-CableRouting-YAM", "CableRoutingEnvCfg")
_register_environment("IsaacContrib-CableRouting-YAM-Peg0-CCW", "CableRoutingPeg0CCWEnvCfg")
_register_environment("IsaacContrib-CableRouting-YAM-Peg1-CW", "CableRoutingPeg1CWEnvCfg")
_register_environment("IsaacContrib-CableRouting-YAM-Tier1-Pegs", "CableRoutingTier1PegsEnvCfg")
_register_environment("IsaacContrib-CableRouting-YAM-SevenGoals", "CableRoutingSevenGoalsEnvCfg")

gym.register(
    id="IsaacContrib-CableRouting-YAM-AVP-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cable_routing_avp_env_cfg:CableRoutingAVPTeleopEnvCfg",
    },
)
