# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 whole-body motion tracking."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Motion-Tracking-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1MotionTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionTrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

for representation, env_class, agent_class in (
    ("GenCoord", "G1ModelBasedEnvCfg", "TrackingModelBasedCfg"),
    ("MaxCoord", "G1MaxCoordModelBasedEnvCfg", "MaxCoordModelBasedCfg"),
):
    agent_entries = {"model_based_cfg_entry_point": f"{__name__}.agents.model_based_cfg:{agent_class}"}
    if representation == "GenCoord":
        agent_entries.update(
            paper_model_based_cfg_entry_point=f"{__name__}.agents.model_based_cfg:PaperTrackingModelBasedCfg",
            paper_no_critic_cfg_entry_point=f"{__name__}.agents.model_based_cfg:PaperNoCriticModelBasedCfg",
        )
    for prefix in ("Isaac", "IsaacContrib"):
        gym.register(
            id=f"{prefix}-Motion-Tracking-{representation}-G1",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.model_based_env_cfg:{env_class}",
                "default_agent": "model_based",
                **agent_entries,
            },
        )
