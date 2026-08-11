# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka two-bowl pour MPM task."""

import gymnasium as gym

from . import agents

_ENV = "isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv"
_CFG = "isaaclab_tasks.contrib.franka_pour.pour_env_cfg"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPourPPORunnerCfg"
_RESET_DATASET_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPourResetDatasetPPORunnerCfg"
_RESET_MIXTURE_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPourResetMixturePPORunnerCfg"


def _register(task_id: str, env_cfg: str, agent_cfg: str | None = None) -> None:
    """Register one Franka Pour task variant."""
    kwargs = {"env_cfg_entry_point": f"{_CFG}:{env_cfg}"}
    if agent_cfg is not None:
        kwargs["rsl_rl_cfg_entry_point"] = agent_cfg
    gym.register(id=task_id, entry_point=_ENV, disable_env_checker=True, kwargs=kwargs)

_register("Isaac-Pour-Franka-v0", "FrankaPourEnvCfg", _AGENT)
_register("Isaac-Pour-Franka-Play-v0", "FrankaPourEnvCfg_PLAY", _AGENT)
_register("Isaac-Pour-Franka-Teleop-v0", "FrankaPourEnvCfg_TELEOP")

for suffix, cfg_name in (
    ("", "FrankaPourEnvCfg_RESET_DATASET"),
    ("-Eval", "FrankaPourEnvCfg_RESET_DATASET_EVAL"),
    ("-Play", "FrankaPourEnvCfg_RESET_DATASET_PLAY"),
):
    _register(f"Isaac-Pour-Franka-Reset-Dataset{suffix}-v0", cfg_name, _RESET_DATASET_AGENT)

# Deprecated task IDs retained for one release. They select the reset-dataset implementation;
# Cartesian-IK checkpoints from the experimental task remain incompatible with its joint policy.
for suffix, cfg_name in (
    ("", "FrankaPourEnvCfg_RESET_MIXTURE"),
    ("-Eval", "FrankaPourEnvCfg_RESET_MIXTURE_EVAL"),
    ("-Play", "FrankaPourEnvCfg_RESET_MIXTURE_PLAY"),
):
    _register(f"Isaac-Pour-Franka-Reset-Mixture{suffix}-v0", cfg_name, _RESET_MIXTURE_AGENT)
