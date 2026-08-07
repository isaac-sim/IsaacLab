# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the opt-in custom coupling example.

Registration lives here rather than in the package ``__init__`` so that importing the
example's library modules, including from the deprecated
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`, does not register the
task.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Lift-Soft-Franka-Custom-Coupling",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__package__}.franka_soft_env_cfg:FrankaSoftCustomCouplingEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg:FrankaDeformablePPORunnerCfg"
        ),
    },
)
