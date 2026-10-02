# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tasks exercising the Newton step program: in-program controllers, Newton actuators, and Newton sensors.

They pair with existing tasks to compare work run inside the captured Newton step against host-side equivalents.
"""

import gymnasium as gym

from isaaclab_tasks.core.reach.config.franka import agents as franka_reach_agents
from isaaclab_tasks.core.velocity.config.anymal_d import agents as anymal_d_agents

gym.register(
    id="IsaacContrib-StepProgram-Reach-Franka-NewtonOSC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_newton_osc_env_cfg:FrankaReachNewtonOSCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_reach_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-StepProgram-Velocity-Flat-AnymalD-Sensors",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_sensors_env_cfg:AnymalDSensorsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{anymal_d_agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
    },
)

for name, cfg in (("DCMotor", "AnymalDDCMotorEnvCfg"), ("MixedActuators", "AnymalDMixedActuatorsEnvCfg")):
    gym.register(
        id=f"IsaacContrib-StepProgram-Velocity-Flat-AnymalD-{name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_actuators_env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{anymal_d_agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        },
    )
