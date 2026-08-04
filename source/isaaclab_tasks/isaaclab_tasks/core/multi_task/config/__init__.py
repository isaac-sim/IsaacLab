# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory task registration.

A single gym environment is registered.  The robot (and every robot-specific
override) is chosen declaratively through the preset system -- e.g.::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task IsaacContrib-Factory-Franka presets=franka,peg_insert_4mm

See :mod:`..factory_presets` for the robot registry and how to add new robots.
"""

import gymnasium as gym

from . import agents

# Imported for its side effect: the module sets each robot-specific field on the
# ``..factory_presets`` classes at import time, and ``presets=franka`` cannot resolve until
# it has run. Keeping it here rather than in ``factory_presets`` leaves that module free of
# any reference to a particular robot.
from . import franka_factory_cfg as _franka_factory_cfg  # noqa: F401

gym.register(
    id="IsaacContrib-Factory-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.multi_task.factory_env_cfg:FactoryBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)
