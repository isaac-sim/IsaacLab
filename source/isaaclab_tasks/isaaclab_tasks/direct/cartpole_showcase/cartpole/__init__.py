# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from isaaclab_tasks.utils import deprecated_task_alias

from . import agents

###########################
# Register Gym environments
###########################

# Canonical proprioceptive showcase task -- selects the (observation, action)
# space combination via the preset CLI (#5587). The default skrl yaml matches
# the canonical ``box_box`` shape; for other variants pass the matching
# ``--agent skrl_<obs>_<action>_cfg_entry_point``. Retired per-shape IDs below
# remain registered for one release as deprecation shims pointing at this task.
_SHOWCASE_CFG_PATH = f"{__name__}.cartpole_env_cfg:CartpoleShowcasePresetsEnvCfg"

gym.register(
    id="Isaac-Cartpole-Showcase-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _SHOWCASE_CFG_PATH,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
        "skrl_box_box_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
        "skrl_box_discrete_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
        "skrl_box_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_box_multidiscrete_ppo_cfg.yaml",
        "skrl_discrete_box_cfg_entry_point": f"{agents.__name__}:skrl_discrete_box_ppo_cfg.yaml",
        "skrl_discrete_discrete_cfg_entry_point": f"{agents.__name__}:skrl_discrete_discrete_ppo_cfg.yaml",
        "skrl_discrete_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_discrete_multidiscrete_ppo_cfg.yaml",
        "skrl_multidiscrete_box_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_box_ppo_cfg.yaml",
        "skrl_multidiscrete_discrete_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_discrete_ppo_cfg.yaml",
        "skrl_multidiscrete_multidiscrete_cfg_entry_point": (
            f"{agents.__name__}:skrl_multidiscrete_multidiscrete_ppo_cfg.yaml"
        ),
        "skrl_dict_box_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
        "skrl_dict_discrete_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
        "skrl_dict_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_dict_multidiscrete_ppo_cfg.yaml",
        "skrl_tuple_box_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
        "skrl_tuple_discrete_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
        "skrl_tuple_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)

# -- Deprecated aliases --------------------------------------------------------

###
# Observation space as Box
###

gym.register(
    id="Isaac-Cartpole-Showcase-Box-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Box-Box-Direct-v0",
            "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=box_box",
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0",
            "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=box_discrete --agent=skrl_box_discrete_cfg_entry_point",
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=box_multidiscrete"
                " --agent=skrl_box_multidiscrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Discrete
###

gym.register(
    id="Isaac-Cartpole-Showcase-Discrete-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Discrete-Box-Direct-v0",
            "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=discrete_box --agent=skrl_discrete_box_cfg_entry_point",
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_discrete_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=discrete_discrete"
                " --agent=skrl_discrete_discrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_discrete_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=discrete_multidiscrete"
                " --agent=skrl_discrete_multidiscrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_discrete_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as MultiDiscrete
###

gym.register(
    id="Isaac-Cartpole-Showcase-MultiDiscrete-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-MultiDiscrete-Box-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=multidiscrete_box"
                " --agent=skrl_multidiscrete_box_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=multidiscrete_discrete"
                " --agent=skrl_multidiscrete_discrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=multidiscrete_multidiscrete"
                " --agent=skrl_multidiscrete_multidiscrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Dict
###

gym.register(
    id="Isaac-Cartpole-Showcase-Dict-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Dict-Box-Direct-v0",
            "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=dict_box --agent=skrl_dict_box_cfg_entry_point",
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=dict_discrete"
                " --agent=skrl_dict_discrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=dict_multidiscrete"
                " --agent=skrl_dict_multidiscrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Tuple
###

gym.register(
    id="Isaac-Cartpole-Showcase-Tuple-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Tuple-Box-Direct-v0",
            "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=tuple_box --agent=skrl_tuple_box_cfg_entry_point",
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=tuple_discrete"
                " --agent=skrl_tuple_discrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0",
            (
                "--task=Isaac-Cartpole-Showcase-Direct-v0 presets=tuple_multidiscrete"
                " --agent=skrl_tuple_multidiscrete_cfg_entry_point"
            ),
            _SHOWCASE_CFG_PATH,
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)
