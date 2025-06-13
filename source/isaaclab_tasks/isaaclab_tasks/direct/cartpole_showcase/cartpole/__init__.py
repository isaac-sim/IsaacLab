# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

###########################
# Register Gym environments
###########################

###
# Observation space as Box
###

gym.register(
    id="Isaac-Cartpole-Showcase-Box-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:BoxBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:BoxDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:BoxMultiDiscreteEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DiscreteBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_discrete_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DiscreteDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_discrete_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DiscreteMultiDiscreteEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:MultiDiscreteBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:MultiDiscreteDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_multidiscrete_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:MultiDiscreteMultiDiscreteEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DictBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DictDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:DictMultiDiscreteEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:TupleBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:TupleDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:TupleMultiDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)
