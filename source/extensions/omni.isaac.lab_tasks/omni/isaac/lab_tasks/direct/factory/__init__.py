import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv, FactoryEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Factory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)
