"""Mobile Manipulator(MM) environment"""
  
import gymnasium as gym

from . import agents

gym.register(
    id="Cabinet-anubis-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AnubisCabinetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    }
)

gym.register(
    id="Cabinet-anubis-teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_cabinet_env_cfg:AnubisCabinetEnvCfg",
         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    }
)