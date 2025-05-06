import gymnasium as gym

##
# Register Gym environments.
##


# ----------------------------------- Position Control ----------------------------------- #

gym.register(
    id="Go2-Velocity-PositionControl-Flat-Async",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-PositionControl-Rough-Async",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)


# ----------------------------------- Position Control RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity-PositionControlRNN-Flat-Async",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg_rnn:Go2FlatEnvCfg_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg_rnn:Go2FlatPPORunnerCfg_rnn",
    },
)

gym.register(
    id="Go2-Velocity-PositionControlRNN-Rough-Async",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg_rnn:Go2RoughEnvCfg_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg_rnn:Go2RoughPPORunnerCfg_rnn",
    },
)


