import gymnasium as gym

##
# Register Gym environments.
##


# ----------------------------------- Position Control ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-PositionControl-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_pos:UnitreeGo2FlatEnvCfg_pos",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_pos:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity_Spot-PositionControl-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_pos:UnitreeGo2RoughEnvCfg_pos",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_pos:UnitreeGo2RoughPPORunnerCfg",
    },
)

# ----------------------------------- Position Control RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-PositionControl_RNN-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg_pos_rnn:UnitreeGo2FlatEnvCfg_pos_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg_pos_rnn:UnitreeGo2FlatPPORunnerCfg_rnn",
    },
)

gym.register(
    id="Go2-Velocity_Spot-PositionControl_RNN-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg_pos_rnn:UnitreeGo2RoughEnvCfg_pos_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg_pos_rnn:UnitreeGo2RoughPPORunnerCfg_rnn",
    },
)

# ----------------------------------- All Position Control RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-AllControl_RNN-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_all_rnn:UnitreeGo2FlatEnvCfg_all_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_all_rnn:UnitreeGo2FlatPPORunnerCfg_all_rnn",
    },
)

gym.register(
    id="Go2-Velocity_Spot-AllControl_RNN-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_all_rnn:UnitreeGo2RoughEnvCfg_all_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_all_rnn:UnitreeGo2RoughPPORunnerCfg_all_rnn",
    },
)

# ----------------------------------- Effort Control ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-EffortControl-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_eff:UnitreeGo2FlatEnvCfg_eff",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_eff:UnitreeGo2FlatPPORunnerCfg_eff",
    },
)

gym.register(
    id="Go2-Velocity_Spot-EffortControl-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_eff:UnitreeGo2RoughEnvCfg_eff",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg_eff:UnitreeGo2RoughPPORunnerCfg_eff",
    },
)

# ----------------------------------- Effort Control RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-EffortControl_RNN-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eff.flat_env_cfg_eff_rnn:UnitreeGo2FlatEnvCfg_eff_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.eff.agents.rsl_rl_ppo_cfg_eff_rnn:UnitreeGo2FlatPPORunnerCfg_eff_rnn",
    },
)

gym.register(
    id="Go2-Velocity_Spot-EffortControl_RNN-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eff.rough_env_cfg_eff_rnn:UnitreeGo2RoughEnvCfg_eff_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.eff.agents.rsl_rl_ppo_cfg_eff_rnn:UnitreeGo2RoughPPORunnerCfg_eff_rnn",
    },
)

# ----------------------------------- Gains Control RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity_Spot-GainsControl_RNN-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gains.flat_env_cfg_gains_rnn:UnitreeGo2FlatEnvCfg_gains_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.gains.agents.rsl_rl_ppo_cfg_gains_rnn:UnitreeGo2FlatPPORunnerCfg_gains_rnn",
    },
)

gym.register(
    id="Go2-Velocity_Spot-GainsControl_RNN-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gains.rough_env_cfg_gains_rnn:UnitreeGo2RoughEnvCfg_gains_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.gains.agents.rsl_rl_ppo_cfg_gains_rnn:UnitreeGo2RoughPPORunnerCfg_gains_rnn",
    },
)

# ----------------------------------- Position Control (Velocity Height) ----------------------------------- #

gym.register(
    id="Go2-Velocity_Height-PositionControl-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_height.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_height.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity_Height-PositionControl-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_height.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_height.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

# ----------------------------------- Position Control (Velocity Height) RNN ----------------------------------- #

gym.register(
    id="Go2-Velocity_Height-PositionControl_RNN-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_height.flat_env_cfg_pos_rnn:UnitreeGo2FlatEnvCfg_pos_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_height.agents.rsl_rl_ppo_cfg_pos_rnn:UnitreeGo2FlatPPORunnerCfg_pos_rnn",
    },
)

gym.register(
    id="Go2-Velocity_Height-PositionControl_RNN-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_height.rough_env_cfg_pos_rnn:UnitreeGo2RoughEnvCfg_pos_rnn",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_height.agents.rsl_rl_ppo_cfg_pos_rnn:UnitreeGo2RoughPPORunnerCfg_pos_rnn",
    },
)



