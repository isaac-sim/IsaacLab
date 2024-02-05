import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Velocity-Flat-Anymal-C-v0",
#     entry_point="omni.isaac.orbit.envs:RLTaskEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.AnymalCFlatEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCFlatPPORunnerCfg,
#         "skrl_cfg_entry_point": "omni.isaac.orbit_tasks.locomotion.velocity.anymal_c.agents:skrl_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Android-v0",
#     entry_point="omni.isaac.orbit.envs:RLTaskEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.AndroidFlatEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AndroidFlatPPORunnerCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
# )
