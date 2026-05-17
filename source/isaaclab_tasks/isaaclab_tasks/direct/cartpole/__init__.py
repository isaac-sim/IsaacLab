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

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_presets_env:CartpoleCameraPresetsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)


# Retired per-data-type camera task IDs. Each is registered as a deprecation
# shim that emits a DeprecationWarning naming the consolidated task with the
# equivalent presets=<name>, then loads the corresponding variant of
# CartpoleCameraPresetsEnvCfg. The shim's default cfg resolution walks every
# nested PresetCfg via ``resolve_presets``, so both the root variant and the
# nested ``tiled_camera`` preset are pinned to ``presets=<name>`` without
# per-call-site wiring.

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-RGB-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=rgb"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Albedo-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Albedo-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=albedo"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=simple_shading_constant_diffuse"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=simple_shading_diffuse_mdl"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=simple_shading_full_mdl"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Depth-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=depth"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Presets-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_presets_env:CartpoleCameraPresetsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Presets-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0"],
            consolidated_cfg_path=f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)
