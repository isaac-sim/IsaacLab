# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

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


# Retired per-data-type camera task IDs. Each carries a ``deprecated`` kwarg
# whose ``alias`` field names the consolidated task with the equivalent
# ``presets=<name>`` -- parse_cfg.load_cfg_from_registry consults that kwarg
# and emits a FutureWarning when the retired ID's env cfg is loaded.
# The ``env_cfg_entry_point`` keeps pointing at the historical per-variant
# cfg so the retired ID stays bit-for-bit identical to develop.

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleRGBCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=rgb"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Albedo-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleAlbedoCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=albedo"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleSimpleShadingConstantCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_constant_diffuse"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleSimpleShadingDiffuseCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_diffuse_mdl"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleSimpleShadingFullCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_full_mdl"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleDepthCameraEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0 presets=depth"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Presets-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_presets_env:CartpoleCameraPresetsEnv",
    disable_env_checker=True,
    kwargs={
        # The retired catch-all points at the same consolidated cfg as the
        # canonical task above; the Hydra resolver applies any user-CLI
        # presets the user passes alongside this ID, matching develop.
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        "deprecated": {"alias": "--task=Isaac-Cartpole-Camera-Direct-v0"},
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)
