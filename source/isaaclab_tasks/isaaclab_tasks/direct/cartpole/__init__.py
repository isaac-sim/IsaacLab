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
# CartpoleCameraPresetsEnvCfg. The nested tiled_camera attribute must be
# pinned alongside the root preset -- see _resolve_camera_variant.
_CAMERA_CFG_PATH = f"{__name__}.cartpole_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg"
_CAMERA_KWARGS = {
    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
}


def _resolve_camera_variant(preset_name: str):
    """Lazy 2-axis resolver: pin both the root cfg variant and the nested
    ``tiled_camera`` preset. Without this the nested PresetCfg's default
    (rgb) wins and a deprecated albedo task would load albedo's
    ``observation_space`` with rgb ``data_types``.

    Returns a zero-arg callable so the import of
    :class:`CartpoleCameraPresetsEnvCfg` is deferred to ``gym.make()``.
    """

    def call():
        from .cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg

        cfg = CartpoleCameraPresetsEnvCfg()
        result = getattr(cfg, preset_name)
        result.tiled_camera = getattr(result.tiled_camera, preset_name)
        return result

    return call


gym.register(
    id="Isaac-Cartpole-Camera-Presets-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_presets_env:CartpoleCameraPresetsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Camera-Presets-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0",
            _CAMERA_CFG_PATH,
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-RGB-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=rgb",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("rgb"),
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-Albedo-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Albedo-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=albedo",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("albedo"),
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_constant_diffuse",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("simple_shading_constant_diffuse"),
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_diffuse_mdl",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("simple_shading_diffuse_mdl"),
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_full_mdl",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("simple_shading_full_mdl"),
        ),
        **_CAMERA_KWARGS,
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Depth-Camera-Direct-v0",
            "--task=Isaac-Cartpole-Camera-Direct-v0 presets=depth",
            _CAMERA_CFG_PATH,
            cfg_factory=_resolve_camera_variant("depth"),
        ),
        **_CAMERA_KWARGS,
    },
)
