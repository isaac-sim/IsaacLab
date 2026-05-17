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


def _lazy_cfg(class_name: str, module: str = "cartpole_camera_env_cfg"):
    """Return a zero-arg callable that lazily imports the named cfg class.

    Deferring the cfg-class import to ``gym.make()`` time keeps the
    eager-import surface of this ``__init__.py`` minimal. Importing the cfg
    classes at module top pulls ``isaaclab.scene`` and friends into
    ``sys.modules`` *before* an ``AppLauncher`` cycle clears and restores
    them, which leaves ``isaaclab.scene`` in ``sys.modules`` but no longer
    bound as an attribute of the ``isaaclab`` package -- breaking string
    ``monkeypatch.setattr("isaaclab.scene...")`` calls in unrelated tests.
    """

    def factory():
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{module}"), class_name)()

    return factory


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
# equivalent presets=<name>, then returns the historical per-variant cfg via
# cfg_factory so the retired ID stays bit-for-bit identical to develop. The
# OLD subclasses ``Cartpole{RGB,Depth,Albedo,SimpleShading*}CameraEnvCfg``
# in ``cartpole_camera_env_cfg.py`` are kept for one release alongside the
# consolidated ``CartpoleCameraPresetsEnvCfg`` and will be removed together
# with these retired task IDs.

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-RGB-Camera-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0", "presets=rgb"],
            cfg_factory=_lazy_cfg("CartpoleRGBCameraEnvCfg"),
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
            cfg_factory=_lazy_cfg("CartpoleAlbedoCameraEnvCfg"),
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
            cfg_factory=_lazy_cfg("CartpoleSimpleShadingConstantCameraEnvCfg"),
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
            cfg_factory=_lazy_cfg("CartpoleSimpleShadingDiffuseCameraEnvCfg"),
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
            cfg_factory=_lazy_cfg("CartpoleSimpleShadingFullCameraEnvCfg"),
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
            cfg_factory=_lazy_cfg("CartpoleDepthCameraEnvCfg"),
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
        # The retired catch-all returns the bare consolidated cfg unresolved
        # so the downstream Hydra resolver can apply any user-CLI presets the
        # user passes alongside this task ID, matching develop's behavior.
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Presets-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Direct-v0"],
            cfg_factory=_lazy_cfg("CartpoleCameraPresetsEnvCfg", module="cartpole_camera_presets_env_cfg"),
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)
