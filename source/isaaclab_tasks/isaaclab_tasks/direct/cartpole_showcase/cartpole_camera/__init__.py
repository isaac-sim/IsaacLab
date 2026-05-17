# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment with camera.
"""

import gymnasium as gym

from isaaclab_tasks.utils import deprecated_task_alias

from . import agents


def _lazy_cfg(class_name: str, module: str = "cartpole_camera_env_cfg"):
    """Return a zero-arg callable that lazily imports the named cfg class.

    Deferring the cfg-class import to ``gym.make()`` time avoids pulling
    ``isaaclab.scene`` (and other heavy modules) into ``sys.modules`` *before*
    an ``AppLauncher`` cycle clears and restores them, which would leave
    ``isaaclab.scene`` in ``sys.modules`` but no longer bound as an attribute
    of the ``isaaclab`` package -- breaking string
    ``monkeypatch.setattr("isaaclab.scene...")`` calls in unrelated tests.
    """

    def factory():
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{module}"), class_name)()

    return factory

###########################
# Register Gym environments
###########################

# Canonical camera-based showcase task -- selects the (observation, action)
# space combination via the preset CLI (#5587). The default skrl yaml matches
# the canonical ``box_box`` shape; for other variants pass the matching
# ``--agent skrl_<obs>_<action>_cfg_entry_point``. Retired per-shape IDs below
# remain registered for one release as deprecation shims pointing at this task.
gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.cartpole_camera_env_cfg:CartpoleCameraShowcasePresetsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
        "skrl_box_box_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
        "skrl_box_discrete_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
        "skrl_box_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_box_multidiscrete_ppo_cfg.yaml",
        "skrl_dict_box_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
        "skrl_dict_discrete_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
        "skrl_dict_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_dict_multidiscrete_ppo_cfg.yaml",
        "skrl_tuple_box_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
        "skrl_tuple_discrete_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
        "skrl_tuple_multidiscrete_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)

# -- Deprecated aliases --------------------------------------------------------

###
# Observation space as Box
###

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
            new_command=["--task=Isaac-Cartpole-Camera-Showcase-Direct-v0", "presets=box_box"],
            cfg_factory=_lazy_cfg("BoxBoxEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_box_discrete_cfg_entry_point",
                "presets=box_discrete",
            ],
            cfg_factory=_lazy_cfg("BoxDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_box_multidiscrete_cfg_entry_point",
                "presets=box_multidiscrete",
            ],
            cfg_factory=_lazy_cfg("BoxMultiDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Dict
###

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_dict_box_cfg_entry_point",
                "presets=dict_box",
            ],
            cfg_factory=_lazy_cfg("DictBoxEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_dict_discrete_cfg_entry_point",
                "presets=dict_discrete",
            ],
            cfg_factory=_lazy_cfg("DictDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_dict_multidiscrete_cfg_entry_point",
                "presets=dict_multidiscrete",
            ],
            cfg_factory=_lazy_cfg("DictMultiDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_multidiscrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Tuple
###

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_tuple_box_cfg_entry_point",
                "presets=tuple_box",
            ],
            cfg_factory=_lazy_cfg("TupleBoxEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_tuple_discrete_cfg_entry_point",
                "presets=tuple_discrete",
            ],
            cfg_factory=_lazy_cfg("TupleDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-Showcase-Direct-v0",
                "--agent=skrl_tuple_multidiscrete_cfg_entry_point",
                "presets=tuple_multidiscrete",
            ],
            cfg_factory=_lazy_cfg("TupleMultiDiscreteEnvCfg"),
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)
