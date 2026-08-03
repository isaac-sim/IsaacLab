# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment with camera.
"""

import gymnasium as gym

from . import agents

_SPACE_PRESETS = (
    "box_box",
    "box_discrete",
    "box_multidiscrete",
    "dict_box",
    "dict_discrete",
    "dict_multidiscrete",
    "tuple_box",
    "tuple_discrete",
    "tuple_multidiscrete",
)

###########################
# Register Gym environments
###########################

# Canonical camera-based showcase task -- selects the (observation, action)
# space combination via the preset CLI (#5587). The default skrl yaml matches
# the canonical ``box_box`` shape; for other variants pass the matching
# ``--agent skrl_<obs>_<action>_cfg_entry_point``.
gym.register(
    id="IsaacContrib-Cartpole-Camera-Showcase-Direct",
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
        "agent_preset_compatibility": {
            "skrl_cfg_entry_point": ("box_box",),
            **{f"skrl_{preset_name}_cfg_entry_point": (preset_name,) for preset_name in _SPACE_PRESETS},
        },
    },
)
