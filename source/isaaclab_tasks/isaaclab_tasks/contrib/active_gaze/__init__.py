# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active vision with a foveated camera: the policy moves the camera's gaze.

The camera is :class:`isaaclab_fovi.sensors.FoveatedCamera`, from the optional ``fovi`` extra
(``uv sync --extra fovi``). The task is registered only when that extra is installed.
"""

from importlib.util import find_spec

import gymnasium as gym

ISAACLAB_FOVI_AVAILABLE = find_spec("isaaclab_fovi") is not None and find_spec("fovi") is not None
"""Whether the ``fovi`` extra is installed: the foveated camera package and the fovi library it builds on."""

if ISAACLAB_FOVI_AVAILABLE:
    gym.register(
        id="IsaacContrib-Cartpole-Camera-Fovi",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_camera_fovi_env_cfg:CartpoleCameraFoviEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:CartpoleCameraFoviPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
