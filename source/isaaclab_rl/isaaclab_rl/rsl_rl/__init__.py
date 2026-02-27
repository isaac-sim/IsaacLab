# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an environment for RSL-RL library.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .exporter import export_policy_as_jit, export_policy_as_onnx
    from .rnd_cfg import RslRlRndCfg
    from .symmetry_cfg import RslRlSymmetryCfg
    from .vecenv_wrapper import RslRlVecEnvWrapper
    from .distillation_cfg import *  # noqa: F403
    from .rl_cfg import *  # noqa: F403

from isaaclab.utils.module import lazy_export

lazy_export(
    ("exporter", ["export_policy_as_jit", "export_policy_as_onnx"]),
    ("rnd_cfg", "RslRlRndCfg"),
    ("symmetry_cfg", "RslRlSymmetryCfg"),
    ("vecenv_wrapper", "RslRlVecEnvWrapper"),
    submodules=["distillation_cfg", "rl_cfg"],
)
