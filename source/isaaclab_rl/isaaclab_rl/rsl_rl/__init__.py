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

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "distillation_cfg": [
            "RslRlDistillationRunnerCfg",
            "RslRlDistillationAlgorithmCfg",
            "RslRlDistillationStudentTeacherCfg",
            "RslRlDistillationStudentTeacherRecurrentCfg",
        ],
        "exporter": ["export_policy_as_jit", "export_policy_as_onnx"],
        "rl_cfg": [
            "RslRlBaseRunnerCfg",
            "RslRlOnPolicyRunnerCfg",
            "RslRlPpoActorCriticCfg",
            "RslRlPpoActorCriticRecurrentCfg",
            "RslRlPpoAlgorithmCfg",
        ],
        "rnd_cfg": ["RslRlRndCfg"],
        "symmetry_cfg": ["RslRlSymmetryCfg"],
        "vecenv_wrapper": ["RslRlVecEnvWrapper"],
    },
)
