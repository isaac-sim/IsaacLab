# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: F401
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401
from skrl.utils.model_instantiators.torch import Shape  # noqa: F401

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

"""
Configuration Parser.
"""


def process_skrl_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(env: ManagerBasedRLEnv):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the :class:`ManagerBasedRLEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.

    Raises:
        ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
        raise ValueError(
            f"The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type: {type(env)}"
        )
    # wrap and return the environment
    return wrap_env(env, wrapper="isaaclab")
