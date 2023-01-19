"""Wrapper to configure an :class:`IsaacEnv` instance to skrl environment

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env

    env = wrap_env(env, wrapper="isaac-orbit")

"""


# skrl
from skrl.envs.torch.wrappers import wrap_env

from omni.isaac.orbit_envs.isaac_env import IsaacEnv

__all__ = ["SkrlVecEnvWrapper"]


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(env: IsaacEnv):
    """Wraps around IsaacSim environment for skrl.

    This function wraps around the IsaacSim environment. Since the :class:`IsaacEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.envs.wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, IsaacEnv):
        raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
    # wrap and return the environment
    return wrap_env(env, wrapper="isaac-orbit")
