omni.isaac.orbit_envs.utils.wrappers
====================================

Wrappers allow you to modify the behavior of an environment without modifying the environment itself.
This is useful for modifying the observation space, action space, or reward function. Additionally,
they can be used to cast a given environment into the respective environment class definition used by
different learning frameworks. This operation may include handling of asymmetric actor-critic observations,
casting the data between different backends such `numpy` and `pytorch`, or organizing the returned data
into the expected data structure by the learning framework.

All wrappers derive from the ``gym.Wrapper``` class. Using a wrapper is as simple as passing the initialized
environment instance to the wrapper constructor. For instance, to wrap an environment in the
`Stable-Baselines3`_ wrapper, you can do the following:

.. code-block:: python

   from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper

   env = Sb3VecEnvWrapper(env)


.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _skrl: https://github.com/Toni-SM/skrl
.. _Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3


RL-Games Wrapper
----------------

.. automodule:: omni.isaac.orbit_envs.utils.wrappers.rl_games
   :members:
   :show-inheritance:

RSL-RL Wrapper
--------------

.. automodule:: omni.isaac.orbit_envs.utils.wrappers.rsl_rl
   :members:
   :show-inheritance:

SKRL Wrapper
------------

.. automodule:: omni.isaac.orbit_envs.utils.wrappers.skrl
   :members:
   :show-inheritance:

Stable-Baselines3 Wrapper
-------------------------

.. automodule:: omni.isaac.orbit_envs.utils.wrappers.sb3
   :members:
   :show-inheritance:
