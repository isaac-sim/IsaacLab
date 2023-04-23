omni.isaac.orbit_envs.isaac_env
===============================

We use OpenAI Gym registry to register the environment and their default configuration file.
The default configuration file is passed to the argument "kwargs" in the Gym specification registry.
The string is parsed into respective configuration container which needs to be passed to the environment
class. This is done using the function :meth:`load_default_env_cfg` in the sub-module
:mod:`omni.isaac.orbit.utils.parse_cfg`.


.. attention::

    There is a slight abuse of kwargs since they are meant to be directly passed into the environment class.
    Instead, we remove the key :obj:`cfg_file` from the "kwargs" dictionary and the user needs to provide
    the kwarg argument :obj:`cfg` while creating the environment.


.. code-block:: python

   import gym
   import omni.isaac.orbit_envs
   from omni.isaac.orbit_envs.utils.parse_cfg import load_default_env_cfg

   task_name = "Isaac-Cartpole-v0"
   cfg = load_default_env_cfg(task_name)
   env = gym.make(task_name, cfg=cfg)


All environments must inherit from :class:`IsaacEnv` class which is defined in the sub-module
:mod:`omni.isaac.orbit_envs.isaac_env`.
The main methods that needs to be implemented by an inherited environment class:

* :meth:`_design_scene`: Design the template environment for cloning.
* :meth:`_reset_idx`: Environment reset function based on environment indices.
* :meth:`_step_impl`: Apply actions into simulation and compute MDP signals.
* :meth:`_get_observations`: Get observations from the environment.

The following attributes need to be set by the inherited class:

* :attr:`action_space`: The Space object corresponding to valid actions
* :attr:`observation_space`: The Space object corresponding to valid observations
* :attr:`reward_range`: A tuple corresponding to the min and max possible rewards. A default reward range set to [-inf, +inf] already exists.

The action and observation space correspond to single environment (and not vectorized).


Base Environment
----------------

.. automodule:: omni.isaac.orbit_envs.isaac_env
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
   :exclude-members: _configure_simulation_flags, _create_viewport_render_product, _last_obs_buf

Base Configuration
---------------------

.. automodule:: omni.isaac.orbit_envs.isaac_env_cfg
   :members:
   :undoc-members:
   :show-inheritance:
