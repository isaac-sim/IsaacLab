Running an RL environment
=========================

In this tutorial, we will learn how to run existing learning environments provided in the ``omni.isaac.orbit_envs``
extension. All the environments included in Orbit follow the ``gym.Env`` interface, which means that they can be used
with any reinforcement learning framework that supports OpenAI Gym. However, since the environments are implemented
in a vectorized fashion, they can only be used with frameworks that support vectorized environments.

Many common frameworks come with their own desired definitions of a vectorized environment and require the returned data
to follow their supported data types and data structures. For example, ``stable-baselines3`` uses ``numpy`` arrays, while
``rsl-rl``, ``rl-games``, or ``skrl`` use ``torch.Tensor``. We provide wrappers for these different frameworks, which can be found
in the ``omni.isaac.orbit_envs.utils.wrappers`` module.


The Code
~~~~~~~~

The tutorial corresponds to the ``zero_agent.py`` script in the ``orbit/source/standalone/environments`` directory.


.. literalinclude:: ../../../source/standalone/environments/zero_agent.py
   :language: python
   :emphasize-lines: 34-35,41-44,49-55
   :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

Using gym registry for environments
-----------------------------------

All environments are registered using the ``gym`` registry, which means that you can create an instance of
an environment by calling ``gym.make``. The environments are registered in the ``__init__.py`` file of the
``omni.isaac.orbit_envs`` extension with the following syntax:

.. code-block:: python

    # Cartpole environment
    gym.register(
        id="Isaac-Cartpole-v0",
        entry_point="omni.isaac.orbit_envs.classic.cartpole:CartpoleEnv",
        kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.classic.cartpole:cartpole_cfg.yaml"},
    )

The ``cfg_entry_point`` argument is used to load the default configuration for the environment. The default
configuration is loaded using the :meth:`omni.isaac.orbit_envs.utils.parse_cfg.load_default_env_cfg` function.
The configuration entry point can correspond to both a YAML file or a python configuration
class. The default configuration can be overridden by passing a custom configuration instance to the ``gym.make``
function as shown later in the tutorial.

To inform the ``gym`` registry with all the environments provided by the ``omni.isaac.orbit_envs`` extension,
we must import the module at the start of the script.

.. literalinclude:: ../../../source/standalone/environments/zero_agent.py
   :language: python
   :lines: 33-35
   :linenos:
   :lineno-start: 33

.. note::

    As a convention, we name all the environments in ``omni.isaac.orbit_envs`` extension with the prefix ``Isaac-``.
    For more complicated environments, we follow the pattern: ``Isaac-<TaskName>-<RobotName>-v<N>``,
    where `N` is used to specify different observations or action spaces within the same task definition. For example,
    for legged locomotion with ANYmal C, the environment is called ``Isaac-Velocity-Anymal-C-v0``.


In this tutorial, the task name is read from the command line. The task name is used to load the default configuration
as well as to create the environment instance. In addition, other parsed command line arguments such as the
number of environments, the simulation device, and whether to render, are used to override the default configuration.

.. literalinclude:: ../../../source/standalone/environments/zero_agent.py
   :language: python
   :lines: 42-45
   :linenos:
   :lineno-start: 42


Running the environment
-----------------------

Once creating the environment, the rest of the execution follows the standard resetting and stepping.

.. literalinclude:: ../../../source/standalone/environments/zero_agent.py
   :language: python
   :lines: 45-55
   :linenos:
   :lineno-start: 45

Similar to previous tutorials, to ensure a safe exit when running the script, we need to add checks
for whether the simulation is stopped or not.

.. literalinclude:: ../../../source/standalone/environments/zero_agent.py
   :language: python
   :lines: 57-59
   :linenos:
   :lineno-start: 57


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32


This should open a stage with a ground plane, lights and 32 cartpoles spawned in a grid. The cartpole
would be falling down since no actions are acting on them. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal.

.. note::
    When running environments with GPU pipeline, the states in the scene are not synced with the USD
    interface. Therefore values in the UI may appear wrong when simulation is running. Although objects
    may be updating in the Viewport, attribute values in the UI will not update along with them.

    To enable USD synchronization, please use the CPU pipeline with ``--cpu`` and disable flatcache by setting
    ``use_flatcache`` to False in the environment configuration.
