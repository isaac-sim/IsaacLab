.. _tutorial-register-rl-env-gym:

Registering an Environment
==========================

.. currentmodule:: isaaclab

In the previous tutorial, we learned how to create a custom cartpole environment. We manually
created an instance of the environment by importing the environment class and its configuration
class.

.. dropdown:: Environment creation in the previous tutorial
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/03_envs/run_cartpole_rl_env.py
      :language: python
      :start-at: # create environment configuration
      :end-at: env = ManagerBasedRLEnv(cfg=env_cfg)

While straightforward, this approach is not scalable as we have a large suite of environments.
In this tutorial, we will show how to use the :meth:`gymnasium.register` method to register
environments with the ``gymnasium`` registry. This allows us to create the environment through
the :meth:`gymnasium.make` function.


.. dropdown:: Environment creation in this tutorial
   :icon: code

   .. literalinclude:: ../../../../scripts/environments/random_agent.py
      :language: python
      :lines: 36-47


The Code
~~~~~~~~

The tutorial corresponds to the ``random_agent.py`` script in the ``scripts/environments`` directory.

.. dropdown:: Code for random_agent.py
   :icon: code

   .. literalinclude:: ../../../../scripts/environments/random_agent.py
      :language: python
      :emphasize-lines: 36-37, 42-47
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The :class:`envs.ManagerBasedRLEnv` class inherits from the :class:`gymnasium.Env` class to follow
a standard interface. However, unlike the traditional Gym environments, the :class:`envs.ManagerBasedRLEnv`
implements a *vectorized* environment. This means that multiple environment instances
are running simultaneously in the same process, and all the data is returned in a batched
fashion.

Similarly, the :class:`envs.DirectRLEnv` class also inherits from the :class:`gymnasium.Env` class
for the direct workflow. For :class:`envs.DirectMARLEnv`, although it does not inherit
from Gymnasium, it can be registered and created in the same way.

Using the gym registry
----------------------

To register an environment, we use the :meth:`gymnasium.register` method. This method takes
in the environment name, the entry point to the environment class, and the entry point to the
environment configuration class.

.. note::
    The :mod:`gymnasium` registry is a global registry. Hence, it is important to ensure that the
    environment names are unique. Otherwise, the registry will throw an error when registering
    the environment.

Manager-Based Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^

For manager-based environments, the following shows the registration
call for the cartpole environment in the ``isaaclab_tasks.manager_based.classic.cartpole`` sub-package:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py
   :language: python
   :lines: 10-
   :emphasize-lines: 4, 11, 12, 15

The ``id`` argument is the name of the environment. As a convention, we name all the environments
with the prefix ``Isaac-`` to make it easier to search for them in the registry. The name of the
environment is typically followed by the name of the task, and then the name of the robot.
For instance, for legged locomotion with ANYmal C on flat terrain, the environment is called
``Isaac-Velocity-Flat-Anymal-C-v0``. The version number ``v<N>`` is typically used to specify different
variations of the same environment. Otherwise, the names of the environments can become too long
and difficult to read.

The ``entry_point`` argument is the entry point to the environment class. The entry point is a string
of the form ``<module>:<class>``. In the case of the cartpole environment, the entry point is
``isaaclab.envs:ManagerBasedRLEnv``. The entry point is used to import the environment class
when creating the environment instance.

The ``env_cfg_entry_point`` argument specifies the default configuration for the environment. The default
configuration is loaded using the :meth:`isaaclab_tasks.utils.parse_env_cfg` function.
It is then passed to the :meth:`gymnasium.make` function to create the environment instance.
The configuration entry point can be both a YAML file or a python configuration class.

Direct Environments
^^^^^^^^^^^^^^^^^^^

For direct-based environments, the environment registration follows a similar pattern. Instead of
registering the environment's entry point as the :class:`~isaaclab.envs.ManagerBasedRLEnv` class,
we register the environment's entry point as the implementation class of the environment.
Additionally, we add the suffix ``-Direct`` to the environment name to differentiate it from the
manager-based environments.

As an example, the following shows the registration call for the cartpole environment in the
``isaaclab_tasks.direct.cartpole`` sub-package:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/__init__.py
   :language: python
   :lines: 10-31
   :emphasize-lines: 5, 12, 13, 16


Creating the environment
------------------------

To inform the ``gym`` registry with all the environments provided by the ``isaaclab_tasks``
extension, we must import the module at the start of the script. This will execute the ``__init__.py``
file which iterates over all the sub-packages and registers their respective environments.

.. literalinclude:: ../../../../scripts/environments/random_agent.py
   :language: python
   :start-at: import isaaclab_tasks  # noqa: F401
   :end-at: import isaaclab_tasks  # noqa: F401

In this tutorial, the task name is read from the command line. The task name is used to parse
the default configuration as well as to create the environment instance. In addition, other
parsed command line arguments such as the number of environments, the simulation device,
and whether to render, are used to override the default configuration.

.. literalinclude:: ../../../../scripts/environments/random_agent.py
   :language: python
   :start-at: # create environment configuration
   :end-at: env = gym.make(args_cli.task, cfg=env_cfg)

Once creating the environment, the rest of the execution follows the standard resetting and stepping.


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


This should open a stage with everything similar to the :ref:`tutorial-create-manager-rl-env` tutorial.
To stop the simulation, you can either close the window, or press ``Ctrl+C`` in the terminal.


.. figure:: ../../_static/tutorials/tutorial_register_environment.jpg
    :align: center
    :figwidth: 100%
    :alt: result of random_agent.py


In addition, you can also change the simulation device from GPU to CPU by setting the value of the ``--device`` flag explicitly:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32 --device cpu

With the ``--device cpu`` flag, the simulation will run on the CPU. This is useful for debugging the simulation.
However, the simulation will run much slower than on the GPU.
