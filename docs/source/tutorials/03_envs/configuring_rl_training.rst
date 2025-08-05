.. _tutorial-configure-rl-training:

Configuring an RL Agent
=======================

.. currentmodule:: isaaclab

In the previous tutorial, we saw how to train an RL agent to solve the cartpole balancing task
using the `Stable-Baselines3`_ library. In this tutorial, we will see how to configure the
training process to use different RL libraries and different training algorithms.

In the directory ``scripts/reinforcement_learning``, you will find the scripts for
different RL libraries. These are organized into subdirectories named after the library name.
Each subdirectory contains the training and playing scripts for the library.

To configure a learning library with a specific task, you need to create a configuration file
for the learning agent. This configuration file is used to create an instance of the learning agent
and is used to configure the training process. Similar to the environment registration shown in
the :ref:`tutorial-register-rl-env-gym` tutorial, you can register the learning agent with the
``gymnasium.register`` method.

The Code
--------

As an example, we will look at the configuration included for the task ``Isaac-Cartpole-v0``
in the ``isaaclab_tasks`` package. This is the same task that we used in the
:ref:`tutorial-run-rl-training` tutorial.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py
   :language: python
   :lines: 18-29

The Code Explained
------------------

Under the attribute ``kwargs``, we can see the configuration for the different learning libraries.
The key is the name of the library and the value is the path to the configuration instance.
This configuration instance can be a string, a class, or an instance of the class.
For example, the value of the key ``"rl_games_cfg_entry_point"`` is a string that points to the
configuration YAML file for the RL-Games library. Meanwhile, the value of the key
``"rsl_rl_cfg_entry_point"`` points to the configuration class for the RSL-RL library.

The pattern used for specifying an agent configuration class follows closely to that used for
specifying the environment configuration entry point. This means that while the following
are equivalent:


.. dropdown:: Specifying the configuration entry point as a string
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole-v0",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
         },
      )

.. dropdown:: Specifying the configuration entry point as a class
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole-v0",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
         },
      )

The first code block is the preferred way to specify the configuration entry point.
The second code block is equivalent to the first one, but it leads to import of the configuration
class which slows down the import time. This is why we recommend using strings for the configuration
entry point.

All the scripts in the ``scripts/reinforcement_learning`` directory are configured by default to read the
``<library_name>_cfg_entry_point`` from the ``kwargs`` dictionary to retrieve the configuration instance.

For instance, the following code block shows how the ``train.py`` script reads the configuration
instance for the Stable-Baselines3 library:

.. dropdown:: Code for train.py with SB3
    :icon: code

    .. literalinclude:: ../../../../scripts/reinforcement_learning/sb3/train.py
      :language: python
      :emphasize-lines: 26-28, 102-103
      :linenos:

The argument ``--agent`` is used to specify the learning library to use. This is used to
retrieve the configuration instance from the ``kwargs`` dictionary. You can manually specify
alternate configuration instances by passing the ``--agent`` argument.

The Code Execution
------------------

Since for the cartpole balancing task, RSL-RL library offers two configuration instances,
we can use the ``--agent`` argument to specify the configuration instance to use.

* Training with the standard PPO configuration:

  .. code-block:: bash

    # standard PPO training
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless \
      --run_name ppo

* Training with the PPO configuration with symmetry augmentation:

  .. code-block:: bash

    # PPO training with symmetry augmentation
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless \
      --agent rsl_rl_with_symmetry_cfg_entry_point \
      --run_name ppo_with_symmetry_data_augmentation

    # you can use hydra to disable symmetry augmentation but enable mirror loss computation
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless \
      --agent rsl_rl_with_symmetry_cfg_entry_point \
      --run_name ppo_without_symmetry_data_augmentation \
      agent.algorithm.symmetry_cfg.use_data_augmentation=false

The ``--run_name`` argument is used to specify the name of the run. This is used to
create a directory for the run in the ``logs/rsl_rl/cartpole`` directory.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
