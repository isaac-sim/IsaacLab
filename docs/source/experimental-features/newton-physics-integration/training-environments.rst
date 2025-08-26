Training Environments
======================

To run training, we follow the standard Isaac Lab workflow. If you are new to Isaac Lab, we recommend that you review the `Quickstart Guide here <https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html#>`_.

The currently supported tasks are as follows:

* Isaac-Cartpole-Direct-v0
* Isaac-Ant-Direct-v0
* Isaac-Humanoid-Direct-v0
* Isaac-Velocity-Flat-Anymal-D-v0
* Isaac-Velocity-Flat-G1-v0
* Isaac-Velocity-Flat-G1-v1 (Sim-to-Real tested)
* Isaac-Velocity-Flat-H1-v0

To launch an environment and check that it loads as expected, we can start by trying it out with zero actions sent to its actuators.
This can be done as follows, where ``TASK_NAME`` is the name of the task you’d like to run, and ``NUM_ENVS`` is the number of instances of the task that you’d like to create.

.. code-block:: shell

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task TASK_NAME --num_envs NUM_ENVS

For cartpole with 128 instances it would look like this:

.. code-block:: shell

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128

To run the same environment with random actions we can use a different script:

.. code-block:: shell

    ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128

To train the environment we provide hooks to different rl frameworks. See the `Reinforcement Learning Scripts documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html>`_ for more information.

Here are some examples on how to run training on several different RL frameworks. Note that we are explicitly setting the number of environments to
4096 to benefit more from GPU parallelization. We also disable the Omniverse UI visualization to train the environment as quickly as possible by using the ``--headless`` option.

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --headless

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --headless

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --headless

Once a policy is trained we can visualize it by using the play scripts. But first, we need to find the checkpoint of the trained policy. Typically, these are stored under:
``logs/NAME_OF_RL_FRAMEWORK/TASK_NAME/DATE``.

For instance with our rsl_rl example it could look like this:
``logs/rsl_rl/cartpole_direct/2025-08-21_15-45-30/model_299.pt``

To then run this policy we can use the following command, note that we reduced the number of environments and removed the ``--headless`` option so that we can see our policy in action!

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --checkpoint logs/rsl_rl/cartpole_direct/2025-08-21_15-45-30/model_299.pt

The same approach applies to all other frameworks.

Note that not all environments are supported in all frameworks. For example, several of the locomotion environments are only supported in the rsl_rl framework.
