.. _tutorial-run-rl-training:

Training with an RL Agent
=========================

.. currentmodule:: isaaclab

In the previous tutorials, we covered how to define an RL task environment, register
it into the ``gym`` registry, and interact with it using a random agent. We now move
on to the next step: training an RL agent to solve the task.

Although the :class:`envs.ManagerBasedRLEnv` conforms to the :class:`gymnasium.Env` interface,
it is not exactly a ``gym`` environment. The input and outputs of the environment are
not numpy arrays, but rather based on torch tensors with the first dimension being the
number of environment instances.

Additionally, most RL libraries expect their own variation of an environment interface.
For example, `Stable-Baselines3`_ expects the environment to conform to its
`VecEnv API`_ which expects a list of numpy arrays instead of a single tensor. Similarly,
`RSL-RL`_, `RL-Games`_ and `SKRL`_ expect a different interface. Since there is no one-size-fits-all
solution, we do not base the :class:`envs.ManagerBasedRLEnv` on any particular learning library.
Instead, we implement wrappers to convert the environment into the expected interface.
These are specified in the :mod:`isaaclab_rl` module.

In this tutorial, we will use `Stable-Baselines3`_ to train an RL agent to solve the
cartpole balancing task.

.. caution::

  Wrapping the environment with the respective learning framework's wrapper should happen in the end,
  i.e. after all other wrappers have been applied. This is because the learning framework's wrapper
  modifies the interpretation of environment's APIs which may no longer be compatible with :class:`gymnasium.Env`.

The Code
--------

For this tutorial, we use the training script from `Stable-Baselines3`_ workflow in the
``scripts/reinforcement_learning/sb3`` directory.

.. dropdown:: Code for train.py
    :icon: code

    .. literalinclude:: ../../../../scripts/reinforcement_learning/sb3/train.py
      :language: python
      :emphasize-lines: 57, 66, 68-70, 81, 90-98, 100, 105-113, 115-116, 121-126, 133-136
      :linenos:

The Code Explained
------------------

.. currentmodule:: isaaclab_rl.utils

Most of the code above is boilerplate code to create logging directories, saving the parsed configurations,
and setting up different Stable-Baselines3 components. For this tutorial, the important part is creating
the environment and wrapping it with the Stable-Baselines3 wrapper.

There are three wrappers used in the code above:

1. :class:`gymnasium.wrappers.RecordVideo`: This wrapper records a video of the environment
   and saves it to the specified directory. This is useful for visualizing the agent's behavior
   during training.
2. :class:`wrappers.sb3.Sb3VecEnvWrapper`: This wrapper converts the environment
   into a Stable-Baselines3 compatible environment.
3. `stable_baselines3.common.vec_env.VecNormalize`_: This wrapper normalizes the
   environment's observations and rewards.

Each of these wrappers wrap around the previous wrapper by following ``env = wrapper(env, *args, **kwargs)``
repeatedly. The final environment is then used to train the agent. For more information on how these
wrappers work, please refer to the :ref:`how-to-env-wrappers` documentation.

The Code Execution
------------------

We train a PPO agent from Stable-Baselines3 to solve the cartpole balancing task.

Training the agent
~~~~~~~~~~~~~~~~~~

There are three main ways to train the agent. Each of them has their own advantages and disadvantages.
It is up to you to decide which one you prefer based on your use case.

Headless execution
""""""""""""""""""

If the ``--headless`` flag is set, the simulation is not rendered during training. This is useful
when training on a remote server or when you do not want to see the simulation. Typically, it speeds
up the training process since only physics simulation step is performed.

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless


Headless execution with off-screen render
"""""""""""""""""""""""""""""""""""""""""

Since the above command does not render the simulation, it is not possible to visualize the agent's
behavior during training. To visualize the agent's behavior, we pass the ``--enable_cameras`` which
enables off-screen rendering. Additionally, we pass the flag ``--video`` which records a video of the
agent's behavior during training.

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video

The videos are saved to the ``logs/sb3/Isaac-Cartpole-v0/<run-dir>/videos/train`` directory. You can open these videos
using any video player.

Interactive execution
"""""""""""""""""""""

.. currentmodule:: isaaclab

While the above two methods are useful for training the agent, they don't allow you to interact with the
simulation to see what is happening. In this case, you can ignore the ``--headless`` flag and run the
training script as follows:

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64

This will open the Isaac Sim window and you can see the agent training in the environment. However, this
will slow down the training process since the simulation is rendered on the screen. As a workaround, you
can switch between different render modes in the ``"Isaac Lab"`` window that is docked on the bottom-right
corner of the screen. To learn more about these render modes, please check the
:class:`sim.SimulationContext.RenderMode` class.

Viewing the logs
~~~~~~~~~~~~~~~~

On a separate terminal, you can monitor the training progress by executing the following command:

.. code:: bash

   # execute from the root directory of the repository
   ./isaaclab.sh -p -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole-v0

Playing the trained agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the training is complete, you can visualize the trained agent by executing the following command:

.. code:: bash

   # execute from the root directory of the repository
   ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Cartpole-v0 --num_envs 32 --use_last_checkpoint

The above command will load the latest checkpoint from the ``logs/sb3/Isaac-Cartpole-v0``
directory. You can also specify a specific checkpoint by passing the ``--checkpoint`` flag.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
.. _`stable_baselines3.common.vec_env.VecNormalize`: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
