:orphan:

.. _tutorial-run-rl-training:

Training with an RL Agent
=========================

.. currentmodule:: isaaclab

Train a Stable-Baselines3 policy to balance Cartpole using the task from
:ref:`tutorial-register-rl-env-gym`.

Isaac Lab environments return batched PyTorch tensors, while each RL library expects
its own environment interface. The wrappers in :mod:`isaaclab_rl` adapt observations,
actions, and episode information to that interface. This tutorial uses the
Stable-Baselines3 ``VecEnv`` wrapper; see :ref:`how-to-env-wrappers` for the other
wrappers and :ref:`reinforcement-learning` for the training command reference.

.. caution::

   Apply Gymnasium wrappers before the Isaac Lab wrapper for your RL library.
   The RL wrapper changes the environment interface and may no longer support
   Gymnasium wrappers.

The Code
--------

For this tutorial, we use the training implementation from `Stable-Baselines3`_ workflow in the
``isaaclab_rl.entrypoints.backends.train_sb3`` module.

.. dropdown:: Code for train_sb3.py
    :icon: code

    .. literalinclude:: ../../../source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_sb3.py
      :language: python
      :linenos:
      :emphasize-lines: 97-100, 104-109, 121-137, 145-157, 164-170

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

Train without a visualizer, watch the simulation live, or record a video using the
options below.

Headless execution
""""""""""""""""""

When no visualizer is requested, no interactive visualizer window is opened during training. This is useful
when training on a remote server or when you do not need live visual feedback, which can add some compute cost.
Rendering can still be active for sensor/camera data capture when enabled by the workflow.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra sb3 isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64


Headless execution with off-screen render
"""""""""""""""""""""""""""""""""""""""""

To inspect behavior from a training run without an interactive window, pass ``--video``.
By default, it records the scene from a headless Newton visualizer. See
:doc:`/source/how-to/record_video` to choose a different visualizer or a camera-sensor output.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra sb3,video isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --video

The videos are saved to the ``logs/sb3/Isaac-Cartpole/<run-dir>/videos/train`` directory. You can open these videos
using any video player.

For tasks with on-scene cameras, you can also save the sensor image outputs directly during training
with ``--capture_env_sensors``. See :doc:`/source/how-to/capture_sensor_frames` for the available
options and output formats.

Interactive execution
"""""""""""""""""""""

.. currentmodule:: isaaclab

While the above two methods are useful for training the agent, they don't allow you to interact with the
simulation to see what is happening. In this case, run the
training command as follows:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra sb3 isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --viz kit

This will open the Kit visualizer window and you can see the agent training in the environment. However, this
can slow down the training process because interactive visual feedback is enabled. As a workaround, you
can switch between different render modes in the ``"Isaac Lab"`` window that is docked on the bottom-right
corner of the screen. To learn more about these render modes, please check the
:class:`sim.SimulationContext.RenderMode` class.

Viewing the logs
~~~~~~~~~~~~~~~~

On a separate terminal, you can monitor the training progress by executing the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # execute from the root directory of the repository
         uv run python -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole

Playing the trained agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the training is complete, you can visualize the trained agent by executing the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # execute from the root directory of the repository
         uv run --extra sb3 isaaclab play --rl_library sb3 --task Isaac-Cartpole --num_envs 32 --viz kit

The above command will load the latest checkpoint from the ``logs/sb3/Isaac-Cartpole``
directory. You can also specify a specific checkpoint by passing the ``--checkpoint`` flag.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
.. _`stable_baselines3.common.vec_env.VecNormalize`: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
