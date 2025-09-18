Newton Visualizer
=================

Newton includes its own built-in visualizer to enable a fast and lightweight way to view the results of simulation.
Many additional features are planned for this system for the future, including the ability to view the results of
training remotely through a web browser. To enable use of the Newton Visualizer use the ``--newton_visualizer`` command line option.

The Newton Visualizer is not capable of or intended to provide camera sensor data for robots being trained. It is solely
intended as a development debugging and visualization tool.

It also currently only supports visualization of collision shapes, not visual shapes.

Both the Omniverse RTX renderer and the Newton Visualizer can be run in parallel, or the Omniverse UI and RTX renderer
can be disabled using the ``--headless`` option.

Using one of our training examples above, training the Cartpole environment, we might choose to disable the Omniverse UI
and RTX renderer using the ``--headless`` option and enable the Newton Visualizer instead as follows:

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --headless --newton_visualizer

In general, we do not recommend using the Omniverse UI while training to ensure the fastest possible training times.
The Newton Visualizer has less of a performance penalty while running, and we aim to bring that overhead even lower in the future.

If we would like to run the Omniverse UI and the Newton Visualizer at the same time, for example when running inference using a
lower number of environments, we can omit the ``--headless`` option while still adding the ``--newton_visualizer`` option, as follows:

.. code-block:: shell

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --checkpoint logs/rsl_rl/cartpole_direct/2025-08-21_15-45-30/model_299.pt --newton_visualizer

These options are available across all the learning frameworks.

For more information about the Newton Visualizer, please refer to the `Newton documentation <https://newton-physics.github.io/newton/guide/visualization.html>`_ .
