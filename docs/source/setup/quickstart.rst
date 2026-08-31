.. _isaac-lab-quickstart:

Quickstart
==========

This page takes you from a fresh checkout to training, replaying, and inspecting
your first task. For prerequisites, see :ref:`installation-system-requirements`.
Run all commands from the Isaac Lab repository root with ``uv run``. ``uv``
creates and manages the project environment for you.

.. _uv-run-training:

Install and run your first task
-------------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__, clone
Isaac Lab, and enter the repository:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

Train Cartpole with the Newton MJWarp physics backend and open the Newton
visualizer:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole --num_envs 16 --viz newton

Training outputs, including checkpoints, are saved under ``logs/``. Add
``--help`` to any command to see its available arguments:

.. code-block:: bash

   uv run isaaclab train --help

.. hint::

    ``uv run`` installs the core dependencies automatically. To use an optional
    integration, add ``--extra <name>`` before ``isaaclab``. You can enable multiple
    extras with a comma-separated list. For example:

   .. code-block:: bash

      uv run --extra ovphysx isaaclab train --task Isaac-Cartpole physics=ovphysx

   Extras make optional capabilities available; task selectors choose which capabilities the task uses
   For example, ``--extra ovphysx`` makes the OV PhysX integration available, while
   ``physics=ovphysx`` selects it for the task. You can combine extras as needed. The ``--extra all``
   shortcut installs a curated set of backends, RL libraries, and visualizers.
   Specialized extras such as ``rlinf``, ``mimic``, ``teleop``, ``tetrahedralization``, ``video``,
   and ``leapp`` are not included; add them explicitly when needed. See
   :ref:`installation-optional-extras` for the complete list.

Choose an RL library
--------------------

Pass ``--rl_library`` to ``train`` or ``play`` to choose the RL framework. If
you omit it, Isaac Lab uses the default registered for the task. Most core tasks
default to ``rsl_rl``, which is included in the standard ``uv run`` environment
and is a good starting point for GPU-based training.

.. list-table::
   :widths: 18 35 47
   :header-rows: 1

   * - ``rsl_rl``
     - Fast GPU training and policy distillation
     - ``uv run isaaclab train --rl_library rsl_rl ...``
   * - ``rl_games``
     - PPO, SAC, and A2C workflows
     - ``uv run --extra rl-games isaaclab train --rl_library rl_games ...``
   * - ``skrl``
     - Broad algorithm support with PyTorch and JAX
     - ``uv run --extra skrl isaaclab train --rl_library skrl ...``
   * - ``sb3``
     - Stable-Baselines3 and CPU-oriented experiments
     - ``uv run --extra sb3 isaaclab train --rl_library sb3 ...``
   * - ``rlinf``
     - VLA model fine-tuning
     - ``uv run --extra rlinf isaaclab train --rl_library rlinf ...``

RL libraries differ in their supported algorithms, tasks, and workflows. See
:doc:`/source/overview/reinforcement-learning/rl_frameworks` for a detailed
comparison.


The five commands to know
-------------------------

All task commands accept ``--task <task_name>``. Start by listing the registered tasks:

.. code-block:: bash

   uv run python scripts/environments/list_envs.py

.. list-table::
   :widths: 18 42 40
   :header-rows: 1

   * - Command
     - Use it to
     - Example
   * - ``train``
     - Train a policy with an RL library.
     - ``uv run isaaclab train --task Isaac-Cartpole``
   * - ``play``
     - Run a trained policy from a checkpoint.
     - ``uv run isaaclab play --task Isaac-Cartpole --checkpoint latest``
   * - ``zero_agent``
     - Run a task with zero actions to verify that it launches correctly.
     - ``uv run isaaclab zero_agent --task Isaac-Cartpole --viz newton``
   * - ``random_agent``
     - Run a task with random actions for a quick interaction smoke test.
     - ``uv run isaaclab random_agent --task Isaac-Cartpole --viz newton``
   * - ``benchmark``
     - Measure environment, training, play, or startup performance.
     - ``uv run isaaclab benchmark runtime --task Isaac-Cartpole``

All supported RL libraries use ``--checkpoint`` to choose a checkpoint for
playback. See :doc:`/source/overview/reinforcement-learning/rl_existing_scripts`
for the complete training and playback reference.


Choose a backend
----------------

Isaac Lab supports multiple physics and rendering backends. Use
``physics=<backend>`` to choose the physics implementation. For camera tasks,
use ``renderer=<backend>`` to choose the renderer.
Available backends depend on the task configuration. Use the task's help output
to see its supported selectors:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole --help

.. list-table::
   :widths: 28 48 24
   :header-rows: 1

   * - Selector
     - Backend
     - Required extra
   * - ``physics=newton_mjwarp``
     - Newton with the MuJoCo-Warp solver.
     - None
   * - ``physics=newton_kamino``
     - Newton with the Kamino solver. This backend is beta and supports a limited set of tasks.
     - None
   * - ``physics=ovphysx``
     - OV PhysX.
     - ``ov`` or ``ovphysx``
   * - ``physics=isaacsim_physx``
     - Isaac Sim PhysX.
     - ``isaacsim``
   * - ``renderer=newton_renderer``
     - Newton Warp renderer.
     - None
   * - ``renderer=ovrtx``
     - OV RTX renderer.
     - ``ov`` or ``ovrtx``
   * - ``renderer=isaacsim_rtx``
     - Isaac Sim RTX renderer.
     - ``isaacsim``
   * - ``renderer=rtx``
     - Automatic RTX renderer selection.
     - ``isaacsim`` or ``ovrtx``

Use ``presets=<name>`` to apply a task-specific configuration preset. For
example:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole-Camera physics=newton_mjwarp renderer=newton_renderer presets=rgb

See :doc:`/source/concepts/backends_and_presets` for backend and preset selection,
and :doc:`/source/features/hydra` for arbitrary configuration overrides.


Visualize a task
----------------

Use ``--viz`` (or ``--visualizer``) to choose one or more visualizers during
training or playback. To use multiple visualizers, pass a comma-separated list
without spaces, such as ``--viz newton,rerun``.

.. list-table::
   :widths: 18 58 24
   :header-rows: 1

   * - Option
     - Use it to
     - Required extra
   * - ``--viz newton``
     - Open the Newton visualizer.
     - None
   * - ``--viz rerun``
     - Stream the task to the Rerun visualizer.
     - ``rerun``
   * - ``--viz viser``
     - Open the web-based Viser visualizer, useful for remote connections.
     - ``viser``
   * - ``--viz kit``
     - Open the Kit visualizer when it is available in your environment.
     - ``isaacsim``
   * - Omit ``--viz`` or use ``--viz none``
     - Run without a visualizer.
     - None

For example, open the same task in both Newton and Rerun:

.. code-block:: bash

   uv run --extra rerun isaaclab random_agent --task Isaac-Cartpole \
      physics=newton_mjwarp --viz newton,rerun

See :doc:`/source/overview/core-concepts/visualization` for visualizer setup and
configuration.


Play a trained policy
---------------------

First, train Cartpole to create a checkpoint:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole

Then play the latest checkpoint in the Newton visualizer:

.. code-block:: bash

   uv run isaaclab play --task Isaac-Cartpole --checkpoint latest --viz newton

Choose a checkpoint with one of the following options:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Option
     - Loads
   * - ``--checkpoint <path>``
     - A checkpoint file at the specified local path. Some libraries also accept a run directory.
   * - ``--checkpoint best``
     - The library-specific best or final checkpoint. Falls back to ``latest``
       if no separate best or final checkpoint was saved.
   * - ``--checkpoint latest``
     - The highest-step checkpoint from the newest compatible run.
   * - ``--checkpoint pretrained``
     - A pretrained checkpoint hosted by Isaac Lab. Available only for
       supported tasks.


Benchmark a task
----------------

``isaaclab benchmark`` takes a workflow name as its first argument. Start with
``runtime`` to measure environment-step capacity without a policy:

.. code-block:: bash

   uv run isaaclab benchmark runtime --task Isaac-Cartpole

.. list-table::
   :widths: 36 64
   :header-rows: 1

   * - Workflow
     - Measures
   * - ``runtime``
     - Environment-step capacity with random actions (no policy).
   * - ``startup``
     - Launch, import, configuration, scene creation, and first-step latency.
   * - ``training``
     - End-to-end learning throughput. Requires ``--rl_library``.
   * - ``play``
     - Trained-policy rollout throughput. Requires ``--rl_library`` and
       ``--checkpoint``.
   * - ``-multigpu``
     - Run ``startup``, ``runtime``, or ``training`` across multiple GPUs by adding the suffix ``-multigpu``. For example, ``runtime-multigpu``.

See :ref:`testing_benchmarks` for warm-up, formatters, multi-GPU details, and
how to read results.

Next steps
----------

- Browse all registered environments: :doc:`/source/overview/environments`
- Learn how backends and presets fit together: :doc:`/source/concepts/backends_and_presets`
- Learn how to override task configuration: :doc:`/source/features/hydra`
- Follow a guided environment-building tutorial: :doc:`/source/tutorials/index`
- Read the installation options and troubleshooting guide: :ref:`isaaclab-installation-root`
