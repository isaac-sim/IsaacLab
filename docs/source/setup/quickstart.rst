.. _isaac-lab-quickstart:

Quickstart
==========

This page takes you from a fresh checkout to training, playing back, and inspecting a task. Run every
command from the Isaac Lab repository root with ``uv run``; ``uv`` creates and
manages the project environment for you.

.. _uv-run-training:

Install and run your first task
-------------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__, clone
Isaac Lab, and enter the checkout:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

Train Cartpole with the Newton MJWarp backend and open the Newton visualizer:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 \
      physics=newton_mjwarp --viz newton

Training outputs, including checkpoints, are written under ``logs/``. Use
``--help`` after any command to see its arguments:

.. code-block:: bash

   uv run isaaclab train --help

.. hint::

   ``uv run`` installs core dependencies automatically. When a command needs an
   optional integration, add ``--extra <name>`` before ``isaaclab``. Pass a
   comma-separated list to enable several extras. For example:

   .. code-block:: bash

      uv run --extra ovphysx isaaclab train --rl_library rsl_rl \
         --task Isaac-Cartpole-Direct physics=ovphysx

   Extras install capabilities; task selectors choose how to use them. For example,
   ``--extra ovphysx`` makes the OV PhysX integration available, while
   ``physics=ovphysx`` selects it for the task. Extras can be combined freely, and
   ``--extra all`` installs every backend, RL library, and visualizer at once. See
   :ref:`installation-optional-extras` for the complete list.


Choose an RL library
--------------------

Pass ``--rl_library`` to ``train`` and ``play`` to choose the learning framework.
Start with ``rsl_rl``: it is included in the default ``uv run`` environment and is
a good choice for most GPU-based training.

.. list-table::
   :widths: 18 35 47
   :header-rows: 1

   * - ``--rl_library``
     - Best starting point
     - Run it with
   * - ``rsl_rl``
     - Fast GPU training and policy distillation.
     - ``uv run isaaclab train --rl_library rsl_rl ...``
   * - ``rl_games``
     - PPO, SAC, and A2C training.
     - ``uv run --extra rl-games isaaclab train --rl_library rl_games ...``
   * - ``skrl``
     - A broad algorithm selection, with PyTorch and JAX support.
     - ``uv run --extra skrl isaaclab train --rl_library skrl ...``
   * - ``sb3``
     - Stable-Baselines3 workflows, including CPU-oriented experiments.
     - ``uv run --extra sb3 isaaclab train --rl_library sb3 ...``

   * - ``rlinf``
     - VLA model fine-tuning.
     - ``uv run --extra rlinf isaaclab train --rl_library rlinf ...``

The libraries support different algorithms, workflows, and tasks. See
:doc:`/source/overview/reinforcement-learning/rl_frameworks` for the full comparison.


The four commands to know
-------------------------

All task commands accept ``--task <task_name>``. Start by listing the tasks
available in your installation:

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
     - ``uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct``
   * - ``play``
     - Run a trained policy from a checkpoint.
     - ``uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --checkpoint latest``
   * - ``zero_agent``
     - Check a task using zero actions; useful for confirming that it launches.
     - ``uv run isaaclab zero_agent --task Isaac-Cartpole-Direct --viz newton``
   * - ``random_agent``
     - Check a task using random actions; useful for a quick interaction smoke test.
     - ``uv run isaaclab random_agent --task Isaac-Cartpole-Direct --viz newton``

All supported RL libraries select a checkpoint with ``--checkpoint``. See :doc:`/source/overview/reinforcement-learning/rl_existing_scripts`
for the complete training and playback reference.


Choose a backend
----------------

Add ``physics=<backend>`` to a task command to select its physics backend. For
camera tasks, you can also choose a renderer backend with
``renderer=<backend>``. The backends available to a task depend on its
configuration; use the task help to see the supported selectors:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole-Direct --help

.. list-table::
   :widths: 28 48 24
   :header-rows: 1

   * - Selector
     - Backend
     - Required extra
   * - ``physics=newton_mjwarp``
     - Newton using the MuJoCo-Warp solver. This is a good default for the quickstart.
     - None
   * - ``physics=newton_kamino``
     - Newton using the Kamino solver. This backend is beta and supports a limited set of tasks.
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

Add task-specific options with ``presets=<name>``; for example:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera-Direct \
      physics=newton_mjwarp renderer=newton_renderer presets=rgb

See :doc:`/source/concepts/backends_and_presets` for backend and preset selection,
and :doc:`/source/features/hydra` for arbitrary configuration overrides.


Visualize a task
----------------

Use ``--viz`` (or ``--visualizer``) to select one or more visualizers during training or playback. Pass a
comma-separated list without spaces, such as ``--viz newton,rerun``.

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
     - Open the web-based Viser visualizer.
     - ``viser``
   * - ``--viz kit``
     - Open the Kit visualizer when it is available in your environment.
     - ``isaacsim``
   * - Omit ``--viz`` or use ``--viz none``
     - Run headlessly.
     - None

For example, view the same task in both the Newton and Rerun visualizers:

.. code-block:: bash

   uv run --extra rerun isaaclab random_agent --task Isaac-Cartpole-Direct \
      physics=newton_mjwarp --viz newton,rerun

See :doc:`/source/overview/core-concepts/visualization` for visualizer setup and
configuration.


Replay a checkpoint
-------------------

Train Cartpole to create a checkpoint:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 \
      physics=newton_mjwarp

Then replay the newest checkpoint in the Newton visualizer:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp \
      --checkpoint latest --viz newton

Choose a checkpoint with one of the following selectors:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Selector
     - Loads
   * - ``--checkpoint <path>``
     - The checkpoint at the specified local path.
   * - ``--checkpoint best``
     - The library-specific best or final checkpoint. If none was saved separately, this resolves to ``latest``.
   * - ``--checkpoint latest``
     - The highest-step checkpoint from the newest compatible run.


Next steps
----------

- Browse all registered environments: :doc:`/source/overview/environments`
- Learn how backends and presets fit together: :doc:`/source/concepts/backends_and_presets`
- Learn how to override task configuration: :doc:`/source/features/hydra`
- Follow a guided environment-building tutorial: :doc:`/source/tutorials/index`
- Read the installation options and troubleshooting guide: :ref:`isaaclab-installation-root`
