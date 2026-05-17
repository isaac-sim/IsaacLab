.. _uv-run-training:

``uv run`` Training and Play (Experimental)
============================================

.. warning::

   This feature is experimental and subject to change in future releases.

Install ``uv`` if you do not have it already:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repo and start training immediately — no virtual environment setup required:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

   # Newton backend training without Isaac Sim
   uv run train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct-v0 --headless presets=newton_mjwarp

   # Add OVRTX/OVPhysX extras only when the workflow needs them
   uv run --extra ov --extra rtx train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct-v0 --headless presets=newton_mjwarp

``uv`` resolves and manages the environment automatically on each invocation. Supported
libraries for ``--rl_library`` are: ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``, and ``rlinf``.

Multi-GPU Training
------------------

Use ``train_multigpu`` for torch distributed training. It defaults to ``rsl_rl``, launches one
process per visible GPU, adds ``--distributed`` automatically, and forwards the remaining
arguments to the selected training library:

.. code-block:: bash

   uv run train_multigpu \
      --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
      --headless --num_envs 4096 --max_iterations 100 \
      --run_name gpu4_vis presets=newton

Override the GPU count or torch distributed settings when needed:

.. code-block:: bash

   uv run train_multigpu --num_gpus 4 --master_port 29504 \
      --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
      --headless --num_envs 4096 --max_iterations 100 \
      --run_name gpu4_vis presets=newton

Use ``--rl_library`` for other distributed-capable libraries: ``rsl_rl``, ``rl_games``, or ``skrl``.
For multi-node jobs, pass torchrun settings such as ``--nnodes``, ``--node_rank``,
``--rdzv_backend``, ``--rdzv_endpoint``, and ``--rdzv_id`` before the training arguments.

Play / Evaluation
-----------------

.. code-block:: bash

   uv run play --rl_library rsl_rl --task <any_task>
