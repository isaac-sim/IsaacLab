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
      --task Isaac-Cartpole-Direct-v0 physics=newton_mjwarp

   # Add OVRTX/OVPhysX extras only when the workflow needs them
   uv run --extra ov --extra rtx train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct-v0 physics=newton_mjwarp

``uv`` resolves and manages the environment automatically on each invocation. Supported
libraries for ``--rl_library`` are: ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``, and ``rlinf``.


Play / Evaluation
-----------------

.. code-block:: bash

   uv run play --rl_library rsl_rl --task <any_task>
