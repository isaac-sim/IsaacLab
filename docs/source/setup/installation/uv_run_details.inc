.. _uv-run-training:

.. rubric:: Complete ``uv run`` reference

.. warning::

   This feature is experimental and subject to change in future releases.

Install ``uv`` if you do not have it already:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repo and start training immediately — no virtual environment setup required:

.. isaaclab-clone-commands::

.. code-block:: bash

   # Newton backend training without Isaac Sim
   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp

   # Add OVRTX/OVPhysX extras only when the workflow needs them
   uv run --extra ovphysx --extra ovrtx isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp

   # PhysX backend: --extra isaacsim pulls in Isaac Sim on the fly
   uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct presets=physx

   # Add automatic tetrahedralization for volume deformables explicitly
   uv run --extra isaacsim --extra tetrahedralization isaaclab train \
      --rl_library rsl_rl --task Isaac-Lift-Soft-Franka presets=physx

``uv`` resolves and manages the environment automatically on each invocation. Supported
libraries for ``--rl_library`` are: ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``, and ``rlinf``.
The ``tetrahedralization`` extra is not included in the default environment or the
``all`` extra.

.. rubric:: Play / Evaluation

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task <any_task>
