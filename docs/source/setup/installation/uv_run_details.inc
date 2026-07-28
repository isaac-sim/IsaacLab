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

   # XR teleoperation: the xr extra aggregates isaacsim and teleop
   uv run --extra xr isaaclab teleop run \
      --task IsaacContrib-PickPlace-Locomanipulation-G1-Abs --visualizer kit --xr

``uv`` resolves and manages the environment automatically on each invocation. Supported
libraries for ``--rl_library`` are: ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``, and ``rlinf``.

Extras compose freely except where their pins genuinely clash. ``teleop`` cannot combine with
``mimic`` or ``all`` (Isaac Teleop needs ``lxml>=5.2.2``, the imitation-learning stack needs
``lxml<5.0.0``), and ``xr`` inherits that plus Isaac Sim's own conflicts with ``ov``,
``viser``, and ``test``. ``uv`` reports an error rather than installing a broken environment.
See :ref:`installation-method-teleop` for the teleoperation prerequisites.


.. rubric:: Play / Evaluation

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task <any_task>
