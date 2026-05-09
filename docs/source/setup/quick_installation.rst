.. _isaac-lab-quick-installation:

Quick Installation
=======================

The fastest path from a fresh clone is to let ``uv`` create and sync the environment for
the command you are running:

.. code-block:: bash

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone Isaac Lab
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

   # Check the unified training entrypoint
   uv run train --help

   # Run training with the default RSL-RL + Newton stack
   uv run train --library rsl_rl \
     --task=Isaac-Cartpole-Direct-v0 \
     --num_envs=16 --max_iterations=10 \
     presets=newton_mjwarp --visualizer newton

The default ``uv`` environment includes the RSL-RL, tasks, and Newton dependency
stacks. Add extras such as ``ov`` or ``rtx`` only when a workflow needs them.
Isaac Sim Kit workflows, including PhysX, should use the existing full installation
path. The ``./isaaclab.sh -i`` installer remains available for users who prefer an
explicit virtual environment setup.


Running Tasks
-------------------

The ``presets=`` Hydra override selects the physics backend and renderer at runtime:

.. code-block:: bash

   # MJWarp (Newton backend, Kit-less)
    ./isaaclab.sh train --library rsl_rl \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=newton_mjwarp \
     --visualizer newton

   # PhysX (Kit — requires Isaac Sim from the full installation path)
    ./isaaclab.sh train --library rsl_rl \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=physx

   # MJWarp with a specific visualizer
    ./isaaclab.sh train --library rsl_rl \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=newton_mjwarp \
     --visualizer viser

Kit-less visualizer options: ``newton``, ``rerun``, ``viser``. Multiple can be
combined: ``--visualizer newton,rerun``.

.. seealso::

   - :doc:`/source/features/hydra` — Hydra presets and configuration overrides
   - :doc:`/source/features/visualization` — Visualizer backends and configuration
   - :doc:`/source/how-to/configure_rendering` — Rendering mode presets and settings
   - :ref:`isaaclab-installation-root` — Full installation guide with all methods

Available Presets
^^^^^^^^^^^^^^^^^

Presets are combined with commas: ``presets=newton_mjwarp,newton_renderer,depth``.

.. code-block:: bash

   presets=newton_mjwarp,newton_renderer,rgb  # presets=physics,renderer,render mode
   presets=newton_mjwarp,newton_renderer,depth
   presets=physx,isaacsim_rtx_renderer,rgb
   presets=physx,isaacsim_rtx_renderer,depth
   presets=physx,isaacsim_rtx_renderer,albedo
   presets=physx,isaacsim_rtx_renderer,simple_shading_constant_diffuse
   presets=physx,isaacsim_rtx_renderer,simple_shading_diffuse_mdl
   presets=physx,isaacsim_rtx_renderer,simple_shading_full_mdl
   presets=newton_mjwarp,ovrtx_renderer,rgb
   presets=newton_mjwarp,ovrtx_renderer,depth
   presets=newton_mjwarp,ovrtx_renderer,albedo
   presets=newton_mjwarp,ovrtx_renderer,simple_shading_constant_diffuse
   presets=newton_mjwarp,ovrtx_renderer,simple_shading_diffuse_mdl
   presets=newton_mjwarp,ovrtx_renderer,simple_shading_full_mdl
