.. _isaac-lab-quickstart:

Quick Installation
=======================

``./isaaclab.sh -i`` installs everything needed to run with Newton Physics out of the box.

.. code-block:: bash

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

.. code-block:: bash

   # Clone Isaac Lab
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

.. code-block:: bash

   # Create environment and install
   uv venv .venv --python 3.12
   source .venv/bin/activate
   ./isaaclab.sh -i

.. code-block:: bash

   # Run training
   ./isaaclab.sh -p scripts/benchmarks/benchmark_rlgames.py \
     --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
     --headless --enable_cameras --num_envs=1225 --max_iterations=10 \
     presets=newton,newton_renderer,depth


Running Tasks
-------------------

The ``presets=`` Hydra override selects the physics backend and renderer at runtime:

.. code-block:: bash

   # Newton (Kit-less)
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=newton \
     --visualizer newton

   # PhysX (Kit — requires Isaac Sim)
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=physx

   # Newton with a specific visualizer
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Cartpole-Direct-v0 \
     --num_envs 4096 \
     presets=newton \
     --visualizer viser

Kit-less visualizer options: ``newton``, ``rerun``, ``viser``. Multiple can be
combined: ``--visualizer newton,rerun``.

Available Presets
^^^^^^^^^^^^^^^^^

Presets are combined with commas: ``presets=newton,newton_renderer,depth``.

.. code-block:: bash

   presets=newton,newton_renderer,rgb  # presets=physics,renderer,render mode
   presets=newton,newton_renderer,depth
   presets=physx,isaacsim_rtx_renderer,rgb
   presets=physx,isaacsim_rtx_renderer,depth
   presets=physx,isaacsim_rtx_renderer,albedo
   presets=physx,isaacsim_rtx_renderer,simple_shading_constant_diffuse
   presets=physx,isaacsim_rtx_renderer,simple_shading_diffuse_mdl
   presets=physx,isaacsim_rtx_renderer,simple_shading_full_mdl
   presets=newton,ovrtx_renderer,rgb
   presets=newton,ovrtx_renderer,depth
   presets=newton,ovrtx_renderer,albedo
   presets=newton,ovrtx_renderer,simple_shading_constant_diffuse
   presets=newton,ovrtx_renderer,simple_shading_diffuse_mdl
   presets=newton,ovrtx_renderer,simple_shading_full_mdl
