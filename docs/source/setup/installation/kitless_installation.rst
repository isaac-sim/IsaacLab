.. _kitless-installation:

Kit-less Installation
=====================

Isaac Lab can be installed and used **without Isaac Sim** using the kit-less mode. This is the
fastest way to get started and is ideal for users who only need the Newton physics backend.

.. code-block:: bash

   # Clone Isaac Lab
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

   # Install Isaac Lab (Newton backend, no Isaac Sim required)
   ./isaaclab.sh --install   # or ./isaaclab.sh -i

   # Kickoff training with MJWarp physics and Newton visualizer
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
   --task=Isaac-Cartpole-Direct-v0 \
   --num_envs=16 --max_iterations=10 \
   presets=newton_mjwarp --visualizer newton


**Features available in kit-less mode (Newton backend, no Isaac Sim):**

- Newton physics simulation (GPU-accelerated, including MuJoCo-Warp solver)
- All manager-based and direct RL environments that support Newton
- RL training with SKRL, RSL-RL, and other frameworks
- Robot assets compatible with Newton

**Features that require Isaac Sim:**

- PhysX physics backend
- Isaac Sim RTX rendering (not ovrtx)
- Kit visualizer
- Photorealistic rendering workflows
- ROS / ROS2 integration
- URDF and MJCF importers (GUI-based)
- Deformable objects and surface gripper (PhysX-only)
- Teleoperation and imitation learning workflows

To install Isaac Sim, use the pip method described in :doc:`pip_installation`.


.. _installation-selective-install:

Selective Install
-----------------

If you want a minimal environment, ``./isaaclab.sh -i`` accepts comma-separated
sub-package names:

.. list-table::
   :header-rows: 1

   * - Option
     - What it does
   * - ``isaacsim``
     - Install Isaac Sim pip package
   * - ``newton``
     - Install Newton physics + Newton visualizer
   * - ``physx``
     - Install PhysX physics runtime
   * - ``ov``
     - Install Omniverse renderer runtime
   * - ``tasks``
     - Install built-in task environments
   * - ``assets``
     - Install robot/object configurations
   * - ``visualizers``
     - Install all visualizer backends
   * - ``rsl_rl``
     - Install RSL-RL framework
   * - ``skrl``
     - Install skrl framework
   * - ``sb3``
     - Install Stable Baselines3 framework
   * - ``rl_games``
     - Install rl_games framework
   * - ``robomimic``
     - Install robomimic framework
   * - ``none``
     - Install only core ``isaaclab`` package

Examples:

.. code-block:: bash

   # Minimal Newton setup
   ./isaaclab.sh -i newton,tasks,assets,ov,rl[rsl_rl]

   # Newton with OVRTX, RSL-RL, and Newton visualizer
   ./isaaclab.sh -i newton,tasks,assets,ov[ovrtx],rl[rsl_rl],visualizers[newton]


.. _installation-ovrtx:

OVRTX Rendering
---------------

OVRTX provides GPU-accelerated rendering for vision tasks without Kit.

.. code-block:: bash

   ./isaaclab.sh -i ov[ovrtx]

   export LD_PRELOAD=$(python -c "import ovrtx, pathlib; print(pathlib.Path(ovrtx.__file__).parent / 'bin/plugins/libcarb.so')")

   ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
     --task Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
     --headless --enable_cameras --num_envs 16 --max_iterations 10 \
     presets=newton_mjwarp,ovrtx_renderer,simple_shading_diffuse_mdl


Running Installation Tests
--------------------------

.. code-block:: bash

   ./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_cli_utils.py -v
