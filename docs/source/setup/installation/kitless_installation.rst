.. _kitless-installation:

Kit-less Installation
=====================

Isaac Lab can be installed and used **without Isaac Sim** using the kit-less mode. This is the
fastest way to get started and is ideal for users who only need the Newton physics backend.

.. include:: include/pip_python_virtual_env.rst

Cloning and installing Isaac Lab
--------------------------------

With the virtual environment activated, clone the repository:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

Then install Isaac Lab (Newton backend, no Isaac Sim required) and kickoff training
with MJWarp physics and the Newton visualizer:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Install Isaac Lab (Newton backend, no Isaac Sim required)
         ./isaaclab.sh --install   # or ./isaaclab.sh -i

         # Kickoff training with MJWarp physics and Newton visualizer
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
         --task=Isaac-Cartpole-Direct-v0 \
         --num_envs=16 --max_iterations=10 \
         physics=newton_mjwarp --visualizer newton

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Install Isaac Lab (Newton backend, no Isaac Sim required)
         :: or: isaaclab.bat -i
         isaaclab.bat --install

         :: Kickoff training with MJWarp physics and Newton visualizer
         isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
         --task=Isaac-Cartpole-Direct-v0 ^
         --num_envs=16 --max_iterations=10 ^
         physics=newton_mjwarp --visualizer newton


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

``./isaaclab.sh -i`` always installs the full core package set (assets, tasks, physx, rl,
visualizers, …).  The argument controls which **optional** submodules and extra feature
dependencies are added on top.

**Optional submodules** (heavy — must be explicitly requested):

.. list-table::
   :header-rows: 1

   * - Token
     - What it installs
   * - ``mimic``
     - ``isaaclab_mimic`` — imitation-learning tools (ipywidgets, h5py)
   * - ``teleop``
     - ``isaaclab_teleop`` — teleoperation SDK (Linux x86 only)

**Optional extra feature sets** (heavy deps on top of always-installed core):

.. list-table::
   :header-rows: 1

   * - Token
     - What it installs
   * - ``newton``
     - Newton physics library (``newton[sim]``) + newton extras across ``isaaclab_newton``, ``isaaclab_physx``, ``isaaclab_visualizers``
   * - ``rl[<framework>]``
     - RL framework. Selectors: ``rsl-rl``, ``skrl``, ``sb3``, ``rl-games``. Omit selector for all.
   * - ``visualizer[<backend>]``
     - Visualizer backend. Selectors: ``rerun``, ``viser``, ``newton``, ``kit``. Omit selector for all.
   * - ``contrib[rlinf]``
     - rlinf extras (ray, diffusers, etc.)
   * - ``ov``
     - OVRTX + OVPhysX extras for Omniverse rendering
   * - ``isaacsim``
     - Isaac Sim pip package
   * - ``all``
     - Core + optional submodules (mimic, teleop) + auto extras (newton, rl, visualizer, ov). Default. Does not include ``contrib``.
   * - ``none``
     - Core packages only — no optional submodules, no extra feature deps

Examples:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Core only (physx, tasks, assets always included — no optional extras)
         ./isaaclab.sh -i core

         # Newton physics + RSL-RL (most common kitless setup)
         ./isaaclab.sh -i newton,'rl[rsl-rl]'

         # Newton + OVRTX renderer + RSL-RL + Newton visualizer
         ./isaaclab.sh -i newton,ov,'rl[rsl-rl]','visualizer[newton]'

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Core only
         isaaclab.bat -i core

         :: Newton physics + RSL-RL
         isaaclab.bat -i newton,rl[rsl-rl]

         :: Newton + OVRTX + RSL-RL + Newton visualizer
         isaaclab.bat -i newton,ov,rl[rsl-rl],visualizer[newton]


.. _installation-ovrtx:

OVRTX Rendering
---------------

OVRTX provides GPU-accelerated rendering for vision tasks without Kit.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./isaaclab.sh -i ov[ovrtx]

         ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
           --task Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
           --headless --enable_cameras --num_envs 16 --max_iterations 10 \
           physics=newton_mjwarp renderer=ovrtx_renderer presets=simple_shading_diffuse_mdl

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -i ov[ovrtx]

         isaaclab.bat -p scripts\benchmarks\benchmark_rsl_rl.py ^
           --task Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 ^
           --headless --enable_cameras --num_envs 16 --max_iterations 10 ^
           physics=newton_mjwarp renderer=ovrtx_renderer presets=simple_shading_diffuse_mdl


Running Installation Tests
--------------------------

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_cli_utils.py -v

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -p -m pytest source\isaaclab\test\cli\test_cli_utils.py -v
