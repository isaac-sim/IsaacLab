.. _kitless-installation:

Kit-less Installation
=====================

Isaac Lab can be installed and used **without Isaac Sim** using the kit-less mode. This is the
fastest way to get started and is ideal for users who only need the Newton physics backend.

Cloning Isaac Lab
-----------------

Clone the repository:

.. isaaclab-clone-commands::

.. include:: include/pip_python_virtual_env.rst

Installing Isaac Lab
--------------------

With the virtual environment activated, install Isaac Lab (Newton backend, no Isaac Sim required)
and kickoff training with MJWarp physics and the Newton visualizer:

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
- ovphysx and ovrtx backends

**Features that require Isaac Sim:**

- Isaac Sim PhysX physics backend (not ovphysx)
- Isaac Sim RTX rendering (not ovrtx)
- Kit visualizer
- URDF and MJCF importers (GUI-based)
- Surface gripper (Isaac Sim PhysX-only)
- PhysX Deformable simulation (Isaac Sim PhysX-only)
- Teleoperation and imitation learning workflows

To install Isaac Sim, use the pip method described in :doc:`pip_installation`.

.. _installation-selective-install:

.. include:: include/selective_install.rst
