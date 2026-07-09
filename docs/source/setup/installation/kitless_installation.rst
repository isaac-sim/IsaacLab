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
         ./isaaclab.sh train --rl_library rsl_rl \
         --task=Isaac-Cartpole-Direct \
         --num_envs=16 --max_iterations=10 \
         physics=newton_mjwarp --visualizer newton

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Install Isaac Lab (Newton backend, no Isaac Sim required)
         :: or: isaaclab.bat -i
         isaaclab.bat --install

         :: Kickoff training with MJWarp physics and Newton visualizer
         isaaclab.bat train --rl_library rsl_rl ^
         --task=Isaac-Cartpole-Direct ^
         --num_envs=16 --max_iterations=10 ^
         physics=newton_mjwarp --visualizer newton


**Features available in kit-less mode (Newton backend, no Isaac Sim):**

- Newton physics simulation (GPU-accelerated, including MuJoCo-Warp solver)
- All manager-based and direct RL environments that support Newton
- RL training with SKRL, RSL-RL, and other frameworks
- Robot assets compatible with Newton
- URDF and MJCF command-line conversion when the standalone importer wheel is installed
- ovphysx and ovrtx backends

**Features that require Isaac Sim:**

- Isaac Sim PhysX physics backend (not ovphysx)
- Isaac Sim RTX rendering (not ovrtx)
- Kit visualizer
- Photorealistic rendering workflows
- ROS / ROS2 integration
- Omniverse Kit GUI workflows for inspecting, editing, or importing assets
- Surface gripper (Isaac Sim PhysX-only)
- PhysX Deformable simulation (Isaac Sim PhysX-only)
- Teleoperation and imitation learning workflows

To install Isaac Sim, use the pip method described in :doc:`pip_installation`.

.. _installation-standalone-importers:

Standalone URDF/MJCF importers
------------------------------

The URDF and MJCF converter scripts can run without Isaac Sim when the standalone
``isaacsim-asset-isolated`` wheel is installed in the active environment:

.. code-block:: bash

   python -m pip install isaacsim-asset-isolated==1.0.0 \
     --find-links <package-index-or-wheel-directory>

After installing the wheel, run conversion without requesting the Kit visualizer:

.. code-block:: bash

   python scripts/tools/convert_urdf.py \
     path/to/robot.urdf path/to/output_dir --merge-joints

   python scripts/tools/convert_mjcf.py \
     path/to/model.xml path/to/output.usd --merge-mesh

If Isaac Sim is installed in the same environment, Isaac Lab uses the Isaac Sim
importer extensions first. The standalone wheel is used only when the full
Isaac Sim runtime is not available.

.. _installation-selective-install:

.. include:: include/selective_install.rst
