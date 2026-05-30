.. _isaac-lab-quickstart:

Quickstart
==========

Isaac Lab is a GPU-accelerated framework for robot learning built on vectorized simulation.
Environments run thousands of parallel copies on the GPU, and a modular manager design lets you
swap robots, sensors, and controllers without rewriting task logic.

This page gets you installed and running a first training job in minutes. For deeper topics
(configurations, project scaffolding, standalone apps, preset catalogs), see
:doc:`quickstart_details`.

Install
-------

Create a Python 3.12 environment, clone Isaac Lab, and install. Choose the path that matches
your workflow:

.. tab-set::

   .. tab-item:: Kit-less (no Isaac Sim)

      Fastest start — Newton physics only, no Isaac Sim download required.

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code-block:: bash

               # Install uv (https://docs.astral.sh/uv/getting-started/installation/)
               curl -LsSf https://astral.sh/uv/install.sh | sh

               git clone https://github.com/isaac-sim/IsaacLab.git
               cd IsaacLab

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate
               ./isaaclab.sh -i

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code-block:: batch

               git clone https://github.com/isaac-sim/IsaacLab.git
               cd IsaacLab

               uv venv --python 3.12 --seed env_isaaclab
               env_isaaclab\Scripts\activate
               isaaclab.bat -i

      See :doc:`/source/setup/installation/kitless_installation` for selective install tokens
      and feature availability without Isaac Sim.

   .. tab-item:: With Isaac Sim (full features)

      Recommended for PhysX, RTX rendering, ROS, URDF/MJCF importers, and the Kit visualizer.

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code-block:: bash

               git clone https://github.com/isaac-sim/IsaacLab.git
               cd IsaacLab

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install --upgrade pip
               uv pip install "isaacsim[all,extscache]==6.0.0" \
                 --extra-index-url https://pypi.nvidia.com \
                 --index-strategy unsafe-best-match --prerelease=allow
               uv pip install -U torch==2.10.0 torchvision==0.25.0 \
                 --index-url https://download.pytorch.org/whl/cu128
               ./isaaclab.sh -i

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code-block:: batch

               git clone https://github.com/isaac-sim/IsaacLab.git
               cd IsaacLab

               uv venv --python 3.12 --seed env_isaaclab
               env_isaaclab\Scripts\activate
               uv pip install --upgrade pip
               uv pip install "isaacsim[all,extscache]==6.0.0" ^
                 --extra-index-url https://pypi.nvidia.com ^
                 --index-strategy unsafe-best-match --prerelease=allow
               uv pip install -U torch==2.10.0 torchvision==0.25.0 ^
                 --index-url https://download.pytorch.org/whl/cu128
               isaaclab.bat -i

      On Linux aarch64 (DGX Spark), use ``cu130`` for PyTorch and see
      :ref:`isaaclab-installation-root` for additional setup notes.

      For conda, binary installs, Docker, and troubleshooting, see
      :ref:`isaaclab-installation-root`.


Run Training
------------

Training scripts live under ``scripts/reinforcement_learning/``. Pass a **task name** and use
``physics=``, ``renderer=``, and ``presets=`` to select backends and task-specific options:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Kit-less: Newton MJWarp physics + Newton visualizer
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Direct-v0 \
           --num_envs=16 --max_iterations=10 \
           physics=newton_mjwarp --visualizer newton

         # With Isaac Sim: PhysX physics (default renderer)
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Direct-v0 \
           --num_envs=4096 \
           physics=physx

         # Camera task: typed physics + renderer + domain preset
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Camera-Presets-Direct-v0 \
           physics=newton_mjwarp renderer=newton_renderer presets=rgb

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
           --task=Isaac-Cartpole-Direct-v0 ^
           --num_envs=16 --max_iterations=10 ^
           physics=newton_mjwarp --visualizer newton

         isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
           --task=Isaac-Cartpole-Direct-v0 ^
           --num_envs=4096 ^
           physics=physx

Add ``--headless`` to disable the GUI. Use ``--help`` on any script to see task-specific
``physics=``, ``renderer=``, and ``presets=`` options.

.. seealso::

   - :doc:`/source/features/hydra` — preset system and Hydra overrides
   - :doc:`quickstart_details` — preset catalog, environments, project generator, and more


Next Steps
----------

- List registered environments: ``./isaaclab.sh -p scripts/environments/list_envs.py``
- Scaffold a new project: ``./isaaclab.sh --new``
- Walk through tutorials: :doc:`/source/tutorials/index`
- Browse all environments: :doc:`/source/overview/environments`


.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart_details
