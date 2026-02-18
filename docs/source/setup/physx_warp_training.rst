.. _physx-warp-training:

PhysX + Warp Training (Isaac Sim 6.0)
=====================================

This document gives step-by-step instructions to set up the environment and run vision-based RL training (Dexsuite Kuka-Allegro, single camera) using PhysX simulation and the Newton Warp renderer.

1. Isaac Sim (6.0) from source
------------------------------

#. Clone and build Isaac Sim:

   .. code-block:: bash

      cd ~/git   # or your preferred parent directory
      git clone https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim.git
      cd omni_isaac_sim
      git checkout b69c05612c11ee0bafe15ea9e8d0189fab3e07f4

#. If needed for the build, add Jinja2 to ``deps/pip_cloud.toml`` (e.g. ``"Jinja2==3.1.5"`` in the packages list).
#. Build:

   .. code-block:: bash

      ./build.sh -r

2. Clone IsaacLab-Physx-Warp and symlink Isaac Sim
--------------------------------------------------

#. Clone the IsaacLab-Physx-Warp repo (if not already), e.g.:

   .. code-block:: bash

      cd ~/git
      git clone git@github.com:bdilinila/IsaacLab.git IsaacLab-Physx-Warp
      cd IsaacLab-Physx-Warp

   Use the appropriate branch or fork URL for your setup.

#. Create the ``_isaac_sim`` symlink to the built Sim:

   .. code-block:: bash

      rm -f _isaac_sim
      ln -sfn /path/to/omni_isaac_sim/_build/linux-x86_64/release _isaac_sim

   Replace ``/path/to/omni_isaac_sim`` with the actual path to your ``omni_isaac_sim`` clone.

3. Conda environment
--------------------

#. From the **IsaacLab-Physx-Warp** repo root, create the conda env (Python 3.12):

   .. code-block:: bash

      ./isaaclab.sh -c physx_dextrah

#. Activate it:

   .. code-block:: bash

      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate physx_dextrah

4. Install IsaacLab and dependencies
-----------------------------------

#. Install ``flatdict`` (version required by this branch):

   .. code-block:: bash

      pip install "flatdict==3.4.0"

#. Install all IsaacLab extensions (this installs the ``isaaclab`` package from ``source/isaaclab`` and other packages under ``source/``):

   .. code-block:: bash

      ./isaaclab.sh -i

   If you only need to reinstall the ``isaaclab`` package (e.g. after editing code in ``source/isaaclab``), you can run from the repo root:

   .. code-block:: bash

      pip install -e source/isaaclab

#. **(Optional)** If you use a local Newton clone:

   .. code-block:: bash

      pip install -e ~/git/newton

   Otherwise, Newton is installed from the Git dependency declared in ``source/isaaclab/setup.py``.

5. Verify installation
----------------------

From the IsaacLab-Physx-Warp root with the conda env activated:

.. code-block:: bash

   python -c "import newton; import warp; print('Newton, Warp OK')"
   python -c "import isaacsim; print('Isaac Sim:', isaacsim.__file__)"
   python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"

6. Run training
---------------

From the **IsaacLab-Physx-Warp** repo root, with the conda env activated:

.. code-block:: bash

   cd /path/to/IsaacLab-Physx-Warp
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate physx_dextrah
   export WANDB_USERNAME="${USERNAME:-$USER}"

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
     --enable_cameras \
     --headless \
     --num_envs=2048 \
     --max_iterations=32 \
     --logger=tensorboard \
     env.scene=64x64newton_depth

.. note::

   2048 environments will lead to a simulation startup time of ~30 minutes before training begins. Redirect stdout/stderr if desired, e.g. ``2>&1 | tee train.log``.

Summary
------

+------+----------------------------------------------------------------------------------------------------------------------------------+
| Step | Action                                                                                                                            |
+======+==================================================================================================================================+
| 1    | Clone and build Isaac Sim 6.0; checkout commit ``b69c05612c11ee0bafe15ea9e8d0189fab3e07f4``                                     |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 2    | Clone IsaacLab-Physx-Warp and set ``_isaac_sim`` symlink                                                                          |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 3    | ``./isaaclab.sh -c physx_dextrah`` then ``conda activate physx_dextrah``                                                            |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 4    | ``pip install "flatdict==3.4.0"``, then ``./isaaclab.sh -i``; optionally ``pip install -e source/isaaclab`` or ``pip install -e ~/git/newton`` |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 5    | Run the verification commands                                                                                                     |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 6    | Run ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...`` as above                                                 |
+------+----------------------------------------------------------------------------------------------------------------------------------+
