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

#. Install all IsaacLab extensions (this installs the ``isaaclab`` package from ``source/isaaclab`` and other packages under ``source/``):

   .. code-block:: bash

      ./isaaclab.sh -i

   If you only need to reinstall the ``isaaclab`` package (e.g. after editing code in ``source/isaaclab``), you can run from the repo root:

   .. code-block:: bash

      pip install -e source/isaaclab

#. Remove Newton and Warp from the Isaac Sim build so the app uses the pip-installed Newton (and avoids version conflicts). In the ``omni_isaac_sim`` build tree, rename or remove the prebundle folders so they are not loaded, e.g.:

   .. code-block:: bash

      cd /path/to/omni_isaac_sim/_build/linux-x86_64/release
      mv pip_prebundle/newton pip_prebundle/newton_bak   # or remove
      mv pip_prebundle/warp pip_prebundle/warp_bak     # if needed

   Replace ``/path/to/omni_isaac_sim`` with your clone path.

#. Install Newton via pip (required for the Newton Warp renderer). Either from the Git commit in ``source/isaaclab/setup.py``:

   .. code-block:: bash

      pip install "newton @ git+https://github.com/newton-physics/newton.git@35657fc"

5. Verify installation
----------------------

From the IsaacLab-Physx-Warp root with the conda env activated:

.. code-block:: bash

   python -c "import newton; import warp; print('Newton, Warp OK')"
   python -c "import isaacsim; print('Isaac Sim:', isaacsim.__file__)"
   python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"

Possible Newton / Warp conflict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim's build ships its own Newton and Warp under ``_build/.../pip_prebundle/``. If you skip the steps in section 4 (removing those folders and pip-installing Newton), the app may load the prebundle and you can see version or API conflicts. Follow the steps in section 4 to remove or rename the ``newton`` and ``warp`` folders in the Isaac Sim build and install Newton via pip.

6. Run training
---------------

Renderer selection
~~~~~~~~~~~~~~~~~~

Camera renderer is chosen via Hydra overrides. Supported values are ``isaac_rtx`` (default) and ``newton_warp``.
They are registered as Hydra config groups; the composed value is normalized to a string and then turned into a
concrete ``RendererCfg`` before env creation (see ``instantiate_renderer_cfg_in_env`` in ``isaaclab_tasks.utils.hydra``).

- **Kuka (manager-based)**: camera is under the scene, use ``env.scene.base_camera.renderer_type=newton_warp``
  (and pick a scene variant with ``env.scene=64x64rgb`` etc.).
- **Cartpole (direct)**: camera is top-level under env, use ``env.tiled_camera.renderer_type=newton_warp``.

If ``renderer_type`` is omitted, it defaults to ``isaac_rtx``. TiledCamera uses the instantiated ``renderer_cfg``;
if it is never set (e.g. no Hydra), it falls back to ``isaac_rtx``.

Example commands
~~~~~~~~~~~~~~~~

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
     env.scene=64x64rgb \
     env.scene.base_camera.renderer_type=newton_warp

.. code-block:: bash

   cd /path/to/IsaacLab-Physx-Warp
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate physx_dextrah
   export WANDB_USERNAME="${USERNAME:-$USER}"

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
     --enable_cameras \
     --headless \
     --num_envs=2048 \
     --max_iterations=32 \
     --logger=tensorboard \
     env.scene=64x64rgb \
     env.tiled_camera.renderer_type=isaac_rtx

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
| 4    | ``./isaaclab.sh -i``; remove/rename ``newton`` and ``warp`` in omni_isaac_sim ``pip_prebundle``; ``pip install`` Newton (git or local) |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 5    | Run the verification commands                                                                                                     |
+------+----------------------------------------------------------------------------------------------------------------------------------+
| 6    | Run ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...`` as above                                                 |
+------+----------------------------------------------------------------------------------------------------------------------------------+
