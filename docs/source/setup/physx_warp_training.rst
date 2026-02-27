.. _physx-warp-training:

PhysX + Warp Training (Isaac Sim 6.0)
=====================================

This document gives step-by-step instructions to set up the environment and run RL training (e.g., Cartpole RGB Camera, Direct) using PhysX simulation and the Isaac RTX or Newton Warp renderer.

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

Camera renderer is chosen via the Hydra ``renderer`` config group. Supported values are ``isaac_rtx`` (default) and ``newton_warp``.
The selected preset is applied to all cameras; each camera's ``data_types`` are set at use time (see ``isaaclab_tasks.utils.render_config_store`` and ``instantiate_renderer_cfg_in_env`` in ``isaaclab_tasks.utils.hydra``).

Use the top-level override ``renderer=isaac_rtx`` or ``renderer=newton_warp``. If omitted, it defaults to ``isaac_rtx``.

Example command
~~~~~~~~~~~~~~~

From the **IsaacLab-Physx-Warp** repo root, with the conda env activated:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
     --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
     --enable_cameras \
     --headless \
     renderer=isaac_rtx

Use ``renderer=newton_warp`` to use the Newton Warp renderer instead. For longer training runs, increase ``--num_envs`` and ``--max_iterations`` (e.g. ``--num_envs=2048 --max_iterations=32``); redirect stdout/stderr if desired, e.g. ``2>&1 | tee train.log``.

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
| 6    | Run ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --enable_cameras --headless ... renderer=isaac_rtx`` as above |
+------+----------------------------------------------------------------------------------------------------------------------------------+
