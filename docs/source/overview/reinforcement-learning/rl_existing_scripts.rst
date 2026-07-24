Reinforcement Learning Workflows
================================

We provide wrappers to different reinforcement libraries. These wrappers convert the data
from the environments into the respective libraries function argument and return types.

Preset Selectors
----------------

All training and play commands accept ``physics=NAME``, ``renderer=NAME``, and
``presets=NAME[,NAME,...]`` tokens appended directly to the command (no leading dashes).
See :doc:`/source/features/hydra` for all available names and how the selectors work.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # Switch physics backend
         uv run isaaclab train --rl_library <library> \
             --task <task-name> physics=newton_mjwarp

         # Switch renderer (camera environments)
         uv run isaaclab train --rl_library rsl_rl \
             --task Isaac-Cartpole-Camera-Direct \
       renderer=newton_renderer

         # Combine selectors freely
         uv run isaaclab train --rl_library rsl_rl \
             --task Isaac-Cartpole-Camera-Direct \
       physics=newton_mjwarp renderer=newton_renderer presets=rgb

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

         # Switch physics backend
         ./isaaclab.sh train --rl_library <library> \
             --task <task-name> physics=newton_mjwarp

         # Switch renderer (camera environments)
         ./isaaclab.sh train --rl_library rsl_rl \
             --task Isaac-Cartpole-Camera-Direct \
       renderer=newton_renderer

         # Combine selectors freely
         ./isaaclab.sh train --rl_library rsl_rl \
             --task Isaac-Cartpole-Camera-Direct \
       physics=newton_mjwarp renderer=newton_renderer presets=rgb

.. note::

   **Not all environments support the Newton backend yet.** Using ``physics=newton_mjwarp`` with an
   environment that has not been configured for Newton will raise an error at launch. See
   :doc:`/source/overview/core-concepts/physical-backends/newton/index`
   for more details, and the :ref:`migrating-to-isaaclab-3-0`
   guide for how to add Newton support to your own environments.

Newton does not require Isaac Sim. See :ref:`installation-legacy-installer` for the legacy installer setup.

Programmatic use
----------------

Downstream applications can use the same dispatcher without constructing an
``argparse.Namespace``:

.. code:: python

   from isaaclab_rl import TrainingRequest, train

   train(TrainingRequest(backend="rsl_rl", task="Isaac-Cartpole", max_iterations=100))

Use :class:`~isaaclab_rl.entrypoints.PlaybackRequest` and :func:`~isaaclab_rl.entrypoints.play` for
playback. Pass backend-specific options through ``backend_args`` and Hydra or
preset selectors through ``hydra_args``.


Observation-mode Presets
------------------------

Some environments support multiple observation modes selectable via ``presets=``.
Unlike physics or renderer presets, **observation-mode presets affect the checkpoint
structure**: you must pass the same preset to both the training and play commands.
Using a different preset (or none) at play time will cause a model-architecture
mismatch when loading the checkpoint.

For example, ``Isaac-Reorient-Cube-Shadow-Camera-Direct`` defaults to RGB + depth
+ segmentation inputs but can be switched to RGB-only with ``presets=rgb``:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # Train with RGB-only observations
         uv run isaaclab train --rl_library rsl_rl \
             --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
             --enable_cameras presets=rgb

         # Play — must use the same preset to load the matching checkpoint
         uv run isaaclab play --rl_library rsl_rl \
             --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
             --enable_cameras presets=rgb

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

         # Train with RGB-only observations
         ./isaaclab.sh train --rl_library rsl_rl \
             --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
             --enable_cameras presets=rgb

         # Play — must use the same preset to load the matching checkpoint
         ./isaaclab.sh play --rl_library rsl_rl \
             --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
             --enable_cameras presets=rgb

Other available presets for this environment: ``albedo``,
``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl``,
``simple_shading_full_mdl``.  The ``depth`` preset is intended for
benchmarking only (see the environment's config for details).

During training, image-like scene sensor outputs from camera tasks can be saved with
``--capture_env_sensors``. See :doc:`/source/how-to/capture_sensor_frames` for the full capture
schedule and output format details.

Agent compatibility
~~~~~~~~~~~~~~~~~~~

An observation or action preset can require a different policy network. Task-specific
``--help`` lists all registered ``--agent`` values for the selected RL library. Tasks
that require a particular pairing additionally show ``compatible presets`` beneath
each agent.

For example, the manager-based Cartpole camera task uses a CNN agent for rendered
images and an MLP agent for frozen vision features:

.. code-block:: bash

   # Raw RGB images: the default agent is compatible
   ./isaaclab.sh train --rl_library rl_games \
       --task Isaac-Cartpole-Camera presets=rgb

   # ResNet18 features: select the feature agent explicitly
   ./isaaclab.sh train --rl_library rl_games \
       --task Isaac-Cartpole-Camera \
       --agent rl_games_feature_cfg_entry_point presets=resnet18

The contributed Cartpole showcase tasks likewise pair each non-default
``presets=<observation>_<action>`` selection with
``--agent skrl_<observation>_<action>_cfg_entry_point``. Other alternate agent
configs, such as RSL-RL symmetry or recurrent policies and skrl's AMP/IPPO/MAPPO
algorithms, are algorithm choices rather than preset requirements.


RL-Games
--------

.. attention::

  When using RL-Games with the Ray workflow for distributed training or hyperparameter tuning,
  please be aware that due to security risks associated with Ray, this workflow is not intended
  for use outside of a strictly controlled network environment.

-  Training an agent with
   `RL-Games <https://github.com/Denys88/rl_games>`__ on ``Isaac-Ant``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # install python module (for rl-games)
                  ./isaaclab.sh -i rl_games
                  # run command for training
                  uv run isaaclab train --rl_library rl_games --task Isaac-Ant
                  # run command for training with Newton backend
                  uv run isaaclab train --rl_library rl_games --task Isaac-Ant physics=newton_mjwarp
                  # run command for playing with 32 environments
                  uv run isaaclab play --rl_library rl_games --task Isaac-Ant --num_envs 32 --checkpoint /PATH/TO/model.pth
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  uv run isaaclab play --rl_library rl_games --task Isaac-Ant --video --video_length 200

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # install python module (for rl-games)
                  ./isaaclab.sh -i rl_games
                  # run command for training
                  ./isaaclab.sh train --rl_library rl_games --task Isaac-Ant
                  # run command for training with Newton backend
                  ./isaaclab.sh train --rl_library rl_games --task Isaac-Ant physics=newton_mjwarp
                  # run command for playing with 32 environments
                  ./isaaclab.sh play --rl_library rl_games --task Isaac-Ant --num_envs 32 --checkpoint /PATH/TO/model.pth
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  ./isaaclab.sh play --rl_library rl_games --task Isaac-Ant --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rl-games)
            isaaclab.bat -i rl_games
            :: run command for training
            isaaclab.bat train --rl_library rl_games --task Isaac-Ant
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library rl_games --task Isaac-Ant physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library rl_games --task Isaac-Ant --num_envs 32 --checkpoint /PATH/TO/model.pth
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library rl_games --task Isaac-Ant --video --video_length 200

RSL-RL
------

-  Training an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Reach-Franka``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # install python module (for rsl-rl)
                  ./isaaclab.sh -i rsl_rl
                  # run command for training
                  uv run isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka
                  # run command for training with Newton backend
                  uv run isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp
                  # run command for playing with 32 environments
                  uv run isaaclab play --rl_library rsl_rl --task Isaac-Reach-Franka --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  uv run isaaclab play --rl_library rsl_rl --task Isaac-Reach-Franka --video --video_length 200

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # install python module (for rsl-rl)
                  ./isaaclab.sh -i rsl_rl
                  # run command for training
                  ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Reach-Franka
                  # run command for training with Newton backend
                  ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp
                  # run command for playing with 32 environments
                  ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Reach-Franka --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Reach-Franka --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run command for training
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Reach-Franka
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Reach-Franka --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Reach-Franka --video --video_length 200

-  Training and distilling an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Velocity-Flat-AnymalD``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # install python module (for rsl-rl)
                  ./isaaclab.sh -i rsl_rl
                  # run command for rl training of the teacher agent
                  uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD
                  # run command for rl training of the teacher agent with Newton backend
                  uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD physics=newton_mjwarp
                  # run command for distilling the teacher agent into a student agent
                  uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
                  # run command for playing the student with 64 environments
                  uv run isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # install python module (for rsl-rl)
                  ./isaaclab.sh -i rsl_rl
                  # run command for rl training of the teacher agent
                  ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD
                  # run command for rl training of the teacher agent with Newton backend
                  ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD physics=newton_mjwarp
                  # run command for distilling the teacher agent into a student agent
                  ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
                  # run command for playing the student with 64 environments
                  ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run command for rl training of the teacher agent
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD
            :: run command for rl training of the teacher agent with Newton backend
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD physics=newton_mjwarp
            :: run command for distilling the teacher agent into a student agent
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
            :: run command for playing the student with 64 environments
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

SKRL
----

-  Training an agent with
   `SKRL <https://skrl.readthedocs.io>`__ on ``Isaac-Reach-Franka``:

   .. tab-set::

      .. tab-item:: PyTorch

            .. tab-set::
               :sync-group: os

               .. tab-item:: :icon:`fa-brands fa-linux` Linux
                  :sync: linux

                  .. tab-set::

                     .. tab-item:: uv (Recommended)

                        .. code:: bash

                           # install python module (for skrl)
                           ./isaaclab.sh -i skrl
                           # run command for training
                           uv run isaaclab train --rl_library skrl --task Isaac-Reach-Franka
                           # run command for training with Newton backend
                           uv run isaaclab train --rl_library skrl --task Isaac-Reach-Franka physics=newton_mjwarp
                           # run command for playing with 32 environments
                           uv run isaaclab play --rl_library skrl --task Isaac-Reach-Franka --num_envs 32 --checkpoint /PATH/TO/model.pt
                           # run command for recording video of a trained agent (requires installing `ffmpeg`)
                           uv run isaaclab play --rl_library skrl --task Isaac-Reach-Franka --video --video_length 200

                     .. tab-item:: isaaclab.sh / isaaclab.bat

                        .. code:: bash

                           # install python module (for skrl)
                           ./isaaclab.sh -i skrl
                           # run command for training
                           ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka
                           # run command for training with Newton backend
                           ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka physics=newton_mjwarp
                           # run command for playing with 32 environments
                           ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka --num_envs 32 --checkpoint /PATH/TO/model.pt
                           # run command for recording video of a trained agent (requires installing `ffmpeg`)
                           ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka --video --video_length 200

               .. tab-item:: :icon:`fa-brands fa-windows` Windows
                  :sync: windows

                  .. code:: batch

                     :: install python module (for skrl)
                     isaaclab.bat -i skrl
                     :: run command for training
                     isaaclab.bat train --rl_library skrl --task Isaac-Reach-Franka
                     :: run command for training with Newton backend
                     isaaclab.bat train --rl_library skrl --task Isaac-Reach-Franka physics=newton_mjwarp
                     :: run command for playing with 32 environments
                     isaaclab.bat play --rl_library skrl --task Isaac-Reach-Franka --num_envs 32 --checkpoint /PATH/TO/model.pt
                     :: run command for recording video of a trained agent (requires installing `ffmpeg`)
                     isaaclab.bat play --rl_library skrl --task Isaac-Reach-Franka --video --video_length 200

      .. tab-item:: JAX

         .. warning::

            It is recommended to `install JAX <https://docs.jax.dev/en/latest/installation.html>`_ manually before proceeding to install skrl and its dependencies, as JAX installs its CPU version by default.
            Visit the **skrl** `installation <https://skrl.readthedocs.io/en/latest/intro/installation.html>`_ page for more details.
            Note that JAX GPU support is only available on Linux x86_64 and Linux aarch64.
            Use the CUDA 12 wheel on Linux x86_64 and the CUDA 13 wheel on Linux aarch64 systems such as DGX Spark.

         .. hint::

            When using JAX its default behavior is to pre-allocate 75% of the GPU memory for its own computations. If you run into memory issues,
            you can set the ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` environment variable to disable this behavior, or reduce the amount of
            pre-allocated memory by setting ``export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`` which will allocate 50% of the GPU memory for JAX.
            Any value between 0 and 1 can be set, where 0 will allocate no memory for JAX and 1 will allocate 100% of the GPU memory for JAX.

         .. tab-set::
            :sync-group: jax-cuda

            .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64, CUDA 12)
               :sync: linux-x86_64-jax-cuda12

               .. tab-set::

                  .. tab-item:: uv (Recommended)

                     .. code:: bash

                        # install python module (for skrl)
                        ./isaaclab.sh -i skrl
                        # install JAX for CUDA 12
                        uv pip install -U "jax[cuda12]"
                        # install skrl dependencies for JAX
                        uv pip install "skrl[jax]"

                  .. tab-item:: isaaclab.sh / isaaclab.bat

                     .. code:: bash

                        # install python module (for skrl)
                        ./isaaclab.sh -i skrl
                        # install JAX for CUDA 12
                        ./isaaclab.sh -p -m pip install -U "jax[cuda12]"
                        # install skrl dependencies for JAX
                        ./isaaclab.sh -p -m pip install "skrl[jax]"

            .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64, CUDA 13)
               :sync: linux-aarch64-jax-cuda13

               .. tab-set::

                  .. tab-item:: uv (Recommended)

                     .. code:: bash

                        # install python module (for skrl)
                        ./isaaclab.sh -i skrl
                        # install JAX for CUDA 13
                        uv pip install -U "jax[cuda13]"
                        # install skrl dependencies for JAX
                        uv pip install "skrl[jax]"

                  .. tab-item:: isaaclab.sh / isaaclab.bat

                     .. code:: bash

                        # install python module (for skrl)
                        ./isaaclab.sh -i skrl
                        # install JAX for CUDA 13
                        ./isaaclab.sh -p -m pip install -U "jax[cuda13]"
                        # install skrl dependencies for JAX
                        ./isaaclab.sh -p -m pip install "skrl[jax]"

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # run command for training
                  uv run isaaclab train --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax
                  # run command for training with Newton backend
                  uv run isaaclab train --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax presets=newton_mjwarp
                  # run command for playing with 32 environments
                  uv run isaaclab play --rl_library skrl --task Isaac-Reach-Franka --num_envs 32  --ml_framework jax --checkpoint /PATH/TO/model.pt
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  uv run isaaclab play --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax --video --video_length 200

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # run command for training
                  ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax
                  # run command for training with Newton backend
                  ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax presets=newton_mjwarp
                  # run command for playing with 32 environments
                  ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka --num_envs 32  --ml_framework jax --checkpoint /PATH/TO/model.pt
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka --ml_framework jax --video --video_length 200

   - Training the multi-agent environment ``Isaac-Shadow-Handover-Direct`` with skrl:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # install python module (for skrl)
                  ./isaaclab.sh -i skrl
                  # run command for training with the MAPPO algorithm (IPPO is also supported)
                  uv run isaaclab train --rl_library skrl --task Isaac-Shadow-Handover-Direct --algorithm MAPPO
                  # run command for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
                  uv run isaaclab play --rl_library skrl --task Isaac-Shadow-Handover-Direct --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # install python module (for skrl)
                  ./isaaclab.sh -i skrl
                  # run command for training with the MAPPO algorithm (IPPO is also supported)
                  ./isaaclab.sh train --rl_library skrl --task Isaac-Shadow-Handover-Direct --algorithm MAPPO
                  # run command for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
                  ./isaaclab.sh play --rl_library skrl --task Isaac-Shadow-Handover-Direct --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for skrl)
            isaaclab.bat -i skrl
            :: run command for training with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat train --rl_library skrl --task Isaac-Shadow-Handover-Direct --algorithm MAPPO
            :: run command for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat play --rl_library skrl --task Isaac-Shadow-Handover-Direct --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

Stable-Baselines3
-----------------

-  Training an agent with
   `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/index.html>`__
   on ``IsaacContrib-Velocity-Flat-UnitreeA1``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  # install python module (for stable-baselines3)
                  ./isaaclab.sh -i sb3
                  # run command for training
                  uv run isaaclab train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1
                  # run command for training with Newton backend
                  uv run isaaclab train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 physics=newton_mjwarp
                  # run command for playing with 32 environments
                  uv run isaaclab play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --num_envs 32 --checkpoint /PATH/TO/model.zip
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  uv run isaaclab play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --video --video_length 200

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  # install python module (for stable-baselines3)
                  ./isaaclab.sh -i sb3
                  # run command for training
                  ./isaaclab.sh train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1
                  # run command for training with Newton backend
                  ./isaaclab.sh train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 physics=newton_mjwarp
                  # run command for playing with 32 environments
                  ./isaaclab.sh play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --num_envs 32 --checkpoint /PATH/TO/model.zip
                  # run command for recording video of a trained agent (requires installing `ffmpeg`)
                  ./isaaclab.sh play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for stable-baselines3)
            isaaclab.bat -i sb3
            :: run command for training
            isaaclab.bat train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --num_envs 32 --checkpoint /PATH/TO/model.zip
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library sb3 --task IsaacContrib-Velocity-Flat-UnitreeA1 --video --video_length 200

RLinf
-----

`RLinf <https://github.com/RLinf/RLinf>`__ is a distributed RL infrastructure for fine-tuning
Vision-Language-Action (VLA) models such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`__.
It uses Ray for distributed computing and FSDP for model parallelism, enabling RL training of
large VLA models that don't fit on a single GPU.

For installation instructions, see :ref:`rlinf-post-training`.

-  Training a VLA agent with RLinf:

   .. tab-set::

      .. tab-item:: uv (Recommended)

         .. code:: bash

            # Train with a specific config
            uv run isaaclab train --rl_library rlinf \
                --config_name isaaclab_ppo_gr00t_assemble_trocar \
                --model_path /path/to/checkpoint

      .. tab-item:: isaaclab.sh / isaaclab.bat

         .. code:: bash

            # Train with a specific config
            ./isaaclab.sh train --rl_library rlinf \
                --config_name isaaclab_ppo_gr00t_assemble_trocar \
                --model_path /path/to/checkpoint

-  Evaluating a trained VLA agent:

   .. tab-set::

      .. tab-item:: uv (Recommended)

         .. code:: bash

            # Evaluate with video recording
            uv run isaaclab play --rl_library rlinf \
                --config_name isaaclab_ppo_gr00t_assemble_trocar \
                --model_path /path/to/checkpoint --video


      .. tab-item:: isaaclab.sh / isaaclab.bat

         .. code:: bash

            # Evaluate with video recording
            ./isaaclab.sh play --rl_library rlinf \
                --config_name isaaclab_ppo_gr00t_assemble_trocar \
                --model_path /path/to/checkpoint --video


All the commands above log the training progress to `Tensorboard`_ in the ``logs`` directory in the root of
the repository. The logs directory follows the pattern ``logs/<library>/<task>/<date-time>``, where ``<library>``
is the name of the learning framework, ``<task>`` is the task name, and ``<date-time>`` is the timestamp at
which the training command was executed.

New training runs also store a ``run.json`` manifest in their run directory. This manifest allows the unified
``train`` and ``play`` commands to resolve a checkpoint without copying its path manually. Pass
``--checkpoint latest`` to select the highest-step checkpoint from the newest compatible run:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole --checkpoint latest

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

         ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole --checkpoint latest

Pass ``--checkpoint best`` to prefer the library-specific best or final checkpoint. For libraries without a
distinct best checkpoint, ``best`` resolves to the same checkpoint as ``latest``. These selectors are supported
by RL-Games, RSL-RL, skrl, and Stable-Baselines3. RSL-RL training resume continues to require ``--resume``.

To view the logs, run:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code:: bash

               # execute from the root directory of the repository
               uv run python -m tensorboard.main --logdir=logs

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code:: bash

               # execute from the root directory of the repository
               ./isaaclab.sh -p -m tensorboard.main --logdir=logs

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: execute from the root directory of the repository
         isaaclab.bat -p -m tensorboard.main --logdir=logs

.. _Tensorboard: https://www.tensorflow.org/tensorboard
