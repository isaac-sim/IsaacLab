Reinforcement Learning Workflows
================================

We provide wrappers to different reinforcement libraries. These wrappers convert the data
from the environments into the respective libraries function argument and return types.

Preset Selectors
----------------

All training and play commands accept ``physics=NAME``, ``renderer=NAME``, and
``presets=NAME[,NAME,...]`` tokens appended directly to the command (no leading dashes).
See :doc:`/source/features/hydra` for all available names and how the selectors work.

.. code:: bash

   # Switch physics backend
   ./isaaclab.sh train --rl_library <library> \
       --task <task-name> --headless physics=newton_mjwarp

   # Switch renderer (camera environments)
   ./isaaclab.sh train --rl_library rsl_rl \
       --task Isaac-Cartpole-Camera-Direct --headless \
       --enable_cameras renderer=newton_renderer

   # Combine selectors freely
   ./isaaclab.sh train --rl_library rsl_rl \
       --task Isaac-Cartpole-Camera-Direct --headless \
       --enable_cameras physics=newton_mjwarp renderer=newton_renderer presets=rgb

.. note::

   **Not all environments support the Newton backend yet.** Using ``physics=newton_mjwarp`` with an
   environment that has not been configured for Newton will raise an error at launch. See
   :doc:`/source/overview/core-concepts/physical-backends/newton/index`
   for more details, and the :ref:`migrating-to-isaaclab-3-0`
   guide for how to add Newton support to your own environments.

Newton does not require Isaac Sim (kit-less mode). See :ref:`kitless-installation` for setup.


Observation-mode Presets
------------------------

Some environments support multiple observation modes selectable via ``presets=``.
Unlike physics or renderer presets, **observation-mode presets affect the checkpoint
structure**: you must pass the same preset to both the training and play commands.
Using a different preset (or none) at play time will cause a model-architecture
mismatch when loading the checkpoint.

For example, ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` defaults to RGB + depth
+ segmentation inputs but can be switched to RGB-only with ``presets=rgb``:

.. code:: bash

   # Train with RGB-only observations
   ./isaaclab.sh train --rl_library rsl_rl \
       --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --headless \
       --enable_cameras presets=rgb

   # Play — must use the same preset to load the matching checkpoint
   ./isaaclab.sh play --rl_library rsl_rl \
       --task Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0 \
       --enable_cameras presets=rgb

Other available presets for this environment: ``albedo``,
``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl``,
``simple_shading_full_mdl``.  The ``depth`` preset is intended for
benchmarking only (see the environment's config for details).


RL-Games
--------

.. attention::

  When using RL-Games with the Ray workflow for distributed training or hyperparameter tuning,
  please be aware that due to security risks associated with Ray, this workflow is not intended
  for use outside of a strictly controlled network environment.

-  Training an agent with
   `RL-Games <https://github.com/Denys88/rl_games>`__ on ``Isaac-Ant-v0``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for rl-games)
            ./isaaclab.sh -i rl_games
            # run command for training
            ./isaaclab.sh train --rl_library rl_games --task Isaac-Ant-v0 --headless
            # run command for training with Newton backend
            ./isaaclab.sh train --rl_library rl_games --task Isaac-Ant-v0 --headless physics=newton_mjwarp
            # run command for playing with 32 environments
            ./isaaclab.sh play --rl_library rl_games --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            # run command for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh play --rl_library rl_games --task Isaac-Ant-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rl-games)
            isaaclab.bat -i rl_games
            :: run command for training
            isaaclab.bat train --rl_library rl_games --task Isaac-Ant-v0 --headless
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library rl_games --task Isaac-Ant-v0 --headless physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library rl_games --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library rl_games --task Isaac-Ant-v0 --headless --video --video_length 200

RSL-RL
------

-  Training an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Reach-Franka-v0``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for rsl-rl)
            ./isaaclab.sh -i rsl_rl
            # run command for training
            ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless
            # run command for training with Newton backend
            ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless physics=newton_mjwarp
            # run command for playing with 32 environments
            ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            # run command for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run command for training
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

-  Training and distilling an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Velocity-Flat-Anymal-D-v0``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for rsl-rl)
            ./isaaclab.sh -i rsl_rl
            # run command for rl training of the teacher agent
            ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
            # run command for rl training of the teacher agent with Newton backend
            ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless physics=newton_mjwarp
            # run command for distilling the teacher agent into a student agent
            ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
            # run command for playing the student with 64 environments
            ./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run command for rl training of the teacher agent
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
            :: run command for rl training of the teacher agent with Newton backend
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless physics=newton_mjwarp
            :: run command for distilling the teacher agent into a student agent
            isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
            :: run command for playing the student with 64 environments
            isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

SKRL
----

-  Training an agent with
   `SKRL <https://skrl.readthedocs.io>`__ on ``Isaac-Reach-Franka-v0``:

   .. tab-set::

      .. tab-item:: PyTorch

            .. tab-set::
               :sync-group: os

               .. tab-item:: :icon:`fa-brands fa-linux` Linux
                  :sync: linux

                  .. code:: bash

                     # install python module (for skrl)
                     ./isaaclab.sh -i skrl
                     # run command for training
                     ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless
                     # run command for training with Newton backend
                     ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless physics=newton_mjwarp
                     # run command for playing with 32 environments
                     ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     # run command for recording video of a trained agent (requires installing `ffmpeg`)
                     ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

               .. tab-item:: :icon:`fa-brands fa-windows` Windows
                  :sync: windows

                  .. code:: batch

                     :: install python module (for skrl)
                     isaaclab.bat -i skrl
                     :: run command for training
                     isaaclab.bat train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless
                     :: run command for training with Newton backend
                     isaaclab.bat train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless physics=newton_mjwarp
                     :: run command for playing with 32 environments
                     isaaclab.bat play --rl_library skrl --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     :: run command for recording video of a trained agent (requires installing `ffmpeg`)
                     isaaclab.bat play --rl_library skrl --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

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

               .. code:: bash

                  # install python module (for skrl)
                  ./isaaclab.sh -i skrl
                  # install JAX for CUDA 12
                  ./isaaclab.sh -p -m pip install -U "jax[cuda12]"
                  # install skrl dependencies for JAX
                  ./isaaclab.sh -p -m pip install "skrl[jax]"

            .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64, CUDA 13)
               :sync: linux-aarch64-jax-cuda13

               .. code:: bash

                  # install python module (for skrl)
                  ./isaaclab.sh -i skrl
                  # install JAX for CUDA 13
                  ./isaaclab.sh -p -m pip install -U "jax[cuda13]"
                  # install skrl dependencies for JAX
                  ./isaaclab.sh -p -m pip install "skrl[jax]"

         .. code:: bash

            # run command for training
            ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless --ml_framework jax
            # run command for training with Newton backend
            ./isaaclab.sh train --rl_library skrl --task Isaac-Reach-Franka-v0 --headless --ml_framework jax presets=newton_mjwarp
            # run command for playing with 32 environments
            ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka-v0 --num_envs 32  --ml_framework jax --checkpoint /PATH/TO/model.pt
            # run command for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh play --rl_library skrl --task Isaac-Reach-Franka-v0 --headless --ml_framework jax --video --video_length 200

   - Training the multi-agent environment ``Isaac-Shadow-Hand-Over-Direct-v0`` with skrl:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for skrl)
            ./isaaclab.sh -i skrl
            # run command for training with the MAPPO algorithm (IPPO is also supported)
            ./isaaclab.sh train --rl_library skrl --task Isaac-Shadow-Hand-Over-Direct-v0 --headless --algorithm MAPPO
            # run command for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
            ./isaaclab.sh play --rl_library skrl --task Isaac-Shadow-Hand-Over-Direct-v0 --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for skrl)
            isaaclab.bat -i skrl
            :: run command for training with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat train --rl_library skrl --task Isaac-Shadow-Hand-Over-Direct-v0 --headless --algorithm MAPPO
            :: run command for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat play --rl_library skrl --task Isaac-Shadow-Hand-Over-Direct-v0 --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

Stable-Baselines3
-----------------

-  Training an agent with
   `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/index.html>`__
   on ``Isaac-Velocity-Flat-Unitree-A1-v0``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for stable-baselines3)
            ./isaaclab.sh -i sb3
            # run command for training
            ./isaaclab.sh train --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
            # run command for training with Newton backend
            ./isaaclab.sh train --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless physics=newton_mjwarp
            # run command for playing with 32 environments
            ./isaaclab.sh play --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            # run command for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh play --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for stable-baselines3)
            isaaclab.bat -i sb3
            :: run command for training
            isaaclab.bat train --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
            :: run command for training with Newton backend
            isaaclab.bat train --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless physics=newton_mjwarp
            :: run command for playing with 32 environments
            isaaclab.bat play --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            :: run command for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat play --rl_library sb3 --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

RLinf
-----

`RLinf <https://github.com/RLinf/RLinf>`__ is a distributed RL infrastructure for fine-tuning
Vision-Language-Action (VLA) models such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`__.
It uses Ray for distributed computing and FSDP for model parallelism, enabling RL training of
large VLA models that don't fit on a single GPU.

For installation instructions, see :ref:`rlinf-post-training`.

-  Training a VLA agent with RLinf:

   .. code:: bash

      # Train with a specific config
      ./isaaclab.sh train --rl_library rlinf \
          --config_name isaaclab_ppo_gr00t_assemble_trocar \
          --model_path /path/to/checkpoint

-  Evaluating a trained VLA agent:

   .. code:: bash

      # Evaluate with video recording
      ./isaaclab.sh play --rl_library rlinf \
          --config_name isaaclab_ppo_gr00t_assemble_trocar \
          --model_path /path/to/checkpoint --video


All the commands above log the training progress to `Tensorboard`_ in the ``logs`` directory in the root of
the repository. The logs directory follows the pattern ``logs/<library>/<task>/<date-time>``, where ``<library>``
is the name of the learning framework, ``<task>`` is the task name, and ``<date-time>`` is the timestamp at
which the training command was executed.

To view the logs, run:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # execute from the root directory of the repository
         ./isaaclab.sh -p -m tensorboard.main --logdir=logs

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: execute from the root directory of the repository
         isaaclab.bat -p -m tensorboard.main --logdir=logs

.. _Tensorboard: https://www.tensorflow.org/tensorboard
