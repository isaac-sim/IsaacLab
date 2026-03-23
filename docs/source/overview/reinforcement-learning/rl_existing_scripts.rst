Reinforcement Learning Scripts
==============================

We provide wrappers to different reinforcement libraries. These wrappers convert the data
from the environments into the respective libraries function argument and return types.

Newton Backend
--------------

All training and play scripts support the **Newton physics backend** via the ``presets=newton``
Hydra override. Appending ``presets=newton`` to any command below switches the physics engine
from the default PhysX to Newton:

.. code:: bash

   # Generic pattern — works with any framework and task that supports Newton
   ./isaaclab.sh -p scripts/reinforcement_learning/<framework>/train.py \
       --task <task-name> --headless presets=newton

.. note::

   **Not all environments support the Newton backend yet.** Using ``presets=newton`` with an
   environment that has not been configured for Newton will raise an error at launch. See
   :doc:`/source/experimental-features/newton-physics-integration/index`
   for more details, and the :ref:`migrating-to-isaaclab-3-0`
   guide for how to add Newton support to your own environments.

Newton does not require Isaac Sim (kit-less mode). See :ref:`kitless-installation` for setup.


Observation-mode Presets
------------------------

Some environments support multiple observation modes — for example different camera
modalities or combinations of state and image observations — selectable via the same
``presets=`` mechanism.  Unlike physics-backend presets, **observation-mode presets
affect the checkpoint structure**, so you must pass the same preset to both the
training script and the play/evaluation script.  Using a different preset (or none)
at play time will cause a model-architecture mismatch when loading the checkpoint.

For example, ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` defaults to RGB + depth
+ segmentation inputs but can be switched to RGB-only (fewer input channels, lighter
model) with ``presets=rgb``:

.. code:: bash

   # Train with RGB-only observations
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --headless \
       --enable_cameras presets=rgb

   # Play — must use the same preset to load the matching checkpoint
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
       --task Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0 \
       --enable_cameras presets=rgb

Other available presets for this environment: ``albedo``,
``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl``,
``simple_shading_full_mdl``.  The ``depth`` preset is intended for
benchmarking only (see the environment's config for details).

Multiple presets can be combined with a comma when they do not conflict —
for instance to switch both the physics backend and the camera modality:

.. code:: bash

   presets=newton_renderer,rgb

See :doc:`/source/features/hydra` for the full preset system documentation.


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
            # run script for training
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Ant-v0 --headless
            # run script for training with Newton backend
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Ant-v0 --headless presets=newton
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rl-games)
            isaaclab.bat -i rl_games
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Ant-v0 --headless
            :: run script for training with Newton backend
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Ant-v0 --headless presets=newton
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            :: run script for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task Isaac-Ant-v0 --headless --video --video_length 200

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
            # run script for training
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-Franka-v0 --headless
            # run script for training with Newton backend
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-Franka-v0 --headless presets=newton
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Reach-Franka-v0 --headless
            :: run script for training with Newton backend
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Reach-Franka-v0 --headless presets=newton
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            :: run script for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

-  Training and distilling an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Velocity-Flat-Anymal-D-v0``:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for rsl-rl)
            ./isaaclab.sh -i rsl_rl
            # run script for rl training of the teacher agent
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
            # run script for rl training of the teacher agent with Newton backend
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless presets=newton
            # run script for distilling the teacher agent into a student agent
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
            # run script for playing the student with 64 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run script for rl training of the teacher agent
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
            :: run script for rl training of the teacher agent with Newton backend
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless presets=newton
            :: run script for distilling the teacher agent into a student agent
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
            :: run script for playing the student with 64 environments
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point

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
                     # run script for training
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless
                     # run script for training with Newton backend
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless presets=newton
                     # run script for playing with 32 environments
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     # run script for recording video of a trained agent (requires installing `ffmpeg`)
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

               .. tab-item:: :icon:`fa-brands fa-windows` Windows
                  :sync: windows

                  .. code:: batch

                     :: install python module (for skrl)
                     isaaclab.bat -i skrl
                     :: run script for training
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Reach-Franka-v0 --headless
                     :: run script for training with Newton backend
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Reach-Franka-v0 --headless presets=newton
                     :: run script for playing with 32 environments
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     :: run script for recording video of a trained agent (requires installing `ffmpeg`)
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

      .. tab-item:: JAX

         .. warning::

            It is recommended to `install JAX <https://jax.readthedocs.io/en/latest/installation.html>`_ manually before proceeding to install skrl and its dependencies, as JAX installs its CPU version by default.
            Visit the **skrl** `installation <https://skrl.readthedocs.io/en/latest/intro/installation.html>`_ page for more details.
            Note that JAX GPU support is only available on Linux.

            JAX 0.6.0 or higher (built on CuDNN v9.8) is incompatible with Isaac Lab's PyTorch 2.7 (built on CuDNN v9.7), and therefore not supported.
            To install a compatible version of JAX for CUDA 12 use ``pip install "jax[cuda12]<0.6.0" "flax<0.10.7"``, for example.

         .. hint::

            When using JAX its default behavior is to pre-allocate 75% of the GPU memory for its own computations. If you run into memory issues,
            you can set the ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` environment variable to disable this behavior, or reduce the amount of
            pre-allocated memory by setting ``export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`` which will allocate 50% of the GPU memory for JAX.
            Any value between 0 and 1 can be set, where 0 will allocate no memory for JAX and 1 will allocate 100% of the GPU memory for JAX.


         .. code:: bash

            # install python module (for skrl)
            ./isaaclab.sh -i skrl
            # install jax<0.6.0 for torch 2.7
            ./isaaclab.sh -p -m pip install "jax[cuda12]<0.6.0" "flax<0.10.7"
            # install skrl dependencies for JAX
            ./isaaclab.sh -p -m pip install skrl["jax"]
            # run script for training
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless --ml_framework jax
            # run script for training with Newton backend
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless --ml_framework jax presets=newton
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32  --ml_framework jax --checkpoint /PATH/TO/model.pt
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --headless --ml_framework jax --video --video_length 200

   - Training the multi-agent environment ``Isaac-Shadow-Hand-Over-Direct-v0`` with skrl:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # install python module (for skrl)
            ./isaaclab.sh -i skrl
            # run script for training with the MAPPO algorithm (IPPO is also supported)
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Shadow-Hand-Over-Direct-v0 --headless --algorithm MAPPO
            # run script for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Shadow-Hand-Over-Direct-v0 --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for skrl)
            isaaclab.bat -i skrl
            :: run script for training with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Shadow-Hand-Over-Direct-v0 --headless --algorithm MAPPO
            :: run script for playing with 32 environments with the MAPPO algorithm (IPPO is also supported)
            isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Shadow-Hand-Over-Direct-v0 --num_envs 32 --algorithm MAPPO --checkpoint /PATH/TO/model.pt

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
            # run script for training
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
            # run script for training with Newton backend
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless presets=newton
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for stable-baselines3)
            isaaclab.bat -i sb3
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
            :: run script for training with Newton backend
            isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless presets=newton
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            :: run script for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

RLinf
-----

`RLinf <https://github.com/RLinf/RLinf>`__ is a distributed RL infrastructure for fine-tuning
Vision-Language-Action (VLA) models such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`__.
It uses Ray for distributed computing and FSDP for model parallelism, enabling RL training of
large VLA models that don't fit on a single GPU.

-  Installation and setup:

   .. code:: bash

      # Step 1: Install RLinf and its dependencies (from isaaclab_contrib)
      pip install -e "source/isaaclab_contrib[rlinf]"

      # Step 2: Clone and install Isaac-GR00T (for VLA model support)
      cd scripts/reinforcement_learning/rlinf
      git clone https://github.com/NVIDIA/Isaac-GR00T.git
      pip install -e Isaac-GR00T/.[base] --no-deps

      # Step 3: Install flash-attn (must be built against the correct PyTorch)
      pip install --no-build-isolation flash-attn==2.7.1.post4

-  Training a VLA agent with RLinf:

   .. code:: bash

      # Train with default config (assemble trocar task with GR00T)
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/train.py

      # Train with a specific config
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/train.py \
          --config_name isaaclab_ppo_gr00t_assemble_trocar

      # Train with task override and custom settings
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/train.py \
          --config_name isaaclab_ppo_gr00t_assemble_trocar \
          --task Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0 \
          --num_envs 64 --max_epochs 1000

      # List available tasks
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/train.py --list_tasks

-  Evaluating a trained VLA agent:

   .. code:: bash

      # Evaluate a trained checkpoint
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/play.py \
          --model_path /path/to/checkpoint

      # Evaluate with video recording
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/play.py \
          --model_path /path/to/checkpoint --video

      # Evaluate with specific number of environments and episodes
      ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/play.py \
          --model_path /path/to/checkpoint --num_envs 8 --num_episodes 10


All the scripts above log the training progress to `Tensorboard`_ in the ``logs`` directory in the root of
the repository. The logs directory follows the pattern ``logs/<library>/<task>/<date-time>``, where ``<library>``
is the name of the learning framework, ``<task>`` is the task name, and ``<date-time>`` is the timestamp at
which the training script was executed.

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
