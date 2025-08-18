Reinforcement Learning Scripts
==============================

We provide wrappers to different reinforcement libraries. These wrappers convert the data
from the environments into the respective libraries function argument and return types.


RL-Games
--------

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
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            # run script for playing a pre-trained checkpoint with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --use_pretrained_checkpoint
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rl-games)
            isaaclab.bat -i rl_games
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Ant-v0 --headless
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
            :: run script for playing a pre-trained checkpoint with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py --task Isaac-Ant-v0 --num_envs 32 --use_pretrained_checkpoint
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
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            # run script for playing a pre-trained checkpoint with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --use_pretrained_checkpoint
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for rsl-rl)
            isaaclab.bat -i rsl_rl
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Reach-Franka-v0 --headless
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint /PATH/TO/model.pt
            :: run script for playing a pre-trained checkpoint with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --use_pretrained_checkpoint
            :: run script for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

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
                     # run script for playing with 32 environments
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     # run script for playing a pre-trained checkpoint with 32 environments
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --use_pretrained_checkpoint
                     # run script for recording video of a trained agent (requires installing `ffmpeg`)
                     ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

               .. tab-item:: :icon:`fa-brands fa-windows` Windows
                  :sync: windows

                  .. code:: batch

                     :: install python module (for skrl)
                     isaaclab.bat -i skrl
                     :: run script for training
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Reach-Franka-v0 --headless
                     :: run script for playing with 32 environments
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt
                     :: run script for playing a pre-trained checkpoint with 32 environments
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --use_pretrained_checkpoint
                     :: run script for recording video of a trained agent (requires installing `ffmpeg`)
                     isaaclab.bat -p scripts\reinforcement_learning\skrl\play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

      .. tab-item:: JAX

         .. warning::

            It is recommended to `install JAX <https://jax.readthedocs.io/en/latest/installation.html>`_ manually before proceeding to install skrl and its dependencies, as JAX installs its CPU version by default.
            Visit the **skrl** `installation <https://skrl.readthedocs.io/en/latest/intro/installation.html>`_ page for more details.
            Note that JAX GPU support is only available on Linux.

            JAX 0.6.0 or higher (built on CuDNN v9.8) is incompatible with Isaac Lab's PyTorch 2.7 (built on CuDNN v9.7), and therefore not supported.
            To install a compatible version of JAX for CUDA 12 use ``pip install "jax[cuda12]<0.6.0"``, for example.

         .. code:: bash

            # install python module (for skrl)
            ./isaaclab.sh -i skrl
            # install skrl dependencies for JAX
            ./isaaclab.sh -p -m pip install skrl["jax"]
            # install jax<0.6.0 for torch 2.7
            ./isaaclab.sh -p -m pip install "jax[cuda12]<0.6.0"
            # run script for training
            ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless --ml_framework jax
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
            # run script for playing with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            # run script for playing a pre-trained checkpoint with 32 environments
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --use_pretrained_checkpoint
            # run script for recording video of a trained agent (requires installing `ffmpeg`)
            ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: install python module (for stable-baselines3)
            isaaclab.bat -i sb3
            :: run script for training
            isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
            :: run script for playing with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip
            :: run script for playing a pre-trained checkpoint with 32 environments
            isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 32 --use_pretrained_checkpoint
            :: run script for recording video of a trained agent (requires installing `ffmpeg`)
            isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video --video_length 200

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
