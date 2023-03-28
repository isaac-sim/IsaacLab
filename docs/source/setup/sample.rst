Running existing scripts
========================

API Demos
---------

The main core interface extension in Orbit ``omni.isaac.orbit`` provides
the main modules for actuators, objects, robots and sensors. We provide
a list of demo scripts. These showcase how to use the provided interfaces
within a code in a minimal way.

A few quick demo scripts to run and checkout:

-  Spawn different quadrupeds, visualize feet markers, and make
   robots stand using position commands:

   .. code:: bash

      ./orbit.sh -p source/standalone/demo/play_quadrupeds.py

-  Spawn multiple Franka arms and apply random position commands:

   .. code:: bash

      ./orbit.sh -p source/standalone/demo/play_arms.py --robot franka_panda

-  Spawn multiple robots and control them using inverse kinematics
   controller:

   .. code:: bash

      ./orbit.sh -p source/standalone/demo/play_ik_control.py --robot franka_panda --num_envs 128

-  Spawn a camera and visualize the obtained pointcloud:

   .. code:: bash

      # CPU
      ./orbit.sh -p source/standalone/demo/play_camera.py
      # GPU
      ./orbit.sh -p source/standalone/demo/play_camera.py --gpu

Environments
------------

With Orbit, we also provide a suite of benchmark environments included
in the ``omni.isaac.orbit_envs`` extension. We use the OpenAI Gym registry
to register these environments. For each environment, we provide a default
configuration file that defines the scene, observations, rewards and action spaces.

The list of environments available registered with OpenAI Gym can be found by running:

.. code:: bash

   ./orbit.sh -p source/standalone/environments/list_envs.py


Basic agents
~~~~~~~~~~~~

These include basic agents that output zero or random agents. They are
useful to ensure that the environments are configured correctly.

-  Zero-action agent on the Cart-pole example

   .. code:: bash

      ./orbit.sh -p source/standalone/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32

-  Random-action agent on the Cart-pole example:

   .. code:: bash

      ./orbit.sh -p source/standalone/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


State machine
~~~~~~~~~~~~~

We include examples on hand-crafted state machines for the environments. These
help in understanding the environment and how to use the provided interfaces.
The state machines are written in `warp <https://github.com/NVIDIA/warp>`__ which
allows efficient execution for large number of environments using CUDA kernels.

.. code:: bash

   ./orbit.sh -p source/standalone/environments/state_machine/play_lift.py --num_envs 32


Teleoperation
~~~~~~~~~~~~~

We provide interfaces for providing commands in SE(2) and SE(3) space
for robot control. In case of SE(2) teleoperation, the returned command
is the linear x-y velocity and yaw rate, while in SE(3), the returned
command is a 6-D vector representing the change in pose.

To play inverse kinematics (IK) control with a keyboard device:

.. code:: bash

   ./orbit.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Franka-v0 --num_envs 1 --cpu --device keyboard

The script prints the teleoperation events configured. For keyboard,
these are as follows:

.. code:: text

   Keyboard Controller for SE(3): Se3Keyboard
       Reset all commands: L
       Toggle gripper (open/close): K
       Move arm along x-axis: W/S
       Move arm along y-axis: A/D
       Move arm along z-axis: Q/E
       Rotate arm along x-axis: Z/X
       Rotate arm along y-axis: T/G
       Rotate arm along z-axis: C/V

Imitation Learning
~~~~~~~~~~~~~~~~~~

Using the teleoperation devices, it is also possible to collect data for
learning from demonstrations (LfD). For this, we support the learning
framework `Robomimic <https://robomimic.github.io/>`__ and allow saving
data in
`HDF5 <https://robomimic.github.io/docs/tutorials/dataset_contents.html#viewing-hdf5-dataset-structure>`__
format.

1. Collect demonstrations with teleoperation for the environment
   ``Isaac-Lift-Franka-v0``:

   .. code:: bash

      # step a: collect data with keyboard
      ./orbit.sh -p source/standalone/workflows/robomimic/collect_demonstrations.py --task Isaac-Lift-Franka-v0 --num_envs 1 --num_demos 10 --device keyboard
      # step b: inspect the collected dataset
      ./orbit.sh -p source/standalone/workflows/robomimic/tools/inspect_demonstrations.py logs/robomimic/Isaac-Lift-Franka-v0/hdf_dataset.hdf5

2. Split the dataset into train and validation set:

   .. code:: bash

      # install python module (for robomimic)
      ./orbit.sh -p -m pip install -e 'source/extensions/omni.isaac.orbit_envs[robomimic]'
      # split data
      ./orbit.sh -p source/standalone//workflows/robomimic/tools/split_train_val.py logs/robomimic/Isaac-Lift-Franka-v0/hdf_dataset.hdf5 --ratio 0.2

3. Train a BC agent for ``Isaac-Lift-Franka-v0`` with
   `Robomimic <https://robomimic.github.io/>`__:

   .. code:: bash

      ./orbit.sh -p source/standalone/workflows/robomimic/train.py --task Isaac-Lift-Franka-v0 --algo bc --dataset logs/robomimic/Isaac-Lift-Franka-v0/hdf_dataset.hdf5

4. Play the learned model to visualize results:

   .. code:: bash

      ./orbit.sh -p source/standalone//workflows/robomimic/play.py --task Isaac-Lift-Franka-v0 --checkpoint /PATH/TO/model.pth

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~

We provide wrappers to different reinforcement libraries. These wrappers convert the data
from the environments into the respective libraries function argument and return types.

-  Training an agent with
   `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/index.html>`__
   on ``Isaac-Cartpole-v0``:

   .. code:: bash

      # install python module (for stable-baselines3)
      ./orbit.sh -p -m pip install -e 'source/extensions/omni.isaac.orbit_envs[sb3]'
      # run script for training
      # note: we enable cpu flag since SB3 doesn't optimize for GPU anyway
      ./orbit.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --headless --cpu
      # run script for playing with 32 environments
      ./orbit.sh -p source/standalone/workflows/sb3/play.py --task Isaac-Cartpole-v0 --num_envs 32 --checkpoint /PATH/TO/model.zip

-  Training an agent with
   `SKRL <https://skrl.readthedocs.io>`__ on ``Isaac-Reach-Franka-v0``:

   .. code:: bash

      # install python module (for skrl)
      ./orbit.sh -p -m pip install -e 'source/extensions/omni.isaac.orbit_envs[skrl]'
      # run script for training
      ./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Reach-Franka-v0 --headless
      # run script for playing with 32 environments
      ./orbit.sh -p source/standalone/workflows/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt

-  Training an agent with
   `RL-Games <https://github.com/Denys88/rl_games>`__ on ``Isaac-Ant-v0``:

   .. code:: bash

      # install python module (for rl-games)
      ./orbit.sh -p -m pip install -e 'source/extensions/omni.isaac.orbit_envs[rl_games]'
      # run script for training
      ./orbit.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Ant-v0 --headless
      # run script for playing with 32 environments
      ./orbit.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth

-  Training an agent with
   `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ on ``Isaac-Reach-Franka-v0``:

   .. code:: bash

      # install python module (for rsl-rl)
      ./orbit.sh -p -m pip install -e 'source/extensions/omni.isaac.orbit_envs[rsl_rl]'
      # run script for training
      ./orbit.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Reach-Franka-v0 --headless
      # run script for playing with 32 environments
      ./orbit.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth

All the scripts above log the training progress to `Tensorboard`_ in the ``logs`` directory in the root of
the repository. The logs directory follows the pattern ``logs/<library>/<task>/<date-time>``, where ``<library>``
is the name of the learning framework, ``<task>`` is the task name, and ``<date-time>`` is the timestamp at
which the training script was executed.

To view the logs, run:

.. code:: bash

   # execute from the root directory of the repository
   ./orbit.sh -p -m tensorboard.main --logdir=logs

.. _Tensorboard: https://www.tensorflow.org/tensorboard
