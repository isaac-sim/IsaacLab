Teleoperation and Imitation Learning
====================================


Teleoperation
~~~~~~~~~~~~~

We provide interfaces for providing commands in SE(2) and SE(3) space
for robot control. In case of SE(2) teleoperation, the returned command
is the linear x-y velocity and yaw rate, while in SE(3), the returned
command is a 6-D vector representing the change in pose.

To play inverse kinematics (IK) control with a keyboard device:

.. code:: bash

   ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

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
framework `Robomimic <https://robomimic.github.io/>`__ (Linux only) and allow saving
data in
`HDF5 <https://robomimic.github.io/docs/tutorials/dataset_contents.html#viewing-hdf5-dataset-structure>`__
format.

1. Collect demonstrations with teleoperation for the environment
   ``Isaac-Lift-Cube-Franka-IK-Rel-v0``:

   .. code:: bash

      # step a: collect data with keyboard
      ./isaaclab.sh -p source/standalone/workflows/robomimic/collect_demonstrations.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --num_demos 10 --teleop_device keyboard
      # step b: inspect the collected dataset
      ./isaaclab.sh -p source/standalone/workflows/robomimic/tools/inspect_demonstrations.py logs/robomimic/Isaac-Lift-Cube-Franka-IK-Rel-v0/hdf_dataset.hdf5

2. Split the dataset into train and validation set:

   .. code:: bash

      # install the dependencies
      sudo apt install cmake build-essential
      # install python module (for robomimic)
      ./isaaclab.sh -i robomimic
      # split data
      ./isaaclab.sh -p source/standalone/workflows/robomimic/tools/split_train_val.py logs/robomimic/Isaac-Lift-Cube-Franka-IK-Rel-v0/hdf_dataset.hdf5 --ratio 0.2

3. Train a BC agent for ``Isaac-Lift-Cube-Franka-IK-Rel-v0`` with
   `Robomimic <https://robomimic.github.io/>`__:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/robomimic/train.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --algo bc --dataset logs/robomimic/Isaac-Lift-Cube-Franka-IK-Rel-v0/hdf_dataset.hdf5

4. Play the learned model to visualize results:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/robomimic/play.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --checkpoint /PATH/TO/model.pth
