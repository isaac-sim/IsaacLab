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

For smoother operation and off-axis operation, we recommend using a SpaceMouse as input device. Providing smoother demonstration will make it easier for the policy to clone the behavior. To use a SpaceMouse, simply change the teleop device accordingly:

.. code:: bash

   ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device spacemouse

.. note::

   If the SpaceMouse is not detected, you may need to grant additional user permissions by running ``sudo chmod 666 /dev/hidraw<#>`` where ``<#>`` corresponds to the device index
   of the connected SpaceMouse.

   To determine the device index, list all ``hidraw`` devices by running ``ls -l /dev/hidraw*``.
   Identify the device corresponding to the SpaceMouse by running ``cat /sys/class/hidraw/hidraw<#>/device/uevent`` on each of the devices listed
   from the prior step.


The script prints the teleoperation events configured. For keyboard,
these are as follows:

.. code:: text

   Keyboard Controller for SE(3): Se3Keyboard
       Reset all commands: R
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
learning from demonstrations (LfD). For this, we provide scripts to collect data into the open HDF5 format.

.. note::

  This tutorial assumes you have a ``datasets`` directory under the ``IsaacLab`` repo. Create this directory by running ``cd IsaacLab`` and ``mkdir datasets``.

1. Collect demonstrations with teleoperation for the environment
   ``Isaac-Stack-Cube-Franka-IK-Rel-v0``:

   .. code:: bash

      # step a: collect data with spacemouse
      ./isaaclab.sh -p source/standalone/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device spacemouse --dataset_file ./datasets/dataset.hdf5 --num_demos 10
      # step b: replay the collected dataset
      ./isaaclab.sh -p source/standalone/tools/replay_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --dataset_file ./datasets/dataset.hdf5


   .. note::

      The order of the stacked cubes should be blue (bottom), red (middle), green (top).

   About 10 successful demonstrations are required in order for the following steps to succeed.

   Here are some tips to perform demonstrations that lead to successful policy training:

   * Keep demonstrations short. Shorter demonstrations mean fewer decisions for the policy, making training easier.
   * Take a direct path. Do not follow along arbitrary axis, but move straight toward the goal.
   * Do not pause. Perform smooth, continuous motions instead. It is not obvious for a policy why and when to pause, hence continuous motions are easier to learn.

   If, while performing a demonstration, a mistake is made, or the current demonstration should not be recorded for some other reason, press the ``R`` key to discard the current demonstration, and reset to a new starting position.

2. Generate additional demonstrations using Isaac Lab Mimic

   Isaac Lab Mimic is a feature in Isaac Lab that allows to generate additional demonstrations automatically, allowing a policy to learn successfully even from just a handful of manual demonstrations.

   In order to use Isaac Lab Mimic with the recorded dataset, first annotate the subtasks in the recording:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/isaac_lab_mimic/annotate_demos.py --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto

   Then, use Isaac Lab Mimic to generate some additional demonstrations:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/isaac_lab_mimic/generate_dataset.py --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5 --num_envs 10 --generation_num_trials 10

   .. note::

      The output_file of the ``annotate_demos.py`` script is the input_file to the ``generate_dataset.py`` script

   .. note::

      Isaac Lab is designed to work with manipulators with grippers. The gripper commands in the demonstrations are extracted separately and temporally replayed during the generation of additional demonstrations.

   Inspect the output of generated data (filename: ``generated_dataset_small.hdf5``), and if satisfactory, generate the full dataset:

      ./isaaclab.sh -p source/standalone/workflows/isaac_lab_mimic/generate_dataset.py --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --num_envs 10 --generation_num_trials 1000 --headless

   The number of demonstrations can be increased or decreased, 1000 demonstrations have been shown to provide good training results for this task.

   Additionally, the number of environments in the ``--num_envs`` parameter can be adjusted to speed up data generation. The suggested number of 10 can be executed even on a laptop GPU. On a more powerful desktop machine, set it to 100 or higher for significant speedup of this step.

3. Setup robomimic for training a policy

   As an example, we will train a BC agent implemented in `Robomimic <https://robomimic.github.io/>`__ to train a policy. Any other framework or training method could be used.

   .. code:: bash

      # install the dependencies
      sudo apt install cmake build-essential
      # install python module (for robomimic)
      ./isaaclab.sh -i robomimic

4. Train a BC agent for ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` using the Mimic generated data:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/robomimic/train.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo bc --dataset ./datasets/generated_dataset.hdf5

   By default, the training script will save a model checkpoint every 100 epochs. The trained models and logs will be saved to logs/robomimic/Isaac-Stack-Cube-Franka-IK-Rel-v0/bc

5. Play the learned model to visualize results:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/workflows/robomimic/play.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint /PATH/TO/desired_model_checkpoint.pth
