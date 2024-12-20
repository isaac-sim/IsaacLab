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

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

For smoother operation and off-axis operation, we recommend using a SpaceMouse as the input device. Providing smoother demonstrations will make it easier for the policy to clone the behavior. To use a SpaceMouse, simply change the teleop device accordingly:

.. code:: bash

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device spacemouse

.. note::

   If the SpaceMouse is not detected, you may need to grant additional user permissions by running ``sudo chmod 666 /dev/hidraw<#>`` where ``<#>`` corresponds to the device index
   of the connected SpaceMouse.

   To determine the device index, list all ``hidraw`` devices by running ``ls -l /dev/hidraw*``.
   Identify the device corresponding to the SpaceMouse by running ``cat /sys/class/hidraw/hidraw<#>/device/uevent`` on each of the devices listed
   from the prior step.

   Only compatible with the SpaceMouse Wireless and SpaceMouse Compact models from 3Dconnexion.

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

For SpaceMouse, these are as follows:

.. code:: text

   SpaceMouse Controller for SE(3): Se3SpaceMouse
      Reset all commands: R
      Toggle gripper (open/close): Click the left button on the SpaceMouse
      Move arm along x/y-axis: Tilt the SpaceMouse
      Move arm along z-axis: Push or pull the SpaceMouse
      Rotate arm: Twist the SpaceMouse

Imitation Learning
~~~~~~~~~~~~~~~~~~

Using the teleoperation devices, it is also possible to collect data for
learning from demonstrations (LfD). For this, we provide scripts to collect data into the open HDF5 format.

.. note::

  This tutorial assumes you have a ``datasets`` directory under the ``IsaacLab`` repo. Create this directory by running ``cd IsaacLab`` and ``mkdir datasets``.

1. Collect demonstrations with teleoperation for the environment
   ``Isaac-Stack-Cube-Franka-IK-Rel-v0``:

   .. code:: bash

      # step a: collect data with a selected teleoperation device. Replace <teleop_device> with your preferred input device.
      # Available options: spacemouse, keyboard
      ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device <teleop_device> --dataset_file ./datasets/dataset.hdf5 --num_demos 10
      # step b: replay the collected dataset
      ./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --dataset_file ./datasets/dataset.hdf5


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

      ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto

   Then, use Isaac Lab Mimic to generate some additional demonstrations:

   .. code:: bash

      ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5 --num_envs 10 --generation_num_trials 10

   .. note::

      The output_file of the ``annotate_demos.py`` script is the input_file to the ``generate_dataset.py`` script

   .. note::

      Isaac Lab is designed to work with manipulators with grippers. The gripper commands in the demonstrations are extracted separately and temporally replayed during the generation of additional demonstrations.

   Inspect the output of generated data (filename: ``generated_dataset_small.hdf5``), and if satisfactory, generate the full dataset:

      ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --num_envs 10 --generation_num_trials 1000 --headless

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

      ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo bc --dataset ./datasets/generated_dataset.hdf5

   By default, the training script will save a model checkpoint every 100 epochs. The trained models and logs will be saved to logs/robomimic/Isaac-Stack-Cube-Franka-IK-Rel-v0/bc

5. Play the learned model to visualize results:

   .. code:: bash

      ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint /PATH/TO/desired_model_checkpoint.pth

Creating Your Own Isaac Lab Mimic Compatible Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use Isaac Lab Mimic to generate additional demonstrations automatically with an existing Isaac Lab environment, the environment
needs to be made "Mimic compatible" by implementing additional functions which are used during data generation.

Mimic compatible environments are derived from the ``ManagerBasedRLMimicEnv`` base class and must implement the following functions:

* ``get_robot_eef_pose``: Returns the current robot end effector pose in the same frame as used by the robot end effector controller.

* ``target_eef_pose_to_action``: Takes a target pose for the end effector controller and returns an action which achieves the target pose.

* ``action_to_target_eef_pos``: Takes an action and returns a target pose for the end effector controller.

* ``action_to_gripper_action``: Takes an action and returns the gripper actuation part of the action.

* ``get_object_poses``: Returns the pose of each object in the scene that is used for data generation.

* ``get_subtask_term_signals``: Returns a dictionary of binary flags for each subtask in a task. The flag of 1 is set when the subtask has been completed and 0 otherwise.

* ``is_success``: Returns a boolean indicator of whether the task has been successfully completed.

The class ``FrankaCubeStackIKRelMimicEnv`` shows an example of creating a Mimic compatible environment from an existing Isaac Lab environment.
It can be found under ``source/isaaclab_mimic/isaaclab_mimic/envs``.

A Mimic compatible environment config class must also be created by extending the existing environment config with additional Mimic required parameters.
All Mimic required config parameters are specified in the ``MimicEnvCfg`` class found under ``source/isaaclab/isaaclab/envs``.
The config class ``FrankaCubeStackIKRelMimicEnvCfg`` shows an example of creating a Mimic compatible environment config class for the Franka stacking task
and can be found under ``source/isaaclab_mimic/isaaclab_mimic/envs``.

Once both Mimic compatible environment and environment config classes have been created, a new Mimic compatible environment can be registered using ``gym.register`` and used
with Isaac Lab Mimic data generation. For the Franka stacking task in the examples above, the Mimic environment is registered as ``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0``.
