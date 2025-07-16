.. _teleoperation-imitation-learning:

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

   We recommend using local deployment of Isaac Lab to use the SpaceMouse. If using container deployment (:ref:`deployment-docker`), you must manually mount the SpaceMouse to the ``isaac-lab-base`` container by
   adding a ``devices`` attribute with the path to the device in your ``docker-compose.yaml`` file:

   .. code:: yaml

      devices:
         - /dev/hidraw<#>:/dev/hidraw<#>

   where ``<#>`` is the device index of the connected SpaceMouse.

   If you are using the IsaacLab + CloudXR container deployment (:ref:`cloudxr-teleoperation`), you can add the ``devices`` attribute under the ``services -> isaac-lab-base`` section of the
   ``docker/docker-compose.cloudxr-runtime.patch.yaml`` file.

   Isaac Lab is only compatible with the SpaceMouse Wireless and SpaceMouse Compact models from 3Dconnexion.


For tasks that benefit from the use of an extended reality (XR) device with hand tracking, Isaac Lab supports using NVIDIA CloudXR to immersively stream the scene to compatible XR devices for teleoperation. Note that when using hand tracking we recommend using the absolute variant of the task (``Isaac-Stack-Cube-Franka-IK-Abs-v0``), which requires the ``handtracking_abs`` device:

.. code:: bash

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Stack-Cube-Franka-IK-Abs-v0 --teleop_device handtracking_abs --device cpu

.. note::

   See :ref:`cloudxr-teleoperation` to learn more about using CloudXR with Isaac Lab.


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
      Reset all commands: Right click
      Toggle gripper (open/close): Click the left button on the SpaceMouse
      Move arm along x/y-axis: Tilt the SpaceMouse
      Move arm along z-axis: Push or pull the SpaceMouse
      Rotate arm: Twist the SpaceMouse

The next section describes how teleoperation devices can be used for data collection for imitation learning.


Imitation Learning
~~~~~~~~~~~~~~~~~~

Using the teleoperation devices, it is also possible to collect data for
learning from demonstrations (LfD). For this, we provide scripts to collect data into the open HDF5 format.

Collecting demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^

To collect demonstrations with teleoperation for the environment ``Isaac-Stack-Cube-Franka-IK-Rel-v0``, use the following commands:

.. code:: bash

   # step a: create folder for datasets
   mkdir -p datasets
   # step b: collect data with a selected teleoperation device. Replace <teleop_device> with your preferred input device.
   # Available options: spacemouse, keyboard, handtracking, handtracking_abs, dualhandtracking_abs
   ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device <teleop_device> --dataset_file ./datasets/dataset.hdf5 --num_demos 10
   # step a: replay the collected dataset
   ./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --dataset_file ./datasets/dataset.hdf5


.. note::

   The order of the stacked cubes should be blue (bottom), red (middle), green (top).

.. tip::

   When using an XR device, we suggest collecting demonstrations with the ``Isaac-Stack-Cube-Frank-IK-Abs-v0`` version of the task and ``--teleop_device handtracking_abs``, which controls the end effector using the absolute position of the hand.

About 10 successful demonstrations are required in order for the following steps to succeed.

Here are some tips to perform demonstrations that lead to successful policy training:

* Keep demonstrations short. Shorter demonstrations mean fewer decisions for the policy, making training easier.
* Take a direct path. Do not follow along arbitrary axis, but move straight toward the goal.
* Do not pause. Perform smooth, continuous motions instead. It is not obvious for a policy why and when to pause, hence continuous motions are easier to learn.

If, while performing a demonstration, a mistake is made, or the current demonstration should not be recorded for some other reason, press the ``R`` key to discard the current demonstration, and reset to a new starting position.

.. note::
   Non-determinism may be observed during replay as physics in IsaacLab are not determimnistically reproducible when using ``env.reset``.

Pre-recorded demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a pre-recorded ``dataset.hdf5`` containing 10 human demonstrations for ``Isaac-Stack-Cube-Franka-IK-Rel-v0``
`here <https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Mimic/dataset.hdf5>`_.
This dataset may be downloaded and used in the remaining tutorial steps if you do not wish to collect your own demonstrations.

.. note::
   Use of the pre-recorded dataset is optional.

Generating additional demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional demonstrations can be generated using Isaac Lab Mimic.

Isaac Lab Mimic is a feature in Isaac Lab that allows generation of additional demonstrations automatically, allowing a policy to learn successfully even from just a handful of manual demonstrations.

In the following example, we will show how to use Isaac Lab Mimic to generate additional demonstrations that can be used to train either a state-based policy
(using the ``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`` environment) or visuomotor policy (using the ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`` environment).

.. important::

   All commands in the following sections must keep a consistent policy type. For example, if choosing to use a state-based policy, then all commands used should be from the "State-based policy" tab.

In order to use Isaac Lab Mimic with the recorded dataset, first annotate the subtasks in the recording:

.. tab-set::
   :sync-group: policy_type

   .. tab-item:: State-based policy
      :sync: state

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
         --device cuda --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto \
         --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5

   .. tab-item:: Visuomotor policy
      :sync: visuomotor

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
         --device cuda --enable_cameras --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 --auto \
         --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5


Then, use Isaac Lab Mimic to generate some additional demonstrations:

.. tab-set::
   :sync-group: policy_type

   .. tab-item:: State-based policy
      :sync: state

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
         --device cuda --num_envs 10 --generation_num_trials 10 \
         --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5

   .. tab-item:: Visuomotor policy
      :sync: visuomotor

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
         --device cuda --enable_cameras --num_envs 10 --generation_num_trials 10 \
         --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5

.. note::

   The output_file of the ``annotate_demos.py`` script is the input_file to the ``generate_dataset.py`` script

Inspect the output of generated data (filename: ``generated_dataset_small.hdf5``), and if satisfactory, generate the full dataset:

.. tab-set::
   :sync-group: policy_type

   .. tab-item:: State-based policy
      :sync: state

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
         --device cuda --headless --num_envs 10 --generation_num_trials 1000 \
         --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5

   .. tab-item:: Visuomotor policy
      :sync: visuomotor

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
         --device cuda --enable_cameras --headless --num_envs 10 --generation_num_trials 1000 \
         --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5


The number of demonstrations can be increased or decreased, 1000 demonstrations have been shown to provide good training results for this task.

Additionally, the number of environments in the ``--num_envs`` parameter can be adjusted to speed up data generation.
The suggested number of 10 can be executed on a moderate laptop GPU.
On a more powerful desktop machine, use a larger number of environments for a significant speedup of this step.

Robomimic setup
^^^^^^^^^^^^^^^

As an example, we will train a BC agent implemented in `Robomimic <https://robomimic.github.io/>`__ to train a policy. Any other framework or training method could be used.

To install the robomimic framework, use the following commands:

.. code:: bash

   # install the dependencies
   sudo apt install cmake build-essential
   # install python module (for robomimic)
   ./isaaclab.sh -i robomimic

Training an agent
^^^^^^^^^^^^^^^^^

Using the Mimic generated data we can now train a state-based BC agent for ``Isaac-Stack-Cube-Franka-IK-Rel-v0``, or a visuomotor BC agent for ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0``:

.. tab-set::
   :sync-group: policy_type

   .. tab-item:: State-based policy
      :sync: state

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
         --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo bc \
         --dataset ./datasets/generated_dataset.hdf5

   .. tab-item:: Visuomotor policy
      :sync: visuomotor

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
         --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo bc \
         --dataset ./datasets/generated_dataset.hdf5

.. note::
   By default the trained models and logs will be saved to ``IssacLab/logs/robomimic``.

Visualizing results
^^^^^^^^^^^^^^^^^^^

By inferencing using the generated model, we can visualize the results of the policy:

.. tab-set::
   :sync-group: policy_type

   .. tab-item:: State-based policy
      :sync: state

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
         --device cuda --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_rollouts 50 \
         --checkpoint /PATH/TO/desired_model_checkpoint.pth

   .. tab-item:: Visuomotor policy
      :sync: visuomotor

      .. code:: bash

         ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
         --device cuda --enable_cameras --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --num_rollouts 50 \
         --checkpoint /PATH/TO/desired_model_checkpoint.pth


Demo: Data Generation and Policy Training for a Humanoid Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Isaac Lab Mimic supports data generation for robots with multiple end effectors. In the following demonstration, we will show how to generate data
to train a Fourier GR-1 humanoid robot to perform a pick and place task.

Optional: Collect and annotate demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collect human demonstrations
""""""""""""""""""""""""""""
.. note::

   Data collection for the GR-1 humanoid robot environment requires use of an Apple Vision Pro headset. If you do not have access to
   an Apple Vision Pro, you may skip this step and continue on to the next step: `Generate the dataset`_.
   A pre-recorded annotated dataset is provided in the next step .

.. tip::
   The GR1 scene utilizes the wrist poses from the Apple Vision Pro (AVP) as setpoints for a differential IK controller (Pink-IK).
   The differential IK controller requires the user's wrist pose to be close to the robot's initial or current pose for optimal performance.
   Rapid movements of the user's wrist may cause it to deviate significantly from the goal state, which could prevent the IK controller from finding the optimal solution.
   This may result in a mismatch between the user's wrist and the robot's wrist.
   You can increase the gain of the all `Pink-IK controller's FrameTasks <https://github.com/isaac-sim/IsaacLab-Internal/blob/devel/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py>`__ to track the AVP wrist poses with lower latency.
   However, this may lead to more jerky motion.
   Separately, the finger joints of the robot are retargeted to the user's finger joints using the `dex-retargeting <https://github.com/dexsuite/dex-retargeting>`_ library.

Set up the CloudXR Runtime and Apple Vision Pro for teleoperation by following the steps in :ref:`cloudxr-teleoperation`.
CPU simulation is used in the following steps for better XR performance when running a single environment.

Collect a set of human demonstrations using the command below.
A success demo requires the object to be placed in the bin and for the robot's right arm to be retracted to the starting position.
The Isaac Lab Mimic Env GR-1 humanoid robot is set up such that the left hand has a single subtask, while the right hand has two subtasks.
The first subtask involves the right hand remaining idle while the left hand picks up and moves the object to the position where the right hand will grasp it.
This setup allows Isaac Lab Mimic to interpolate the right hand's trajectory accurately by using the object's pose, especially when poses are randomized during data generation.
Therefore, avoid moving the right hand while the left hand picks up the object and brings it to a stable position.
We recommend 10 successful demonstrations for good data generation results. An example of a successful demonstration is shown below:

.. figure:: ../_static/tasks/manipulation/gr-1_pick_place.gif
   :width: 100%
   :align: center
   :alt: GR-1 humanoid robot performing a pick and place task

Collect demonstrations by running the following command:

.. code:: bash

   ./isaaclab.sh -p scripts/tools/record_demos.py \
   --device cpu \
   --task Isaac-PickPlace-GR1T2-Abs-v0 \
   --teleop_device dualhandtracking_abs \
   --dataset_file ./datasets/dataset_gr1.hdf5 \
   --num_demos 10 --enable_pinocchio

.. tip::
   If a demo fails during data collection, the environment can be reset using the teleoperation controls panel in the XR teleop client
   on the Apple Vision Pro or via voice control by saying "reset". See :ref:`teleoperate-apple-vision-pro` for more details.

   The robot uses simplified collision meshes for physics calculations that differ from the detailed visual meshes displayed in the simulation. Due to this difference, you may occasionally observe visual artifacts where parts of the robot appear to penetrate other objects or itself, even though proper collision handling is occurring in the physics simulation.

.. warning::
   When first starting the simulation window, you may encounter the following ``DeprecationWarning`` and ``UserWarning`` error:

   .. code-block:: text

      DeprecationWarning: get_prim_path is deprecated and will be removed
      in a future release. Use get_path.
      UserWarning: Sum of faceVertexCounts (25608) does not equal sum of
      length of GeomSubset indices (840) for prim
      '/GR1T2_fourier_hand_6dof/waist_pitch_link/visuals/waist_pitch_link/mesh'.
      Material mtl files will not be created.

   This error can be ignored and will not affect the data collection process.
   The error will be patched in a future release of Isaac Sim.

You can replay the collected demonstrations by running the following command:

.. code:: bash

   ./isaaclab.sh -p scripts/tools/replay_demos.py \
   --device cpu \
   --task Isaac-PickPlace-GR1T2-Abs-v0 \
   --dataset_file ./datasets/dataset_gr1.hdf5 --enable_pinocchio

.. note::
   Non-determinism may be observed during replay as physics in IsaacLab are not determimnistically reproducible when using ``env.reset``.


Annotate the demonstrations
"""""""""""""""""""""""""""

Unlike the prior Franka stacking task, the GR-1 pick and place task uses manual annotation to define subtasks.
Each demo requires a single annotation between the first and second subtask of the right arm. This annotation ("S" button press) should be done when the right robot arm finishes the "idle" subtask and begins to
move towards the target object. An example of a correct annotation is shown below:

.. figure:: ../_static/tasks/manipulation/gr-1_pick_place_annotation.jpg
   :width: 100%
   :align: center

Annotate the demonstrations by running the following command:

.. code:: bash

   ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
   --device cpu \
   --task Isaac-PickPlace-GR1T2-Abs-Mimic-v0 \
   --input_file ./datasets/dataset_gr1.hdf5 \
   --output_file ./datasets/dataset_annotated_gr1.hdf5 --enable_pinocchio

.. note::

   The script prints the keyboard commands for manual annotation and the current subtask being annotated:

   .. code:: text

      Annotating episode #0 (demo_0)
         Playing the episode for subtask annotations for eef "right".
         Subtask signals to annotate:
            - Termination:	['idle_right']

         Press "N" to begin.
         Press "B" to pause.
         Press "S" to annotate subtask signals.
         Press "Q" to skip the episode.

.. tip::

   If the object does not get placed in the bin during annotation, you can press "N" to replay the episode and annotate again. Or you can press "Q" to skip the episode and annotate the next one.

Generate the dataset
^^^^^^^^^^^^^^^^^^^^

If you skipped the prior collection and annotation step, download the pre-recorded annotated dataset ``dataset_annotated_gr1.hdf5`` from
`here <https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Mimic/dataset_annotated_gr1.hdf5>`_.
Place the file under ``IsaacLab/datasets`` and run the following command to generate a new dataset with 1000 demonstrations.

.. code:: bash

   ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
   --device cuda --headless --num_envs 10 --generation_num_trials 1000 --enable_pinocchio \
   --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/generated_dataset_gr1.hdf5

Train a policy
^^^^^^^^^^^^^^

Use Robomimic to train a policy for the generated dataset.

.. code:: bash

   ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
   --task Isaac-PickPlace-GR1T2-Abs-v0 --algo bc \
   --normalize_training_actions \
   --dataset ./datasets/generated_dataset_gr1.hdf5

The training script will normalize the actions in the dataset to the range [-1, 1].
The normalization parameters are saved in the model directory under ``PATH_TO_MODEL_DIRECTORY/logs/normalization_params.txt``.
Record the normalization parameters for later use in the visualization step.

.. note::
   By default the trained models and logs will be saved to ``IssacLab/logs/robomimic``.

Visualize the results
^^^^^^^^^^^^^^^^^^^^^

Visualize the results of the trained policy by running the following command, using the normalization parameters recorded in the prior training step:

.. code:: bash

   ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
   --device cuda \
   --enable_pinocchio \
   --task Isaac-PickPlace-GR1T2-Abs-v0 \
   --num_rollouts 50 \
   --norm_factor_min <NORM_FACTOR_MIN> \
   --norm_factor_max <NORM_FACTOR_MAX> \
   --checkpoint /PATH/TO/desired_model_checkpoint.pth

.. note::
   Change the ``NORM_FACTOR`` in the above command with the values generated in the training step.

Common Pitfalls when Generating Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Demonstrations are too long:**

* Longer time horizon is harder to learn for a policy
* Start close to the first object and minimize motions

**Demonstrations are not smooth:**

* Irregular motion is hard for policy to decipher
* Better teleop devices result in better data (i.e. SpaceMouse is better than Keyboard)

**Pauses in demonstrations:**

* Pauses are difficult to learn
* Keep the human motions smooth and fluid

**Excessive number of subtasks:**

* Minimize the number of defined subtasks for completing a given task
* Less subtacks results in less stitching of trajectories, yielding higher data generation success rate

**Lack of action noise:**

* Action noise makes policies more robust

**Recording cropped too tight:**

* If recording stops on the frame the success term triggers, it may not re-trigger during replay
* Allow for some buffer at the end of recording

**Non-deterministic replay:**

* Physics in IsaacLab are not deterministically reproducible when using ``env.reset`` so demonstrations may fail on replay
* Collect more human demos than needed, use the ones that succeed during annotation
* All data in Isaac Lab Mimic generated HDF5 file represent a successful demo and can be used for training (even if non-determinism causes failure when replayed)


Creating Your Own Isaac Lab Mimic Compatible Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How it works
^^^^^^^^^^^^

Isaac Lab Mimic works by splitting the input demonstrations into subtasks. Subtasks are user-defined segments in the demonstrations that are common to all demonstrations. Examples for subtasks are "grasp an object", "move end effector to some pre-defined position", "release object" etc.. Note that most subtasks are defined with respect to some object that the robot interacts with.

Subtasks need to be defined, and then annotated for each input demonstration. Annotation can either happen algorithmically by defining heuristics for subtask detection, as was done in the example above, or it can be done manually.

With subtasks defined and annotated, Isaac Lab Mimic utilizes a small number of helper methods to then transform the subtask segments, and generate new demonstrations by stitching them together to match the new task at hand.

For each thusly generated candidate demonstration, Isaac Lab Mimic uses a boolean success criteria to determine whether the demonstration succeeded in performing the task, and if so, add it to the output dataset. Success rate of candidate demonstrations can be as high as 70% in simple cases, and as low as <1%, depending on the difficulty of the task, and the complexity of the robot itself.

Configuration and subtask definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subtasks, among other configuration settings for Isaac Lab Mimic, are defined in a Mimic compatible environment configuration class that is created by extending the existing environment config with additional Mimic required parameters.

All Mimic required config parameters are specified in the :class:`~isaaclab.envs.MimicEnvCfg` class.

The config class :class:`~isaaclab_mimic.envs.FrankaCubeStackIKRelMimicEnvCfg` serves as an example of creating a Mimic compatible environment config class for the Franka stacking task that was used in the examples above.

The ``DataGenConfig`` member contains various parameters that influence how data is generated. It is initially sufficient to just set the ``name`` parameter, and revise the rest later.

Subtasks are a list of :class:`~isaaclab.envs.SubTaskConfig` objects, of which the most important members are:

* ``object_ref`` is the object that is being interacted with. This will be used to adjust motions relative to this object during data generation. Can be ``None`` if the current subtask does not involve any object.
* ``subtask_term_signal`` is the ID of the signal indicating whether the subtask is active or not.

For multi end-effector environments, subtask ordering between end-effectors can be enforced by specifying subtask constraints. These constraints are defined in the :class:`~isaaclab.envs.SubTaskConstraintConfig` class.

Subtask annotation
^^^^^^^^^^^^^^^^^^

Once the subtasks are defined, they need to be annotated in the source data. There are two methods to annotate source demonstrations for subtask boundaries: Manual annotation or using heuristics.

It is often easiest to perform manual annotations, since the number of input demonstrations is usually very small. To perform manual annotations, use the ``annotate_demos.py`` script without the ``--auto`` flag. Then press ``B`` to pause, ``N`` to continue, and ``S`` to annotate a subtask boundary.

For more accurate boundaries, or to speed up repeated processing of a given task for experiments, heuristics can be implemented to perform the same task. Heuristics are observations in the environment. An example how to add subtask terms can be found in ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py``, where they are added as an observation group called ``SubtaskCfg``. This example is using prebuilt heuristics, but custom heuristics are easily implemented.


Helpers for demonstration generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helpers needed for Isaac Lab Mimic are defined in the environment. All tasks that are to be used with Isaac Lab Mimic are derived from the :class:`~isaaclab.envs.ManagerBasedRLMimicEnv` base class, and must implement the following functions:

* ``get_robot_eef_pose``: Returns the current robot end effector pose in the same frame as used by the robot end effector controller.

* ``target_eef_pose_to_action``: Takes a target pose and a gripper action for the end effector controller and returns an action which achieves the target pose.

* ``action_to_target_eef_pose``: Takes an action and returns a target pose for the end effector controller.

* ``actions_to_gripper_actions``: Takes a sequence of actions and returns the gripper actuation part of the actions.

* ``get_object_poses``: Returns the pose of each object in the scene that is used for data generation.

* ``get_subtask_term_signals``: Returns a dictionary of binary flags for each subtask in a task. The flag of true is set when the subtask has been completed and false otherwise.

The class :class:`~isaaclab_mimic.envs.FrankaCubeStackIKRelMimicEnv` shows an example of creating a Mimic compatible environment from an existing Isaac Lab environment.

Registering the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once both Mimic compatible environment and environment config classes have been created, a new Mimic compatible environment can be registered using ``gym.register``. For the Franka stacking task in the examples above, the Mimic environment is registered as ``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0``.

The registered environment is now ready to be used with Isaac Lab Mimic.
