Training and Deploying a G1 Visual Locomotion Policy
====================================================

.. currentmodule:: isaaclab

This guide follows one policy from training to a walking robot: a teacher trained with PPO, a
student distilled from it that sees through a chest-mounted depth camera, a LEAPP export, and a
deployment using Isaac ROS Deploy and the official Unitree G1 ros2_control hardware interface.

:doc:`../05_leapp/deploying_exported_policies_with_leapp` ends at
:class:`~envs.LeappDeploymentEnv`, which runs an exported policy back in simulation. This guide
continues from there onto hardware using the `Isaac ROS Deploy node runtime
<https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/concepts/ros2_nodes_runtime.html>`__.
Its ``InputBuilderNode``, ``TritonNode`` and ``OutputBuilderNode`` execute the exported graph.
``g1_deploy.py`` configures the official hardware driver, message converters and controllers for this policy.

.. note::

   The pipeline below is specific to a 29-DoF Unitree G1 with a RealSense D435i on the chest. The
   camera calibration and policy settings are specific to that setup. The ROS controller integration
   must be validated on your robot before unsupported walking.

.. code-block:: text

   1. train      Isaac-Velocity-Rough-G1
   2. distill    Isaac-Velocity-Rough-G1-Depth-Student
   3. export     pipeline .yaml + .onnx + _initial_values.safetensors
   4. deploy     scripts/deploy/g1_deploy.py

.. raw:: html

   <div style="display: flex; flex-direction: column; gap: 1rem; margin: 1rem 0;">
     <a href="https://github.com/Double7sBurger/IsaacLab/releases/download/g1-visual-locomotion-media/block_physx_vs_newton_vs_real.gif" style="flex: 1; min-width: 0;">
       <img src="https://github.com/Double7sBurger/IsaacLab/releases/download/g1-visual-locomotion-media/block_physx_vs_newton_vs_real.gif" alt="G1 on blocks: PhysX, Newton (MJWarp), and real robot" style="width: 100%; height: auto;">
     </a>
     <a href="https://github.com/Double7sBurger/IsaacLab/releases/download/g1-visual-locomotion-media/ramp_physx_vs_newton_vs_real.gif" style="flex: 1; min-width: 0;">
       <img src="https://github.com/Double7sBurger/IsaacLab/releases/download/g1-visual-locomotion-media/ramp_physx_vs_newton_vs_real.gif" alt="G1 on ramps: PhysX, Newton (MJWarp), and real robot" style="width: 100%; height: auto;">
     </a>
   </div>


Prerequisites
-------------

Start with the :doc:`Isaac Lab installation </source/isaaclab/installation>`. From the repository
root, install the LEAPP export dependencies into its Python environment:

.. code-block:: bash

   uv sync --extra leapp

For hardware deployment, use the `Isaac ROS 5.0 setup
<https://nvidia-isaac-ros.github.io/getting_started/index.html>`__: an Ubuntu 24.04 environment with
ROS 2 Lyrical and access to a supported NVIDIA GPU with compatible drivers and CUDA libraries.
Install the official G1 hardware stack, inference runtime and image conversion packages there:

.. code-block:: bash

   sudo apt-get install ros-lyrical-unitree-g1-bringup ros-lyrical-isaac-ros-deploy-bringup \
       ros-lyrical-isaac-ros-triton ros-lyrical-isaac-ros-tensor-proc ros-lyrical-cv-bridge
   source /opt/ros/lyrical/setup.bash

The deployment scripts use the environment's Python, NumPy and PyYAML. They do not require the
Unitree Python SDK, a Python DDS binding, or a separate Python inference manager.
Connect the deployment machine to the robot's Ethernet subnet. Follow the official
`Isaac ROS Robots G1 setup <https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_robots/tree/release-5.0/unitree_g1>`__
for hardware access and ros2_control configuration. The camera computer needs a compatible ROS 2
environment and the RealSense ROS driver, as described in `Starting the Camera`_ below.


Training the Teacher
--------------------

The teacher reads a terrain height scan and its own base linear velocity. A real G1 publishes
neither, which is why the second stage exists.

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-G1

``Isaac-Velocity-Rough-G1`` uses ``G1RoughTeacherEnvCfg`` with ``g1_29dof.usd``. Its rewards
reference that asset's three-axis waist, elbow, wrist and hand joints. The flat G1 task retains
its existing ``G1_MINIMAL_CFG`` asset.


The default configuration uses 4096 environments, 6000 PPO iterations, Newton mesh terrain
with 5–35 cm stairs, per-joint action scales and left-right data augmentation. Simulation torque
limits include 20 N·m at the ankles and are configured separately from hardware limits.
Use ``--seed`` to set the random seed.


Distilling the Depth Student
----------------------------

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-G1-Depth-Student \
       --checkpoint <TEACHER_CHECKPOINT>

The student runs in the teacher's own environment and drops the privileged observation terms. In
their place it reads five frames of proprioception and three frames of 64x38 depth, rendered from
a camera placed where the hardware's RealSense sits.

``obs_groups`` is where the two contracts meet: ``student`` reads the deployable groups and
``teacher`` the privileged one. ``rsl_rl`` splits observation groups by rank, sends the rank-4
depth group through a convolutional encoder, and concatenates the latent with the 1D groups before
the MLP.


Distillation starts a new student for 24000 iterations. The teacher observation is noise-free;
the student normalizes five frames per proprioceptive term. A second, reflected camera supplies
mirror augmentation during training only. PPO checkpoints load only the teacher, without restoring
student parameters, optimizer state or iteration. Evaluate checkpoints across the target terrains
to select a policy for deployment.


Exporting the Pipeline
----------------------

.. code-block:: bash

   uv run --extra leapp isaaclab leapp export --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-G1-Depth-Student \
       physics=newton_mjwarp

The export declares six inputs and three outputs:

.. code-block:: text

   inputs    robot_joint_pos          robot_joint_vel
             robot_root_quat_w        robot_root_ang_vel_b
             base_velocity            depth_camera_output_distance_to_image_plane

   outputs   joint_pos                joint_pos_kp_gains       joint_pos_kd_gains

It also declares fifteen feedback connections. The proprioception history, the depth stack, the
previous action and the observation normalization all live inside the graph, and the
``_initial_values.safetensors`` file holds their reset state. Nothing outside the graph recomputes
any of it.


Running on the Robot
--------------------

Official Runtime and Hardware Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script creates temporary configuration for the exported policy and launches official nodes:

* ``UnitreeHgSystemInterface`` handles Unitree DDS and the 29 body motor interfaces.
* ``JointStateBroadcaster`` and ``IMUSensorBroadcaster`` publish ROS robot state.
* Isaac ROS Deploy's ``InputBuilderNode`` converts ``JointState``, ``Imu`` and ``Twist`` inputs,
  selects joints by the export's names, and assembles the inputs at 50 Hz. The broadcaster's
  ``extra_joints`` supplies zero state for the unactuated fingers present in the export.
* ``ImageToTensorNode`` converts the preprocessed ``32FC1`` depth image without dividing by 255.
* Triton executes the LEAPP ONNX graph; ``OutputBuilderNode`` publishes ``JointCommand`` targets and gains.
* ``ForwardJointCommandController`` passes the 29 body commands through ``SafetyController``
  to the hardware at 200 Hz. Hand commands are not sent to the robot.

The exported models, graph connections and feedback initial values are not modified. Only the
external port metadata and ROS controller configuration are adapted. The stock G1 launch configures
NVIDIA's AGILE controller group, so this script composes the same official components for the
visual policy's different inputs and outputs.

A small watchdog checks input/output age, finite state and commands, joint layout and orientation.
Missing input or output for 0.2 seconds latches a stop and requests the official ``FreezeController``,
configured with zero stiffness and damping gain 3. The official input builder only warns about
stale inputs, so this check is still needed. A separate startup window allows Triton to produce its
first result. Restart deployment after a stop; receiving data again does not automatically re-arm it.

Camera and IMU Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The official driver and IMU converter provide the graph's ``(x, y, z, w)`` quaternion convention.
The policy expects orientation and angular velocity in the pelvis frame. If the IMU mounting needs
calibration, supply a corrected ROS ``Imu`` topic with ``--imu_topic``; the old ``--imu_tilt``
Python correction is no longer applied by this launcher.

The depth preprocessor retains the policy-specific operations: centered crop toward the trained
87.0° horizontal and 58.8° vertical field of view, valid-pixel averaging to 64x38, millimetre
quantization, a three-metre clip and missing-return value, and 180° rotation to compensate the
exported camera orientation. ``--depth-rotation 0`` disables that last operation. Standard image
resizing alone would change how invalid pixels contribute to the policy's depth input.

Starting the Camera
^^^^^^^^^^^^^^^^^^^

On the computer attached to the chest RealSense, start the official ROS driver:

.. code-block:: bash

   source /opt/ros/lyrical/setup.bash
   ros2 launch realsense2_camera rs_launch.py enable_color:=false enable_depth:=true \
       depth_module.depth_profile:=848x480x90

Install ``ros-lyrical-realsense2-camera`` and its device dependencies on that computer first.
If the camera cannot sustain 90 Hz, select a supported 60 Hz profile. Camera and deployment
processes must use the same ROS domain, discover each other on the robot network, and have
synchronized clocks: freshness checks use the original image capture timestamp.

On the deployment machine, run the preprocessor in another terminal:

.. code-block:: bash

   source /opt/ros/lyrical/setup.bash
   python3 scripts/deploy/depth_publisher.py

It subscribes to ``/camera/camera/depth/image_rect_raw`` and
``/camera/camera/depth/camera_info`` and publishes ``/g1_policy/depth/image``.
Use ``--depth-topic`` and ``--camera-info-topic`` if the driver uses different names.
There is no custom UDP camera transport.

Starting and Stopping the Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Put the robot in debug mode and support it in the policy's starting pose before launch. The official
safety controller initially holds the activation pose with ``blend_ratio=0``; this launcher does not
move the robot through the former Python startup ramp. Use the exported ONNX bundle with its YAML,
models and initial-state file kept together:

.. code-block:: bash

   source /opt/ros/lyrical/setup.bash
   python3 scripts/deploy/g1_deploy.py \
       --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> --interface <ROBOT_ETHERNET_INTERFACE>

Supply a continuous ``geometry_msgs/msg/Twist`` command on ``/cmd_vel``. Only ``linear.x``,
``linear.y`` and ``angular.z`` enter the policy. For an initial zero command, use another terminal:

.. code-block:: bash

   ros2 topic pub --rate 50 /cmd_vel geometry_msgs/msg/Twist '{}'

After the launcher reports ``Fresh policy output ready``, the operator can enable policy control:

.. code-block:: bash

   ros2 param set /safety_controller blend_ratio 1.0

The official controller blends from the activation pose toward the policy with a maximum blend
rate of 0.3 per second. Replace the zero-command publisher with a ROS teleoperation source when
needed; use only one command source and keep its publication interval below 0.2 seconds.
The former ``go``/``go2``, keyboard heading control and Python remote-button handling are not used.

To request a latched stop, publish:

.. code-block:: bash

   ros2 topic pub --once /g1_policy/stop std_msgs/msg/Empty '{}'

``Ctrl+C`` also requests damping and waits for the controller switch before shutting down the ROS
processes. Setting ``blend_ratio`` back to zero holds the activation pose; it is not the damping stop.

.. warning::

   Keep the robot supported during bringup. ROS service-based stopping depends on a responsive
   controller manager and hardware connection; it is not a hardware emergency stop. Validate the
   robot's independent stopping procedure before hardware use.


Further Reading
---------------

- :doc:`../05_leapp/deploying_exported_policies_with_leapp`
- `Isaac ROS Deploy <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html>`__
- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- :class:`~envs.LeappDeploymentEnv` API reference
