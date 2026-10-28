Training and Deploying a G1 Visual Locomotion Policy
====================================================

.. currentmodule:: isaaclab

This guide follows one policy from training to a walking robot: a teacher trained with PPO, a
student distilled from it that sees through a chest-mounted depth camera, a LEAPP export, and a
control loop that drives a physical Unitree G1 over DDS.

:doc:`../05_leapp/deploying_exported_policies_with_leapp` ends at
:class:`~envs.LeappDeploymentEnv`, which runs an exported policy back in simulation. This guide
continues from there onto hardware using the `Isaac ROS Deploy node runtime
<https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/concepts/ros2_nodes_runtime.html>`__.
Its ``InputBuilderNode``, ``TritonNode`` and ``OutputBuilderNode`` execute the exported graph.
``g1_deploy.py`` connects that runtime to the robot's Unitree DDS interface and depth stream.

.. note::

   The pipeline below is specific to a 29-DoF Unitree G1 with a RealSense D435i on the chest. The
   structure generalizes, but every number in it was measured on that robot.

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

For hardware deployment, prepare an `Isaac ROS 5.0 environment
<https://nvidia-isaac-ros.github.io/getting_started/index.html>`__ with GPU access and install
``isaac_ros_deploy_bringup`` and ``isaac_ros_triton`` with their runtime dependencies. The deployment
environment must also reach the robot's Ethernet subnet and receive the UDP depth stream.
From the repository root inside that environment, create a Python environment that can import
the installed ROS packages:

.. code-block:: bash

   source /opt/ros/lyrical/setup.bash
   uv venv --python /usr/bin/python3 --system-site-packages .venv-g1-deploy

Install CycloneDDS and the Unitree Python SDK in this deployment environment.
Use version ``0.10.2`` for compatibility with the robot. Python 3.12 requires building
the DDS library first, following the `Unitree SDK installation instructions
<https://github.com/unitreerobotics/unitree_sdk2_python#installation>`__:

.. code-block:: bash

   sudo apt-get install -y build-essential cmake git
   git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds-0.10.2
   export CYCLONEDDS_HOME="$HOME/.local/cyclonedds-0.10.2"
   cmake -S /tmp/cyclonedds-0.10.2 -B /tmp/cyclonedds-0.10.2/build -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME"
   cmake --build /tmp/cyclonedds-0.10.2/build --target install --parallel
   uv pip install --python .venv-g1-deploy/bin/python numpy pyyaml "cyclonedds==0.10.2" \
       "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"

Source the ROS setup file and set ``CYCLONEDDS_HOME`` to the same path in each deployment terminal.
Check the ROS and SDK imports before connecting to the robot:

.. code-block:: bash

   .venv-g1-deploy/bin/python -c \
       "import rclpy; from isaac_ros_deploy_interfaces.msg import JointCommand; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('ROS and SDK ready')"

Connect the workstation to the robot's subnet over Ethernet. The robot's onboard computer also
needs the RealSense drivers and a depth publisher, as described in `Starting the Camera`_ below.


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

Resolving the Connections Against Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~envs.LeappDeploymentEnv` resolves each ``isaaclab_connection`` in the exported YAML
against a simulated scene. Hardware deployment resolves the same connections against the robot:

.. list-table::
   :header-rows: 1
   :widths: 44 26 30

   * - Connection
     - Simulation
     - Hardware
   * - ``state:robot:joint_pos``
     - articulation
     - ``rt/lowstate``
   * - ``state:robot:joint_vel``
     - articulation
     - ``rt/lowstate``
   * - ``state:robot:root_quat_w``
     - scene root
     - IMU
   * - ``state:robot:root_ang_vel_b``
     - scene root
     - IMU gyroscope
   * - ``state:depth_camera:output:distance_to_image_plane``
     - rendered
     - UDP depth stream
   * - ``command:base_velocity``
     - ``CommandManager``
     - keyboard
   * - ``write:robot:set_joint_position_target_index``
     - articulation
     - ``rt/lowcmd``
   * - ``write:robot:write_joint_stiffness_to_sim_index``
     - articulation
     - ``rt/lowcmd`` Kp
   * - ``write:robot:write_joint_damping_to_sim_index``
     - articulation
     - ``rt/lowcmd`` Kd

The Control Loop
^^^^^^^^^^^^^^^^

The script publishes sensor and command tensors as ROS ``TensorList`` messages. Isaac ROS Deploy
runs the ONNX graph and returns ``JointCommand`` messages containing targets and gains. The script
checks their timestamps, joint layout and finite values before writing the 29 motor commands.
The official input builder is configured for 50 Hz; the loop waits for a fresh result each step.

.. literalinclude:: ../../../../scripts/deploy/g1_deploy.py
   :language: python
   :start-at: while teleop.running:
   :end-at: link.write(joint_pos, kp, kd)
   :dedent: 16

Two details about the ends of this loop.

Joint targets are mapped to the 29 body motors by joint name using the exported ``element_names``.

Before enabling the policy, the robot transitions to the predefined starting pose over three seconds
and holds it until the operator confirms. The inference graph is warmed up before taking control;
its enable gate stays closed during the ramp and hold, and feedback resets when the policy is enabled.

Conversions
^^^^^^^^^^^

The following conversions adapt robot sensor data to the graph's input format.

**Quaternion order.** The IMU publishes ``(w, x, y, z)`` and the graph expects ``(x, y, z, w)``.

**IMU mounting offset.** ``--imu_tilt=PITCH,ROLL`` supplies the IMU pitch and roll in degrees
measured with the pelvis level. Both orientation and angular velocity are corrected before reaching
the graph. The default is ``0,0``.

**Invalid depth.** A real camera returns zero where it got no return. The graph reads zero as a
very close surface, so an unprocessed frame plants obstacles at the robot's feet. Pixels with no
return are filled with the clip range instead.

**Field of view.** The publisher crops to the trained field of view before downsampling. Resizing
the whole frame compresses the horizontal axis by about 3%, and the array is 64x38 either way, so
nothing downstream can detect it.

Starting the Camera
^^^^^^^^^^^^^^^^^^^

Run ``scripts/deploy/depth_publisher.py`` on the G1 computer connected to the chest RealSense
camera (PC2). The script is standalone and requires only NumPy and ``pyrealsense2``; Isaac Lab
does not need to be installed on the robot.

From the Isaac Lab repository root on the workstation, copy the script to PC2. The commands below
use ``192.168.123.164`` as PC2's address; replace it if your robot uses a different address.

.. code-block:: bash

   scp scripts/deploy/depth_publisher.py unitree@192.168.123.164:~/depth_publisher.py

Install the Python dependencies on PC2 using the same interpreter that will run the publisher.
The RealSense drivers must be installed and the camera available to the ``unitree`` user.

.. code-block:: bash

   ssh unitree@192.168.123.164 \
       '/usr/bin/python3 -m pip install --user numpy pyrealsense2'

Start the publisher from the workstation and leave this SSH session running. Replace
``<WORKSTATION_IP>`` with the workstation's address on the robot network, which receives the
depth stream and runs the policy in a second terminal.

.. code-block:: bash

   ssh unitree@192.168.123.164 \
       '/usr/bin/python3 ~/depth_publisher.py --host <WORKSTATION_IP>'

The publisher requests 848x480 depth at 90 Hz, crops using the camera intrinsics toward the
trained 87.0° horizontal and 58.8° vertical field of view, and averages valid pixels into 64x38
frames. It sends unsigned 16-bit millimetres to UDP port 5601 and reports the publishing rate
every five seconds. Missing returns stay zero; the control loop fills them with the clip range.
If camera startup fails, the script retries and reduces the requested rate to 60 Hz on its final
attempt. ``--port`` must match the policy's ``--depth_port`` if you change the default port.
Press ``Ctrl+C`` to stop the publisher after stopping the policy.

Starting the Policy
^^^^^^^^^^^^^^^^^^^

Put the robot in debug mode and support it on a stand before starting. Run from the repository root
in the Isaac ROS deployment environment prepared above. Use the exported ONNX bundle, keeping its
YAML, models and initial-state file together; the script creates temporary ROS port metadata without
modifying the source bundle.

.. code-block:: bash

   source /opt/ros/lyrical/setup.bash
   .venv-g1-deploy/bin/python scripts/deploy/g1_deploy.py \
       --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
       --interface <NETWORK_INTERFACE> --domain_id 0

The loop takes control in two steps. Typing ``go`` ramps the robot to the policy's start pose over
three seconds and then holds it. Type ``go2`` within 300 seconds to engage the policy three seconds
later; other input is ignored. ``rt/lowcmd`` keeps flowing throughout, so there is no watchdog gap
between the two. The robot must be upright after the ramp and before policy engagement.

The policy starts with zero forward speed and adopts the robot's current heading. Keyboard controls
take effect without Enter:

.. code-block:: text

   w/s    forward speed +/- 0.1 m/s (0 to 1 m/s)
   q/e    heading target +/- 15 degrees
   space  zero forward speed
   r      adopt the current heading
   x      quit
   ?      show controls

The yaw-rate command is half the wrapped heading error, limited to ±1 rad/s. Space leaves the
heading target unchanged. ``Ctrl+C``, ``L2+B``, ``L1+A`` or ``select`` stops the run with damping,
including during the ramp and hold. Exiting the policy restores the terminal settings.

.. warning::

   Keep the robot supported until the policy is engaged. The start pose is held with PD control and
   no policy, and it is not statically stable at these gains.


Further Reading
---------------

- :doc:`../05_leapp/deploying_exported_policies_with_leapp`
- `Isaac ROS Deploy <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html>`__
- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- :class:`~envs.LeappDeploymentEnv` API reference
