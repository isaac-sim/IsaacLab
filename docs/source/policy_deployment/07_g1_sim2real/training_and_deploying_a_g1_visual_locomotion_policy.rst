Training and Deploying a G1 Visual Locomotion Policy
====================================================

.. currentmodule:: isaaclab

This guide follows one policy from training to a walking robot: a teacher trained with PPO, a
student distilled from it that sees through a chest-mounted depth camera, a LEAPP export, and a
control loop that drives a physical Unitree G1 over DDS.

:doc:`../05_leapp/deploying_exported_policies_with_leapp` ends at
:class:`~envs.LeappDeploymentEnv`, which runs an exported policy back in simulation. This guide
continues from there onto hardware. The inference stage is unchanged — it is the same
``leapp.InferenceManager`` — and what this guide adds is the robot at both ends of it.

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

In addition to the ``leapp`` extra required by export, hardware deployment needs
``unitree_sdk2py`` with ``cyclonedds==0.10.2``, a wired connection to the robot's subnet, and a
depth publisher running on the robot's onboard computer.

.. note::

   The ``cyclonedds`` pin is load-bearing. A newer CycloneDDS enforces XTypes type consistency at
   match time, and the SDK's Python IDL hashes differently from the robot's C++ IDL. Readers then
   never match. This passes every simulation test, because both ends share the Python serializer,
   and fails only against hardware.


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

The gains are graph outputs, not configuration: the deployment writes the Kp and Kd the policy
asks for on each step.


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

.. literalinclude:: ../../../../scripts/deploy/g1_deploy.py
   :language: python
   :start-at: while teleop.running:
   :end-at: link.wait(CONTROL_DT)
   :dedent: 8

Two details about the ends of this loop.

The graph writes **every joint of the articulation**, fingers included, so its 43 outputs are
matched to the 29 motors ``rt/lowcmd`` carries by name, using the ``element_names`` the export
records. The two orders agree for the first 22 joints and then diverge, which is the kind of
mismatch that produces a walking robot with the wrong arm.

The ramp target is a fixed constant, not something read back from the policy. The operator squares
the robot up in that pose and it has to be the same on every run.

Conversions
^^^^^^^^^^^

Four conversions sit between the robot and the graph. Each is silent when wrong: the arrays keep
their shapes and the robot simply walks badly.

**Quaternion order.** The IMU publishes ``(w, x, y, z)`` and the graph expects ``(x, y, z, w)``.

**IMU mounting offset.** Subtracted before the quaternion reaches the graph.

**Invalid depth.** A real camera returns zero where it got no return. The graph reads zero as a
very close surface, so an unprocessed frame plants obstacles at the robot's feet. Pixels with no
return are filled with the clip range instead.

**Field of view.** The publisher crops to the trained field of view before downsampling. Resizing
the whole frame compresses the horizontal axis by about 3%, and the array is 64x38 either way, so
nothing downstream can detect it.

Starting the Camera
^^^^^^^^^^^^^^^^^^^

The camera runs on the robot's onboard computer, which carries the RealSense drivers. It crops,
downsamples and publishes 64x38 frames as unsigned 16-bit millimetres over UDP.

.. code-block:: bash

   ssh unitree@192.168.123.164 \
       '/usr/bin/python3 ~/depth_publisher.py --host <WORKSTATION_IP>'

Starting the Policy
^^^^^^^^^^^^^^^^^^^

Put the robot in debug mode and support it on a stand before starting.

.. code-block:: bash

   uv run --extra leapp python scripts/deploy/g1_deploy.py \
       --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
       --interface <NETWORK_INTERFACE> --domain_id 0

The loop takes control in two steps. Typing ``go`` ramps the robot to the policy's start pose over
three seconds and then holds it. Typing ``go2`` engages the policy three seconds later. ``rt/lowcmd``
keeps flowing throughout, so there is no watchdog gap between the two.

.. warning::

   Keep the robot supported until the policy is engaged. The start pose is held with PD control and
   no policy, and it is not statically stable at these gains.


Further Reading
---------------

- :doc:`../05_leapp/deploying_exported_policies_with_leapp`
- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- :class:`~envs.LeappDeploymentEnv` API reference
