Using a task-space controller
=============================

.. currentmodule:: isaaclab

In the previous tutorials, we have joint-space controllers to control the robot. However, in many
cases, it is more intuitive to control the robot using a task-space controller. For example, if we
want to teleoperate the robot, it is easier to specify the desired end-effector pose rather than
the desired joint positions.

In this tutorial, we will learn how to use a task-space controller to control the robot.
We will use the :class:`controllers.DifferentialIKController` class to track a desired
end-effector pose command.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_diff_ik.py`` script in the
``scripts/tutorials/05_controllers`` directory.


.. dropdown:: Code for run_diff_ik.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
      :language: python
      :emphasize-lines: 98-100, 121-136, 155-157, 161-171
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

While using any task-space controller, it is important to ensure that the provided
quantities are in the correct frames. When parallelizing environment instances, they are
all existing in the same unique simulation world frame. However, typically, we want each
environment itself to have its own local frame. This is accessible through the
:attr:`scene.InteractiveScene.env_origins` attribute.

In our APIs, we use the following notation for frames:

- The simulation world frame (denoted as ``w``), which is the frame of the entire simulation.
- The local environment frame (denoted as ``e``), which is the frame of the local environment.
- The robot's base frame (denoted as ``b``), which is the frame of the robot's base link.

Since the asset instances are not "aware" of the local environment frame, they return
their states in the simulation world frame. Thus, we need to convert the obtained
quantities to the local environment frame. This is done by subtracting the local environment
origin from the obtained quantities.


Creating an IK controller
-------------------------

The :class:`~controllers.DifferentialIKController` class computes the desired joint
positions for a robot to reach a desired end-effector pose. The included implementation
performs the computation in a batched format and uses PyTorch operations. It supports
different types of inverse kinematics solvers, including the damped least-squares method
and the pseudo-inverse method. These solvers can be specified using the
:attr:`~controllers.DifferentialIKControllerCfg.ik_method` argument.
Additionally, the controller can handle commands as both relative and absolute poses.

In this tutorial, we will use the damped least-squares method to compute the desired
joint positions. Additionally, since we want to track desired end-effector poses, we
will use the absolute pose command mode.

.. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :start-at: # Create controller
   :end-at: diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

Obtaining the robot's joint and body indices
--------------------------------------------

The IK controller implementation is a computation-only class. Thus, it expects the
user to provide the necessary information about the robot. This includes the robot's
joint positions, current end-effector pose, and the Jacobian matrix.

While the attribute :attr:`assets.ArticulationData.joint_pos` provides the joint positions,
we only want the joint positions of the robot's arm, and not the gripper. Similarly, while
the attribute :attr:`assets.ArticulationData.body_state_w` provides the state of all the
robot's bodies, we only want the state of the robot's end-effector. Thus, we need to
index into these arrays to obtain the desired quantities.

For this, the articulation class provides the methods :meth:`~assets.Articulation.find_joints`
and :meth:`~assets.Articulation.find_bodies`. These methods take in the names of the joints
and bodies and return their corresponding indices.

While you may directly use these methods to obtain the indices, we recommend using the
:attr:`~managers.SceneEntityCfg` class to resolve the indices. This class is used in various
places in the APIs to extract certain information from a scene entity. Internally, it
calls the above methods to obtain the indices. However, it also performs some additional
checks to ensure that the provided names are valid. Thus, it is a safer option to use
this class.

.. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :start-at: # Specify robot-specific parameters
   :end-before: # Define simulation stepping


Computing robot command
-----------------------

The IK controller separates the operation of setting the desired command and
computing the desired joint positions. This is done to allow for the user to
run the IK controller at a different frequency than the robot's control frequency.

The :meth:`~controllers.DifferentialIKController.set_command` method takes in
the desired end-effector pose as a single batched array. The pose is specified in
the robot's base frame.

.. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :start-at: # reset controller
   :end-at: diff_ik_controller.set_command(ik_commands)

We can then compute the desired joint positions using the
:meth:`~controllers.DifferentialIKController.compute` method.
The method takes in the current end-effector pose (in base frame), Jacobian, and
current joint positions. We read the Jacobian matrix from the robot's data, which uses
its value computed from the physics engine.


.. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :start-at: # obtain quantities from simulation
   :end-at: joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

The computed joint position targets can then be applied on the robot, as done in the
previous tutorials.

.. literalinclude:: ../../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :start-at: # apply actions
   :end-at: scene.write_data_to_sim()


The Code Execution
~~~~~~~~~~~~~~~~~~


Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --robot franka_panda --num_envs 128

The script will start a simulation with 128 robots. The robots will be controlled using the IK controller.
The current and desired end-effector poses should be displayed using frame markers. When the robot reaches
the desired pose, the command should cycle through to the next pose specified in the script.

.. figure:: ../../_static/tutorials/tutorial_task_space_controller.jpg
    :align: center
    :figwidth: 100%
    :alt: result of run_diff_ik.py

To stop the simulation, you can either close the window,  or press ``Ctrl+C`` in the terminal.
