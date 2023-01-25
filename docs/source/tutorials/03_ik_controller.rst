Using a task-space controller
=============================

In the previous tutorials, we learned how to create a scene and control a robotic arm using a
joint-space controller. In this tutorial, we will learn how to use a task-space controller to
control the robot. More specifically, we will use the :class:`DifferentialInverseKinematics` class
to track a desired end-effector pose command.

The Code
~~~~~~~~

The tutorial corresponds to the ``play_ik_control.py`` script in the ``orbit/source/standalone/demo`` directory.


.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :emphasize-lines: 44-47,130-138,147,201-217
   :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

While using any task-space controller, it is important to ensure that the provided
quantities are in the correct frames. When parallelizing environment instances using the cloner,
each environment itself can be thought of having its own local frame. However, from the way
the physics engine is implemented in Isaac Sim, all environments exist on the same stage and
thus there is a unique global frame for the entire simulation. In summary, there are the three
main frames that are used in Orbit:

- The simulation world frame (denoted as ``w``), which is the frame of the entire simulation.
- The local environment frame (denoted as ``e``), which is the frame of the local environment.
- The robot's base frame (denoted as ``b``), which is the frame of the robot's base link.

In the current scenario, where the robot is mounted on the table, the base frame of the robot coincides with
the local environment frame. However, this is not always the case. For example, in a scenario where the robot
is a floating-base system. The location of the environment frames are obtained from the
:attr:`envs_positions` value returned by the cloner.


Creating an IK controller
-------------------------

Computing the inverse kinematics (IK) of a robot is a common task in robotics.
The :class:`DifferentialInverseKinematics` class computes the desired joint positions
for a robot to reach a desired end-effector pose. The included implementation performs
the computation in a batched format and is optimized for speed.

Since in many robots the end-effector is not a rigid body, the simulator does not provide
the pose and Jacobian of the end-effector directly. Instead, the obtained Jacobian
is that of the parent body and not the end-effector. Thus, the IK controller takes in
as input the end-effector offset from the parent frame. This offset is typically specified
in the robot's configuration instance and thus is obtained from there.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 130-138
   :linenos:
   :lineno-start: 130


Computing robot command
-----------------------

The IK controller separates the operation of setting the desired command and
computing the desired joint positions. This is done to allow for the user to
run the IK controller at a different frequency than the robot's control frequency.

The :attr:`set_command` method takes in the desired end-effector pose as a single
batched array. The first three columns correspond to the desired position and the
last four columns correspond to the desired quaternion orientation in ``(w, x, y, z)``
order.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 201-202
   :linenos:
   :lineno-start: 201

We can then compute the desired joint positions using the :attr:`compute` method.
The method takes in the current end-effector position, orientation, Jacobian, and
current joint positions. We read the Jacobian matrix from the robot's data, which uses
its value computed from the physics engine.


.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 203-209
   :linenos:
   :lineno-start: 203

While the IK controller returns the desired joint positions, we need to convert
them to the robot's action space. This is done by subtracting joint positions
offsets from the desired joint positions. The joint offsets are obtained from the
robot's data which is a constant value obtained from the robot's configuration.
For more details, we suggest reading the :doc:`/source/api/orbit.actuators.group` tutorial.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 210-215
   :linenos:
   :lineno-start: 210

These actions can then be applied on the robot, as done in the previous tutorials.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 216-219
   :linenos:
   :lineno-start: 216

Using markers for displaying frames
-----------------------------------

We will use the :class:`StaticMarker` class to display the current and desired end-effector poses.
The marker class takes as input the associated prim name, the number of markers to display, the
USD file corresponding to the marker, and the scale of the marker. By default, it uses a frame marker
to display the pose.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 81-83
   :linenos:
   :lineno-start: 81

We can then set the pose of the marker using the :attr:`set_world_poses` method.
It is important to ensure that the set poses are in the simulation world frame and not the
local environment frame.

.. literalinclude:: ../../../source/standalone/demo/play_ik_control.py
   :language: python
   :lines: 223-229
   :linenos:
   :lineno-start: 223


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demo/play_ik_control.py --robot franka_panda --num_envs 128

The script will start a simulation with 128 robots. The robots will be controlled using a task-space controller.
The current and desired end-effector poses should be displayed using frame markers. When the robot reaches the
desired pose, the command should cycle through to the next pose specified in the script.
To stop the simulation, you can either close the window, or press the ``STOP`` button in the UI, or
press ``Ctrl+C`` in the terminal.
