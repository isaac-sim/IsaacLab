Adding a robotic arm to the scene
=================================

In the previous tutorial, we explained the basic working of the standalone script and how to
play the simulator. This tutorial shows how to add a robotic arm into the stage and control the
arm by providing random joint commands.

The tutorial will cover how to use the robot classes provided in Orbit. This includes spawning,
initializing, resetting, and controlling the robot.


The Code
~~~~~~~~

The tutorial corresponds to the ``play_arms.py`` script in the ``orbit/source/standalone/demo`` directory.


.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :emphasize-lines: 45-47,92-95,100-103,130-133,144-147,152-154
   :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Most of the code is the same as the previous tutorial. The only difference is that we are adding a robot to the scene.
We explain the changes below.

Designing the simulation scene
------------------------------

The single arm manipulators refer to robotic arms with a fixed base. These robots are at the ground height, i.e. `z=0`.
Accordingly, the robots are placed on a table that define their workspace. In this tutorial, we spawn two robots on two
different tables. The first robot is placed on the left table ``/World/Table_1`` and the second robot is placed on the
right table ``/World/Table_2``.

The tables are loaded from their respective USD file which is hosted on Omniverse Nucleus server

.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 80-83
   :linenos:
   :lineno-start: 80

Next, we create an instance of the single arm robot class. The robot class is initialized with a configuration object
that contains information on the associated USD file, default initial state, actuator models for different joints, and
other meta-information about the robot kinematics. The robot class provides method to spawn the robot in the scene at
a given position and orientation, if provided.

In this tutorial, we disambiguate the robot configuration to load through the parsed command line argument. In Orbit,
we include pre-configured instances of the configuration class to simplify usage. After creating
the robot instance, we can spawn it at the origin defined by the table's location.


.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 84-95
   :linenos:
   :lineno-start: 84

Initializing the robot
----------------------

Prims in the scene with physics schemas enabled on them have an associated physics handle created by the
physics engine. These handles are created only when the simulator starts playing. After that, the allocated
tensor buffers used to store the associated prim's state are accessible. This data is exposed through *physics
views* that provide a convenient interface to access the state of the prims. The physics views can be used to
group or encapsulate multiple prims and obtain their data in a batched manner.

Using the robot class in Orbit, a user can initialize the physics views to obtain views over the articulation
and essential rigid bodies (such as the end-effector) as shown below. Multiple prims are grouped together by
specifying their paths as regex patterns.

.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 97-103
   :linenos:
   :lineno-start: 97

Running the simulation loop
---------------------------

The robot class provides a method to obtain the default state of the robot. This state is the initial state of the
robot when it is spawned in the scene. The default state is a tuple of two tensors, one for the joint positions and
the other for the joint velocities. It is used to reset the robot to its initial state at a pre-defined interval of
steps.

.. danger::
    Since the underlying physics engine in Isaac Sim does not separate the kinematics forwarding and dynamics stepping,
    the robot's state does not take into affect until after stepping the simulation.

.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 126-133
   :linenos:
   :lineno-start: 126


At the start of an episode, we randmly generate the joint commands for the arm. However, we toggle the gripper command
at a regular interval to simulate a grasp and release action. The robot class provides a method to apply the joint
commands. The type of command is configured in the robot configuration object.

.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 140-147
   :linenos:
   :lineno-start: 140

After stepping the simulator, we can obtain the current state of the robot. The robot class provides a method to
update the buffers by reading the data through the physics views. By default, the simulation engine provides all data
in the world frame. Thus, the update method also takes care of transforming quantities to other frames such as the
base frame of the robot.

.. literalinclude:: ../../../source/standalone/demo/play_arms.py
   :language: python
   :lines: 151-154
   :linenos:
   :lineno-start: 151

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demo/play_arms.py --robot franka_panda


This should open a stage with a ground plane, lights, tables and robots.
The simulation should be playing with the robot arms going to random joint configurations. The
gripper, if present, should be opening or closing at regular intervals. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal

In addition to the demo script for playing single arm manipulators, we also provide scripts
for playing other robots such as quadrupeds or mobile manipulators. You can run these as follows:

.. code-block:: bash

    # Quadruped -- Spawns ANYmal C, ANYmal B, Unitree A1 on one stage
   ./orbit.sh -p source/standalone/demo/play_quadrupeds.py

   # Mobile manipulator -- Spawns Franka Panda on Clearpath Ridgeback
   ./orbit.sh -p source/standalone/demo/play_ridgeback_franka.py

In this tutorial, we saw how to spawn a robot multiple times and initialize the physics views to access
the simulation state of the robots. In the next tutorial, we will see how to simplify duplicating a simulation
scene multiple times by using the cloner APIs.
