.. _how_to_articulation_label:

Interacting with an articulation
================================

In the previous tutorial, we explained the essential workings of the standalone script and how to
play the simulator. This tutorial shows how to add a robotic arm to the stage and control the
arm by providing random joint commands.

The tutorial will cover how to use the robot classes provided in Orbit. This includes spawning,
initializing, resetting, and controlling the robot.


The Code
~~~~~~~~

The tutorial corresponds to the ``arms.py`` script in the ``orbit/source/standalone/demos`` directory.

.. dropdown:: :fa:`eye,mr-1` Code for `arms.py`

   .. literalinclude:: ../../../../source/standalone/demos/arms.py
      :language: python
      :emphasize-lines: 74-85,92-95,105-120,121-125,126-135,146-147
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The scene is designed similarly to the previous tutorial.

Here, we add two articulated robots to the scene and apply some actions to them. You can choose from franka_panda or ur10 robots.

In the following, we will detail the ``add_robots`` function, which is responsible for adding the robots to the scene, and the ``run_simulator`` function, which steps the simulator, applies some actions to the robot, and handles their reset.

Adding the robots
-----------------

We create an instance of the single-arm robot class using a pre-defined configuration object in `Orbit`. This object contains information on the associated USD file, default initial state, actuator models for different joints, and other meta information about the robot's kinematics.
All pre-defined config files can be found in `orbit/source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config`.

Single arm manipulators refer to robotic arms with a fixed base. These robots are at the ground height, i.e., `z=0`. Similar to previous objects, a spawn function is defined in each configuration, which is used to place the robot in the scene. Again, we provide the prim path, the spawn configuration, and the translation to the spawn function.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 74-83
   :linenos:
   :lineno-start: 74

As we want to articulate the robot and enable it to move, we need to model it as a combination of fixed and articulated joints. While the articulated joints are defined in the configuration, we want to initialize their physics handles. In `Orbit`, we provide this behavior in the form of an :class:`Articulation` class that allows us to set actions to the articulated joints and retrieve the robot's current state.
Here, multiple prims can be grouped by specifying their paths as regex patterns.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 84-85
   :linenos:
   :lineno-start: 84

Please note that the physics handles are initialized when the simulation is played. Thus, we must call ``sim.reset()`` before accessing the physics handles.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 146-147
   :linenos:
   :lineno-start: 146


Running the simulation loop
---------------------------

In this tutorial, we step the simulation, apply some actions to the robot, and reset the robot at regular intervals.

At first, we generate a joint position target that should be achieved. Every articulation class contains a :class:`ArticulationData` object that contains the current state of the robot. This object can be used to retrieve the current state of the robot as well as some default values. Here, we use it to get the default joint positions and add a slight random random offset to get a target position.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 92-95
   :linenos:
   :lineno-start: 92

As long as the simulation runs, we reset the robot regularly.
We first acquire the default joint position and velocity from the data buffer to perform the reset. By calling the :meth:`Articulation.write_joint_state_to_sim` method, we directly write these values into the PhysX buffer. Then, we call :meth:`Articulation.reset` to reset the robot to its default state.
Following the reset, a new target position of the robot and, if present, the gripper is generated.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 105-120
   :linenos:
   :lineno-start: 105


If a gripper is present, we toggle the command regularly to simulate a grasp and release action.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 121-125
   :linenos:
   :lineno-start: 121

The joint position target is set to the Articulation object by calling the :meth:`Articulation.set_joint_target` method. Similar methods exist to set velocity and effort targets depending on the use case. Afterward, the values are again written into the PhysX buffer before the simulation is stepped. Finally, we update the articulation object's internal buffers to reflect the robot's new state.

.. literalinclude:: ../../../../source/standalone/demos/arms.py
   :language: python
   :lines: 126-135
   :linenos:
   :lineno-start: 126


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demos/arms.py --robot franka_panda


This command should open a stage with a ground plane, lights, and robots.
The simulation should play with the robot arms going to random joint configurations. The
gripper, if present, should be opening or closing at regular intervals. To stop the simulation,
you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal

In addition to the demo script for playing single-arm manipulators, we also provide a script
for spawning a few quadrupeds:

.. code-block:: bash

    # Quadruped -- Spawns ANYmal C, ANYmal B, Unitree A1 on one stage
   ./orbit.sh -p source/standalone/demos/quadrupeds.py

In this tutorial, we saw how to spawn a robot multiple times and wrap it in an Articulation class to initialize all physics handles, and that lets us control the robot. We also saw how to reset the robot and set joint targets.
