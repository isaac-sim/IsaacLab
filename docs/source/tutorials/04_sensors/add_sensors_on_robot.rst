Adding Sensors on a Robot
=========================

This tutorial demonstrates how to simulate various sensors onboard the quadruped robot ANYmal-C (ANYbotics) using the ORBIT framework. The included sensors are:

- USD-Camera
- Height Scanner
- Contact Sensor

Please review their how-to guides before proceeding with this guide.

The Code
~~~~~~~~

The tutorial corresponds to the ``add_sensors_on_robot.py`` script in the
``orbit/source/standalone/tutorials/04_sensors`` directory.

.. dropdown:: Code for add_sensors_on_robot.py
   :icon: code

   .. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
      :language: python
      :emphasize-lines: 72-90, 116-123, 125-139, 150-151
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The script designs a scene with a ground plane, lights, and two instances of the ANYmal-C robot.
This guide will not explain the individual sensors; please refer to their corresponding how-to guides for more details (see :ref:`Height-Scanner How-to-Guide <how_to_ray_caster_label>` and :ref:`Camera How-to-Guide <how_to_camera_label>`).
Furthermore, how to spawn such an articulated robot in the scene is explained in the :ref:`Articulation How-to-Guide <how_to_articulation_label>`.

In the following, we will detail the ``add_sensor`` function, which is responsible for adding the sensors on the robot.
The ``run_simulator`` function updates the sensors and provides some information on them.

Adding the sensors
------------------

For each sensor, the corresponding config class has to be created, and the corresponding parameters have to be set.
To add the sensors to the robot, the prim paths must be set as a child prim of the robot.
In this case, the sensor will move with the robot. The offsets have to be provided w.r.t. the parent frame on the robot.
The resulting configurations and initialization calls are shown below.

Camera sensor:

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
   :language: python
   :lines: 88-102
   :linenos:
   :lineno-start: 88

Height scanner sensor:

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
   :language: python
   :lines: 103-114
   :linenos:
   :lineno-start: 103

Contact sensor:

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
   :language: python
   :lines: 115-120
   :linenos:
   :lineno-start: 115

Please note that the buffers, physics handles for the camera and robot, and other aspects are initialized when the simulation is played. Thus, we must call ``sim.reset()``.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
   :language: python
   :lines: 181-182
   :linenos:
   :lineno-start: 181


Running the simulation loop
---------------------------

For every simulation step, the sensors are updated and we print some information.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
   :language: python
   :lines: 150-168
   :linenos:
   :lineno-start: 150


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/04_sensors/add_sensors_on_robot.py


This command should open a stage with a ground plane, lights, and two quadrupedal robots.
To stop the simulation, you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal.

In this guide, we saw how to add sensors to a robot and how to update them in the simulation loop.
