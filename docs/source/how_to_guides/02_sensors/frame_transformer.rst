.. _how_to_frame_transformer_label:

Using the Frame Transformer Sensor in Orbit
===========================================

This tutorial demonstrates using the :class:`FrameTransformer` sensor in the ORBIT framework.
The :class:`FrameTransformer` sensor is used to report the transformation of one or more frames (target frames) with respect to another frame (source frame)


The Code
~~~~~~~~

The tutorial corresponds to the ``frame_transformer.py`` script in the ``orbit/source/standalone/tutorials/02_sensors`` directory.

.. dropdown:: :fa:`eye,mr-1` Code for `frame_transformer.py`

   .. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/frame_transformer.py
      :language: python
      :emphasize-lines: 72-90, 116-123, 125-139, 150-151
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

As usual, we design a minimal scene for this example. In addition to the GroundPlane and a Distant Light, we spawn a quadrupedal robot.
Spawning such an articulated robot in the scene is explained in the :ref:`Articulation How-to-Guide <_how_to_articulation_label>`.

In the following, we will detail the ``add_sensor`` function, responsible for adding the ray caster sensor to the scene.
The ``run_simulator`` function visualizes the different frames given the computed transforms from the sensor and passes the default actions to the robot.

Adding the frame transformer sensor
-----------------------------------

As usual, the frame transformer is defined over its config class, :class:`FrameTransformerCfg`.
We need to specify the source frame (`prim_path`) and the target frames for the Frame Transformer sensor.
The source frame is the frame with respect to which the translations are reported.
We can specify a list of frames for the target frames and include regex patterns to match multiple frames.
In some cases, the target frame is not an individual prim; its relation to a "parent" prim can be defined over an offset.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/frame_transformer.py
   :language: python
   :lines: 72-90
   :linenos:
   :lineno-start: 72

Please note that the buffers, physics handles for the robot, and other aspects are initialized when the simulation is played. Thus, we must call ``sim.reset()``.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/frame_transformer.py
   :language: python
   :lines: 150-151
   :linenos:
   :lineno-start: 150


Running the simulation loop
---------------------------

After each step call, we must update the frame transformer sensor to get the latest transforms.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/frame_transformer.py
   :language: python
   :lines: 116-123
   :linenos:
   :lineno-start: 116

To visualize the transforms, we visualize a different frame in a regular interval based on the transform reported by the sensor.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/frame_transformer.py
   :language: python
   :lines: 125-139
   :linenos:
   :lineno-start: 125


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/02_sensors/frame_transformer.py


This should open a stage with a ground plane, lights, and a quadrupedal robot. Consistently, one frame should be visualized.
To stop the simulation, you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal.

In this guide, we saw how to use a frame transformer sensor. In the following how-to guides, other sensors will be introduced, and how to place sensors on the robot will be explained.
