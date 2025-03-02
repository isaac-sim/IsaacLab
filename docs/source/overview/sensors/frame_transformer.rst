.. _overview_sensors_frame_transformer:

Frame Transformer
====================

.. figure:: ../../_static/overview/overview_sensors_frame_transformer.jpg
    :align: center
    :figwidth: 100%
    :alt: A diagram outlining the basic geometry of frame transformations

..
  Do YOU want to know where things are relative to other things at a glance?  Then the frame transformer is the sensor for you!*

One of the most common operations that needs to be performed within a physics simulation is the frame transformation: rewriting a vector or quaternion in the basis of an arbitrary euclidean coordinate system. There are many ways to accomplish this within Isaac and USD, but these methods can be cumbersome to implement within Isaac Lab's GPU based simulation and cloned environments. To mitigate this problem, we have designed the Frame Transformer Sensor, that tracks and calculate the relative frame transformations for rigid bodies of interest to the scene.

The sensory is minimally defined by a source frame and a list of target frames.  These definitions take the form of a prim path (for the source) and list of regex capable prim paths the rigid bodies to be tracked (for the targets).

.. literalinclude:: ../../../../scripts/demos/sensors/frame_transformer_sensor.py
    :language: python
    :lines: 38-86

We can now run the scene and query the sensor for data

.. code-block:: python

  def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    .
    .
    .
    # Simulate physics
    while simulation_app.is_running():
      .
      .
      .

      # print information from the sensors
      print("-------------------------------")
      print(scene["specific_transforms"])
      print("relative transforms:", scene["specific_transforms"].data.target_pos_source)
      print("relative orientations:", scene["specific_transforms"].data.target_quat_source)
      print("-------------------------------")
      print(scene["cube_transform"])
      print("relative transform:", scene["cube_transform"].data.target_pos_source)
      print("-------------------------------")
      print(scene["robot_transforms"])
      print("relative transforms:", scene["robot_transforms"].data.target_pos_source)

Let's take a look at the result for tracking specific objects. First, we can take a look at the data coming from the sensors on the feet

.. code-block:: bash

  -------------------------------
  FrameTransformer @ '/World/envs/env_.*/Robot/base':
          tracked body frames: ['base', 'LF_FOOT', 'RF_FOOT']
          number of envs: 1
          source body frame: base
          target frames (count: ['LF_FOOT', 'RF_FOOT']): 2

  relative transforms: tensor([[[ 0.4658,  0.3085, -0.4840],
          [ 0.4487, -0.2959, -0.4828]]], device='cuda:0')
  relative orientations: tensor([[[ 0.9623,  0.0072, -0.2717, -0.0020],
          [ 0.9639,  0.0052, -0.2663, -0.0014]]], device='cuda:0')

.. figure:: ../../_static/overview/overview_sensors_ft_visualizer.jpg
    :align: center
    :figwidth: 100%
    :alt: The frame transformer visualizer

By activating the visualizer, we can see that the frames of the feet are rotated "upward" slightly.  We can also see the explicit relative positions and rotations by querying the sensor for data, which returns these values as a list with the same order as the tracked frames.  This becomes even more apparent if we examine the transforms specified by regex.

.. code-block:: bash

  -------------------------------
  FrameTransformer @ '/World/envs/env_.*/Robot/base':
          tracked body frames: ['base', 'LF_FOOT', 'LF_HIP', 'LF_SHANK', 'LF_THIGH', 'LH_FOOT', 'LH_HIP', 'LH_SHANK', 'LH_THIGH', 'RF_FOOT', 'RF_HIP', 'RF_SHANK', 'RF_THIGH', 'RH_FOOT', 'RH_HIP', 'RH_SHANK', 'RH_THIGH', 'base']
          number of envs: 1
          source body frame: base
          target frames (count: ['LF_FOOT', 'LF_HIP', 'LF_SHANK', 'LF_THIGH', 'LH_FOOT', 'LH_HIP', 'LH_SHANK', 'LH_THIGH', 'RF_FOOT', 'RF_HIP', 'RF_SHANK', 'RF_THIGH', 'RH_FOOT', 'RH_HIP', 'RH_SHANK', 'RH_THIGH', 'base']): 17

  relative transforms: tensor([[[ 4.6581e-01,  3.0846e-01, -4.8398e-01],
          [ 2.9990e-01,  1.0400e-01, -1.7062e-09],
          [ 2.1409e-01,  2.9177e-01, -2.4214e-01],
          [ 3.5980e-01,  1.8780e-01,  1.2608e-03],
          [-4.8813e-01,  3.0973e-01, -4.5927e-01],
          [-2.9990e-01,  1.0400e-01,  2.7044e-09],
          [-2.1495e-01,  2.9264e-01, -2.4198e-01],
          [-3.5980e-01,  1.8780e-01,  1.5582e-03],
          [ 4.4871e-01, -2.9593e-01, -4.8277e-01],
          [ 2.9990e-01, -1.0400e-01, -2.7057e-09],
          [ 1.9971e-01, -2.8554e-01, -2.3778e-01],
          [ 3.5980e-01, -1.8781e-01, -9.1049e-04],
          [-5.0090e-01, -2.9095e-01, -4.5746e-01],
          [-2.9990e-01, -1.0400e-01,  6.3592e-09],
          [-2.1860e-01, -2.8251e-01, -2.5163e-01],
          [-3.5980e-01, -1.8779e-01, -1.8792e-03],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00]]], device='cuda:0')

Here, the sensor is tracking all rigid body children of ``Robot/base``, but this expression is **inclusive**, meaning that the source body itself is also a target. This can be seen both by examining the source and target list, where ``base`` appears twice, and also in the returned data, where the sensor returns the relative transform to itself, (0, 0, 0).

.. dropdown:: Code for frame_transformer_sensor.py
   :icon: code

   .. literalinclude:: ../../../../scripts/demos/sensors/frame_transformer_sensor.py
      :language: python
      :linenos:
