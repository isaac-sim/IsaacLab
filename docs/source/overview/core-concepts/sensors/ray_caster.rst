.. _overview_sensors_ray_caster:

.. currentmodule:: isaaclab

Ray Caster
=============

.. figure:: ../../../_static/overview/sensors/raycaster_patterns.jpg
    :align: center
    :figwidth: 100%
    :alt: A diagram outlining the basic geometry of frame transformations

The Ray Caster sensor (and the ray caster camera) are similar to RTX based rendering in that they both involve casting rays.  The difference here is that the rays cast by the Ray Caster sensor return strictly collision information along the cast, and the direction of each individual ray can be specified.  They do not bounce, nor are they affected by things like materials or opacity. For each ray specified by the sensor, a line is traced along the path of the ray and the location of first collision with the specified mesh is returned. This is the method used by some of our quadruped examples to measure the local height field.

To keep the sensor performant when there are many cloned environments, the line tracing is done directly in `Warp <https://nvidia.github.io/warp/>`_. This is the reason why specific meshes need to be identified to cast against: that mesh data is loaded onto the device by warp when the sensor is initialized. As a consequence, the current iteration of this sensor only works for literally static meshes (meshes that *are not changed from the defaults specified in their USD file*).  This constraint will be removed in future releases.

Using a ray caster sensor requires a **pattern** and a parent xform to be attached to.  The pattern defines how the rays are cast, while the prim properties defines the orientation and position of the sensor (additional offsets can be specified for more exact placement).  Isaac Lab supports a number of ray casting pattern configurations, including a generic LIDAR and grid pattern.

.. literalinclude:: ../../../../../scripts/demos/sensors/raycaster_sensor.py
    :language: python
    :lines: 40-71

Notice that the units on the pattern config is in degrees! Also, we enable visualization here to explicitly show the pattern in the rendering, but this is not required and should be disabled for performance tuning.

.. figure:: ../../../_static/overview/sensors/raycaster_visualizer.jpg
    :align: center
    :figwidth: 100%
    :alt: Lidar Pattern visualized

Querying the sensor for data can be done at simulation run time like any other sensor.

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
        print(scene["ray_caster"])
        print("Ray cast hit results: ", scene["ray_caster"].data.ray_hits_w)


.. code-block:: bash

    -------------------------------
    Ray-caster @ '/World/envs/env_.*/Robot/base/lidar_cage':
            view type            : <class 'isaacsim.core.prims.xform_prim.XFormPrim'>
            update period (s)    : 0.016666666666666666
            number of meshes     : 1
            number of sensors    : 1
            number of rays/sensor: 18000
            total number of rays : 18000
    Ray cast hit results:  tensor([[[-0.3698,  0.0357,  0.0000],
            [-0.3698,  0.0357,  0.0000],
            [-0.3698,  0.0357,  0.0000],
            ...,
            [    inf,     inf,     inf],
            [    inf,     inf,     inf],
            [    inf,     inf,     inf]]], device='cuda:0')
    -------------------------------

Here we can see the data returned by the sensor itself.  Notice first that there are 3 closed brackets at the beginning and the end: this is because the data returned is batched by the number of sensors. The ray cast pattern itself has also been flattened, and so the dimensions of the array are ``[N, B, 3]`` where ``N`` is the number of sensors, ``B`` is the number of cast rays in the pattern, and 3 is the dimension of the casting space. Finally, notice that the first several values in this casting pattern are the same: this is because the lidar pattern is spherical and we have specified our FOV  to be hemispherical, which includes the poles. In this configuration, the "flattening pattern" becomes apparent: the first 180 entries will be the same because it's the bottom pole of this hemisphere, and there will be 180 of them because our horizontal FOV is 180 degrees with a resolution of 1 degree.

You can use this script to experiment with pattern configurations and build an intuition about how the data is stored by altering the ``triggered`` variable on line 81.

.. dropdown:: Code for raycaster_sensor.py
   :icon: code

   .. literalinclude:: ../../../../../scripts/demos/sensors/raycaster_sensor.py
      :language: python
      :linenos:
