Using the Ray-Caster Camera Sensor in Orbit
===========================================

This tutorial demonstrates using the :class:`RayCasterCamera` sensor as a depth camera from the Orbit framework.
We already saw how to use the USD camera sensor (see :ref:`Camera How-to-Guide <_how_to_camera_label>`) and the ray-caster sensor when used as a height scanner (see :ref:`Height Scanner How-to-Guide <_how_to_ray_caster_label>`).
As the current implementation of the ray-caster is faster and more memory efficient than the USD camera sensor, it is a good alternative when only geometric information is required.
The interfaces for both cameras are identical, including the data buffers they use. However, the initialization configuration differs slightly as no spawn configuration is required for the ray-caster camera.

The Code
~~~~~~~~

The tutorial corresponds to the ``ray_caster_camera.py`` script in the ``orbit/source/standalone/tutorials/02_sensors`` directory.

.. dropdown:: :fa:`eye,mr-1` Code for `ray_caster_camera.py`

   .. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/ray_caster_camera.py
      :language: python
      :emphasize-lines: 68-88,94-106,111-145,156-157
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

We designed a minimal scene for this example, composed of a rough terrain and a distant light.
In the following, we will detail the ``add_sensor`` function, which is responsible for adding the raycaster sensor to the scene, and the ``run_simulator`` function, which steps the simulator and saves the rendered images.

Adding the ray-caster camera sensor
-----------------------------------

As usual, the ray-caster camera is defined over its config class, :class:`RayCasterCameraCfg`.
Like the ray-casting-based height scanner and unlike the USD camera, the ray-caster operates as a virtual sensor and does not require a spawn within the scene. Instead, it is solely attached to a prim, employed to specify its location, along with a potential offset.
For the ray-caster camera, we specify the pinhole camera pattern. The pattern config includes the camera intrinsics and the image dimension, which will define the direction of the rays.
Other parameters are defined in the general config, as they are independent of the pattern. These include data types, offset, or the mesh to be ray-casted against.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/ray_caster_camera.py
   :language: python
   :lines: 68-88
   :linenos:
   :lineno-start: 68

Please note that the buffers and other aspects are initialized when the simulation is played. Thus, we must call ``sim.reset()``.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/ray_caster_camera.py
   :language: python
   :lines: 156-157
   :linenos:
   :lineno-start: 156

.. attention::

    Currently, ray-casting is only supported against a single mesh. We are working on extending this functionality to multiple meshes.


Running the simulation loop
---------------------------

In this tutorial, we step the simulation and efficiently render and save the camera images. To save the images, we use the Replicator BasicWriter (more information `here <www.docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_recorder.html?highlight=basic%20writer#writer-parameters>`_).

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/usd_camera.py
   :language: python
   :lines: 94-96
   :linenos:
   :lineno-start: 94

The camera's position and orientation can be set within the config by defining the offset relative to the parent frame. Alternatively, we can set the position and orientation of the camera directly in the scene. In this example, we provide two options for the latter: either by providing the camera center and a target point (:meth:`RayCasterCamera:set_world_poses_from_view`) or by providing the camera center and a rotation quaternion (:meth:`RayCasterCamera:set_world_poses_from_view`). To allow for maximum flexibility, the provided quaternions can be provided in three conventions:

* `"opengl"` - forward axis: -Z - up axis +Y - OpenGL (Usd.Camera) convention
* `"ros"`    - forward axis: +Z - up axis -Y - ROS convention
* `"world"`  - forward axis: +X - up axis +Z - World Frame convention

This behavior is the same as for the USD camera.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/usd_camera.py
   :language: python
   :lines: 98-106
   :linenos:
   :lineno-start: 98

While stepping the simulator, we update the camera and write the images to the defined folder. Therefore, we first convert them to numpy arrays before packing them in a dictionary, which the BasicWriter can handle.

.. literalinclude:: ../../../../source/standalone/tutorials/02_sensors/usd_camera.py
   :language: python
   :lines: 111-145
   :linenos:
   :lineno-start: 111


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/02_sensors/ray_caster_camera.py


This call should open a stage with a ground plane, lights, and a visualization of the points where the rays hit the mesh.
To stop the simulation, you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal.

In this guide, we saw how to use a ray-caster camera sensor. Moving forward, we will see how sensors can be used further and how to combine and place them on a robot.
