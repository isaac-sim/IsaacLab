.. _how_to_camera_label:


Using the Camera Sensor
=======================

This tutorial demonstrates using the :class:`Camera` from the Orbit framework. The camera sensor is created and interfaced through the Omniverse Replicator API.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_usd_camera.py`` script in the
``orbit/source/standalone/tutorials/04_sensors`` directory.

.. dropdown:: Code for run_usd_camera.py
   :icon: code

   .. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
      :language: python
      :emphasize-lines: 102-112,113-114,121-123,125-133,135-139,140-141,155-179,181-215
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

As usual, we design a minimal scene for this example. In addition to the GroundPlane and a Distant Light, we add some simple shapes to the scene for this how-to guide.

In the following, we will detail the ``add_sensor`` function, responsible for adding the camera to the scene, and the ``run_simulator`` function, which steps the simulator and saves the rendered images.

Adding the camera sensor
------------------------

As Orbit is a config-driven framework, the camera is defined over its config class, :class:`CameraCfg`. Whereas parameters that do not depend on the used camera type are direct arguments of this class, all camera type-related arguments are defined in the spawn config. With the camera type, we refer to either PinholeCamera or FisheyeCamera. The type-independent parameters are, e.g., the data types to capture (e.g., "rgb," "distance_to_image_plane," "normals," "motion_vectors," "semantic_segmentation"), the width and height of the image and its offset. The spawn configurations defined parameters such as the aperture or the focus distance. These are given together with all other spawn-related configs under :class:`PinholeCameraCfg` and :class:`FisheyeCameraCfg`.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 102-112
   :linenos:
   :lineno-start: 102

While in previous how-to guides, we had to manually call the function to spawn the object into the scene, sensors already include this functionality when initialized. Consequently, we only have to pass the ``camera_cfg`` to the :class:`Camera` class.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 113-114
   :linenos:
   :lineno-start: 113

Please note that the Replicator Render Products, camera buffers and other aspects are initialized when the simulation is played. Thus, we must call ``sim.play()`` before rendering camera images.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 140-141
   :linenos:
   :lineno-start: 140


Running the simulation loop
---------------------------

In this tutorial, we step the simulation and efficiently render and save the camera images. To save the images, we use the Replicator BasicWriter (more information `here <www.docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_recorder.html?highlight=basic%20writer#writer-parameters>`_).

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 121-123
   :linenos:
   :lineno-start: 121

The camera's position and orientation can be set within the config by defining the offset relative to the parent frame. Alternatively, we can set the position and orientation of the camera directly in the scene. In this example, we provide two options for the latter: either by providing the camera center and a target point (:meth:`Camera:set_world_poses_from_view`) or by providing the camera center and a rotation quaternion (:meth:`Camera:set_world_poses_from_view`). To allow for maximum flexibility, the provided quaternions can be provided in three conventions:

* ``"opengl"`` - forward axis: -Z - up axis +Y - OpenGL (Usd.Camera) convention
* ``"ros"``    - forward axis: +Z - up axis -Y - ROS convention
* ``"world"``  - forward axis: +X - up axis +Z - World Frame convention

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 125-133
   :linenos:
   :lineno-start: 125

While stepping the simulator, we write the images to the defined folder. Therefore, we first convert them to numpy arrays before packing them in a dictionary, which the BasicWriter can handle.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 155-179
   :linenos:
   :lineno-start: 155

In addition, we provide the functionality to project the depth image into 3D space. This reprojection is done by using the camera intrinsics and the depth image. The resulting point cloud is visualized using the ``_debug_draw`` extension of Isaac Sim.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 181-215
   :linenos:
   :lineno-start: 99

.. attention::

    For all replicator buffers to be filled with the latest data, we may need to render the simulation multiple times.

    .. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
        :language: python
        :lines: 135-139
        :linenos:
        :lineno-start: 135

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py


This should open a stage with a ground plane, lights, and some slowly falling down objects.
To stop the simulation, you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal.

This guide showed how to spawn a camera into the scene and save data. The following guides present other sensors and the possibility of putting them on a robot.
