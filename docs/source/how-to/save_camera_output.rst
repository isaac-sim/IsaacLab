:orphan:

.. _how-to-save-images-and-3d-reprojection:


Saving rendered images and 3D re-projection
===========================================

.. currentmodule:: isaaclab

This guide accompanied with the ``run_usd_camera.py`` script in the ``IsaacLab/scripts/tutorials/04_sensors``
directory.

.. dropdown:: Code for run_usd_camera.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
      :language: python
      :emphasize-lines: 173-175, 224-229, 231-250
      :linenos:


Saving the images to file
-------------------------

To save camera outputs, we use the :func:`~isaaclab.sensors.camera.utils.save_images_to_file` utility.
It writes a batch of images as a PNG file and does not depend on the renderer backend. The script creates
the output folder once, before the simulation loop:

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # Create the output directory
   :end-at: os.makedirs(output_dir, exist_ok=True)

While stepping the simulator, the outputs of the camera at ``camera_index`` are saved once per data type and
frame: 8-bit color outputs (the RGB image and the colorized segmentations) as PNG files, and floating-point
outputs, such as depth and normals, as NumPy ``.npy`` files.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # Save the camera outputs at camera_index
   :end-at: np.save(


Projection into 3D Space
------------------------

We include utilities to project the depth image into 3D Space. The re-projection operations are done using
PyTorch operations which allows faster computation.

.. code-block:: python

   from isaaclab.utils.math import transform_points, unproject_depth

   # Pointcloud in world frame
   points_3d_cam = unproject_depth(
      camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
   )

   points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)

Alternately, we can use the :meth:`isaaclab.sensors.camera.utils.create_pointcloud_from_depth` function
to create a point cloud from the depth image and transform it to the world frame.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # Derive pointcloud from camera at camera_index
   :end-before: # In the first few steps, things are still being instanced and Camera.data

The resulting point cloud can be visualized with Isaac Lab's :class:`~isaaclab.markers.VisualizationMarkers`.
The accompanying tutorial uses ``RAY_CASTER_MARKER_CFG`` and calls
:meth:`~isaaclab.markers.VisualizationMarkers.visualize` to draw the points in the viewport.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # In the first few steps, things are still being instanced and Camera.data
   :end-at: pc_markers.visualize(translations=pointcloud)


Executing the script
--------------------

To run the accompanying script, execute the following command:

.. code-block:: bash

   # Usage with saving and drawing
   uv run python scripts/tutorials/04_sensors/run_usd_camera.py --save --draw --viz kit

   # Usage with saving only (no visualizer)
   uv run python scripts/tutorials/04_sensors/run_usd_camera.py --save


The simulation should start, and you can observe different objects falling down. An output folder will be created
in the ``IsaacLab/scripts/tutorials/04_sensors`` directory, where the images will be saved as PNG files. Additionally,
you should see the point cloud in the 3D space drawn on the viewport.

To stop the simulation, close the window, or use ``Ctrl+C`` in the terminal.
