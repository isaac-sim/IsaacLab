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
      :emphasize-lines: 171-179, 229-247, 251-264
      :linenos:


Saving using Replicator Basic Writer
------------------------------------

To save camera outputs, we use the basic write class from Omniverse Replicator. This class allows us to save the
images in a numpy format. For more information on the basic writer, please check the
`documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html>`_.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: rep_writer = rep.BasicWriter(
   :end-before: # Camera positions, targets, orientations

While stepping the simulator, the images can be saved to the defined folder. Since the BasicWriter only supports
saving data using NumPy format, we first need to convert the PyTorch sensors to NumPy arrays before packing
them in a dictionary.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # Save images from camera at camera_index
   :end-at: single_cam_info = camera.data.info[camera_index]

After this step, we can save the images using the BasicWriter.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # Pack data back into replicator format to save them using its writer
   :end-at: rep_writer.write(rep_output)


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

The resulting point cloud can be visualized using the :mod:`isaacsim.util.debug_draw` extension from Isaac Sim.
This makes it easy to visualize the point cloud in the 3D space.

.. literalinclude:: ../../../scripts/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :start-at: # In the first few steps, things are still being instanced and Camera.data
   :end-at: pc_markers.visualize(translations=pointcloud)


Executing the script
--------------------

To run the accompanying script, execute the following command:

.. code-block:: bash

   # Usage with saving and drawing
   ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --save --draw --enable_cameras

   # Usage with saving only in headless mode
   ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --save --headless --enable_cameras


The simulation should start, and you can observe different objects falling down. An output folder will be created
in the ``IsaacLab/scripts/tutorials/04_sensors`` directory, where the images will be saved. Additionally,
you should see the point cloud in the 3D space drawn on the viewport.

To stop the simulation, close the window, or use ``Ctrl+C`` in the terminal.
