.. _how-to-estimate-how-cameras-can-run:


Find How Many/What Cameras You Should Train With
================================================

.. currentmodule:: omni.isaac.lab

Currently in Isaac Lab, there are several camera types; USD Cameras (standard), Tiled Cameras,
and Ray Caster cameras. These camera types differ in functionality and performance. The ``benchmark_cameras.py``
script can be used to understand the difference in cameras types, as well to characterize their relative performance
at different parameters such as camera quantity, image dimensions, and replicator types.

This utility is provided so that one easily can find the camera type/parameters that are the most performant
while meeting the requirements of the user's scenario. This utility also helps estimate
the maximum number of cameras one can realistically run, assuming that one wants to maximize the number
of environments while minimizing step time.

This guide accompanies the ``benchmark_cameras.py`` script in the ``IsaacLab/source/standalone/tutorials/04_sensors``
directory.

.. dropdown:: Code for benchmark_cameras.py
   :icon: code

   .. literalinclude:: ../../../source/standalone/tutorials/04_sensors/benchmark_cameras.py
      :language: python
      :emphasize-lines: 197-286, 450-480
      :linenos:


Possible Parameters
-------------------

First, run

.. code-block:: bash

   ./isaaclab.sh -p source/standalone/tutorials/04_sensors/benchmark_cameras.py -h

to see all possible parameters you can vary with this utility.

Possible replicators to try are as follows:

(Note: the standard camera supports all replicators, while tiled camera and ray caster camera
support subsets of this list. If a replicator isn't supported by the tiled or ray caster camera,
and it is supplied, then the script should throw an error related to this, and you
should choose a different replicator.)

   - ``"rgb"``: A rendered color image.
   - ``"distance_to_camera"``: An image containing the distance to camera optical center.
   - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
   - ``"normals"``: An image containing the local surface normal vectors at each pixel.
   - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
   - ``"semantic_segmentation"``: The semantic segmentation data.
   - ``"instance_segmentation_fast"``: The instance segmentation data.
   - ``"instance_id_segmentation_fast"``: The instance id segmentation data.


WARNING: If the ``distance_to_camera`` replicator is used to generate depth images, then they should be
converted with ``convert_perspective_depth_image_to_orthogonal_depth_image`` from ``isaac.lab.utils.math``
prior to creating any Point Cloud, as the ``unproject_depth`` from ``isaac.lab.utils.math`` currently assumes
orthogonal depth.

Compare Camera Type Output Through Visualization
------------------------------------------------
It is possible to visualize the result of the camera(s) replicators. However,
when visualizing, the benchmark results are not meaningful due to rendering
and the GPU to CPU conversion. Regardless, this provides a way to easily and quickly sanity check
camera outputs.

.. code-block:: bash

   ./isaaclab.sh -p source/standalone/tutorials/04_sensors/benchmark_cameras.py \
   --height 100 --width 100 --num_tiled_cameras 1 --num_standard_cameras 1 \
   --num_ray_caster_cameras 1 --visualize

This should save several images in the same directory that the script is run, showing the labelled output
of the cameras. If depth is enabled, point clouds are generated from depth, and the rendered point cloud
result is saved as an image as well. After the simulations stops it can be closed with CTRL C.

Compare Camera Type and Performance Under Different Parameters By Benchmarking
------------------------------------------------------------------------------

If one doesn't supply the ``--visualize`` flag, then ``benchmark_cameras.py``
can be used to estimate average rendering time for the number of desired cameras.

Currently, tiled cameras are the most performant camera that can handle multiple dynamic objects.

For example, to see how a system can handle 1000 tiled cameras, that are only in depth mode,
with 1000 objects, try

.. code-block:: bash

   ./isaaclab.sh -p source/standalone/tutorials/04_sensors/benchmark_cameras.py \
   --height 100 --width 100 --num_tiled_cameras 1000 --num_standard_cameras 0 \
   --num_ray_caster_cameras 0 --tiled_camera_replicators depth --num_objects 1000

If your system cannot handle this due to performance reasons, then the process will be killed.
It's recommended to monitor CPU/RAM utilization and GPU utilization while running this script, to get
an idea of how many resources rendering the desired camera requires. In Ubuntu, you can use tools like ``htop`` and ``nvtop``
to live monitor resources while running this script, and in Windows, you can use the Task Manager.

If your system has a hard time handling the desired cameras, you can try the following

   - Switch to headless mode (supply ``--headless``)
   - Ensure you are using the GPU pipeline not CPU!
   - If you aren't using Tiled Cameras, switch to Tiled Cameras
   - Decrease camera resolution
   - Decrease how many replicators there are for each camera.
   - Decrease the number of cameras
   - Decrease the number of objects in the scene

If your system is able to handle the amount of cameras, then the time statistics will be printed to the terminal.
After the simulations stops it can be closed with CTRL C.
