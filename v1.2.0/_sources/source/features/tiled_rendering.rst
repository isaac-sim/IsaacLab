Tiled-Camera Rendering
======================

.. currentmodule:: omni.isaac.lab

.. note::

    This feature is only available from Isaac Sim version 4.2.0 onwards.

    Tiled rendering in combination with image processing networks require heavy memory resources, especially
    at larger resolutions. We recommend running at 512 cameras in the scene on RTX 4090 GPUs or similar.


Tiled rendering APIs provide a vectorized interface for collecting data from camera sensors.
This is useful for reinforcement learning environments requiring vision in the loop.
Tiled rendering works by concatenating camera outputs from multiple cameras and rendering
one single large image instead of multiple smaller images that would have been produced
by each individual camera. This reduces the amount of time required for rendering and
provides a more efficient API for working with vision data.

Isaac Lab provides tiled rendering APIs for RGB, depth, along with other annotators through the :class:`~sensors.TiledCamera`
class. Configurations for the tiled rendering APIs can be defined through the :class:`~sensors.TiledCameraCfg`
class, specifying parameters such as the regex expression for all camera paths, the transform
for the cameras, the desired data type, the type of cameras to add to the scene, and the camera
resolution.

.. code-block:: python

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )

To access the tiled rendering interface, a :class:`~sensors.TiledCamera` object can be created and used
to retrieve data from the cameras.

.. code-block:: python

    tiled_camera = TiledCamera(cfg.tiled_camera)
    data_type = "rgb"
    data = tiled_camera.data.output[data_type]

The returned data will be transformed into the shape (num_cameras, height, width, num_channels), which
can be used directly as observation for reinforcement learning.

When working with rendering, make sure to add the ``--enable_cameras`` argument when launching the
environment. For example:

.. code-block:: shell

    python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras


Annotators and Data Types
-------------------------

Both :class:`~sensors.TiledCamera` and :class:`~sensors.Camera` classes provide APIs for retrieving various types annotator data from replicator:

* ``"rgb"``: A 3-channel rendered color image.
* ``"rgba"``: A 4-channel rendered color image with alpha channel.
* ``"distance_to_camera"``: An image containing the distance to camera optical center.
* ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
* ``"depth"``: The same as ``"distance_to_image_plane"``.
* ``"normals"``: An image containing the local surface normal vectors at each pixel.
* ``"motion_vectors"``: An image containing the motion vector data at each pixel.
* ``"semantic_segmentation"``: The semantic segmentation data.
* ``"instance_segmentation_fast"``: The instance segmentation data.
* ``"instance_id_segmentation_fast"``: The instance id segmentation data.

RGB and RGBA
~~~~~~~~~~~~

``rgb`` data type returns a 3-channel RGB colored image of type ``torch.uint8``, with dimension (B, H, W, 3).

``rgba`` data type returns a 4-channel RGBA colored image of type ``torch.uint8``, with dimension (B, H, W, 4).

To convert the ``torch.uint8`` data to ``torch.float32``, divide the buffer by 255.0 to obtain a ``torch.float32`` buffer containing data from 0 to 1.

Depth and Distances
~~~~~~~~~~~~~~~~~~~

``distance_to_camera`` returns a single-channel depth image with distance to the camera optical center. The dimension for this annotator is (B, H, W, 1) and has type ``torch.float32``.

``distance_to_image_plane`` returns a single-channel depth image with distances of 3D points from the camera plane along the camera's Z-axis. The dimension for this annotator is (B, H, W, 1) and has type ``torch.float32``.

``depth`` is provided as an alias for ``distance_to_image_plane`` and will return the same data as the ``distance_to_image_plane`` annotator, with dimension (B, H, W, 1) and type ``torch.float32``.

Normals
~~~~~~~

``normals`` returns an image containing the local surface normal vectors at each pixel. The buffer has dimension (B, H, W, 3), containing the (x, y, z) information for each vector, and has data type ``torch.float32``.

Motion Vectors
~~~~~~~~~~~~~~

``motion_vectors`` returns the per-pixel motion vectors in image space, with a 2D array of motion vectors representing the relative motion of a pixel in the camera’s viewport between frames. The buffer has dimension (B, H, W, 2), representing x - the motion distance in the horizontal axis (image width) with movement to the left of the image being positive and movement to the right being negative and y - motion distance in the vertical axis (image height) with movement towards the top of the image being positive and movement to the bottom being negative. The data type is ``torch.float32``.

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

``semantic_segmentation`` outputs semantic segmentation of each entity in the camera’s viewport that has semantic labels. In addition to the image buffer, an ``info`` dictionary can be retrieved with ``tiled_camera.data.info['semantic_segmentation']`` containing ID to labels information.

- If ``colorize_semantic_segmentation=True`` in the camera config, a 4-channel RGBA image will be returned with dimension (B, H, W, 4) and type ``torch.uint8``. The info ``idToLabels`` dictionary will be the mapping from color to semantic labels.

- If ``colorize_semantic_segmentation=False``, a buffer of dimension (B, H, W, 1) of type ``torch.int32`` will be returned, containing the semantic ID of each pixel. The info ``idToLabels`` dictionary will be the mapping from semantic ID to semantic labels.

Instance ID Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

``instance_id_segmentation_fast`` outputs instance ID segmentation of each entity in the camera’s viewport. The instance ID is unique for each prim in the scene with different paths. In addition to the image buffer, an ``info`` dictionary can be retrieved with ``tiled_camera.data.info['instance_id_segmentation_fast']`` containing ID to labels information.

The main difference between ``instance_id_segmentation_fast`` and ``instance_segmentation_fast`` are that instance segmentation annotator goes down the hierarchy to the lowest level prim which has semantic labels, where instance ID segmentation always goes down to the leaf prim.

- If ``colorize_instance_id_segmentation=True`` in the camera config, a 4-channel RGBA image will be returned with dimension (B, H, W, 4) and type ``torch.uint8``. The info ``idToLabels`` dictionary will be the mapping from color to USD prim path of that entity.

- If ``colorize_instance_id_segmentation=False``, a buffer of dimension (B, H, W, 1) of type ``torch.int32`` will be returned, containing the instance ID of each pixel. The info ``idToLabels`` dictionary will be the mapping from instance ID to USD prim path of that entity.

Instance Segmentation
"""""""""""""""""""""

``instance_segmentation_fast`` outputs instance segmentation of each entity in the camera’s viewport. In addition to the image buffer, an ``info`` dictionary can be retrieved with ``tiled_camera.data.info['instance_segmentation_fast']`` containing ID to labels and ID to semantic information.

- If ``colorize_instance_segmentation=True`` in the camera config, a 4-channel RGBA image will be returned with dimension (B, H, W, 4) and type ``torch.uint8``. The info ``idToLabels`` dictionary will be the mapping from color to USD prim path of that semantic entity. The info ``idToSemantics`` dictionary will be the mapping from color to semantic labels of that semantic entity.

- If ``colorize_instance_segmentation=False``, a buffer of dimension (B, H, W, 1) of type ``torch.int32`` will be returned, containing the instance ID of each pixel. The info ``idToLabels`` dictionary will be the mapping from instance ID to USD prim path of that semantic entity. The info ``idToSemantics`` dictionary will be the mapping from instance ID to semantic labels of that semantic entity.
