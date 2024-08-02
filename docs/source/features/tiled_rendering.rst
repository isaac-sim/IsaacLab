Tiled Rendering and Recording
=============================

.. currentmodule:: omni.isaac.lab


Tiled Rendering
---------------

.. note::

    This feature is only available from Isaac Sim version 4.0.0 onwards.

    Tiled rendering requires heavy memory resources. We recommend running at most 256 cameras in the scene.

Tiled rendering APIs provide a vectorized interface for collecting data from camera sensors.
This is useful for reinforcement learning environments requiring vision in the loop.
Tiled rendering works by concatenating camera outputs from multiple cameras and rendering
one single large image instead of multiple smaller images that would have been produced
by each individual camera. This reduces the amount of time required for rendering and
provides a more efficient API for working with vision data.

Isaac Lab provides tiled rendering APIs for RGB and depth data through the :class:`~sensors.TiledCamera`
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


Recording during training
-------------------------

Isaac Lab supports recording video clips during training using the `gymnasium.wrappers.RecordVideo <https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/>`_ class.

This feature can be enabled by installing ``ffmpeg`` and using the following command line arguments with the training script:

* ``--video`` - enables video recording during training
* ``--video_length`` - length of each recorded video (in steps)
* ``--video_interval`` - interval between each video recording (in steps)

Make sure to also add the ``--enable_cameras`` argument when running headless.
Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.

Example usage:

.. code-block:: shell

    python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500

Recorded videos will be saved in the same directory as the training checkpoints, under ``IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train``.
