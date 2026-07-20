.. _how_to_record_video:

Recording Video
===============

Isaac Lab records video by driving :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`
entries inside ``env.step()`` — no gym wrapper or ``render_mode`` argument required.
Add one or more :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` entries to
``env_cfg.video_recorders`` and the env handles the rest.

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir="videos/")
    ]

Clips are written to ``output_dir/clip_NNNN.mp4`` via `moviepy <https://pypi.org/project/moviepy/>`_
when the clip reaches ``clip_length`` steps or when ``env.close()`` is called.


Source types
------------

The ``source`` string selects what to capture:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Source string
     - Captures from
   * - ``"visualizer"``
     - First active recording-capable visualizer (auto)
   * - ``"visualizer:kit"``
     - Kit viewport camera (PhysX only — errors with Newton physics)
   * - ``"visualizer:newton"``
     - Newton GL visualizer framebuffer
   * - ``"visualizer:newton/tiled"``
     - Newton tiled camera panel
   * - ``"sensor:<name>"``
     - ``env.scene.sensors[name]``, RGB channel
   * - ``"sensor:<name>/depth"``
     - ``env.scene.sensors[name]``, depth channel

The camera angle, resolution and other visualizer settings are configured on the
corresponding :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` or
:class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`, not on the recorder.

.. note::

   ``source="visualizer:kit"`` does not work with Newton physics — Kit Replicator
   cannot read Newton Fabric transforms and the recorder logs an error.
   Use ``source="visualizer:newton"`` instead when Newton is active.


Common use cases
----------------

**Record the Kit viewport during PhysX training**

Configure a Kit visualizer and point the recorder at it:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(8.0, 0.0, 5.0))]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir="videos/")
    ]

**Record with Newton physics**

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=1280, window_height=720)]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton", output_dir="videos/")
    ]

**Record from a scene camera sensor**

Any ``CameraCfg`` field on the scene can be used as a recording source by name:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    # Assumes env_cfg.scene.tiled_camera is configured with data_types=["rgb"]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="sensor:tiled_camera", output_dir="videos/")
    ]

For depth or other AOVs, append the channel name:

.. code-block:: python

    VideoRecorderCfg(source="sensor:wrist_cam/depth", output_dir="videos/depth/")

**Multiple simultaneous streams**

Each entry in ``video_recorders`` is independent — different sources, different output dirs:

.. code-block:: python

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit",       output_dir="videos/viewport/"),
        VideoRecorderCfg(source="sensor:wrist_cam",     output_dir="videos/wrist/"),
        VideoRecorderCfg(source="sensor:tiled_camera",  output_dir="videos/overhead/"),
    ]


Clip control
------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``clip_length``
     - ``200``
     - Env steps per clip
   * - ``clip_trigger_step``
     - ``0``
     - ``0`` = one clip starting at step 1; ``N > 0`` = new clip every N steps
   * - ``fps``
     - ``30``
     - Output frame rate
   * - ``output_dir``
     - ``"videos"``
     - Directory for ``clip_NNNN.mp4`` files (created on demand)

**Record one clip at the start of training:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", clip_length=500, clip_trigger_step=0)

**Record a 200-step clip every 1 000 env steps:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", clip_length=200, clip_trigger_step=1000)


Requirements
------------

* `moviepy <https://pypi.org/project/moviepy/>`_ 1.x and ``ffmpeg`` must be installed:

  .. code-block:: bash

      pip install "moviepy<2"  # already in Isaac Lab's dependencies

* For ``source="visualizer:kit"``: the Kit app must be launched with ``--enable_cameras``
  (done automatically by :class:`~isaaclab.app.AppLauncher` when a Kit visualizer is configured).

* For ``source="visualizer:newton"``: an active :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`
  must be in ``env_cfg.sim.visualizer_cfgs``.

* For ``source="sensor:<name>"``: the named field must exist on the scene config and have ``"rgb"``
  (or the specified channel) in its ``data_types``.


See also
--------

* :doc:`/source/overview/core-concepts/visualization` — configuring interactive visualizers
* :doc:`capture_sensor_frames` — saving per-frame sensor outputs as images
