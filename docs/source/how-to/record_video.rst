.. _how_to_record_video:

Recording Video
===============

.. currentmodule:: isaaclab

Isaac Lab can record video from a Kit, Newton GL, or Newton RTX visualizer, or directly from a
scene camera sensor, by adding :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg`
entries to the environment config. Each recorder captures from a configurable source and writes
``mp4`` clips to disk independently. Streaming visualizers (Rerun and Viser) do not support local
frame capture; see `Visualizer compatibility`_ below.

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir="videos/")
    ]

This guide is accompanied by the ``run_video_recording.py`` tutorial script in
``IsaacLab/scripts/tutorials/07_visualizers``.  Pass ``--example 1``, ``--example 2``,
or ``--example 3`` to select which recording configuration to run.

.. dropdown:: Code for run_video_recording.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_video_recording.py
      :language: python
      :linenos:

.. |kit_viewport| image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_example_kit_viewport.gif
   :width: 100%
   :alt: Kit viewport — 4 Shadow Hand environments (RTX)

.. |newton_viewport| image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_example_newton_viewport.gif
   :width: 100%
   :alt: Newton GL viewport — 4 Shadow Hand environments

.. |tiled_viewport| image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_example_tiled_kit_viewport.gif
   :width: 100%
   :alt: Kit tiled-camera grid — per-environment views

.. |sensor| image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_example_sensor.gif
   :width: 100%
   :alt: Scene tiled-camera sensor recording

.. list-table::
   :widths: 50 50

   * - |kit_viewport|
     - |newton_viewport|
   * - Kit visualizer
     - Newton GL visualizer
   * - |tiled_viewport|
     - |sensor|
   * - Kit visualizer tiled streaming
     - Scene sensor


Tutorial examples
-----------------

All three examples use the Shadow Hand cube-reorientation task
(``Isaac-Reorient-Cube-Shadow-Camera-Direct``), which ships with a built-in tiled camera
sensor.  Example 1 and Example 2 each demonstrate one recording source; Example 3
combines all of them simultaneously.


Example 1: Kit viewport
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 1 --num_envs 4

Records the Kit interactive viewport (RTX renderer) showing 4 parallel environments.
One clip is written to ``videos/recording_tutorial/example_1/kit_viewport_0000.mp4``.


Example 2: Scene sensor, headless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 2 --num_envs 16

No visualizer window opens.  Frames are read directly from the ``tiled_camera`` sensor,
writing one clip to ``videos/recording_tutorial/example_2/sensor_0000.mp4``.

``source="sensor:tiled_camera"`` refers to the key under which the camera is registered
in ``env.scene.sensors``.  The sensor must have ``"rgb"`` in its ``data_types``; only the
``rgb`` channel is currently supported for sensor sources.


Example 3: All sources simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 3 --num_envs 4

Four independent clips are written to ``videos/recording_tutorial/example_3/``:

* ``kit_viewport_0000.mp4`` — Kit interactive viewport (RTX renderer).
* ``tiled_kit_viewport_0000.mp4`` — Kit tiled-camera grid (per-environment views).
* ``newton_viewport_0000.mp4`` — Newton GL viewer framebuffer.
* ``sensor_0000.mp4`` — scene tiled-camera sensor (offline render).

Each ``VideoRecorderCfg`` entry is fully independent — different sources write different
files at their own cadence.  There is no limit on the number of simultaneous recorders.


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
     - Kit visualizer viewport
   * - ``"visualizer:kit:streaming_view"``
     - Kit streaming camera panel (requires ``streaming_view=True``)
   * - ``"visualizer:newton"``
     - Newton GL visualizer viewport
   * - ``"visualizer:newton_rtx"``
     - Newton OVRTX path-traced viewport
   * - ``"visualizer:newton:streaming_view"``
     - Newton GL streaming camera panel (requires ``streaming_view=True``)
   * - ``"sensor:<name>"``
     - ``env.scene.sensors[name]``, RGB (default)
   * - ``"sensor:<name>:rgb"``
     - RGB channel
   * - ``"sensor:<name>:depth"``
     - Depth, turbo colormap (range: ``depth_colormap_min`` … ``depth_colormap_max``)
   * - ``"sensor:<name>:segmentation"``
     - Segmentation, colorized
   * - ``"sensor:<name>:normals"``
     - Surface normals, colorized

The camera angle, resolution, and other visualizer settings are configured on the corresponding
visualizer config, not on the recorder.

.. note::

   The Newton RTX viewer framebuffer can be recorded with ``"visualizer:newton_rtx"``, but
   recording its streaming view is not supported.


Clip control
------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``video_length``
     - ``200``
     - Env steps per clip
   * - ``video_interval``
     - ``0``
     - ``0`` = one clip starting at step 1; ``N > 0`` = new clip every N steps
   * - ``fps``
     - ``None``
     - Output frame rate; ``None`` resolves from ``env.metadata["render_fps"]`` or ``1 / step_dt``
   * - ``output_dir``
     - ``"videos"``
     - Directory for output files (created on demand)
   * - ``output_filename_prefix``
     - ``"clip"``
     - File stem; output is ``<prefix>_NNNN.mp4``
   * - ``keep_last_n_clips``
     - ``None``
     - Delete older clips; ``None`` keeps all

**One clip at the start of a run:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", video_length=500, video_interval=0)

**Recurring clips every 1 000 env steps:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", video_length=200, video_interval=1000)

**Keep only the most recent clip on disk:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", video_length=200, video_interval=1000,
                     keep_last_n_clips=1)


Recording from an independent camera angle
------------------------------------------

Configure the recording angle on the visualizer rather than on the recorder.
To open a headless Newton visualizer at a different angle alongside an interactive Kit viewer:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(4.0, 4.0, 2.0)),
        NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton", output_dir="videos/"),
    ]

Alternatively, use a :class:`~isaaclab.sensors.CameraCfg` sensor in the scene and record
with ``source="sensor:<name>"``, which gives full control over the recording viewpoint
without requiring a second interactive visualizer.


Requirements
------------

* Install the ``video`` extra to provide `moviepy <https://pypi.org/project/moviepy/>`_ 1.x
  and its ``ffmpeg`` runtime. In a uv checkout, add ``--extra video`` to the command.

* For ``source="visualizer:kit"`` or ``"visualizer:kit:streaming_view"``: the Kit app is
  launched automatically by :class:`~isaaclab.app.AppLauncher`.  In **headless mode**
  (``--headless``), you must also pass ``--enable_cameras`` (or set ``ENABLE_CAMERAS=1``) to
  activate the Replicator offscreen render pipeline; without it, captured frames are black.
  The ``--video`` flag sets ``--enable_cameras`` automatically when no explicit recorder
  source is configured.

* For ``source="visualizer:newton"`` or ``"visualizer:newton_gl"`` /
  ``"visualizer:newton:streaming_view"``:
  an active :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg` must be in
  ``env_cfg.sim.visualizer_cfgs``.  Newton GL uses pyglet's EGL backend and works headlessly
  without ``--enable_cameras``.

* For ``source="visualizer:newton_rtx"``: the OVRTX runtime and an active
  :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg` are required. Capturing the
  path-traced LDR framebuffer performs a GPU-to-CPU readback.

* For ``source="sensor:<name>"``: the named field must exist on the scene config and
  have ``"rgb"`` in its ``data_types``.


Visualizer compatibility
------------------------

**kit**, **newton_gl**, and **newton_rtx** support frame capture and can run headless.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Visualizer
     - ``--video``
     - Notes
   * - ``kit``
     - ✓
     - Kit/Omniverse viewport; supports headless mode
   * - ``newton_gl``
     - ✓
     - Newton OpenGL viewport; supports headless mode
   * - ``newton_rtx``
     - ✓
     - Newton OVRTX path-traced viewport; native-resolution LDR readback
   * - ``rerun``
     - ✗
     - Remote streaming tool; no local frame-capture API
   * - ``viser``
     - ✗
     - Browser streaming tool; no local frame-capture API

Passing ``--video`` alongside ``--viz rerun`` or ``--viz viser`` raises an error when no other
recording-capable visualizer is configured.

To run a streaming visualizer and record video simultaneously, add a headless capture backend
alongside it in ``sim.visualizer_cfgs``:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.rerun import RerunVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        RerunVisualizerCfg(...),                 # streaming — for monitoring
        KitVisualizerCfg(headless=True),         # headless — provides frames for --video
    ]

Alternatively, record directly from a scene camera sensor without any visualizer:

.. code-block:: python

    VideoRecorderCfg(source="sensor:<name>")    # add to env_cfg.video_recorders


Limitations
-----------

* ``source="visualizer:kit"`` and ``source="visualizer:kit:streaming_view"`` require cubric
  to propagate Newton Fabric scene transforms to the RTX renderer.  Without cubric, a warning
  is logged and a black-frame warning is emitted at clip write time.  Use
  ``source="visualizer:newton"`` for guaranteed capture with Newton physics.

* ``source="visualizer:newton:streaming_view"`` and ``source="visualizer:kit:streaming_view"``
  require ``streaming_view=True`` on the corresponding visualizer cfg.  A
  :class:`~RuntimeError` is raised at the first capture attempt if it is not set.


See also
--------

* :doc:`/source/overview/core-concepts/visualization` — configuring interactive visualizers
* :doc:`visualizer_tiled_camera` — tiled camera panel setup
* :doc:`capture_sensor_frames` — saving per-frame sensor outputs as images
