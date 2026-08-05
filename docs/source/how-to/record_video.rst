.. _how_to_record_video:

Recording Video
===============

.. currentmodule:: isaaclab

Isaac Lab can record video from any active visualizer or scene camera sensor by adding
:class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` entries to the environment
config.  Each recorder captures from a configurable source — a Kit viewport, a Newton GL window,
or a tiled camera sensor — and writes ``mp4`` clips to disk independently.

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
   * - ``"visualizer:kit:tiled"``
     - Kit visualizer tiled-camera grid panel
   * - ``"visualizer:newton"``
     - Newton GL visualizer viewport
   * - ``"visualizer:newton:tiled"``
     - Newton GL visualizer tiled-camera panel
   * - ``"sensor:<name>"``
     - ``env.scene.sensors[name]``, RGB channel

The camera angle, resolution, and other visualizer settings are configured on the
corresponding :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` or
:class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`, not on the recorder.


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
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(4.0, 4.0, 2.0)),
        NewtonVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton", output_dir="videos/"),
    ]

Alternatively, use a :class:`~isaaclab.sensors.CameraCfg` sensor in the scene and record
with ``source="sensor:<name>"``, which gives full control over the recording viewpoint
without requiring a second interactive visualizer.


Requirements
------------

* `moviepy <https://pypi.org/project/moviepy/>`_ 1.x and ``ffmpeg`` must be installed
  (both are already in Isaac Lab's dependencies).

* For ``source="visualizer:kit"`` or ``"visualizer:kit:tiled"``: the Kit app is launched
  automatically by :class:`~isaaclab.app.AppLauncher`; cameras are auto-enabled when a
  Kit visualizer is configured.

* For ``source="visualizer:newton"`` or ``"visualizer:newton:tiled"``: an active
  :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg` must be in
  ``env_cfg.sim.visualizer_cfgs``.

* For ``source="sensor:<name>"``: the named field must exist on the scene config and
  have ``"rgb"`` in its ``data_types``.


Limitations
-----------

* ``source="visualizer:kit"`` and ``source="visualizer:kit:tiled"`` require cubric to
  propagate Newton Fabric scene transforms to the RTX renderer.  Without cubric, a warning
  is logged and captured frames may be black.  Use ``source="visualizer:newton"`` for
  guaranteed capture with Newton physics.

* ``source="visualizer:newton:tiled"`` captures the Newton GL window framebuffer when
  ``tiled_cam_view=True`` is set on :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`.
  Newton does not implement a separate ``render_tiled_rgb_array()`` path.


See also
--------

* :doc:`/source/overview/core-concepts/visualization` — configuring interactive visualizers
* :doc:`visualizer_tiled_camera` — tiled camera panel setup
* :doc:`capture_sensor_frames` — saving per-frame sensor outputs as images
