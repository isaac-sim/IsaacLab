.. _how_to_record_video:

Recording Video
===============

.. currentmodule:: isaaclab

Isaac Lab records video by driving :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`
entries inside ``env.step()`` — no gym wrapper or ``render_mode`` argument required.
Add one or more :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` entries to
``env_cfg.video_recorders`` and the env handles the rest.

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir="videos/")
    ]

Clips are written to ``output_dir/<prefix>_NNNN.mp4`` via
`moviepy <https://pypi.org/project/moviepy/>`_ when the clip reaches ``video_length``
steps or when ``env.close()`` is called.

This guide is accompanied by the ``run_video_recording.py`` tutorial script in
``IsaacLab/scripts/tutorials/07_visualizers``.  Pass ``--example 1``, ``--example 2``,
or ``--example 3`` to select which recording configuration to run.

.. dropdown:: Code for run_video_recording.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_video_recording.py
      :language: python
      :linenos:


Tutorial examples
-----------------

All three examples use the Shadow Hand cube-reorientation task
(``Isaac-Reorient-Cube-Shadow-Camera-Direct``), which ships with a built-in tiled camera
sensor.  Each example adds one more recording source, so you can run them in order to
build intuition progressively.


Example 1: Kit viewport and tiled-camera grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with:

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 1 --num_envs 256

Two clips are written to ``videos/recording_tutorial/example_1/``:

* ``viewport_0000.mp4`` — the main Kit interactive viewport camera.
* ``tiled_0000.mp4`` — the tiled-camera grid of 36 environment views.

.. figure:: ../_static/how-to/record_video/example1_viewport.gif
   :width: 100%
   :alt: Kit viewport recording — Shadow Hand cube reorientation

   Kit viewport recording: ``viewport_0000.mp4``

.. figure:: ../_static/how-to/record_video/example1_tiled.gif
   :width: 100%
   :alt: Kit tiled-camera grid recording — all 16 environments

   Kit tiled-camera grid recording: ``tiled_0000.mp4``

The script sets ``KitVisualizerCfg(tiled_cam_view=True, tiled_cam_num=16)`` to open the
tiled panel alongside the main viewport, then points two independent recorders at
``"visualizer:kit"`` and ``"visualizer:kit:tiled"``.  The ``output_filename_prefix``
field distinguishes the two output files in the same directory.


Example 2: Scene sensor, headless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with:

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 2 --num_envs 16

No visualizer window opens.  The recorder reads frames directly from the
``tiled_camera`` sensor in the Shadow Hand scene, writing one clip to
``videos/recording_tutorial/example_2/sensor_0000.mp4``.

.. figure:: ../_static/how-to/record_video/example2_sensor.gif
   :width: 100%
   :alt: Sensor recording — tiled camera grid headless

   Headless sensor recording: ``sensor_0000.mp4``

``source="sensor:tiled_camera"`` refers to the key under which the camera is registered
in ``env.scene.sensors``.  The sensor must have ``"rgb"`` in its ``data_types``; depth or
other channels can be recorded by appending the channel name (e.g.
``"sensor:tiled_camera:depth"``).


Example 3: Kit viewport, Newton viewport, and sensor simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with:

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 3 --num_envs 16

Three clips are written to ``videos/recording_tutorial/example_3/``:

* ``kit_viewport_0000.mp4`` — Kit interactive viewport (RTX renderer).
* ``newton_viewport_0000.mp4`` — Newton GL viewer framebuffer.
* ``sensor_0000.mp4`` — scene tiled-camera sensor (offline render).

.. figure:: ../_static/how-to/record_video/example3_kit_viewport.gif
   :width: 100%
   :alt: Kit viewport recording (Example 3)

   Kit viewport: ``kit_viewport_0000.mp4``

.. figure:: ../_static/how-to/record_video/example3_newton_viewport.gif
   :width: 100%
   :alt: Newton viewport recording (Example 3)

   Newton viewport: ``newton_viewport_0000.mp4``

.. figure:: ../_static/how-to/record_video/example3_sensor.gif
   :width: 100%
   :alt: Sensor recording (Example 3)

   Scene sensor: ``sensor_0000.mp4``

Each ``VideoRecorderCfg`` entry is fully independent — different sources write
different files at their own cadence.  There is no limit on the number of simultaneous
recorders.


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
   * - ``"visualizer:kit:tiled"``
     - Kit tiled-camera grid panel
   * - ``"visualizer:newton"``
     - Newton GL visualizer framebuffer
   * - ``"visualizer:newton:tiled"``
     - Newton tiled camera panel
   * - ``"sensor:<name>"``
     - ``env.scene.sensors[name]``, RGB channel

The camera angle, resolution, and other visualizer settings are configured on the
corresponding :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` or
:class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`, not on the recorder.

.. note::

   ``source="visualizer:kit"`` does not work with Newton physics — Kit Replicator
   cannot read Newton Fabric transforms and the recorder logs an error.
   Use ``source="visualizer:newton"`` instead when Newton is active.


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
     - ``30``
     - Output frame rate
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


See also
--------

* :doc:`/source/overview/core-concepts/visualization` — configuring interactive visualizers
* :doc:`visualizer_tiled_camera` — tiled camera panel setup
* :doc:`capture_sensor_frames` — saving per-frame sensor outputs as images
