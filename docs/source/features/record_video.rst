.. _how_to_record_video:

Recording Video
===============

.. currentmodule:: isaaclab

Isaac Lab supports video recording from visualizers and sensor data streams from renderers.
Recordings output as ``mp4`` clips.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_shadow_hand_tiled_gt.mp4" type="video/mp4">
   </video>
   <p class="viz-cap" style="text-align:center; font-style:italic;">Recorded clip of the
   Shadow Hand cube-reorientation task, recording the streaming view of the Rerun
   visualizer with 4 streaming view envs and 4 GT types</p>


Quick Start
-----------

Add a ``VideoRecorderCfg`` to ``env_cfg.video_recorders``:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="viz:kit", output_dir="videos/")
    ]

Or pass ``--video [SOURCE]`` on the command line to record without editing the environment config. ``--video``
adds a single recorder; to record several sources at once, list one ``VideoRecorderCfg`` per source in
``env_cfg.video_recorders``, which then takes precedence over the ``--video`` source (``--video_length`` and
``--video_interval`` still apply to every recorder):

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --video

.. _record_video_cli:

``--video`` records from the source it names, and ``--viz`` still decides which visualizers open a window:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Command line
     - Records from
   * - ``--video`` (same as ``--video viz``)
     - the first capture-capable visualizer ``--viz`` selected, in its window; if none is selected, or only
       streaming ones such as ``viser`` or ``rerun``, a headless ``newton_gl``
   * - ``--video viz:newton_rtx``
     - a ``newton_rtx``: the ``--viz``-selected one if selected, else an extra headless one
   * - ``--video --viz newton_rtx``
     - the selected ``newton_rtx`` window
   * - ``--video viz:newton_gl --viz viser``
     - ``viser`` runs; an extra headless ``newton_gl`` is added only for recording
   * - ``--video sensor:wrist_camera[:rgb|depth|...]``
     - that scene sensor; no visualizer is added

A visualizer added for the recording reuses the configured visualizer of its type in
``sim.visualizer_cfgs``, e.g. its camera pose, and runs headless. Recorders declared in
``env_cfg.video_recorders`` take precedence over the ``--video`` source; ``--video`` then only enables them
and applies ``--video_length`` and ``--video_interval``. A Hydra override after ``--video`` is never taken as
its source: ``--video presets=newton_mjwarp`` records from ``viz`` and applies the preset.

A bare visualizer type is shorthand for ``viz:<type>``, so ``--video newton_gl``, ``--video kit`` and
``--video newton_rtx`` work directly. ``viser`` and ``rerun`` resolve the same way, but recording from these
streaming visualizers is unsupported because they have no frame capture.

See `Source types`_ for the full list of recordable sources and `Clip control`_ for length and
interval options.


Overview
--------

.. raw:: html

   <style>
   .viz-cap { text-align:center; font-style:italic; margin-top:0.4em; font-size:0.9em; }
   .viz-grid { display:flex; gap:16px; align-items:flex-start; margin: 0.5em 0 1em; }
   .viz-grid > div { flex:1 1 0; min-width:0; }
   .viz-grid video { display:block; width:100%; }
   .viz-grid-natural > div:first-child { flex:0 0 auto; width:40%; }
   .viz-grid-match { justify-content:center; }
   .viz-grid-match > div { flex:0 0 auto; }
   .viz-clip-crop { aspect-ratio:55/36; height:280px; overflow:hidden; margin:0 auto; }
   .viz-clip-crop video { width:100%; height:100%; object-fit:cover; object-position:44.44% center; }
   .viz-clip-crop-lg { height:340px; }
   .viz-clip-square { aspect-ratio:1/1; height:280px; overflow:hidden; }
   .viz-clip-square video { width:100%; height:100%; object-fit:cover; }
   .viz-clip-sensor { aspect-ratio:64/59; overflow:hidden; }
   .viz-clip-sensor video { width:100%; height:100%; object-fit:cover; }
   </style>

Each ``VideoRecorderCfg`` entry is independent: different sources write different files at
their own cadence, with no limit on simultaneous recorders. The examples below record all
four sources from the same run.


Examples
--------

All three examples use the Shadow Hand cube-reorientation task,
``Isaac-Reorient-Cube-Shadow-Camera-Direct``, which ships with a built-in tiled camera
sensor. Examples 1 and 2 each demonstrate one recording source; Example 3 combines all four.

.. dropdown:: Code for run_video_recording.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_video_recording.py
      :language: python
      :linenos:


Example 1: Kit viewport
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 1 --num_envs 4 --viz kit

* Records the Kit interactive viewport (RTX renderer)
* Shows 4 parallel environments
* One clip is written to ``videos/recording_tutorial/example_1/kit_viewport_0000.mp4``

.. raw:: html

   <div class="viz-clip-crop viz-clip-crop-lg">
     <video autoplay loop muted playsinline controls preload="auto">
       <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_kit_viewport.mp4" type="video/mp4">
     </video>
   </div>
   <p class="viz-cap">Kit visualizer</p>


Example 2: Scene sensor, headless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 2 --num_envs 16

.. raw:: html

   <div class="viz-grid viz-grid-natural">
     <div>
       <div class="viz-clip-sensor">
         <video autoplay loop muted playsinline controls preload="auto">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_sensor.mp4" type="video/mp4">
         </video>
       </div>
       <p class="viz-cap">Scene sensor</p>
     </div>
     <div>

* No visualizer window opens; frames are read directly from the ``tiled_camera`` sensor
* One clip is written to ``videos/recording_tutorial/example_2/sensor_0000.mp4``
* ``source="sensor:tiled_camera"`` is the key under which the camera is registered in
  ``env.scene.sensors``
* The sensor must have ``"rgb"`` in its ``data_types``; only the ``rgb`` channel is
  currently supported for sensor sources

.. raw:: html

     </div>
   </div>


Example 3: All sources simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 3 --num_envs 4 --viz kit,newton_gl

Four independent clips are written to ``videos/recording_tutorial/example_3/``:

* ``kit_viewport_0000.mp4``: Kit interactive viewport (RTX renderer)
* ``tiled_kit_viewport_0000.mp4``: Kit tiled-camera grid (per-environment views)
* ``newton_viewport_0000.mp4``: Newton GL viewer framebuffer
* ``sensor_0000.mp4``: scene tiled-camera sensor (offline render)

.. raw:: html

   <div class="viz-grid viz-grid-match">
     <div>
       <div class="viz-clip-crop">
         <video autoplay loop muted playsinline controls preload="auto">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_newton_viewport.mp4" type="video/mp4">
         </video>
       </div>
       <p class="viz-cap">Newton GL visualizer</p>
     </div>
     <div>
       <div class="viz-clip-square">
         <video autoplay loop muted playsinline controls preload="auto">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_tiled_kit_viewport.mp4" type="video/mp4">
         </video>
       </div>
       <p class="viz-cap">Kit visualizer tiled streaming</p>
     </div>
   </div>


Usage
-----

Source types
~~~~~~~~~~~~

The ``source`` string selects what to capture:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Source string
     - Captures from
   * - ``"viz"`` (default)
     - First capture-capable visualizer ``--viz`` selected, else a headless Newton GL visualizer
   * - ``"viz:kit"``
     - Kit visualizer viewport
   * - ``"viz:kit:streaming_view"``
     - Kit streaming camera panel, requires ``streaming_view=True``
   * - ``"viz:newton_gl"``
     - Newton GL visualizer viewport
   * - ``"viz:newton_rtx"``
     - Newton OVRTX path-traced viewport
   * - ``"viz:newton_gl:streaming_view"``
     - Newton GL streaming camera panel, requires ``streaming_view=True``
   * - ``"sensor:<name>"``
     - ``env.scene.sensors[name]``, RGB (default)
   * - ``"sensor:<name>:rgb"``
     - RGB channel
   * - ``"sensor:<name>:depth"``
     - Depth, turbo colormap, range ``depth_colormap_min`` … ``depth_colormap_max``
   * - ``"sensor:<name>:segmentation"``
     - Segmentation, colorized
   * - ``"sensor:<name>:normals"``
     - Surface normals, colorized

The camera angle, resolution, and other visualizer settings are configured on the corresponding
visualizer config, not on the recorder. A ``viz:<type>`` visualizer that ``--viz`` does not select runs
headless, only for the recording, with the settings of its config in ``sim.visualizer_cfgs``.

``visualizer`` is accepted as the long form of the ``viz`` prefix (``"visualizer:kit"`` is ``"viz:kit"``). The
deprecated ``newton`` type (``"viz:newton"``) still works with a warning; use ``newton_gl``. Bare visualizer
types are accepted here too, e.g. ``VideoRecorderCfg(source="newton_gl")`` is equivalent to
``VideoRecorderCfg(source="viz:newton_gl")``.

.. note::

   The Newton RTX viewer framebuffer can be recorded with ``"viz:newton_rtx"``, but
   recording its streaming view is not supported.


Clip control
~~~~~~~~~~~~

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

    VideoRecorderCfg(source="viz:kit", video_length=500, video_interval=0)

**Recurring clips every 1 000 env steps:**

.. code-block:: python

    VideoRecorderCfg(source="viz:kit", video_length=200, video_interval=1000)

**Keep only the most recent clip on disk:**

.. code-block:: python

    VideoRecorderCfg(source="viz:kit", video_length=200, video_interval=1000,
                     keep_last_n_clips=1)


Recording from an independent camera angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure the recording angle on the visualizer, not the recorder. To record from a headless Newton
visualizer at a different angle alongside an interactive Kit viewer, run with ``--viz kit`` and
configure both; the recorder adds the Newton visualizer, headless:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(4.0, 4.0, 2.0)),
        NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0)),
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="viz:newton_gl", output_dir="videos/"),
    ]

Alternatively, use a :class:`~isaaclab.sensors.CameraCfg` sensor in the scene and record
with ``source="sensor:<name>"``, which gives full control over the recording viewpoint
without requiring a second interactive visualizer.


Limitations and compatibility
------------------------------

* ``source="viz:kit"`` and ``source="viz:kit:streaming_view"`` require cubric
  to propagate Newton Fabric scene transforms to the RTX renderer.  Without cubric, a warning
  is logged and a black-frame warning is emitted at clip write time.  Use
  ``source="viz:newton_gl"`` for guaranteed capture with Newton physics.

* ``source="viz:newton_gl:streaming_view"`` and ``source="viz:kit:streaming_view"``
  require ``streaming_view=True`` on the corresponding visualizer cfg.  A
  :class:`~RuntimeError` is raised at the first capture attempt if it is not set.

* For ``source="sensor:<name>"``, the named field must exist on the scene config with
  ``"rgb"`` in its ``data_types``.

.. list-table::
   :widths: 18 10 72
   :header-rows: 1

   * - Visualizer
     - ``--video``
     - Notes
   * - ``kit``
     - ✓
     - Launches Isaac Sim; the launch enables camera rendering for the recording.
   * - ``newton_gl``
     - ✓
     - Uses pyglet's EGL backend when headless; the default for ``--video`` without a
       capture-capable ``--viz``.
   * - ``newton_rtx``
     - ✓
     - Starts the OVRTX runtime; capture performs a GPU-to-CPU readback of the path-traced
       framebuffer.
   * - ``rerun``
     - ✗
     - Remote streaming tool; no local frame-capture API. ``viz:rerun`` is an error;
       ``--video --viz rerun`` records from a headless ``newton_gl``.
   * - ``viser``
     - ✗
     - Browser streaming tool; no local frame-capture API. ``viz:viser`` is an error;
       ``--video --viz viser`` records from a headless ``newton_gl``.

To record video while streaming with Rerun or Viser, pass ``--video`` or name a capture-capable
visualizer, e.g. ``--viz viser --video viz:kit``; it runs headless next to the streaming visualizer.
Alternatively, record directly from a scene camera sensor without any visualizer:

.. code-block:: python

    VideoRecorderCfg(source="sensor:<name>")    # add to env_cfg.video_recorders


See also
--------

* :doc:`/source/concepts/visualization`: configuring interactive visualizers
* :doc:`visualizer_tiled_camera`: tiled camera panel setup
* :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs as images
