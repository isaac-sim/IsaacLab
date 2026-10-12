:orphan:

.. _how_to_record_video:

Recording video
===============

.. currentmodule:: isaaclab

This guide shows how to record ``mp4`` clips from a visualizer or a scene sensor during a run.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_shadow_hand_tiled_gt.mp4" type="video/mp4">
   </video>
   <p class="viz-cap" style="text-align:center; font-style:italic;">Recorded clip of the
   Shadow Hand cube-reorientation task, recording the streaming view of the Rerun
   visualizer with 4 streaming view envs and 4 GT types</p>

Each ``VideoRecorderCfg`` entry is independent: different sources write different files at their own
cadence, with no limit on simultaneous recorders.


Quick Start
-----------

Add a ``VideoRecorderCfg`` to ``env_cfg.video_recorders``:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir="videos/")
    ]

Or pass ``--video`` on the command line to record without editing the environment config. It records the
first capture-capable visualizer selected by ``--viz``. Without ``--viz``, it uses the first configured
visualizer (preferring one that supports capture), or creates a headless Kit visualizer when none is
configured. To record several sources at once, list one ``VideoRecorderCfg`` per source in
``env_cfg.video_recorders``; ``--video_length`` and ``--video_interval`` apply to every recorder:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --video

See :ref:`the --video sources <record_video_cli>` for the command-line selection rules, and `Sources`_
and `Clip control`_ below for the ``VideoRecorderCfg`` options. To record from a different camera
angle than the interactive view, see `Recording from an independent camera angle`_.


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
* The sensor must have the recorded channel in its ``data_types``; ``rgb`` is the default, and
  ``depth``, ``segmentation``, and ``normals`` are also available (see `Sources`_)

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


Recipes
-------

The recipes below are command-line and configuration snippets that are not part of the tutorial
script.

Recording during RL training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record a short clip at regular intervals while training, without opening a window:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --video \
       --video_length 200 --video_interval 2000

With no ``--viz`` or configured visualizer, ``--video`` records from a headless Kit visualizer. This writes a
200-step clip every 2,000 steps. The ``train`` and ``play`` entrypoints put clips in ``videos/``
under the run's log directory unless a recorder sets ``output_dir``; a script that builds its own
recorders writes to ``output_dir`` (default ``videos``). Add ``--viz kit`` to watch the run in a
window while recording the same view.

Recording depth or segmentation from a scene sensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Append a channel to the sensor source to record something other than RGB. Each recorder writes its
own file, so several channels of one camera can be recorded in the same run:

.. code-block:: python

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="sensor:tiled_camera:rgb", output_filename_prefix="rgb"),
        VideoRecorderCfg(source="sensor:tiled_camera:depth", output_filename_prefix="depth"),
    ]

Depth uses the turbo colormap over ``depth_colormap_min`` to ``depth_colormap_max``. Use the
same prefix-per-recorder pattern for ``segmentation`` and ``normals``.

Recording while streaming with Viser or Rerun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Viser and Rerun stream to a browser or viewer and have no frame capture, so they cannot be
recorded. Pair them with a capture-capable visualizer that runs headless next to them:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        ViserVisualizerCfg(),
        NewtonGLVisualizerCfg(headless=True),
    ]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl")]

Alternatively, record directly from a scene camera with ``source="sensor:<name>"``, which needs no
visualizer.

.. _how_to_record_video_angle:

Recording from an independent camera angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure the recording angle on the visualizer, not the recorder. To record from a headless Newton
visualizer at a different angle alongside an interactive Kit viewer, configure both and omit ``--viz``
so the configuration selects them:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(4.0, 4.0, 2.0)),
        NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton_gl", output_dir="videos/"),
    ]

Alternatively, use a :class:`~isaaclab.sensors.CameraCfg` sensor in the scene and record
with ``source="sensor:<name>"``, which gives full control over the recording viewpoint
without requiring a second interactive visualizer.


Sources
-------

The ``source`` string selects what to capture:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Source string
     - Captures from
   * - ``"visualizer"`` (default)
     - First active recording-capable visualizer
   * - ``"visualizer:kit"``
     - Kit visualizer viewport
   * - ``"visualizer:kit:streaming_view"``
     - Kit streaming camera panel, requires ``streaming_view=True``
   * - ``"visualizer:newton_gl"``
     - Newton GL visualizer viewport
   * - ``"visualizer:newton_rtx"``
     - Newton OVRTX path-traced viewport
   * - ``"visualizer:newton_gl:streaming_view"``
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
visualizer config, not on the recorder. A named visualizer source must also have a matching active
visualizer in ``sim.visualizer_cfgs``. Set its ``headless=True`` when it should capture without a window.
The deprecated ``visualizer:newton`` source still selects Newton GL; use ``visualizer:newton_gl``.

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

**Recurring clips every 1,000 env steps:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", video_length=200, video_interval=1000)

**Keep only the most recent clip on disk:**

.. code-block:: python

    VideoRecorderCfg(source="visualizer:kit", video_length=200, video_interval=1000,
                     keep_last_n_clips=1)


Limitations
-----------

* ``source="visualizer:kit"`` and ``source="visualizer:kit:streaming_view"`` require cubric
  to propagate Newton Fabric scene transforms to the RTX renderer.  Without cubric, a warning
  is logged and a black-frame warning is emitted at clip write time.  Use
  ``source="visualizer:newton_gl"`` for guaranteed capture with Newton physics.

* ``source="visualizer:newton_gl:streaming_view"`` and ``source="visualizer:kit:streaming_view"``
  require ``streaming_view=True`` on the corresponding visualizer cfg.  A
  :class:`~RuntimeError` is raised at the first capture attempt if it is not set.

* For ``source="sensor:<name>"``, the named sensor must exist on the scene config with the recorded
  channel (``"rgb"`` by default) in its ``data_types``.

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
     - Remote streaming tool; no local frame-capture API. ``--video --viz rerun`` raises an
       error; select it together with a capture-capable visualizer to record.
   * - ``viser``
     - ✗
     - Browser streaming tool; no local frame-capture API. ``--video --viz viser`` raises an
       error; select it together with a capture-capable visualizer to record.

To record video while streaming with Rerun or Viser, see `Recording while streaming with Viser or
Rerun`_.


See also
--------

* :doc:`/source/concepts/visualization`: configuring interactive visualizers and recording from them
* :doc:`/source/how-to/visualizer_streaming_camera_view`: tiled camera panel setup
* :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs as images
