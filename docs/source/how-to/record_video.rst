:orphan:

.. _how_to_record_video:

Recording video
===============

.. currentmodule:: isaaclab

This guide shows how to record ``mp4`` clips from a visualizer or a scene sensor during a run.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
     <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/how_to/record_video_newton_viewport.mp4" type="video/mp4">
   </video>
   <p class="viz-cap" style="text-align:center; font-style:italic;">Recorded clip of the
   Shadow Hand cube-reorientation task from the Newton visualizer</p>

Each ``VideoRecorderCfg`` entry is independent: different sources write different files at their own
cadence, with no limit on simultaneous recorders.


Quick Start
-----------

Add a ``VideoRecorderCfg`` to ``env_cfg.video_recorders``:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg.video_recorders = [
        VideoRecorderCfg(source="viz:newton", output_dir="videos/")
    ]

``source`` also accepts an ``ImageViewCfg``: use ``VideoRecorderCfg(source=view)`` to share an image
selection and frame with a window. See :ref:`how-to-visualizer-streaming-camera-view` for sensor and
perspective examples.

Or pass ``--video [SOURCE]`` on the command line to record without editing the environment config. ``--video``
adds a single recorder; to record several sources at once, list one ``VideoRecorderCfg`` per source in
``env_cfg.video_recorders``, which then takes precedence over the ``--video`` source (``--video_length`` and
``--video_interval`` still apply to every recorder):

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --video

``--video`` records from the source it names, and ``--viz`` still decides which visualizers open a
window. See :ref:`the --video sources <record_video_cli>` for every command-line form, and `Sources`_ and `Clip control`_
below for the ``VideoRecorderCfg`` options. To record from a different camera angle than the
interactive view, see `Recording from an independent camera angle`_.


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
sensor. Examples 1 and 2 each demonstrate one recording source; Example 3 combines three sources.

The Newton RTX default records ``newton_viewport``, ``tiled_camera_view`` and ``sensor`` clips.
The deprecated Kit mode retains the original four streams and filenames, including
``kit_viewport`` and ``tiled_kit_viewport``:

.. code-block:: bash

   uv run --extra isaacsim --extra video python scripts/tutorials/07_visualizers/run_video_recording.py \
      --example 3 --num_envs 4 --viz kit,newton_gl

Pass ``--viz newton_gl`` to record an OpenGL viewport instead of the default Newton RTX viewport.
The scene camera uses OVRTX by default. Install the ``ovrtx`` extra, or ``ov`` for both OvPhysX
and OVRTX. The Newton viewport uses RTX by default, independently of the scene-camera recording.
Use ``--viz newton_gl`` to select its lightweight OpenGL view.

.. dropdown:: Code for run_video_recording.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_video_recording.py
      :language: python
      :linenos:


Example 1: Newton viewport
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
       --example 1 --num_envs 4 --viz newton_rtx

* Records the Newton interactive viewport using RTX
* Shows 4 parallel environments
* One clip is written to ``videos/recording_tutorial/example_1/newton_viewport_0000.mp4``

.. raw:: html

   <div class="viz-clip-crop viz-clip-crop-lg">
     <video autoplay loop muted playsinline controls preload="auto">
       <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/how_to/record_video_newton_viewport.mp4" type="video/mp4">
     </video>
   </div>
   <p class="viz-cap">Newton visualizer (OpenGL recording; the example now defaults to RTX)</p>


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
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/how_to/record_video_sensor.mp4" type="video/mp4">
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
       --example 3 --num_envs 4 --viz newton_rtx

Three independent clips are written to ``videos/recording_tutorial/example_3/``:

* ``newton_viewport_0000.mp4``: Newton viewport showing its OVRTX camera
* ``tiled_camera_view_0000.mp4``: an ``ImageViewCfg`` composing the scene camera's per-environment images
* ``sensor_0000.mp4``: scene tiled-camera sensor (offline render)

RTX viewport recording uses the OVRTX runtime and works with Newton or OvPhysX physics.
Set the visualizer's ``headless=True`` to record without opening a window.
The tiled ``ImageViewCfg`` and scene sensor remain independent recording sources while the
main viewport shows RTX. Naming the scene camera in the image view fixes the recorded source
even if the window switches to another camera.


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

With no ``--viz``, ``--video`` records from a headless Newton visualizer. This writes a
200-step clip every 2,000 steps. The ``train`` and ``play`` entrypoints put clips in ``videos/``
under the run's log directory unless a recorder sets ``output_dir``; a script that builds its own
recorders writes to ``output_dir`` (default ``videos``). Add ``--viz newton_rtx`` to watch and record
an RTX viewport with Newton or OvPhysX physics.

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

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz viser --video viz:newton

Alternatively, record directly from a scene camera with ``source="sensor:<name>"``, which needs no
visualizer.

.. _how_to_record_video_angle:

Recording from an independent camera angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure a separate perspective view for the recording alongside the interactive Newton window.
The recorder identifies its producer through the shared ``ImageViewCfg``:

.. code-block:: python

    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, VisualizerCfg

    recording_view = ImageViewCfg(source=PerspectiveCameraCfg(eye=(12.0, 0.0, 6.0)))
    env_cfg.sim.visualizer_cfgs = [
        VisualizerCfg(eye=(4.0, 4.0, 2.0), backend=NewtonGLBackendCfg(initial_renderer="rtx")),
        VisualizerCfg(view=recording_view, headless=True, backend=NewtonGLBackendCfg()),
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source=recording_view, output_dir="videos/"),
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
   * - ``"viz"`` (default)
     - First capture-capable visualizer ``--viz`` selected, else a headless Newton visualizer
   * - ``"viz:kit"``
     - Kit visualizer viewport
   * - ``"viz:kit:streaming_view"``
     - Kit streaming camera panel, requires ``streaming_view=True``
   * - ``"viz:newton"``
     - Newton visualizer viewport, in whichever renderer the window shows (OpenGL or RTX)
   * - ``"viz:newton:streaming_view"``
     - Newton streaming camera panel, requires ``streaming_view=True``
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
``newton_gl`` type (``"viz:newton_gl"``) is still accepted as an alias for ``newton``. Bare visualizer
types are accepted here too, e.g. ``VideoRecorderCfg(source="newton")`` is equivalent to
``VideoRecorderCfg(source="viz:newton")``.

.. note::

   Use ``NewtonGLBackendCfg(initial_renderer="rtx")`` to record the RTX viewport, including
   headless runs. The recorder source remains ``"viz:newton"``; ``newton_rtx`` selects the
   renderer when launching the visualizer.


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

    VideoRecorderCfg(source="viz:newton", video_length=500, video_interval=0)

**Recurring clips every 1,000 env steps:**

.. code-block:: python

    VideoRecorderCfg(source="viz:newton", video_length=200, video_interval=1000)

**Keep only the most recent clip on disk:**

.. code-block:: python

    VideoRecorderCfg(source="viz:newton", video_length=200, video_interval=1000,
                     keep_last_n_clips=1)


Limitations
-----------

* ``source="viz:newton:streaming_view"`` and ``source="viz:kit:streaming_view"``
  require ``streaming_view=True`` on the corresponding visualizer cfg.  A
  :class:`~RuntimeError` is raised when the recorder is constructed if it is not set.

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
   * - ``newton``
     - ✓
     - Uses pyglet's EGL backend when headless; the default for ``--video`` without a
       capture-capable ``--viz``.
   * - ``rerun``
     - ✗
     - Remote streaming tool; no local frame-capture API. ``viz:rerun`` is an error;
       ``--video --viz rerun`` records from a headless ``newton``.
   * - ``viser``
     - ✗
     - Browser streaming tool; no local frame-capture API. ``viz:viser`` is an error;
       ``--video --viz viser`` records from a headless ``newton``.

To record video while streaming with Rerun or Viser, see `Recording while streaming with Viser or
Rerun`_.


See also
--------

* :doc:`/source/concepts/visualization`: configuring interactive visualizers and recording from them
* :doc:`/source/how-to/visualizer_streaming_camera_view`: tiled camera panel setup
* :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs as images
