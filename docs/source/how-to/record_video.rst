:orphan:

.. _how_to_record_video:

Recording video
===============

.. currentmodule:: isaaclab

This guide shows how to record ``mp4`` clips from a visualizer or a scene sensor during a run. For
how recording sources, clips, and visualizers fit together, see :doc:`/source/concepts/video_recording`.


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

``--video`` records from the source it names, and ``--viz`` still decides which visualizers open a
window. See :ref:`the --video sources <record_video_cli>` for every command-line form, and the
:doc:`concept page </source/concepts/video_recording>` for source types and clip control.


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


See also
--------

* :doc:`/source/concepts/video_recording`: source types, clip control, and compatibility
* :doc:`/source/concepts/visualization`: configuring interactive visualizers
* :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs as images
