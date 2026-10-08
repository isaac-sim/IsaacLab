:orphan:

.. _how-to-visualizer-streaming-camera-view:

Streaming a camera view in a visualizer
=======================================

.. currentmodule:: isaaclab

This guide shows how to show ground-truth camera frames from many environments in one live
panel of a visualizer. For how the view works and which visualizers support it, see
:ref:`visualization-streaming-camera-view`.

Quick Start
-----------

Run the ``run_tiled_camera_visualizer.py`` script in ``IsaacLab/scripts/tutorials/07_visualizers``:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
             --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py ^
             --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

.. dropdown:: Code for run_tiled_camera_visualizer.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py
      :language: python
      :linenos:

See `Examples`_ below for the two ways to run the script, and `Configuration`_ for the
``VisualizerCfg`` fields that customize streaming, display sources, and troubleshooting.


.. raw:: html

   <style>
   .viz-cap { text-align:center; font-style:italic; margin-top:0.4em; font-size:0.9em; }
   </style>

Examples
--------

Running ``run_tiled_camera_visualizer.py`` demonstrates two ways to use the streaming camera
view:

- scene-declared cameras attached to AnymalD robot bases, shown in the Kit visualizer
- streaming from existing wrist-mounted robot cameras, shown in the Newton visualizer


Example 1: Following AnymalD Robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
             --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py ^
             --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

The script adds a ``CameraCfg`` beneath each robot's base before constructing the environment.
Its offset defines the view relative to the base, so the camera follows the robot through the
normal sensor lifecycle. Of the 256 environments, 36 are sampled for display.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="auto" style="width:100%; display:block;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_kit_anymal_interactive.mp4" type="video/mp4">
   </video>
   <p class="viz-cap">Kit visualizer: interactive viewport</p>
   <video autoplay loop muted playsinline controls preload="auto" style="width:100%; display:block; margin-top:1em;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_kit_anymal_tiled.mp4" type="video/mp4">
   </video>
   <p class="viz-cap">Kit visualizer: streaming camera view</p>


Example 2: Streaming from Robot-Mounted Cameras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run --extra teleop python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
             --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor --num_envs 25 --viz newton_gl

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run --extra teleop python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py ^
             --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor --num_envs 25 --viz newton_gl

The Galbot cube-stacking environment ships with wrist-mounted cameras giving an egocentric
view of the gripper, table, and cubes. The script's ``NewtonGLVisualizerCfg`` streams from the
existing sensor at ``/World/envs/env_.*/Robot/head_camera_sim_view_frame/head_camera``; edit
``cameras=[SceneCameraCfg(prim_path=...)]`` to show a different camera. Of the 25 environments, 12 camera
feeds are shown by default.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="auto" style="width:100%; display:block;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_newton_galbot_interactive.mp4" type="video/mp4">
   </video>
   <p class="viz-cap">Newton visualizer: interactive viewport</p>
   <video autoplay loop muted playsinline controls preload="auto" style="width:100%; display:block; margin-top:1em;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_newton_galbot_tiled.mp4" type="video/mp4">
   </video>
   <p class="viz-cap">Newton visualizer: streaming camera view</p>


Configuration
-------------

Declare camera resolution, pose, renderer, and output types on ``CameraCfg`` in the scene.
Then choose what to display in ``VisualizerCfg``:

* ``streaming_sensor_prim_path`` selects a scene :class:`~isaaclab.sensors.Camera` by its configured
  prim path. The ``{ENV_REGEX_NS}`` macro uses the scene's environment template.
  If omitted, the first camera supporting the requested channels is selected; without a compatible
  camera the panel stays empty.
* ``streaming_envs`` controls how many environment tiles are shown. Pass an ``int`` to randomly
  sample that many environments, or a ``list[int]`` to pin specific environment indices.
* ``streaming_gt_types`` selects which ground-truth types are shown, e.g.
  ``["rgb", "depth", "segmentation", "normals"]``.
* ``streaming_depth_min`` / ``streaming_depth_max`` set the depth colormap range in metres.

Image layouts and color tables are prepared when the selection or layout changes. Colorization and
tiling run on the source device and reuse the output buffer. Kit presents CUDA images directly;
recording and web consumers read back the composed RGB image through ``render_tiled_rgb_array()``.


Display sources
---------------

``VisualizerCfg.cameras`` accepts ``PerspectiveCameraCfg`` for the interactive view and
``SceneCameraCfg(prim_path=...)`` for a camera sensor already declared in the scene.
Kit, Newton GL, Rerun, and Viser display scene-camera output in a streaming panel.
Newton RTX accepts perspective sources only. Every explicit scene source must provide
the requested ``streaming_gt_types`` channels.


Troubleshooting
---------------

* If a view reports no matching camera, declare a ``CameraCfg`` in the scene and set
  ``streaming_sensor_prim_path`` to its prim path.
* Each ``streaming_gt_types`` entry must have a corresponding output in the selected
  camera's ``data_types``. An explicit incompatible source raises an error. Automatic selection
  skips incompatible sources, and RGB display accepts a camera's RGBA output.
* If the depth panel shows a flat color, adjust ``streaming_depth_min`` and
  ``streaming_depth_max`` to bracket the expected depth range in your scene.
* If the view is too expensive, reduce ``streaming_envs``, ``--num_envs``, or the camera
  resolution.


Migrating from generated cameras
--------------------------------

Visualizers no longer create, move, or destroy camera sensors. Replace
``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
with a scene ``CameraCfg``: place its ``prim_path`` under the desired parent, set its
``offset``, and supply ``renderer_cfg`` there. Select that camera through
``streaming_sensor_prim_path``. This puts the camera in the clone plan and the normal sensor
initialization, update, and teardown lifecycle. The full option mapping is in the
:doc:`3.0 migration guide </source/migration/migrating_to_isaaclab_3-0>`.

Closing a visualizer does not affect the camera or other visualizers reading it.
``streaming_envs`` selects displayed tiles, not camera allocation or capture resolution.


See also
--------

* :ref:`visualization-streaming-camera-view`: how the streaming view works and which visualizers support it
* :doc:`/source/concepts/visualization`: visualizer configuration and UI controls
* :doc:`/source/how-to/record_video`: recording the streaming view to video
* :doc:`/source/how-to/configure_rendering`: customizing RTX rendering settings
