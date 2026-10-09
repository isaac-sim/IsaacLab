.. _how-to-visualizer-tiled-camera:

Visualizer Streaming Camera View
=================================

.. currentmodule:: isaaclab

For general visualizer documentation, see :doc:`/source/concepts/visualization`.

The visualizer streaming camera view is a live monitoring and debugging tool. It combines
ground-truth camera frames from multiple environments (RGB, depth, segmentation, or surface
normals) into a single panel. Cameras are declared in the scene before cloning; visualizers only
read their output. Multiple visualizers can display the same sensor with different tile selections.

Image layouts and color tables are prepared when the selection or layout changes. Colorization
and tiling run on the source device and reuse the output buffer. Kit presents CUDA images
directly; recording and web consumers read back the composed RGB image through
``render_tiled_rgb_array()``.

.. note::

   The streaming camera view is supported in the Kit, Newton GL, Rerun, and Viser visualizers.
   Newton RTX supports perspective views only and rejects explicit scene-camera sources.


Quick Start
-----------

This guide is accompanied by the ``run_tiled_camera_visualizer.py`` script in
``IsaacLab/scripts/tutorials/07_visualizers``:

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

See `Examples`_ below for the two ways the script can be run, and `Usage`_ for the
``VisualizerCfg`` fields that customize streaming behavior.


Overview
--------

.. raw:: html

   <style>
   .viz-cap { text-align:center; font-style:italic; margin-top:0.4em; font-size:0.9em; }
   </style>

**Kit** launches the streaming view as a separate **Streaming View** viewport, selectable from
the Viewport tabs; it can also be placed side by side with the default interactive viewport
for dual monitoring.

**Newton GL** shows a **Streaming View** section in the HUD sidebar with a **Hide** / **Open**
toggle to show or hide the panel, and a source dropdown to select between different camera
sensors.

**Rerun** and **Viser** push the composed image to their viewers every step. Rerun and Viser cannot be
recorded, but Kit and Newton GL can: ``--video viz:<type>:streaming_view`` records the panel.


Examples
--------

Running ``run_tiled_camera_visualizer.py`` demonstrates two ways to use the streaming camera
view:

- a tracking camera that follows each AnymalD robot, shown in the Kit visualizer
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

The script declares a ``SceneCameraCfg`` in the visualizer config; see `Created and tracking cameras`_. The
launcher adds a camera to every environment, and the camera follows its robot's position and smoothed
heading. Of the 256 environments, 36 are sampled for display.

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


Usage
-----

Configuration notes
~~~~~~~~~~~~~~~~~~~~

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


Display sources
~~~~~~~~~~~~~~~

``VisualizerCfg.cameras`` accepts ``PerspectiveCameraCfg`` for the interactive view and ``SceneCameraCfg`` for a
scene camera sensor. Kit, Newton GL, Rerun, and Viser display scene-camera output in a streaming panel.
Newton RTX accepts perspective sources only. Every explicit scene source must provide the requested
``streaming_gt_types`` channels.

``SceneCameraCfg(prim_path=...)`` refers to a camera the scene already declares, e.g. a wrist camera, and creates
nothing. A path that matches no scene camera raises an error.


Created and tracking cameras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``create=True``, ``SceneCameraCfg`` declares the camera instead of referring to one. The launcher adds it to
every environment when a visualizer is selected, or a video records a ``streaming_view`` source; runs that display
nothing do not pay for it. The camera stays fixed relative to each environment, or follows an asset when
``track_path`` is set, without changing the interactive camera, which stays free to move. Set it on a task through
``sim.default_visualizer_cfg``, or on any visualizer config:

.. code-block:: python

    from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

    self.sim.default_visualizer_cfg = VisualizerCfg(
        streaming_envs=[0],
        cameras=[
            SceneCameraCfg(
                create=True,
                eye=(1.8, -3.0, 1.1),
                lookat=(0.15, 0.0, 0.0),
                track_path="robot",
                follow_heading=True,
                heading_smoothing_time_constant=0.2,
            )
        ],
    )

These fields apply only with ``create=True``:

* ``eye`` and ``lookat`` are offsets [m] from the tracked asset, or from the environment origin when
  ``track_path`` is None.
* ``track_path`` names a scene asset (``"robot"``) or one of its bodies (``"robot/base"``). Tracking moves the
  camera, so it needs ``create=True``: a camera the scene declares is never moved.
* ``follow_heading`` rotates the offsets with the asset's yaw, keeping the horizon level.
  ``heading_smoothing_time_constant`` [s] damps rapid turns; zero follows immediately.
* ``focal_length`` [mm], ``resolution`` (width, height) and ``data_types`` set the optics and outputs.
* ``renderer_cfg`` sets the camera's renderer and so its quality, e.g. ``NewtonWarpRendererCfg`` with
  shadows enabled. Without it the camera uses the Newton Warp renderer, except with a Kit visualizer on other
  physics, where it uses the camera default.

The scene holds one camera per environment and the renderer draws every environment when the view reads its
image, so the cost grows with the environment count and the resolution, and a view that is hidden costs nothing.
Use few environments, or a smaller ``resolution``, for 1080p views.


Troubleshooting
~~~~~~~~~~~~~~~~

* If a view reports no matching camera, declare a ``CameraCfg`` in the scene and set
  ``streaming_sensor_prim_path`` to its prim path.
* Each ``streaming_gt_types`` entry must have a corresponding output in the selected
  camera's ``data_types``. An explicit incompatible source raises an error. Automatic selection
  skips incompatible sources, and RGB display accepts a camera's RGBA output.
* If the depth panel shows a flat color, adjust ``streaming_depth_min`` and
  ``streaming_depth_max`` to bracket the expected depth range in your scene.
* If the view is too expensive, reduce ``streaming_envs``, ``--num_envs``, or the camera
  resolution.

Migration from generated streaming cameras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizers no longer create, move, or destroy camera sensors. Replace
``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
with a scene ``CameraCfg``: place its ``prim_path`` under the desired parent, set its
``offset``, and supply ``renderer_cfg`` there. Select that camera through
``streaming_sensor_prim_path``. This puts the camera in the clone plan and the normal sensor
initialization, update, and teardown lifecycle.

Closing a visualizer does not affect the camera or other visualizers reading it.
``streaming_envs`` selects displayed tiles, not camera allocation or capture resolution.

.. note::

   A camera parented under a robot in the scene follows it only when the physics backend writes poses
   to USD, e.g. PhysX with Kit. With Newton physics it stays where it was spawned; use a
   ``SceneCameraCfg`` instead, which moves the camera explicitly.


See also
--------

* :doc:`/source/concepts/visualization`: visualizer configuration and UI controls
* :doc:`/source/how-to/configure_rendering`: customizing RTX rendering settings
