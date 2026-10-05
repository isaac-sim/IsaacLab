.. _how-to-visualizer-tiled-camera:

Visualizer Streaming Camera View
=================================

.. currentmodule:: isaaclab

For general visualizer documentation, see :doc:`/source/concepts/visualization`.

The visualizer streaming camera view is a live monitoring and debugging tool. It combines
ground-truth camera frames from multiple environments (RGB, depth, segmentation, or surface
normals) into a single image. Cameras are declared in the scene before cloning; visualizers borrow
their output. Multiple visualizers can display the same sensor with different tile selections.

.. note::

   The streaming camera view is supported in the Kit, Newton GL, Rerun, and Viser visualizers.
   The Newton RTX visualizer accepts the configuration but does not display the panel (experimental).


Quick Start
-----------

This guide is accompanied by the ``run_tiled_camera_visualizer.py`` script in
``IsaacLab/scripts/tutorials/07_visualizers``:

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
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

**Newton GL** has a **Camera View** selector in the sidebar. Selecting a scene camera displays
its tiled output in the main viewport instead of rendering a perspective view behind it.
Selecting a perspective camera restores normal interactive rendering and picking.

Without an explicit ``camera`` list, the viewer starts in perspective and offers scene cameras
that provide every requested ``streaming_gt_types`` channel. For example, set
``streaming_gt_types=("depth",)`` to offer depth-only cameras. Output aliases count: a camera
configured for RGBA that also publishes RGB can supply the RGB view. No images are captured to
build the selector. Explicitly configured scene-camera choices must support the requested channels;
incompatible choices raise a configuration error during initialization, including inactive choices.

Use ``camera`` to declare one view or a list of selectable views; the first starts active:

.. code-block:: python

   from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg
   from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

   viewer_cfg = NewtonGLVisualizerCfg(
       camera=[
           SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/FrontCamera"),
           SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/BackCamera"),
           PerspectiveCameraCfg(eye=(4.0, -4.0, 3.0)),
       ],
       streaming_envs=list(range(16)),
   )

Both scene cameras must already be declared as ``CameraCfg`` entries in the scene. Keyboard
and mouse navigation moves the selected sensor's copies by the same camera-local translation
and rotation across all environments, including copies not displayed. It does not move other
camera choices. These are real sensor pose changes, so policies and other viewers using that
sensor observe the new viewpoint too. Navigation reads the live sensor pose, including parent-body
motion, independently of ``CameraCfg.update_latest_camera_pose``. Pausing rendering freezes the
displayed image and sensor navigation; pausing training alone does not.

With ``InteractiveSceneCfg.lazy_sensor_update=True``, the viewer reads only the selected camera.
Unselected cameras are still allocated and may render if another consumer reads them or the
scene uses eager sensor updates. ``streaming_envs`` selects displayed tiles, not capture work.


Examples
--------

Running ``run_tiled_camera_visualizer.py`` demonstrates two ways to use the streaming camera
view:

- scene-declared cameras attached to AnymalD robot bases, shown in the Kit visualizer
- streaming from existing wrist-mounted robot cameras, shown in the Newton visualizer


Example 1: Following AnymalD Robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
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

.. code-block:: bash

   uv run --extra teleop python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
       --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor --num_envs 25 --viz newton_gl

The Galbot cube-stacking environment ships with wrist-mounted cameras giving an egocentric
view of the gripper, table, and cubes. The script's ``NewtonGLVisualizerCfg`` streams from the
existing sensor at ``/World/envs/env_.*/Robot/head_camera_sim_view_frame/head_camera``; edit
``streaming_sensor_prim_path`` to show a different camera. Of the 25 environments, 12 camera
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
  If omitted, the first scene camera is selected; without cameras the panel stays empty.
* ``streaming_envs`` controls how many environment tiles are shown. Pass an ``int`` to randomly
  sample that many environments, or a ``list[int]`` to pin specific environment indices.
* ``streaming_gt_types`` selects which ground-truth types are shown, e.g.
  ``["rgb", "depth", "segmentation", "normals"]``.
* ``streaming_depth_min`` / ``streaming_depth_max`` set the depth colormap range in metres.


Troubleshooting
~~~~~~~~~~~~~~~~

* If a view reports no matching camera, declare a ``CameraCfg`` in the scene and set
  ``streaming_sensor_prim_path`` to its prim path.
* Each ``streaming_gt_types`` entry must have a corresponding output in the selected
  camera's ``data_types``. Missing channels raise an error rather than silently changing the view.
* If the depth panel shows a flat color, adjust ``streaming_depth_min`` and
  ``streaming_depth_max`` to bracket the expected depth range in your scene.
* If the view is too expensive, reduce ``streaming_envs``, ``--num_envs``, or the camera
  resolution.

Migration from generated streaming cameras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizers no longer create or destroy camera sensors. Replace
``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
with a scene ``CameraCfg``: place its ``prim_path`` under the desired parent, set its
``offset``, and supply ``renderer_cfg`` there. Select that camera through
``streaming_sensor_prim_path``, or ``camera=SceneCameraCfg(...)`` in Newton GL. This puts the camera
in the clone plan and the normal sensor initialization, update, and teardown lifecycle.

Closing a visualizer does not affect the camera or other visualizers reading it.
``streaming_envs`` selects displayed tiles, not camera allocation or capture resolution.


See also
--------

* :doc:`/source/concepts/visualization`: visualizer configuration and UI controls
* :doc:`/source/how-to/configure_rendering`: customizing RTX rendering settings
