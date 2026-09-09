.. _how-to-visualizer-tiled-camera:

Visualizer Streaming Camera View
=================================

.. currentmodule:: isaaclab

For general visualizer documentation, see :doc:`/source/concepts/visualization`.

The visualizer streaming camera view is a live monitoring and debugging tool. It combines
ground-truth camera frames from multiple environments (RGB, depth, segmentation, or surface
normals) into a single panel that updates every step, either following robots automatically or
streaming from existing scene camera sensors.

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
      :emphasize-lines: 72-78,81-83,89-96,107-109
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


Examples
--------

Running ``run_tiled_camera_visualizer.py`` demonstrates two ways to use the streaming camera
view:

- auto-created cameras pointed at and following moving AnymalD robots, shown in the Kit visualizer
- streaming from existing wrist-mounted robot cameras, shown in the Newton visualizer


Example 1: Following AnymalD Robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
       --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

The script's ``KitVisualizerCfg`` creates cameras that point at and follow each robot's base
prim, offset by ``streaming_cam_eye`` (here ``(3.0, 3.0, 3.0)``; try ``(0, 0, 5)`` for a
top-down view). Of the 256 environments, 36 are randomly sampled for the camera view.

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

   uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
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

To customize streaming camera behavior, edit the highlighted ``VisualizerCfg`` fields in
``run_tiled_camera_visualizer.py``:

* For auto-created cameras, ``streaming_cam_target_prim_path`` chooses the followed prim and
  ``streaming_cam_eye`` sets the camera offset from that prim.  Defaults to ``None``, which
  causes the visualizer to adopt the first scene camera it discovers at init; no explicit path
  is needed when a ``TiledCamera`` sensor is already in the scene.
* For existing scene cameras, ``streaming_sensor_prim_path`` must match an Isaac Lab
  :class:`~isaaclab.sensors.Camera` sensor prim path in the selected task.
* ``streaming_envs`` controls how many environment tiles are shown. Pass an ``int`` to randomly
  sample that many environments, or a ``list[int]`` to pin specific environment indices.
* ``streaming_gt_types`` selects which ground-truth types are shown, e.g.
  ``["rgb", "depth", "segmentation", "normals"]``.
* ``streaming_depth_min`` / ``streaming_depth_max`` set the depth colormap range in metres.


Troubleshooting
~~~~~~~~~~~~~~~~

* If a generated view fails with a missing prim error, verify that
  ``streaming_cam_target_prim_path`` resolves in each selected environment; common template
  forms are ``/World/envs/*/...`` and ``/World/envs/env_.*/...``.  In most cases you can leave
  it as ``None`` and let the visualizer adopt an existing scene camera automatically.
* If an existing-camera view reports that no Isaac Lab camera owns the prim, check that
  ``streaming_sensor_prim_path`` matches a :class:`~isaaclab.sensors.Camera` sensor in the task.
* If the depth panel shows a flat color, adjust ``streaming_depth_min`` and
  ``streaming_depth_max`` to bracket the expected depth range in your scene.
* If the view is too expensive, reduce ``streaming_envs``, ``--num_envs``, or the camera
  resolution.

.. warning::

   **Newton MJWarp with** ``replicate_physics=True`` **and auto-created cameras**

   With ``replicate_physics=True``, only ``env_0`` has a USD prim after physics
   initialization. Cameras for the remaining environments (``env_1`` through ``env_{N-1}``)
   are dropped, causing initialization to fail::

       RuntimeError: Number of camera prims in the view (1) does not match
       the number of environments (N).

   **Workaround**: set ``streaming_sensor_prim_path`` to a scene camera that was declared in
   the scene config before physics init (for example, a ``TiledCamera`` on a vision-based
   task).


See also
--------

* :doc:`/source/concepts/visualization`: visualizer configuration and UI controls
* :doc:`/source/how-to/configure_rendering`: customizing RTX rendering settings
