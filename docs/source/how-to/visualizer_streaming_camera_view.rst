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

See `Examples`_ below for the two ways to run the script. For the ``VisualizerCfg`` fields that
customize streaming, sources, and troubleshooting, see :ref:`visualization-streaming-camera-view`.


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


See also
--------

* :ref:`visualization-streaming-camera-view`: configuration, display sources, and troubleshooting
* :doc:`/source/concepts/visualization`: visualizer configuration and UI controls
* :doc:`/source/how-to/record_video`: recording the streaming view to video
* :doc:`/source/how-to/configure_rendering`: customizing RTX rendering settings
