.. _concepts_sensors_lidar:
.. _overview_sensors_lidar:

.. currentmodule:: isaaclab

Lidar
=====

A :class:`~sensors.Lidar` publishes one scan per environment from an ``OmniLidar`` prim.
The prim is a LidarWpm sensor. ``scan_type`` selects a solid-state or rotary
scan and defaults to rotary, matching the OmniLidar schemas. Those prims share one render
product, and one renderer step returns a point cloud that is split back into
per-environment buffers.

``num_rays`` is the full scan. A solid-state scan fires the pattern once. A
rotary scan repeats that pattern at every ``rot_pattern_firing_res_deg`` step across
the configured azimuth window. Each firing keeps ``max_returns`` slots.
:attr:`~sensors.LidarData.points` has shape ``(num_envs, num_rays, 3)`` in the
Torch view. ``elements_coords_type`` defaults to ``SPHERICAL``. ``CARTESIAN``
elements are positions in meters, in the frame selected by
``output_frame_of_reference`` (``WORLD`` or ``SENSOR``).
``include_invalid_points`` defaults to false, matching the schemas, so missed
rays are dropped. Set it true to keep a non-finite slot for every ray.
The other scan defaults follow the schemas too: a 0.1° rotary step, two
returns per ray, the ``SENSOR`` frame, a 200 m far range, and ``instant_lidar``
disabled.
:attr:`~sensors.LidarData.intensity` and :attr:`~sensors.LidarData.valid` have
shape ``(num_envs, num_rays)``. Tile ``i`` is environment ``i``.

The OVRTX renderer is the current capture implementation. Use
:class:`~isaaclab_ov.renderers.OVRTXRendererCfg` and set ``motion_bvh`` to ``"auto"`` or
``"enable"`` so a non-visual sensor is rendered.
Cameras that use the same renderer configuration are submitted in that step.
Camera-only scenes leave ``motion_bvh`` unset and do not build a lidar product.

The cartpole camera tasks demonstrate the sensor. ``presets=lidar`` adds the
lidar and gives the policy two stacked range images of the scan. Misses are 0
and a hit at the far range is 1. The scene defaults to 64 environments; pass
``--num_envs 512`` for the larger camera-preset count. The other camera presets
are unchanged:

.. code-block:: bash

   uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera --num_envs 4 presets=lidar
   uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera-Direct --num_envs 4 presets=lidar

Newton GL and Viser draw every visualized environment's finite hits as points
every tenth step, and Kit draws them in the viewport. The camera backs up until
those environments fit in view.
