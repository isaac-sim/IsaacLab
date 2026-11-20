.. _concepts_sensors_lidar:
.. _overview_sensors_lidar:

.. currentmodule:: isaaclab

Lidar
=====

A :class:`~sensors.Lidar` is a tiled OVRTX solid-state lidar. Each environment
gets one ``OmniLidar`` prim. Those prims share one render product, and one
renderer step returns a point cloud that is split back into per-environment
buffers.

:attr:`~sensors.LidarData.points` has shape ``(num_envs, num_rays, 3)`` in the
Torch view. The values are Cartesian positions, in meters, in the frame
selected by ``frame_of_reference`` (``WORLD`` or ``SENSOR``). Misses are
non-finite, so every environment keeps the same ray slots.
:attr:`~sensors.LidarData.intensity` and :attr:`~sensors.LidarData.valid` have
shape ``(num_envs, num_rays)``. Tile ``i`` is environment ``i``.

The sensor requires :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`. Set
``motion_bvh`` to ``"auto"`` or ``"enable"`` so a non-visual sensor is rendered.
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
