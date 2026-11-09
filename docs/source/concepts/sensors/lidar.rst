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
Torch view. Misses are non-finite, so every environment keeps the same ray
slots. :attr:`~sensors.LidarData.intensity` and :attr:`~sensors.LidarData.valid`
use that same layout. Tile ``i`` is environment ``i``.

The sensor requires :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`. Set
``motion_bvh`` to ``"auto"`` or ``"enable"`` so a non-visual sensor is rendered.
Cameras that use the same renderer configuration are submitted in that step.
Camera-only scenes leave ``motion_bvh`` unset and do not build a lidar product.

The manager-based cartpole camera task demonstrates the sensor. ``presets=lidar``
keeps the side RGB camera, adds the lidar, and appends the point cloud to the
policy observation. The other camera presets are unchanged:

.. code-block:: bash

   uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera --num_envs 4 presets=lidar
