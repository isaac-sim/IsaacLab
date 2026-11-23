* **Breaking:** Replaced ``CameraRenderSpec`` and ``LidarRenderSpec`` with
  :class:`~isaaclab.renderers.sensor_render_spec.SensorRenderSpec`. Construct one spec for a camera
  or a lidar and pass sensor prims as ``prim_paths``. Ray count and sampling interval are no longer
  spec fields; the renderer derives them from the lidar config and the physics step.
