* **Breaking:** Renamed ``BaseRenderer.update_camera`` to ``update_sensor`` so camera and lidar poses share one method.
  Call ``update_sensor`` instead of ``update_camera``. ``render`` accepts an optional ``force`` flag and ignores it.
* **Breaking:** Isaac RTX camera render data reads ``SensorRenderSpec.prim_paths``. A lidar config is
  rejected with ``NotImplementedError``.
