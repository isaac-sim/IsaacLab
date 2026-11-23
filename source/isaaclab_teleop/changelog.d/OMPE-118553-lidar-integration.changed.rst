* **Breaking:** Renamed ``BaseRenderer.update_camera`` to ``update_sensor`` so camera and lidar poses share one method.
  Call ``update_sensor`` instead of ``update_camera``. ``render`` accepts an optional ``force`` flag and ignores it.
* **Breaking:** OVXR camera panels read ``SensorRenderSpec.prim_paths``.
