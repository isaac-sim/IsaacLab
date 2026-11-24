* Deprecated ``CameraRenderSpec`` as a subclass of ``SensorRenderSpec``. Construct
  ``SensorRenderSpec`` for a camera or a lidar and pass sensor prims as ``prim_paths``.
  ``CameraRenderSpec.camera_prim_paths`` still reads and sets that field.
* Deprecated ``BaseRenderer.update_camera`` as a wrapper of ``BaseRenderer.update_sensor``.
  Call ``update_sensor`` instead. A subclass that still implements ``update_camera`` is
  called from ``update_sensor``.
* ``RenderContext.render_into_sensors`` groups prepared camera and lidar captures by renderer
  and reads each output into ``CameraData`` or ``LidarData``.
* Deprecated ``RenderContext.render_into_camera`` and ``RenderContext.render_into_cameras`` as
  wrappers of ``RenderContext.render_into_sensor`` and ``RenderContext.render_into_sensors``.
  Call the sensor methods instead.
