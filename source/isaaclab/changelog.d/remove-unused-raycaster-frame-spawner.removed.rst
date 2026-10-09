* **Breaking:** Removed the unused ``RayCasterCfg.spawn``, ``SensorFrameCfg``, and
  ``spawn_sensor_frame`` APIs. Ray casters track existing bodies or frames; author
  a separate USD Xform explicitly when an attachment frame is needed.
