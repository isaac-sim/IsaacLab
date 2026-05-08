Changed
^^^^^^^

* :class:`~isaaclab_contrib.sensors.tacsl_sensor.VisuotactileSensor` now accesses
  :attr:`~isaaclab.sensors.camera.CameraData.output` entries through their
  :attr:`~isaaclab.utils.warp.proxy_array.ProxyArray.torch` view to match the
  Warp-first :class:`CameraData` storage.
