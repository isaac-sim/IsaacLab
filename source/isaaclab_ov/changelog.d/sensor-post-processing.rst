Changed
^^^^^^^

* Routed OVRTX image processing through the sensor-owned processor chain, preserving HDR buffer bindings
  and neutral exposure. Existing ``CameraCfg.isp_cfg`` configurations remained supported; new pipelines
  used ``CameraCfg.post_processors``. HDR transfers between devices reused persistent storage.
