Changed
^^^^^^^

* Routed Newton Warp image processing through the sensor-owned processor chain, preserving direct HDR
  output bindings. Existing ``CameraCfg.isp_cfg`` configurations remained supported; new pipelines used
  ``CameraCfg.post_processors``.
