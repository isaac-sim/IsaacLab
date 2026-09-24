Changed
^^^^^^^

* Routed Isaac RTX image processing through the sensor-owned processor chain. Renderer inputs and
  neutral exposure were selected through generic sensor requirements. Existing ``CameraCfg.isp_cfg``
  configurations remained supported; new pipelines used ``CameraCfg.post_processors``.
