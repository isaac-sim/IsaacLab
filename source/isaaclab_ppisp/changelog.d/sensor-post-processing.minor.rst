Added
^^^^^

* Added ``PpispProcessorCfg`` for the camera's ordered visual processing chain,
  including USD configuration discovery and controller buffer initialization.

Changed
^^^^^^^

* Moved PPISP execution and controller state from renderer instances to camera
  sensors. Existing ``CameraCfg.isp_cfg`` configurations remained supported;
  composable pipelines can use
  ``post_processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` instead.
