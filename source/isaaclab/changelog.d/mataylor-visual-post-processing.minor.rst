Added
^^^^^

* Added an ordered visual-sensor post-processing pipeline with explicit buffer requirements,
  persistent intermediates, and per-camera lifecycle callbacks. Moved PPISP execution to the
  shared camera pipeline while retaining ``CameraCfg.isp_cfg`` compatibility. New configurations
  can use ``CameraCfg.post_processors=[PpispProcessorCfg(isp_cfg=...)]`` and append additional
  processors without modifying renderer implementations.
