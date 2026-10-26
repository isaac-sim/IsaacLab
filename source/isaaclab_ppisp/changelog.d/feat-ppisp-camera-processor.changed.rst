* Changed PPISP to run as a camera processor instead of inside renderer backends. ``CameraISPMode`` values are
  accepted by ``PpispProcessorCfg.isp_cfg``, and :class:`~isaaclab_ppisp.PpispPipeline` remains the kernel
  dispatcher used by the processor.
