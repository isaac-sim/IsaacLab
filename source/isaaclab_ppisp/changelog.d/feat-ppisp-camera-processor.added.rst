* Added :class:`~isaaclab_ppisp.PpispProcessor` and :class:`~isaaclab_ppisp.PpispProcessorCfg`, which apply
  PPISP as a camera tensor processor with any renderer that publishes ``rgb_hdr``. During camera construction,
  the processor resolves USD-authored coefficients and, on Isaac RTX and OVRTX, disables RTX automatic exposure
  on its camera prims so the HDR input is not exposed twice.
