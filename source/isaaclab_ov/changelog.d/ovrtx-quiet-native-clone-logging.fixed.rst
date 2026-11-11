* Fixed OVRTX camera tasks stalling when their output is read slowly, for example by a benchmark harness. The
  default :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.log_level` is now ``"error"``: lower levels print an
  OVRTX warning for each attribute of every cloned environment (over a million lines at 8192 environments).
