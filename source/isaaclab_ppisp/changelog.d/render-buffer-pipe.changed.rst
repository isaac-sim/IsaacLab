* Separated PPISP numerical execution from camera and renderer discovery. Configured processing with
  ``CameraCfg.pipeline=[PpispStepCfg(params={"cfg": resolved_cfg})]`` instead of ``CameraCfg.isp_cfg``;
  resolved USD settings with ``ppisp_cfg_from_usd_camera`` or ``resolve_and_normalize`` first.
