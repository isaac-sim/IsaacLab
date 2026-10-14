* Retired execution through ``CameraCfg.isp_cfg``. Configure a ``PpispStepCfg`` in
  ``CameraCfg.pipeline`` or an observation pipe instead. Resolve USD-authored PPISP parameters
  explicitly before configuring the step. Non-None legacy ``isp_cfg`` values raise a migration error.
