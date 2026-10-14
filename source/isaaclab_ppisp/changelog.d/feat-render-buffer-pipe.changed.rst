* **Breaking:** Moved PPISP into camera and observation pipelines through ``PpispStepCfg``.
  Resolve USD settings explicitly with ``ppisp_cfg_from_usd_camera`` or ``resolve_and_normalize``
  before configuring the step. Replaced ``apply_rtx_exposure_overrides`` with renderer-managed
  exposure when requesting ``rgb_radiance``.
