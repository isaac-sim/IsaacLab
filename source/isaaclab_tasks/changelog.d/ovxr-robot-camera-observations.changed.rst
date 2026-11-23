* Changed Kit ``--xr`` head-panel uploads for the fixed-base G1 task to a 20 Hz cap.
* Changed the fixed-base G1 policy observation selection and XR camera feeds to presets. Programmatic
  callers should use ``resolve_task_config`` (or ``resolve_presets(cfg)`` after direct construction)
  before reading or modifying them. The resolved default keeps the state-only schema. Camera-panel
  upload caps do not change the camera sensor's sampling period.
