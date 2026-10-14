* **Breaking:** Removed renderer-owned PPISP processing. Request ``rgb_radiance`` and apply PPISP
  through the render buffer pipe instead of configuring ``CameraCfg.isp_cfg``.
