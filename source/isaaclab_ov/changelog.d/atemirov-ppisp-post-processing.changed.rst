* Removed PPISP execution from OVRTX while preserving HDR buffer bindings for observation modifiers.
  Added ``rgb_radiance`` with PPISP's existing camera-wide exposure setup;
  requesting both ``rgb_hdr`` and ``rgb_radiance`` shared the active HDR source. PPISP configuration
  moved from ``CameraCfg.isp_cfg`` to ``PpispModifierCfg(isp_cfg=existing_cfg)`` in the observation
  term's ``modifiers`` list.
  HDR transfers between devices reused persistent storage.
