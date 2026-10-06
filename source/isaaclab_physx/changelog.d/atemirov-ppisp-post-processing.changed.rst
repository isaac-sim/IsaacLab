* Removed PPISP execution from Isaac RTX. Added ``rgb_radiance`` for processing before exposure and
  camera response in observation modifiers. Radiance requests retained PPISP's
  camera-wide exposure setup; requesting both ``rgb_hdr`` and ``rgb_radiance`` shared the active HDR
  source. PPISP configuration moved from ``CameraCfg.isp_cfg`` to
  ``PpispModifierCfg(isp_cfg=existing_cfg)`` in the observation term's ``modifiers`` list.
