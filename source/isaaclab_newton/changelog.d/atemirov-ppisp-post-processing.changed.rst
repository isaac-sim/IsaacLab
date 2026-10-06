* Removed PPISP execution from Newton Warp while preserving direct HDR output bindings for
  observation modifiers. Added ``rgb_radiance`` for processing before exposure
  and camera response. PPISP configuration moved from ``CameraCfg.isp_cfg`` to
  ``PpispModifierCfg(isp_cfg=existing_cfg)`` in the observation term's ``modifiers`` list.
