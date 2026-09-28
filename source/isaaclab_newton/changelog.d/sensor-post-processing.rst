Changed
^^^^^^^

* Removed PPISP execution from Newton Warp while preserving direct HDR output bindings for
  ``mdp.processed_image`` observation terms. Added ``rgb_radiance`` for processing before exposure
  and camera response. PPISP configuration moved from ``CameraCfg.isp_cfg`` to
  ``PpispProcessorCfg(isp_cfg=existing_cfg)`` in the observation term's ``processors`` list.
