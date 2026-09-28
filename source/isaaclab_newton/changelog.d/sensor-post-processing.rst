Changed
^^^^^^^

* Removed PPISP execution from Newton Warp while preserving direct HDR output bindings for
  ``mdp.processed_image`` observation terms. Added ``rgb_radiance`` for processing before exposure
  and camera response. Existing ``CameraCfg.isp_cfg`` configurations remained
  supported through a compatibility adapter.
