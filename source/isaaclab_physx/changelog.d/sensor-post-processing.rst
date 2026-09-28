Changed
^^^^^^^

* Removed PPISP execution from Isaac RTX. Added ``rgb_radiance`` for processing before exposure and
  camera response in ``mdp.processed_image`` observation terms. Radiance requests retained PPISP's
  camera-wide exposure setup; requesting both ``rgb_hdr`` and ``rgb_radiance`` shared the active HDR
  source. Existing ``CameraCfg.isp_cfg`` configurations remained supported through a compatibility adapter.
