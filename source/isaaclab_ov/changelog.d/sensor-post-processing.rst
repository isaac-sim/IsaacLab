Changed
^^^^^^^

* Removed PPISP execution from OVRTX while preserving HDR buffer bindings and neutral exposure for
  ``mdp.processed_image`` observation terms. Existing ``CameraCfg.isp_cfg`` configurations remained
  supported through a compatibility adapter. HDR transfers between devices reused persistent storage.
