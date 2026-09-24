Changed
^^^^^^^

* Removed PPISP execution from Isaac RTX. Generic camera input requirements selected HDR and neutral
  exposure for processing in ``mdp.processed_image`` observation terms. Existing ``CameraCfg.isp_cfg``
  configurations remained supported through a compatibility adapter.
