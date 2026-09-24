Added
^^^^^

* Added ``mdp.processed_image`` for ordered image-processing chains owned by observation terms,
  with explicit buffer requirements, persistent intermediates, cached frame processing, and
  partial-reset and cleanup callbacks. Added early ``ManagerTermBase.prepare_scene`` preparation
  so terms could request HDR and neutral exposure before renderer setup. Retained
  ``CameraCfg.isp_cfg`` through a compatibility adapter; managed environments can migrate by
  setting it to ``None`` and adding ``PpispProcessorCfg(isp_cfg=existing_cfg)`` to the observation
  term's ``processors`` parameter.
