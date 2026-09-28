Added
^^^^^

* Added ``mdp.processed_image`` for ordered image-processing chains owned by observation terms,
  with explicit buffer requirements, persistent intermediates, cached frame processing, and
  partial-reset and cleanup callbacks. Added early ``ManagerTermBase.prepare_scene`` preparation
  so terms could resolve renderer inputs before setup. Added ``rgb_radiance`` for scene-linear RGB
  before exposure and camera response, in renderer-relative intensity units. Preserved existing
  ``rgb_hdr`` settings when radiance was not requested. Prepared all cameras' renderer inputs before
  shared stage export, including public outputs and legacy ISP discovery. Retained
  ``CameraCfg.isp_cfg`` through a compatibility adapter; managed environments can migrate by
  setting it to ``None`` and adding ``PpispProcessorCfg(isp_cfg=existing_cfg)`` to the observation
  term's ``processors`` parameter.
