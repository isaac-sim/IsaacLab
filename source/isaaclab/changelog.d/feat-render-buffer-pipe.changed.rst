* **Breaking:** Replaced ``CameraCfg.isp_cfg`` with reusable image pipelines. Configure camera
  processing with ``CameraCfg.pipeline=[PpispStepCfg(...)]``; use ``mdp.processed_image`` or
  ``RenderBufferPipeModifierCfg`` for observation processing with the same operations.
