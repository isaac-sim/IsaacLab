* **Breaking:** Replaced ``CameraCfg.isp_cfg`` with reusable buffer pipelines. Configure camera
  processing with ``CameraCfg.pipeline=[PpispStepCfg(...)]``; use ``mdp.processed_sensor`` or
  ``ProcessingBufferPipeModifierCfg`` for observation processing with the same operations. The shared
  ``isaaclab.utils.processing.ProcessingBufferPipe`` accepts named ``ProxyArray`` buffers for both image and
  nonvisual sensor data.
