* Deprecated ``CameraCfg.isp_cfg`` in favor of ``isaaclab_ppisp.PpispProcessorCfg`` in
  ``CameraCfg.tensor_processors``. Camera construction moves a set value to that processor, sets neutral camera
  exposure, and issues a ``FutureWarning``.
