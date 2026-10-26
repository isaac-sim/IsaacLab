* Deprecated ``CameraCfg.isp_cfg`` in favor of ``isaaclab_ppisp.PpispProcessorCfg`` in
  ``CameraCfg.tensor_processors``. The field forwards its value to that processor with a ``FutureWarning``
  and is then reset to ``None``; the resolved configuration is on that processor's ``isp_cfg``.
