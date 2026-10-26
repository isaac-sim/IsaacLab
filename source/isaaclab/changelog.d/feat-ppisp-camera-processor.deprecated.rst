* Deprecated :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` in favor of adding
  ``isaaclab_ppisp.PpispProcessorCfg(isp_cfg=...)`` to :attr:`~isaaclab.sensors.camera.CameraCfg.tensor_processors`.
  Camera construction emits a ``FutureWarning`` and forwards the value to a PPISP processor that produces the
  requested ``rgb`` and ``rgba`` outputs.
