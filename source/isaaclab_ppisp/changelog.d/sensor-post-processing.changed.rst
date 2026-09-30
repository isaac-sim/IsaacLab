* **Breaking:** Moved PPISP execution and controller state from renderer instances to processing chains.
  Chains are owned by ``mdp.processed_image`` observation terms. To migrate, remove
  ``CameraCfg.isp_cfg`` and pass
  ``processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` in the ``params`` of an
  ``ObservationTermCfg(func=mdp.processed_image, ...)``. Read processed images from the
  environment's observations instead of ``camera.data.output``. Replace
  ``isaaclab.sensors.camera.CameraISPMode`` imports with ``isaaclab_ppisp.PpispDiscoveryMode``.
