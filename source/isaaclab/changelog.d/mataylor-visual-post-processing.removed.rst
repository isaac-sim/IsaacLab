* Removed ``CameraCfg.isp_cfg`` and ``isaaclab.sensors.camera.CameraISPMode``.
  PPISP processing is configured through observation terms.
  Remove the camera's ``isp_cfg`` argument and configure
  ``processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` in the ``params`` of an
  ``ObservationTermCfg(func=mdp.processed_image, ...)``. Read processed images from the
  environment's observations instead of ``camera.data.output``. Replace discovery mode imports
  with ``isaaclab_ppisp.PpispDiscoveryMode``, which retained ``AUTO_CAMERA`` and ``AUTO_ANY``.
