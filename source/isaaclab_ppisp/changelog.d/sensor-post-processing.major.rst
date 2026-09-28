Added
^^^^^

* Added ``PpispProcessorCfg`` for ordered visual observation processing,
  including USD configuration discovery and controller buffer initialization. Declared
  ``rgb_radiance`` as its input so either the renderer or an earlier processor could supply
  scene-linear RGB before exposure and camera response.
* Added ``PpispDiscoveryMode`` in ``isaaclab_ppisp`` with the existing ``AUTO_CAMERA`` and
  ``AUTO_ANY`` discovery modes.

Changed
^^^^^^^

* **Breaking:** Moved PPISP execution and controller state from renderer instances to
  processing chains owned by ``mdp.processed_image`` observation terms and removed
  ``CameraCfg.isp_cfg``. Remove that camera argument and pass
  ``processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` in the ``params`` of an
  ``ObservationTermCfg(func=mdp.processed_image, ...)``. Read processed images from the
  environment's observations instead of ``camera.data.output``. Replace
  ``isaaclab.sensors.camera.CameraISPMode`` imports with ``isaaclab_ppisp.PpispDiscoveryMode``.
