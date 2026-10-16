* **Breaking:** Replaced renderer-owned PPISP with dictionary processors.
  Replace ``CameraCfg.isp_cfg`` with ``PpispProcessorCfg`` and configure
  ``CameraCfg.exposure=CameraExposureCfg()`` for neutral RTX ``rgb_hdr`` input.
