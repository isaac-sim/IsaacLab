* **Breaking:** Replaced renderer-owned PPISP with ``DictTensorProcessor`` operations.
  Replace ``CameraCfg.isp_cfg`` with ``PpispStepCfg`` and configure
  ``CameraCfg.exposure=CameraExposureCfg()`` for neutral RTX ``rgb_hdr`` input.
