* **Breaking:** Added ``DictTensorProcessor`` for named buffers and renamed the modifiers API to
  ``tensor_processors``, ``TensorProcessor``, and ``TensorProcessorCfg``. Replace observation
  ``modifiers`` with ``tensor_processors``; use ``DictTensorProcessorAdapterCfg`` for dictionary operations.
* Replaced renderer-owned PPISP with camera and observation processors consuming ``rgb_hdr``.
  Set ``CameraCfg.exposure=CameraExposureCfg()`` explicitly for neutral RTX HDR before PPISP.
