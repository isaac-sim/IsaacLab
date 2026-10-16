* **Breaking:** Added ``DictTensorProcessor`` for named buffers and renamed the modifiers API to
  ``tensor_processors``, ``TensorProcessor``, and ``TensorProcessorCfg``. Replace observation
  ``modifiers`` with ``tensor_processors``; use ``DictTensorProcessorAdapterCfg`` for dictionary operations.
* Replaced renderer-owned PPISP with camera and observation processors consuming ``rgb_hdr``.
  Set ``CameraCfg.exposure=CameraExposureCfg()`` explicitly for neutral RTX HDR before PPISP.
* Colocated dictionary and single-tensor processors and removed preparation wrappers and role mappings.
  Dictionary processors now own their output tensors and expose a direct dictionary-to-dictionary call.
