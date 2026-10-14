* Added ``RenderBufferPipe`` and prepared operations over named image buffers, with explicit
  shape, dtype, device, layout and encoding contracts, persistent destinations, and operation lifecycle hooks.
* Added ``CameraCfg.pipeline``, the ``processed_image`` observation term, and
  ``RenderBufferPipeModifierCfg`` so cameras and observations use the same processing operations.
  Observation dependencies are collected before renderer initialization; private inputs do not need
  to be exposed as public camera outputs.
