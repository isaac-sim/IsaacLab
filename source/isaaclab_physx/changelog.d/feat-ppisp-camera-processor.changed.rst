* Removed PPISP composition and its internal HDR buffer from the Isaac RTX renderer. Cameras apply PPISP
  through ``CameraCfg.tensor_processors`` and request ``rgb_hdr`` from the renderer when a processor reads it.
