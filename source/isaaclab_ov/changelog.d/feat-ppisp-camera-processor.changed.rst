* Removed PPISP composition and its internal HDR buffer from the OVRTX renderer. Cameras apply PPISP
  through ``CameraCfg.tensor_processors`` and request ``rgb_hdr`` from the renderer when a processor reads it.
