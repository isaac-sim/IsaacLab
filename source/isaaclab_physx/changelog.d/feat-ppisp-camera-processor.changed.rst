* Removed PPISP from the Isaac RTX renderer; cameras apply it through ``CameraCfg.tensor_processors``.
* Changed Gaussian splats to be tonemapped with the rest of the frame when the camera uses neutral exposure
  (``CameraCfg.exposure``), which PPISP requires, instead of when it sets an ISP.
