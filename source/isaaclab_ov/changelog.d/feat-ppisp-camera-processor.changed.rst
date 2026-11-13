* Removed PPISP from the OVRTX renderer; cameras apply it through ``CameraCfg.tensor_processors``.
* Changed Gaussian splats to be tonemapped with the rest of the frame when the camera requests ``rgb_hdr``,
  which PPISP reads, instead of when it sets an ISP.
