# Isaac Lab PPISP

This extension provides PPISP (Physically Plausible Image Signal Processing) as a
camera tensor processor for Isaac Lab.

Add `PpispProcessorCfg` to `CameraCfg.tensor_processors` to convert the camera's
scene-linear `rgb_hdr` output into LDR `rgb` / `rgba` after each capture, with any
renderer that publishes `rgb_hdr`. On Isaac RTX and OVRTX, the processor disables
RTX automatic exposure on its camera prims during camera construction, so the HDR
input is not exposed twice.
