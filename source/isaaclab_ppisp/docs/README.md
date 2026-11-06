# Isaac Lab PPISP

This extension provides PPISP (Physically Plausible Image Signal Processing) as a
camera tensor processor for Isaac Lab.

Add `PpispProcessorCfg` to `CameraCfg.tensor_processors` to convert the camera's
scene-linear `rgb_hdr` output to LDR `rgb` / `rgba` after each capture. With Isaac RTX
and OVRTX, also set `CameraCfg.exposure = ExposureCfg(mode="neutral")`: PPISP applies
its own exposure, so the renderer must not apply one first.
