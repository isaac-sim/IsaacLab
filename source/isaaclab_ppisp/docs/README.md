# Isaac Lab PPISP

This extension provides a renderer-backend-agnostic PPISP (Physically Plausible
Image Signal Processing) pipeline for Isaac Lab camera outputs.

PPISP consumes HDR scene-linear color from renderer backends and writes LDR
`rgb` / `rgba` camera output after each render tick when `CameraCfg.isp_cfg` is
set.
