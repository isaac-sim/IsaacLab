# Isaac Lab PPISP

This extension provides a renderer-independent PPISP (Physically Plausible
Image Signal Processing) pipeline for Isaac Lab camera outputs.

PPISP converts a camera's scene-linear `rgb_radiance` output to LDR `rgb` /
`rgba`. Add `PpispModifierCfg` to `CameraCfg.modifiers` to process each captured
image once, or to an observation term's modifiers:

```python
CameraCfg(data_types=["rgb"], modifiers={"rgb_radiance": [PpispModifierCfg()]}, ...)
```
