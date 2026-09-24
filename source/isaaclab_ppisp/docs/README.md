# Isaac Lab PPISP

This extension provides a renderer-backend-agnostic PPISP (Physically Plausible
Image Signal Processing) pipeline for Isaac Lab camera outputs.

PPISP consumes scene-linear HDR from Isaac RTX, OVRTX, or Newton Warp and writes
LDR `rgb` / `rgba` through the camera's ordered processing chain. Each sensor
owns its processor state, including controller weights and scratch buffers.

```python
from isaaclab.sensors.camera import CameraCfg
from isaaclab_ppisp import PpispCfg, PpispProcessorCfg

camera = CameraCfg(
    prim_path="/World/Camera",
    width=640,
    height=480,
    data_types=["rgb"],
    post_processors=[PpispProcessorCfg(isp_cfg=PpispCfg())],
)
```

`CameraCfg(isp_cfg=PpispCfg(), ...)` remains supported and configures the
equivalent PPISP processor. To compose additional stages, put
`PpispProcessorCfg` first in `post_processors` and append processors consuming
its RGB output. Use one entry point per camera.

`PpispProcessorCfg()` discovers attributes on the camera itself. Pass
`isp_cfg=CameraISPMode.AUTO_ANY` to allow stage-wide discovery, or
`isp_cfg=PpispCfg(camera_prim_path="/World/ReferenceCamera")` to import a
particular camera's attributes. Discovery that finds no matching attributes
disables the processor. Explicit values and exported controller weights retain
the behavior of `CameraCfg.isp_cfg`.

The chain requests HDR and neutral renderer exposure before renderer setup.
HDR and RGBA intermediate buffers are allocated even when absent from
`data_types`; RGB aliases RGBA. Processing reuses persistent Warp arrays and
runs only when the sensor fetches a fresh frame. The PPISP controller computes
parameters from the current image and has no temporal state to reset.

`PpispPipeline` remains available to callers that apply PPISP kernels directly.
Calling `initialize(hdr)` preallocates its controller buffers; `apply(hdr,
rgba)` also supports the existing lazy allocation behavior. Call `close()` to
release cached controller buffers.
