# Isaac Lab PPISP

This extension provides a renderer-backend-agnostic PPISP (Physically Plausible
Image Signal Processing) pipeline for Isaac Lab camera outputs.

PPISP consumes scene-linear HDR from Isaac RTX, OVRTX, or Newton Warp and writes
LDR `rgb` / `rgba` through an observation term's ordered processing chain. Each
term owns its processor state, including controller weights and scratch buffers.

```python
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab_ppisp import PpispCfg, PpispProcessorCfg

camera_image = ObservationTermCfg(
    func=mdp.processed_image,
    params={
        "sensor_cfg": SceneEntityCfg("camera"),
        "processors": [PpispProcessorCfg(isp_cfg=PpispCfg())],
        "data_type": "rgb",
    },
)
```

Add the term to an observation group in a manager-based environment whose scene
contains a camera named `camera`. Append additional processor configurations to
`processors` to consume PPISP's RGB result. Processor factories and buffer
declarations live in `isaaclab.utils.visual_processing`; no renderer changes
are required to add a stage.

`normalize=False` returns persistent `uint8` output by default. For RGB/RGBA,
`normalize=True` returns reusable `float32` output with the same division by 255
and spatial mean subtraction as `mdp.image`. `permute=True` selects an `NCHW`
view instead of the default `NHWC` layout.

`CameraCfg(isp_cfg=PpispCfg(), ...)` remains supported through a compatibility
adapter for direct camera consumers. To migrate a managed environment, move
the existing value into `PpispProcessorCfg(isp_cfg=existing_cfg)` on the
observation term and leave `CameraCfg.isp_cfg=None`.

`PpispProcessorCfg()` discovers attributes on the camera itself. Pass
`isp_cfg=CameraISPMode.AUTO_ANY` to allow stage-wide discovery, or
`isp_cfg=PpispCfg(camera_prim_path="/World/ReferenceCamera")` to import a
particular camera's attributes. Discovery that finds no matching attributes
disables the processor. Explicit values and exported controller weights retain
the behavior of `CameraCfg.isp_cfg`.

The term prepares the chain after scene spawning, before the first simulation
reset, so HDR and neutral exposure are requested before renderer setup. HDR
and RGBA intermediate buffers are allocated even when absent from camera
`data_types`; RGB aliases RGBA. Repeated observation reads of the same camera
frame reuse the processed result. The observation manager forwards partial
resets and closes processor state with the environment. The PPISP controller
computes parameters from the current image and has no temporal state to reset.

Neutral exposure applies to the source camera. Its raw outputs and all other
observations using that camera also reflect the neutralized renderer settings.

`PpispPipeline` remains available to callers that apply PPISP kernels directly.
Calling `initialize(hdr)` preallocates its controller buffers; `apply(hdr,
rgba)` also supports the existing lazy allocation behavior. Call `close()` to
release cached controller buffers.
