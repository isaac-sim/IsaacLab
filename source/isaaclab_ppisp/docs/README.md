# Isaac Lab PPISP

This extension provides a renderer-backend-agnostic PPISP (Physically Plausible
Image Signal Processing) pipeline for Isaac Lab camera outputs.

PPISP consumes `rgb_radiance`: scene-linear RGB before exposure and camera response,
in renderer-relative intensity units. Isaac RTX, OVRTX, and Newton Warp supply this
signal. PPISP writes LDR `rgb` / `rgba` through an observation term's ordered
processing chain. Each term owns its processor state, including controller weights
and scratch buffers.

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

**Breaking change:** `CameraCfg.isp_cfg` was removed. Remove the `isp_cfg`
argument from your camera configuration and pass
`processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]` in the `params` of an
`ObservationTermCfg(func=mdp.processed_image, ...)`, as shown above. Read the
processed image from the environment's observations instead of
`camera.data.output`, which now contains only raw renderer outputs. Set an
observation group's `concatenate_terms=False` to access the image by term name.

`isaaclab.sensors.camera.CameraISPMode` was also removed. Replace its imports
with `from isaaclab_ppisp import PpispDiscoveryMode`; `AUTO_CAMERA` and `AUTO_ANY`
retain their discovery behavior.

`PpispProcessorCfg()` discovers attributes on the camera itself. Pass
`isp_cfg=PpispDiscoveryMode.AUTO_ANY` to allow stage-wide discovery, or
`isp_cfg=PpispCfg(camera_prim_path="/World/ReferenceCamera")` to import a
particular camera's attributes. Discovery that finds no matching attributes
disables the processor. Explicit values and exported controller weights remain
supported through `PpispCfg`.

The term prepares the chain after scene spawning, before the first simulation
reset, and resolves PPISP's `rgb_radiance` input. An earlier processor can supply
that signal; otherwise the term requests it from the camera and the renderer
prepares the required exposure setup. Radiance and RGBA intermediate buffers
are allocated even when absent from camera
`data_types`; RGB aliases RGBA. Repeated observation reads of the same camera
frame reuse the processed result. The observation manager forwards partial
resets and closes processor state with the environment. The PPISP controller
computes parameters from the current image and has no temporal state to reset.

Plain `rgb_hdr` retains the existing camera settings. Requesting `rgb_radiance`
from Isaac RTX or OVRTX applies exposure overrides to the entire source camera,
as the previous PPISP integration did. Its other outputs, including `rgb_hdr`,
also reflect those settings. When both raw names are requested they alias the
same active HDR source. Use separate cameras for separate exposure settings.

`PpispPipeline` remains available to callers that apply PPISP kernels directly.
Calling `initialize(hdr)` preallocates its controller buffers; `apply(hdr,
rgba)` also supports the existing lazy allocation behavior. Call `close()` to
release cached controller buffers.
