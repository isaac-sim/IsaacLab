# Isaac Lab PPISP

This extension provides PPISP (Physically Plausible Image Signal Processing) as
an operation in the shared render-buffer pipeline used by cameras and observations.

`PpispStepCfg` reads float32 NHWC scene-linear `rgb_radiance` and writes uint8
`rgb` with `camera_response` encoding. For camera processing, configure
`CameraCfg.pipeline=[PpispStepCfg(params={"cfg": ppisp_cfg})]`. The camera requests
required radiance internally; only the requested public outputs are published.

Resolve authored settings explicitly before constructing the pipeline:

```python
from isaaclab_ppisp import PpispStepCfg, ppisp_cfg_from_usd_camera

ppisp_cfg = ppisp_cfg_from_usd_camera(source_camera_prim)
step = PpispStepCfg(params={"cfg": ppisp_cfg})
```

The same step can run in `isaaclab.utils.images.RenderBufferPipe` outside a camera.
Its input/output bindings can be changed independently of PPISP math. For RGBA,
use `outputs={"color": "rgba"}` and `params={"cfg": ppisp_cfg, "channels": 4}`.
PPISP consumes the preceding operation's radiance, so an earlier exposure or scale
operation changes its result. Display-encoded RGB is rejected as PPISP input.

Each operation owns its controller scratch and output buffers. Optional
`params["controller_weights"]=PpispControllerWeights(device_weights)` can supply
shared, read-only device weights; configuration copies preserve this resource's identity. The
caller owns upload synchronization, and mutable controller workspace stays private.
The controller requires CUDA; static PPISP also runs on CPU. `PpispPipeline` remains
available for direct numerical use with explicitly supplied Warp buffers.
