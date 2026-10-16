# Isaac Lab PPISP

This extension provides PPISP (Physically Plausible Image Signal Processing) as
a `DictTensorProcessor` used by cameras and observations. `DictTensorProcessor` and
`TensorProcessor` live together in `isaaclab.utils.tensor_processors`; their call signatures are
dictionary-to-dictionary and tensor-to-tensor respectively.

`PpispProcessorCfg` reads scene-linear `rgb_hdr` from a float32 `ProxyArray` with
shape `(N, H, W, 3)` and writes uint8 `rgb`. For camera processing, configure
`CameraCfg.pipeline=[PpispProcessorCfg(params={"cfg": ppisp_cfg})]`. The camera requests
required HDR internally; only the requested public outputs are published. Set
`CameraCfg.exposure=CameraExposureCfg()` (from `isaaclab.sensors.camera`) before initialization
for neutral RTX HDR input. Output selection does not alter exposure. The same setting is required
when PPISP runs only as an observation tensor processor through `DictTensorProcessorAdapterCfg`.
Set PPISP exposure in stops through `PpispCfg.inputs["exposureOffset"]`.

Resolve authored settings explicitly before constructing the pipeline:

```python
from isaaclab_ppisp import PpispProcessorCfg, ppisp_cfg_from_usd_camera

ppisp_cfg = ppisp_cfg_from_usd_camera(source_camera_prim)
step = PpispProcessorCfg(params={"cfg": ppisp_cfg})
```

`PpispProcessor` can also run outside a camera. Construct it with its configuration and an
actual dictionary of `ProxyArray` tensors; construction validates shape and dtype and allocates
persistent output once on the same device. Input/output names can be changed independently of PPISP math. For RGBA,
use `outputs=["rgba"]` and `params={"cfg": ppisp_cfg, "channels": 4}`.
PPISP consumes the preceding operation's radiance, so an earlier exposure or scale
operation changes its result. The producer must supply scene-linear color before
camera exposure and response. Array storage cannot establish that semantic contract;
float32 display-encoded RGB must not be supplied as radiance.

Each operation owns its controller scratch and output buffers. Optional
`params["controller_weights"]=PpispControllerWeights(device_weights)` can supply
shared, read-only device weights; configuration copies preserve this resource's identity. The
caller owns upload synchronization, and mutable controller workspace stays private.
The controller requires CUDA; static PPISP also runs on CPU. `PpispPipeline` remains
available for direct numerical use with explicitly supplied Warp buffers.

For direct use, construct `PpispProcessor(cfg, {"rgb_hdr": hdr_tensor})`, then call
`processor({"rgb_hdr": hdr_tensor})` to obtain `{"rgb": color_tensor}`. It owns its persistent
output and numerical workspace; call `processor.close()` when finished. No preparation factory
or prepared-operation wrapper is needed.
