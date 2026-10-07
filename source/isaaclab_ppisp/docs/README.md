# Isaac Lab PPISP

This extension applies Physically Plausible Image Signal Processing to Isaac Lab camera images. PPISP consumes scene-linear `rgb_radiance` from Isaac RTX, OVRTX, or Newton Warp and produces uint8 RGB and RGBA.

Use `PpispModifierCfg` in an observation term's existing modifier list:

```python
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab_ppisp import PpispCfg, PpispModifierCfg, ppisp_camera_input

ppisp = PpispModifierCfg(sensor_cfg=SceneEntityCfg("camera"), isp_cfg=PpispCfg())
camera_image = ObservationTermCfg(
    func=ppisp_camera_input,
    params={"modifier_cfg": ppisp},
    modifiers=[ppisp],
)
```

The environment resolves USD attributes and requests private radiance before simulation reset. The modifier owns its output buffers and runs once for each published camera capture. Set `output="rgba"` to select RGBA, or use separate observation terms for RGB and RGBA. `normalize=True` divides by 255 and subtracts each image's spatial mean; `permute=True` returns NCHW. Each observation term owns independent state and cleanup. With `input_source="previous"`, supply an observation source and earlier modifier that produce scene-linear radiance instead of using `ppisp_camera_input`.

For use outside observations, create a `PpispModifierCfg`, call `cfg.func.prepare_scene(cfg, env)` before `sim.reset()`, then create `modifier = cfg.func(cfg, (num_envs, height, width, 3), env=env)`. After rendering, call `modifier(env, ppisp_camera_input(env, cfg))`. Call `modifier.close()` when done. The result uses reusable storage; clone it to retain a frame.

`CameraCfg.isp_cfg` and `CameraISPMode` were removed. Use `PpispModifierCfg` and `PpispDiscoveryMode` instead. `AUTO_CAMERA` reads the camera's PPISP attributes; `AUTO_ANY` may find attributes elsewhere on the stage. When discovery finds none, the modifier passes through raw camera RGB/RGBA. Camera `data.output` remains the raw renderer output.

On Isaac RTX and OVRTX, requesting `rgb_radiance` neutralizes exposure for all outputs from that camera prim. Use a separate prim if the authored-exposure image is also needed. `PpispPipeline` remains available for callers that apply PPISP Warp kernels directly.
