# Isaac Lab Fovi

Foveated camera sensors for Isaac Lab. A foveated camera samples the scene densely
at the point of fixation and sparsely in the periphery, instead of on a uniform
pixel grid. The sample layout comes from the [fovi](https://pypi.org/project/fovi/)
library.

## Installation

From the Isaac Lab repository root:

```bash
uv sync --extra fovi
```

The `fovi` extra installs this package, the `fovi` library and OpenEXR, which the
RTX renderers use to write lens lookup textures.

## Usage

Add a `FoveatedCameraCfg` to a scene like any other camera. `FoveatedCamera`
picks its implementation from the renderer: it supports `newton_warp`,
`isaac_rtx` and `ovrtx`.

```python
import isaaclab.sim as sim_utils
from isaaclab_fovi.sensors import FoveatedCameraCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

camera = FoveatedCameraCfg(
    prim_path="{ENV_REGEX_NS}/Eye",
    data_types=["rgb"],
    renderer_cfg=NewtonWarpRendererCfg(),
    spawn=sim_utils.PinholeCameraCfg(),
    foveated_res=64,
)
```

With the default grid style the sensor publishes an upright image of shape
`(num_envs, foveated_res, foveated_res, channels)` in `camera.data.output`.

### The physical camera

The retina looks through a physical camera. It places its samples by visual
angle inside that camera's field of view, and its rays are cast directly at
those angles, so no pinhole image is resampled. The retinal window spans the
physical camera's image, and `mask_to_sensor_fov` masks against its edge. On a
real camera, fovi's software sampler takes the same `camera_model` fields.

By default it is the camera that `spawn` creates:

- `PinholeCameraCfg` optics give an ideal pinhole. Without a vertical
  aperture the image is square.
- An OpenCV model in `spawn.distortion` gives that calibrated camera.
  `OpenCvFisheyeDistortionCfg` with zero coefficients is an ideal equidistant
  fisheye:

  ```python
  spawn=sim_utils.PinholeCameraCfg(
      distortion=sim_utils.OpenCvFisheyeDistortionCfg(
          fx=229.2, fy=229.2, cx=319.5, cy=319.5, image_size=(640, 640)  # 160 degrees across
      )
  )
  ```

To describe any other calibrated camera, or a prim without spawn
configuration, set `camera_model` to
[`fovi.sensing.projection.CameraModel`](https://pypi.org/project/fovi/) fields:

```python
camera_model={
    "model": "pinhole",  # or "fisheye"
    "image_size": (480, 640),  # (height, width)
    "intrinsics": (550.0, 550.0, 319.5, 239.5),  # fx, fy, cx, cy [px]
}
```

### Foveation strength

`cmf_a_deg` is the cortical magnification constant in degrees. Smaller values
concentrate more samples at the fixation. The default, 0.591, is an estimate
from human data. A value much larger than the field of view gives nearly
uniform sampling, which is a matched non-foveated baseline with the same
resolution, field of view and gaze:

```bash
env.scene.eye.cmf_a_deg=1000
```

### Gaze

The retina can turn while the camera prim stays in place:

```python
# (tilt, pan) [deg] from the physical camera's optical axis, shape (num_envs, 2).
# Positive tilt looks up and positive pan looks left.
camera.set_gaze_angles(angles)
```

How a policy drives the gaze, for example as per-step changes normalized to
the retinal window, belongs to the task's action term.

Set `mask_to_sensor_fov=True` to zero the samples that fall outside the
physical camera's image at the current gaze.

### Advanced options

- Sample-set styles such as `foveated_style="isotropic"` publish
  `(num_envs, num_samples, 1, channels)`. Log-polar styles are not available,
  because fovi does not support them in spherical coordinates.
- `fov_type` and `radius_norm` select circular or square coverage. See
  [square-foveated rendering](square_foveated.md).
- `pool_enabled=True` renders a denser grid and pools it to the output grid.

## Example

```bash
uv run python source/isaaclab_fovi/examples/render_foveated.py --backend newton --steps 8
```

Add `--backend rtx` for the RTX renderer, `--lens fisheye` for an ideal
equidistant fisheye, or `--cmf-a-deg 1000` for the non-foveated baseline.
