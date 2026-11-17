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
    # The physical camera the retina looks through.
    camera_calibration={
        "model": "pinhole",
        "image_size": (480, 640),  # (height, width)
        "intrinsics": (550.0, 550.0, 319.5, 239.5),  # fx, fy, cx, cy [px]
    },
    foveated_res=64,
)
```

With the default grid style the sensor publishes an upright image of shape
`(num_envs, foveated_res, foveated_res, channels)` in `camera.data.output`.

### Foveation strength

`cmf_a_deg` is the cortical magnification constant in degrees. Smaller values
concentrate more samples at the fixation. The default, 0.591, is an estimate
from human data. A value much larger than the field of view gives nearly
uniform sampling, which is a matched non-foveated baseline with the same
resolution, field of view and gaze:

```bash
env.scene.eye.cmf_a_deg=1000
```

### Field geometry

- `field_geometry="spherical"` (default) places samples by visual angle inside
  the calibrated camera's field of view. It needs `camera_calibration`.
- `field_geometry="planar"` places samples on the image plane of the spawn
  optics. It needs no calibration and has no gaze.

### Gaze

In spherical geometry the fixation can move while the camera prim stays in
place:

```python
# Normalized (row, column) position in the calibrated image, shape (num_envs, 2).
camera.set_calibrated_fixation(fixation)
# Or mount angles (tilt, pan) in radians, shape (num_envs, 2).
camera.set_calibrated_angles(angles)
```

Set `mask_to_sensor_fov=True` to zero the samples that fall outside the
calibrated camera's image at the current gaze.

### Advanced options

- Sample-set styles such as `foveated_style="isotropic"` publish
  `(num_envs, num_samples, 1, channels)`.
- `fov_type` and `radius_norm` select circular or square coverage. See
  [square-foveated rendering](square_foveated.md).
- `pool_enabled=True` renders a denser grid and pools it to the output grid.

## Example

```bash
uv run python source/isaaclab_fovi/examples/render_foveated.py --backend newton --steps 8
```

Add `--backend rtx` for the RTX renderer, `--planar` for planar sampling, or
`--cmf-a-deg 1000` for the non-foveated baseline.
