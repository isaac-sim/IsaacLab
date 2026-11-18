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
`newton_optix` (needs the `optix` extra), `isaac_rtx` and `ovrtx`.

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

The retina turns relative to the physical camera's optical axis:

```python
# (tilt, pan) [deg] from the physical camera's optical axis, shape (num_envs, 2).
# Positive tilt looks up and positive pan looks left.
camera.set_gaze_angles(angles)
```

Newton and OVRTX turn the rendered camera pose in each pose update, so the
camera prim keeps its mount pose. Isaac RTX renders from the prims, so it
writes the turned pose into them. On every renderer the gaze stays relative to
the mount when `set_world_poses` moves it.

Set `mask_to_sensor_fov=True` to zero the samples that fall outside the
physical camera's image at the current gaze.

## Active gaze

`isaaclab_fovi.mdp` has the manager terms for a policy that moves the gaze:

- `GazeActionCfg`: two actions that move the gaze by up to `step_size` field
  radii (half the retina's field of view) per step, within `gaze_limit` field
  radii of the optical axis. Make it the last action term. With the default
  `mount="fixed"` the teacher aims from the orientation the camera has when the
  action is created, and turning the mount raises; `mount="tracked"` lets the
  camera ride a moving body,
  such as a turning head, and the teacher follows each camera's current mount
  (Newton or OVRTX).
- `gaze_state`: the gaze, in field radii, and its velocity, in field radii per
  second.
- `gaze_teacher_labels`: a teacher's gaze action toward a point fixed to an
  asset, and whether that point is a valid target.
- `motor_action_l2`: the stock action penalty without the gaze.

`isaaclab_fovi.rsl_rl.GazeSupervisedPPOCfg` trains PPO with an added loss that
pulls the means of the gaze actions toward the teacher's. The labels come from
an `oracle` observation group that the actor and critic never see.

### Tasks

Each task is its stock camera counterpart with the camera foveated at the same
mount and optics, two gaze actions, and gaze supervision. The vision model and
the PPO settings are the stock ones.

| Task | Retina | Teacher looks at |
| --- | --- | --- |
| `Isaac-Cartpole-Camera-Fovi` (`presets=depth` for depth) | 32 × 32 | the pole's tip |
| `Isaac-Lift-KukaAllegro-Camera-Fovi` | the camera preset's resolution | the object |
| `Isaac-Reorient-KukaAllegro-Camera-Fovi` | the camera preset's resolution | the object |

The Kuka-Allegro tasks take the stock base-camera presets, such as
`presets=depth128`. Set the retina's resolution with, for example,
`env.scene.base_camera.foveated_res=96`.

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera-Fovi
```

The tasks are registered through the `isaaclab.tasks` entry point of this
package, so they are available from a source checkout.

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
