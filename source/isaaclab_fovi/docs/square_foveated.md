# Square-foveated rendering

Both RTX and Newton foveated cameras accept this configuration:

```python
FoveatedCameraCfg(
    prim_path="/World/Camera",
    foveated_style="warped_cartesian_as_grid",
    radius_norm=float("inf"),
    fov_type="square",
    foveated_res=64,
)
```

Fovi owns the concentric-square mapping and its analytic inverse. Direction
textures, exit textures, virtual gaze, and software retinalization use that
geometry. The public image is upright `(batch, height, width, channels)`;
RGB, depth, and segmentation retain their existing dtypes and buffer lifetimes.
There are no cells masked by the sensor footprint. Camera visibility and
out-of-frame software sampling can still invalidate pixels.

The mode works with the existing planar and spherical chart conventions and
pinhole/equidistant ray projections. It retains a two-dimensional native
topology; the square footprint refers to the visual chart used by the projection.
The diagonal derivative seams remain relevant when assessing temporal filtering.

Configure policy image preprocessing with the same field geometry, radius norm,
and sampling resolution as the camera. Checkpoints must use the sampling
configuration they were trained with.

A bounded renderer example is:

```bash
uv run --no-sync python source/isaaclab_fovi/examples/render_foveated.py \
  --backend rtx --style warped_cartesian_as_grid --radius-norm inf --fov-type square \
  --resolution 64 --steps 8 --visualizer none --device cuda:0 --output /tmp/square-foveated.pt
```

Install the pinned Fovi dependency through the repository's `fovi` extra.
The implementation uses Fovi's upright-grid and analytic native-mapping APIs.
