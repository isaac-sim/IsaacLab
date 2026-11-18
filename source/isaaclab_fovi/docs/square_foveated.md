# Square-foveated rendering

By default the foveated grid covers a circular field: cells outside the circle
are masked. A square field uses every cell. Both the RTX and Newton foveated
cameras accept this configuration:

```python
FoveatedCameraCfg(
    prim_path="/World/Camera",
    foveated_style="warped_cartesian_as_grid",
    radius_norm=float("inf"),
    fov_type="square",
    foveated_res=64,
)
```

`fovi` owns the concentric-square mapping and its analytic inverse. The lens
lookup textures and the gaze use that geometry. The published image
is upright, `(num_envs, height, width, channels)`, and RGB, depth and
segmentation keep their usual dtypes.

No cell is masked by the sensor footprint. With `mask_to_sensor_fov=True`,
samples outside the physical camera's image are still zeroed.

The camera samples spherical field coordinates only; the lens lookup rejects
planar coordinates. The grid keeps its two-dimensional topology; "square"
refers to the region of the visual field the grid covers.
The mapping is not smooth across the diagonals of the square, which matters
when a temporal filter such as DLSS is applied to the rendered grid.

A policy that consumes the image has to be configured with the same field
geometry, `radius_norm` and resolution as the camera.
