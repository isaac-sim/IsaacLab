# Newton MPM tuning demos

These standalone examples exercise Newton implicit MPM through Isaac Lab. They
are interactive simulation demos; video capture and presentation overlays are
intentionally kept outside this directory.

## Material parameters

`material_parameters.py` compares up to three specimens while changing one
quantity at a time. All variants in a preset share the same geometry, initial
state, particle placement, and fixed material values.

```bash
uv run python scripts/demos/mpm/tuning/material_parameters.py \
  --preset young_modulus --visualizer kit
```

Available presets:

| Preset | Quantity |
|---|---|
| `young_modulus` | Elastic stiffness |
| `poisson_ratio` | Compressibility |
| `friction` | Granular friction |
| `tensile_yield_ratio` | Tensile-to-compressive yield ratio |
| `yield_pressure` | Pressure yielding |
| `hardening` | Plastic hardening |
| `dilatancy` | Shear dilatancy |
| `yield_stress` | Cohesive von Mises yield stress |
| `viscosity` | Plastic viscosity |
| `particle_jitter` | Initial lattice packing |

Pass `--variant_index 0`, `1`, or `2` to run one material variant. The
`particle_jitter` preset has only variants `0` and `1`.

## Rigid-body equivalence

`rigid_body_equivalence.py` places matched spheres, cubes, and capsules on two
inclines. MJWarp rigid bodies run on the left and near-rigid MPM discretizations
run on the right.

```bash
uv run python scripts/demos/mpm/tuning/rigid_body_equivalence.py \
  --visualizer kit
```

## G1 particle coupling

`g1_coupling.py` deploys the published `Isaac-Velocity-Flat-G1` policy across
successive sand, snow, and clay strips. The robot starts on the rigid runway
before reaching the particles. `one_way` lets the robot move particles without
receiving their reaction forces; `two_way` enables feedback.

```bash
uv run --extra rsl-rl python scripts/demos/mpm/tuning/g1_coupling.py \
  --coupling two_way --visualizer kit
```

The default uses the complete G1 collision asset, 40 mm MPM voxels, two
particles per voxel axis, and 1.2 m by 1.5 m by 0.16 m material strips. Use
`--proxy_bodies feet` or `--proxy_bodies lower_legs` only for focused coupling
experiments.

## Surface reconstruction

`surface_reconstruction.py` drops a water blob into a shallow pool and exposes
four reusable reconstruction presets. Surface meshes are supported by the
Newton GL and Newton RTX visualizers; Kit can display the particle baseline.

```bash
uv run python scripts/demos/mpm/tuning/surface_reconstruction.py \
  --surface_preset balanced --visualizer newton_gl

uv run --extra ovrtx python scripts/demos/mpm/tuning/surface_reconstruction.py \
  --surface_preset heavy_smoothing --visualizer newton_rtx
```

Available surface presets are `balanced`, `coarse_grid`, `isotropic`, and
`heavy_smoothing`. Individual reconstruction values can also be overridden with
the `--surface_*` arguments shown by `--help`.
