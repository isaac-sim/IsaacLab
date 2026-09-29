# Newton MPM tutorials

These standalone tutorials provide reproducible experiments
for Newton implicit MPM. Each script owns a complete scene and changes one
controlled dimension. The maintained methodology, parameter interpretation,
and example videos are documented in the MPM solver guide proposed in PR #7897.

Run these scripts from a source checkout. Reusable
simulation configuration belongs in `isaaclab_newton`; task-specific learning
code belongs in a registered environment. Video capture tooling, encoded media,
and slide overlays remain outside this directory.

## Quick start

Kit is the default for the material and rigid-equivalence tutorials. The
surface tutorial defaults to Newton GL; Newton RTX requires the `ovrtx` extra.

```bash
# Controlled material comparison
uv run python scripts/tutorials/08_mpm/material_parameters.py \
  --preset young_modulus --visualizer kit

# Matched MJWarp rigid and nearly rigid MPM primitives
uv run python scripts/tutorials/08_mpm/rigid_body_equivalence.py \
  --visualizer kit

# Water surface reconstruction
uv run python scripts/tutorials/08_mpm/surface_reconstruction.py \
  --surface_preset balanced --visualizer newton_gl
```

Run any script with `--help` for its complete CLI. For the packaged G1
comparison, run `uv run isaaclab example mpm-g1-coupling --help`.

## Material parameters

`material_parameters.py` compares up to three specimens while changing one
quantity. All variants share the same geometry, initial state, seeded particle
placement, and fixed material values.

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

Use `--variant_index 0`, `1`, or `2` to run one centered material variant. The
`particle_jitter` preset has two variants. Constitutive presets use identical
deterministic jitter by default; only `particle_jitter` intentionally compares
aligned and irregular packing.

## Rigid-body equivalence

`rigid_body_equivalence.py` places matched spheres, cubes, and capsules on two
inclines. MJWarp rigid bodies run on the left and high-stiffness MPM
discretizations run on the right. The default Kit view preserves each object's
authored comparison color. The example uses public solver and scene APIs only;
it does not apply a demo-specific post-step particle correction.

## Surface reconstruction

`surface_reconstruction.py` drops a water blob into a shallow pool and exposes
four controlled reconstruction presets. The MPM state remains identical across
presets. Surface meshes are supported by Newton GL and Newton RTX; Kit can show
only the `--fluid_render_mode particles` baseline.

```bash
uv run --extra ovrtx python scripts/tutorials/08_mpm/surface_reconstruction.py \
  --surface_preset heavy_smoothing --visualizer newton_rtx
```

Available presets are `balanced`, `coarse_grid`, `isotropic`, and
`heavy_smoothing`. Override individual reconstruction values with the
`--surface_*` arguments.

## Experiment hygiene

- Keep the seed, geometry, camera, solver settings, and initial particles fixed.
- Tune voxel size and timestep before interpreting material parameters.
- Let the simulation reach its settled state; do not compare only impact frames.
- Record the resolved CLI, particle count, hardware, and code revision with a run.
- Keep generated videos and run artifacts out of Git; publish only selected media
  through the documentation asset pipeline.
