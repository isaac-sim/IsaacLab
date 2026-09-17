# Newton MPM tuning demos

These standalone examples provide reproducible, presentation-ready experiments
for Newton implicit MPM. Each script owns a complete scene and changes one
controlled dimension. The maintained methodology, parameter interpretation,
and video publication slots are in the [MPM guide](../../../../docs/source/concepts/using_mpm.rst).

This is the right location for runnable demonstrations. Reusable simulation
configuration belongs in `isaaclab_newton`; task-specific learning code belongs
in a registered environment. Video capture tooling, encoded media, and slide
overlays intentionally remain outside this directory.

## Quick start

Kit is the default for the material, rigid-equivalence, and G1 demos. Surface
meshes require Newton GL or Newton RTX.

```bash
# Controlled material comparison
uv run python scripts/demos/mpm/tuning/material_parameters.py \
  --preset young_modulus --visualizer kit

# Matched MJWarp rigid and nearly rigid MPM primitives
uv run python scripts/demos/mpm/tuning/rigid_body_equivalence.py \
  --visualizer kit

# Published G1 policy with particle reaction forces
uv run --extra rsl-rl python scripts/demos/mpm/tuning/g1_coupling.py \
  --coupling two_way --visualizer kit

# Water surface reconstruction
uv run python scripts/demos/mpm/tuning/surface_reconstruction.py \
  --surface_preset balanced --visualizer newton_gl
```

Run any script with `--help` for its complete CLI.

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

## G1 particle coupling

`g1_coupling.py` deploys the published `Isaac-Velocity-Flat-G1` policy across
successive sand, snow, and clay strips. The robot starts on a rigid runway before
reaching the particles. `one_way` lets the robot move particles without
receiving their reaction forces; `two_way` enables feedback.

The default uses the complete G1 collision asset, 40 mm MPM voxels, two
particles per voxel axis, and 1.2 m by 1.5 m by 0.16 m material strips. Use
`--proxy_bodies feet` or `--proxy_bodies lower_legs` only for focused coupling
experiments. Use `--checkpoint` to make a local checkpoint explicit instead of
downloading the published policy.

## Surface reconstruction

`surface_reconstruction.py` drops a water blob into a shallow pool and exposes
four controlled reconstruction presets. The MPM state remains identical across
presets. Surface meshes are supported by Newton GL and Newton RTX; Kit can show
only the `--fluid_render_mode particles` baseline.

```bash
uv run --extra ovrtx python scripts/demos/mpm/tuning/surface_reconstruction.py \
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
