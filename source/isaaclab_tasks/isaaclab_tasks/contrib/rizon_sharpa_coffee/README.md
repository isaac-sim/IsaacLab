# Rizon–Sharpa coffee scooping

`IsaacContrib-Rizon-Sharpa-Coffee` is a manager-based tool-use environment with a
Rizon4s arm, a 22-joint right Sharpa Wave hand, a shallow steel spoon, an open
coffee bowl and a ceramic cup. Newton MPM simulates the coffee and optional
water; MJWarp simulates the robot. A scripted IK controller performs two scoops
and pours, followed by stirring when water is enabled. Apple Vision Pro input
uses the same action interface.

![An earlier Newton RTX capture of the coffee setup](../../../../../docs/source/_static/tasks/rizon_sharpa_coffee.jpg)

The image illustrates the setup; the current motion enters steeply, advances
about 15 mm while rotating flat **inside** the pile, then lifts level. The two
entries are offset by −4 and +4 mm. The original 50 × 32 × 10 mm spoon and
1.8 mm shell are retained. A delivered dose of 8–10 g per scoop has not been
established with this spoon.

## Task and demo

The Gym registration points to `CoffeeEnv`, a `ManagerBasedRLEnv`, and
`CoffeeEnvCfg`. The task owns the action interface, observations, rewards,
terminations, scene resets and coupled physics. `mdp.py` defines its manager
terms; `dissolution.py` performs physical material conversion independently of
rendering. This is the reusable environment for future learning experiments.

`scripts/demos/rizon_sharpa_coffee.py` is the demonstration runner. It selects
scripted IK or AVP input, configures optional presentation, follows the spoon
with a camera and records video. Both controllers send actions through the same
task interface. Grain rendering, colour transport and surface reconstruction
are optional presentation resources; learning can run with no visualizer and
`grains_per_particle = 0`.

## Requirements and assets

Use the repository's uv-managed environment with Newton and a CUDA-capable
NVIDIA GPU. Install the `ovrtx` extra for standalone RTX rendering, an Isaac Sim
runtime for Kit, and the `teleop` extra for live hand tracking. The commands below
use `--no-sync` to preserve an already prepared simulation environment.

The robot reuses the licensed NVIDIA Fabrics-Sim USD and textures from the
workstation's RJ45 insertion task. They are not distributed in this repository.
On another workstation, set `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` to the complete
asset directory. `robot.py` declares the default cache location and validates
the USD-plus-texture bundle's SHA-256. A cached USD override repairs malformed
visual normals without modifying the source asset. The procedural tableware
is generated locally. Retargeting asset provenance and licenses are in
[assets/NOTICE](assets/NOTICE).

Live headset operation additionally requires IsaacTeleop, CloudXR and a
configured headset connection. Follow the maintained
[CloudXR guide](../../../../../docs/source/how-to/cloudxr_teleoperation.rst).
Physics, rendering and headset transport are separate runtime requirements.

## Run and record

Start the dry Newton GL preview with five visual grains per physical particle:

```bash
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py --keep_open
```

For the wet Kit demonstration and a video, use the coarser profile validated on
this workstation:

```bash
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --mode scripted --water --visualizer kit --headless --video \
    --voxel_size 0.003 --physics_hz 200 --mpm_iterations 100 \
    --mpm_grid_capacity 49152 --grains_per_particle 5 \
    --render_samples 4 --width 1280 --height 720 \
    --output_dir logs/coffee_video
```

`--video` defaults to Newton RTX unless `--visualizer` is explicit. It requires
FFmpeg and writes `coffee.mp4`, phase images, `run.json`, `metrics.json` and
`spoon_poses.jsonl` to the output directory. The MP4 uses H.264, CRF 18 and
`yuv420p`; its 25 fps timebase plays at simulation speed, independently of the
wall time required to render. Failed runs finalize the partial video. Do not
reuse an output directory whose recording needs to be retained.

Kit requires its own USD libraries. When a separately installed USD SDK takes
precedence, run with the Kit USD directory first on `PYTHONPATH` and its `bin`
directory plus the Kit runtime directory first on `LD_LIBRARY_PATH`:

```bash
COFFEE_KIT_ROOT="$(uv run --no-sync python -c 'from importlib.metadata import distribution; print(distribution("isaacsim").locate_file("isaacsim").resolve())')"
COFFEE_KIT_USD="$(uv run --no-sync python -c 'from pathlib import Path; from importlib.metadata import distribution; print(next(distribution("isaacsim").locate_file("isaacsim/extscache").glob("omni.usd.libs-*.cp312")))')"
env OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONPATH="$COFFEE_KIT_USD${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="$COFFEE_KIT_USD/bin:$COFFEE_KIT_ROOT/kit${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --water --visualizer kit --headless --video --output_dir logs/coffee_video
```

The desktop **Reset** control restarts the entire episode and script. With
`--keep_open`, the viewer remains open after completion. `--camera fixed` holds
the initial view; the following camera raises its view into the cup after the
first pour and moves closer for pouring and stirring. `--max_steps 8000` gives
80 simulated seconds at the 100 Hz control rate.

## Physics and collision settings

The environment configuration defaults to a 2 mm grid and 800 Hz physics. The
demo selects 400 Hz for a dry mounted scripted run at 2 mm; water, teleoperation,
contact grasping and other resolutions default to 800 Hz. `--physics_hz` selects
200, 400 or 800 Hz, and `--mpm_substeps` multiplies the material step rate. Rigid
integration remains at 800 Hz for the mounted spoon and 1,600 Hz for contact
grasping through solver-entry substeps.

The MPM profile uses sparse CUDA graph capture, `P1d` strain interpolation,
`pic27` particle-sampled contacts, `backward` collider velocities and native
background-air regularization. Hard particle projection is disabled. The
default iteration cap is 100. `--mpm_iterations` and `--mpm_grid_capacity` permit
explicit overrides; zero grid capacity requests automatic sizing. Grid overflow
must be checked when changing the fill, resolution or motion.

Broad spoon faces use a margin of 1.2 physics voxels; container margins are
0.3 voxels. The spoon's bowl margin tapers over its outer 15% to one-ninth of the
broad-face margin: **0.4 mm at a 3 mm grid**, or approximately 0.267 mm at 2 mm.
Native per-face collider materials implement the taper before graph capture;
the closed shell and handle retain their geometry. `CoffeeEnvCfg.spoon_margin_voxels`
and `--spoon_rim_margin_scale` control these values. Reducing the timestep rate or
margin requires checking loaded-tool containment as well as the complete path.
Particle speed is capped at half a physics voxel per MPM substep, up to 4 m/s.
This keeps coarse-step advection within the native collider query band and
limits fast pressure-driven outliers without projecting particles out of walls.

At 2 mm, the default source contains 456,836 physical particles at 1 mm spacing,
with 205.5762 g total mass and density 450 kg/m³. Coarser grids resample the fill
at half-voxel spacing while retaining density. Grain rendering only decorates
those physical particles. `--grains_per_particle 0` disables presentation for
learning; factors 1, 5, 10 and 20 change visual sampling without increasing
physical contact resolution.

## Water, dissolution and surface reconstruction

`--water` initializes a fuller cup with a 75 mm seed height and a 37 mm seed
radius. The free surface settles below the seed height within the cup's 39 mm
interior radius; headroom remains for pouring and stirring.
`--water_fill_height` overrides the task default within 12–79 mm. For example,
`--water_fill_height 0.060` reduces the initial fill. The procedural cup handle
has upper and lower wall attachments and remains visual decoration.

Soluble coffee accumulates wet contact time and converts after 0.12 s into the
contacted liquid's MPM material.
Conversion retains mass, position, velocity and particle identity, sets the
reference volume from liquid density, and clears granular stress. The physical
conversion runs inside captured MPM stepping even when presentation is absent.
World-local resets restore the original material and wetting history. The
installed Warp build needs distinct wetting and rendering hash-grid descriptor
types to keep rendering-grid updates from interfering with captured wet queries;
the graph-replay regression covers that dependency behavior.

Converted coffee joins the reconstructed liquid surface and stops rendering as
dry grains. Its mass initializes a colour tracer on the new liquid carriers.
Particle motion advects the tracer; symmetric, bounded neighbour exchanges at
25 Hz conserve its mass while mixing it. This is phenomenological optical
mixing, not calibrated molecular diffusion or a solver for arbitrary material
property diffusion.

Surface reconstruction uses Newton's native `ParticleSurface` density field,
anisotropic kernels, spatial field smoothing and mesh smoothing at 0.3 times
the physics voxel size. A task-local **60 ms exponential filter** blends density
at fixed spatial voxel coordinates before marching cubes. It does not blend
vertex indices, which change between meshes. New voxels use the current field;
reset discards history. `--surface_smoothing_time 0` disables temporal filtering.
The setting affects rendering only and introduces a short visual response lag.

Geometry is reconstructed once per physics state and reused by repeated viewer
and capture calls. Kit and RTX use shared mesh vertices with interpolated local
colours and normals; GL uses the viewer's texture atlas. Kit's water material
uses the tracer colour for optical absorption. The render field uses exact
allocation to handle dispersed droplets, independently of the bounded captured
physics grid. No Newton source modification is required.

## Teleoperation and grasp resets

```bash
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --mode avp --water --visualizer newton_rtx \
    --cloudxr_env /path/to/cloudxr.env --keep_open
```

The calibrated OpenXR wrist controls the palm. Its first valid sample anchors
to the measured simulated pose; reset re-clutches. Translation is bounded to
0.25 m/s and rotation to 1.2 rad/s. DexPilot retargets the raw right-hand skeleton
to the 22 named Sharpa joints using pinned geometry and limits. The XR anchor is
applied to wrist control once; it is not reapplied to finger retargeting. Missing
wrist tracking holds the previous arm target, and missing finger tracking holds
the last complete hand command. A desktop rollout does not validate live headset
transport.

The default spoon follows the **measured** palm through a graph-safe kinematic
attachment at every physics tick, with the correct centre-of-mass twist. It
does not apply reaction force to the hand. The 29 actions are absolute palm XYZ
and XYZW orientation in the robot-base frame, followed by 22 absolute finger
angles. IK targets remain within the articulation's soft joint limits.

Author a named grasp candidate with independent initial positions and motor
preload targets:

```bash
uv run --no-sync python scripts/tools/create_coffee_grasp_reset.py \
    --output /tmp/coffee_grasp.json
uv run --no-sync python scripts/tools/create_coffee_grasp_reset.py \
    --input /tmp/coffee_grasp.json --output /tmp/coffee_grasp.json --validate
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --grasp_reset /tmp/coffee_grasp.json
```

The optional validator screens a free 25 g spoon under gravity with the imported
collision meshes and MuJoCo CPU physics. A stationary screen does not certify
grasp acquisition or moving MPM scooping. `--grasp_mode contact` is experimental:
it removes the mount, loads an authored contact candidate and uses two-way MPM
proxy coupling with finite proxy inertia. Moving grasp retention and hand/bowl
clearance remain limitations of that mode; the mounted script is the supported
demonstration.

## Learning and validation

The task provides observation, reward, action, reset and termination managers;
it does not bundle a trained policy or training framework configuration. The
observations include joints, palm and spoon poses, coffee fractions and the
previous action. Success requires the configured coffee fraction in the cup
for a dwell period. Rewards use delivery, retention above the source and spill
fractions. Dose reporting is independent of the success threshold.

Focused tests are in `source/isaaclab_tasks/test/contrib/test_rizon_sharpa_coffee_*.py`.
They cover asset geometry and clearance, native loaded-spoon containment,
batched mounted dynamics and partial reset, contact hold/release, physical
dissolution during graph replay, conservative colour transport, spatial surface
history, GL texture upload and retargeting conventions. Robot integration tests
require the licensed asset bundle; the other contracts use generated fixtures.
Run each file in a separate process with `uv run --no-sync python -m pytest`.

For complete recordings, check finite particle state, unchanged total mass,
converted-particle material and volume, colour conservation, physics and render
grid status, actual delivered grams, spill and the complete decoded MP4.
Loaded-spoon watch regions measure only their enrolled particles and cannot
establish an exhaustive no-spill guarantee. Rendering cost, physics throughput
and combined recording real-time factor are different measurements; use a
controlled GPU workload when comparing performance.
