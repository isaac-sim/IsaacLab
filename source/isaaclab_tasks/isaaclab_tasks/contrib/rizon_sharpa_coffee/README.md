# Rizon–Sharpa coffee scooping

A manager-based tool-use environment: a Rizon arm and Sharpa hand carry a coffee scoop
through dark coffee granules, lift a scoop, and tip it into a ceramic cup.
The source is an open ceramic bowl. The scene uses a walnut tabletop, a linen
runner, ivory and teal glaze, polished steel, and a satin sage cup.
Enable `--water` to start with water in the cup, convert wetted soluble coffee
into liquid, diffuse its colour, and stir after the two scoops.

The current scene retains the original shallow 50 × 32 × 10 mm spoon, with
its handle centered 10 mm above the cavity floor and no raised neck. The larger
oval source bowl and coffee pile are retained. The spoon shape is fixed rather
than resized to meet a dose target; its motion uses a steep entry, a short
forward scoop while rotating flat within the coffee, and a level lift. Current checks and their scope are
identified in Runtime validation below. Earlier recordings and measurements,
including the reference image, use previous geometry or motion and do not
validate the current complete sequence.

![Rizon–Sharpa hand holding the previous teaspoon above the coffee setup](../../../../../docs/source/_static/tasks/rizon_sharpa_coffee.jpg)

Reference Newton RTX capture with the shallow teaspoon after the first lift in an earlier two-scoop demo,
using brighter studio lighting, the following camera, a 2 mm physics grid, and
five visible grains per physical particle. The hand holds the shaft farther
back to clear the source bowl.

The default task starts with the spoon attached to the hand, isolating scooping
and transport. An experimental `--grasp_mode contact` instead starts from a
preloaded finger grasp with a free spoon; see the contact-grasp section below.
Learning to pick up the spoon is a separate task. The initial pinch was checked against the
imported collision surfaces: the thumb elastomer and index distal surface oppose
one another at the same handle cross-section, with approximately 0.8 mm
cross-finger clearance and no tool penetration. The index contact is on its
distal shell, not its recessed elastomer pad. The final grip places that contact
at spoon-local X = −93 mm, moving the hand 25 mm farther back along the shaft
while preserving the finger angles. This static check does not establish
retention under gravity; the spoon remains mounted. Scripted operation commands
the robot through IK. Pose-IK drive targets are bounded by the articulation's
soft joint limits, with the existing 29-dimensional action interface unchanged.
Apple Vision Pro operation uses hand tracking and robot retargeting.
The robot resets to a measured coffee-ready posture. The scripted approach targets
the level spoon above the source before moving to the local entry pose.
Task success requires at least 0.5% of the initial coffee particles in the cup
for the configured dwell time, which targets a small scoop rather than emptying
the source bowl.
The restored spoon's delivered dose is measured independently of this threshold.

The Gym registration is `IsaacContrib-Rizon-Sharpa-Coffee`. Physics is fixed to the
task's coupled Newton configuration; there is no physics-preset selector or
bundled trained policy. The observation, action, reward and termination managers
provide the starting point for a later RL training configuration.

## Requirements

The robot is the existing 7-axis Rizon4s with the 22-joint right Sharpa Wave hand
used by the workstation's RJ45 task. Its NVIDIA Fabrics-Sim USD and textures remain
in the local asset cache and are not redistributed here. The resolver verifies the
pinned USD-plus-texture bundle. On another workstation, set
`ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` to that complete, licensed asset directory;
copying only the USD omits required textures. The default location and expected
SHA-256 are declared in `robot.py`. Hand-retargeting asset provenance is recorded
in `assets/NOTICE`.

Use the repository's uv-managed Python environment with Newton and a CUDA-capable
NVIDIA GPU. The `ovrtx` extra supplies the desktop RTX renderer. Live Apple Vision
Pro input also needs the `teleop` extra, its IsaacTeleop retargeters, the CloudXR
runtime and a configured headset connection. Follow the maintained
[CloudXR setup guide](../../../../../docs/source/how-to/cloudxr_teleoperation.rst)
for runtime, network, headset and platform requirements. The desktop renderer,
robot bundle, and headset transport are separate dependencies; a working physics
rollout alone does not verify all three.

## Run

From the repository root, start the desktop demo with the current demonstration
defaults and keep the viewer open afterward:

```bash
uv run --extra ovrtx python scripts/demos/rizon_sharpa_coffee.py \
    --mode scripted --keep_open
```

Live preview defaults to Newton GL with five grains per particle; `--video`
defaults to RTX. An explicit `--visualizer` overrides either choice.
`--visualizer kit` uses Isaac Sim's Kit renderer while retaining Newton physics.
It requires an installed Isaac Sim runtime; the `ovrtx` extra alone supplies
the standalone renderer, not Kit.
The demo defaults use a 2 mm physics grid, `--grains_per_particle 5`, and
`--render_samples 4`, with one MPM substep per 400 Hz physics tick for the dry
mounted script. The source coffee seed reaches 65 mm above the tabletop,
including an above-rim mound; gravity determines its settled height. The solver retains
Newton's native background-air
regularization (`air_drag=1.0`) to stabilize nearly empty grid nodes, with
100 solver iterations, `P1d` strain interpolation, `pic27` particle contact
sampling and `backward` collider velocities from measured pose differences.
Hard particle projection is disabled. Five decorative grains per physical particle improve
the powder's appearance; the 2 mm grid supplies the physical contact resolution.
The dry source contains 456,836 physical particles at 1 mm spacing, producing
2,284,180 decorative coffee grains with the demo's factor of five. Density stays
at 450 kg/m³ and each physical particle carries 0.00045 g: 205.5762 g initially.
Restoring the original spoon leaves the source seed and particle mass unchanged.
Water is off by default. The desktop **Reset** control resets the scene
and restarts the scripted controller; `--keep_open` also allows restarting after
the scripted sequence finishes.

Scripted operation performs two short scoops in one episode with mirrored local
draw directions and zero lateral bank. Entry lane offsets are −4 and +4 mm in X,
blending toward the shared lift point as the spoon rises.
Coffee remains in the bowl and
accumulates in the cup between scoops; Reset restarts the complete sequence.
The default `--max_steps 8000` provides 80 simulated seconds of rollout budget.

`--physics_hz` selects 200, 400 or 800 coupled physics ticks per simulated second;
the demo selects 400 Hz for dry scripted operation with a mounted spoon and a
2 mm grid. Contact mode, Apple Vision Pro input, water and other voxel sizes
default to 800 Hz; `CoffeeEnvCfg` also retains its 800 Hz default. An explicit
`--physics_hz` overrides the demo selection. MPM executes
`physics_hz × mpm_substeps` steps per second, while control remains at 100 Hz.
Mounted rigid integration stays at 800 Hz, and contact mode retains 1,600 Hz
rigid integration through entry-specific
substeps. Lower MPM rates require validating the loaded spoon and complete motion;
slowing the arm does not slow particle motion under gravity.

To investigate 400 Hz with an explicit contact-proxy inertia multiplier:

```bash
uv run python scripts/demos/rizon_sharpa_coffee.py \
    --grasp_mode contact --physics_hz 400 --spoon_proxy_mass_scale 40 \
    --visualizer newton_gl --keep_open
```

This contact-mode configuration remains experimental. The
default proxy multiplier remains 20 at every rate. `--mpm_substeps 2` at 400 Hz
runs 800 material steps per simulated second, while coupling remains at 400 Hz.
The mounted spoon's body pose does not advance between those MPM substeps, so
this configuration is not equivalent to 800 Hz coupling. Use `--physics_hz 800`
for that coupling rate; additional MPM substeps require their own moving-tool
validation.

The default `--camera follow` gently tracks the spoon while keeping the pinch
and destination in frame. It moves about 30% closer during pickup, with smooth
motion based on simulation time. Use `--camera fixed` for a camera that the
script does not move. A broad warm key and subdued fill soften the tabletop
shadows and preserve the steel and ceramic highlights.
With `--water`, the first emptying phase begins a smooth camera rise and a small
orbit toward the cup's +X side. This steeper view persists through the second
scoop and clears the near rim to show the liquid. Both pours use an elevated
close-up about 38 cm from the cup. Stirring moves to about 29 cm and centres
the opening with a 55-degree view from its -X/+Y side, showing the spoon and
exposed liquid while keeping the forearm out of the opening. The stirring
view was selected using the complete scoop path rather than a reset hand pose.
Reset restores the initial view. `run.json` records this as `camera_cup_reveal`.

Use `--grains_per_particle 0` to display only native physical particles and skip
the presentation allocation. The manager-based environment configuration keeps
that zero-grain default for RL; the interactive demo opts into five.

The dry, scripted 2 mm demo reserves 98,304 sparse-grid cells to reduce padded
solver work. This leaves the 456,836 physical particles, voxel resolution,
collision margins, solver iterations, and selected timestep unchanged. Use
`--mpm_grid_capacity 0` to restore the conservative particle-count budget, or
pass a positive cell count explicitly. The RL configuration, live teleoperation,
water, and other voxel sizes retain the conservative automatic budget unless
overridden. Sparse-grid overflow raises an error; increase the budget if changing
the fill or motion causes particles to spread beyond it.

Use Newton GL when inspecting collision geometry and wireframes:

```bash
uv run python scripts/demos/rizon_sharpa_coffee.py \
    --mode scripted --visualizer newton_gl --keep_open
```

The **Visualization Markers** section opens by default and contains **Show
Collision**, **Show Visual**, **Show Edges**, and **Gap + Margin**. Newton's native
**Rendering Options** section contains **Wireframe**, **Sky**, **Shadows**, and
color settings. These GL inspection controls are separate from the RTX material
presentation.

The 13.2-second scoop enters at 60 degrees nose-down, then rotates flat during
a short 15 mm forward draw with zero roll and no commanded lateral sweep.
The cavity reaches 30 mm height at 3.8 seconds, curls through 40 degrees at
5.2 seconds, and reaches zero pitch and 36.5 mm height by 8.2 seconds while
within the pile. It then lifts level through 48.5, 105 and 165 mm at
9.2, 10.4 and 11.7 seconds, reaching 220 mm transfer height at 13.2 seconds.
Position, velocity and acceleration interpolate continuously through the
quintic segments. The second scoop uses the opposite lateral entry; its X lane
converges with the first from 9.2 to 10.4 seconds. Actual submersion, clearance and
retained dose depend on the settled powder and measured robot motion.

The cup center is `(0.60, 0.04, 0.0)` m, near the source bowl to reduce loaded
travel and wrist-limit stalls. Each level transfer takes at least 4.5
seconds, followed by 1.4 seconds of lowering, a 2.4-second tip, a 1.3-second empty
hold and 3 seconds of recovery. The return for the second scoop takes 4 seconds.
Phase transitions also require measured arrival, so actual motion can take longer.
Pouring uses a bounded 90-degree tip with the commanded cavity frame 125 mm above
the tabletop and 14.72 mm toward the far side of the cup, positioning the pouring lip
over its mouth.

For a denser grain presentation, opt into more rendering work while keeping the
2 mm contact grid:

```bash
uv run --extra ovrtx python scripts/demos/rizon_sharpa_coffee.py \
    --mode scripted --visualizer newton_rtx --voxel_size 0.002 \
    --grains_per_particle 20 --render_samples 16
```

More render samples improve image convergence; extra visible grains improve
appearance. Neither changes physical resolution. Coarser 3 mm and 4 mm grids
showed true floor crossings in isolated spoon-containment tests and should not
be used to judge leakage with this thin utensil.

For a two-scoop 1920 × 1080 MP4 with the following camera, RTX materials, and
16 render samples, use `--video`:

```bash
uv run --extra ovrtx --extra video python scripts/demos/rizon_sharpa_coffee.py \
    --mode scripted --visualizer newton_rtx --headless --video --camera follow \
    --voxel_size 0.002 --grains_per_particle 5 \
    --width 1920 --height 1080 --render_samples 16 \
    --output_dir logs/coffee_local_scoop_video
```

To record the water-and-stir scene through Kit, use:

```bash
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --visualizer kit --headless --video --water --deterministic \
    --voxel_size 0.003 --physics_hz 200 --mpm_iterations 50 \
    --mpm_grid_capacity 49152 --grains_per_particle 5 --render_samples 4 \
    --width 1280 --height 720 --max_steps 10000 \
    --output_dir logs/coffee_kit_video
```

Kit receives native grain positions and a reconstructed water mesh with local
vertex colours. Its water MDL material uses those colours to vary optical
absorption, with an index of refraction of 1.333 and no diffuse scattering.
The cup has an opaque sage exterior and ivory interior. The demo enables
translucency and twelve ray bounces for the water, without extra glass ray
splitting. `--render_samples` controls Kit's RTPT samples too.
Camera warmup advances rendering without stepping physics.

On this workstation, Isaac Sim 6.1 uses USD 25.11 while standalone USD Exchange
3.0 supplies USD 26.08. Select Kit's bindings **before** importing task modules
when both are installed. Prefix the command above with process-local paths:

```bash
COFFEE_KIT_ROOT="$(uv run --no-sync python -c 'from importlib.metadata import distribution; print(distribution("isaacsim").locate_file("isaacsim").resolve())')"
COFFEE_KIT_USD="$(uv run --no-sync python -c 'from importlib.metadata import distribution; print(next(distribution("isaacsim").locate_file("isaacsim/extscache").glob("omni.usd.libs-*.cp312")))')"
env PYTHONPATH="$COFFEE_KIT_USD${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="$COFFEE_KIT_USD/bin:$COFFEE_KIT_ROOT/kit${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --visualizer kit --headless --video --water --deterministic \
    --voxel_size 0.003 --physics_hz 200 --mpm_iterations 50 \
    --mpm_grid_capacity 49152 --width 1280 --height 720 \
    --output_dir logs/coffee_kit_video
```

These paths isolate Kit's USD runtime for that process and preserve the installed
Warp/Newton versions. The coarse physics settings have the containment limits
reported in Runtime validation below.

The result is `coffee.mp4` in the output directory. Omit `--output_dir` to create
`logs/coffee_video/YYYY-MM-DD_HH-MM-SS/` automatically. The movie plays at **1×
simulation speed**, independent of how long rendering takes: one frame every four
100 Hz control steps produces 25 fps. Extra phase-transition PNGs do not add movie
frames. Encoding uses H.264, CRF 18, `yuv420p`, and fast-start MP4 metadata. The
movie keeps its initial output dimensions if the desktop window is resized.

Alongside the movie, the launcher saves phase PNGs, `latest.png`, rollout metrics,
`run.json` with the simulation timebase and settings, and `spoon_poses.jsonl`
with a measured spoon pose every five control steps.
`run.json` also records initial coffee mass and physical particle mass in grams;
rollout metrics include measured `carried_g` and `delivered_g`, derived from the
physical-particle classification and total mass. Decorative grain count does not
enter those measurements. The `carried_g` mask covers the cavity and extends
only 4 mm above its rim, so it can exclude a higher heap; it does not measure
the full heaped dose. Dose validation instead uses the increase in measured
glass mass after each emptying phase. The restored spoon's dose, spill and
retention require a complete physical pickup, carry and release check.

Capture files are finalized after the complete two-scoop sequence, even with
`--keep_open`; desktop Reset can then replay the scene without overwriting that
recording. A stopped or failed capture also saves its available metrics.

The launcher uses system FFmpeg when available, otherwise the encoder supplied by
the video extra. Explicit `--video` fails early when neither encoder is available.
Using `--output_dir` without that flag retains the existing capture behavior,
including PNGs and metrics when no encoder is installed.
Shorter runs, such as `--max_steps 1200`, capture only part of the demonstration.
`--headless` is an explicit demo option: it removes the desktop window while still
allowing RTX captures. Use `--visualizer none --headless` for a physics-only rollout.
Use `--help` for the current input and capture options. Physical resolution and
rendering density can be tuned independently, with different compute costs.
Start with one environment when judging the scene or recording a demonstration.

After configuring CloudXR, start right-hand teleoperation with:

```bash
uv run --extra ovrtx --extra teleop python scripts/demos/rizon_sharpa_coffee.py \
    --mode avp --visualizer newton_rtx --cloudxr_env /path/to/avp-cloudxr.env
```

Omit `--cloudxr_env` to use the repository's Apple Vision Pro profile. This path
uses a standalone IsaacTeleop/CloudXR tracking session with desktop RTX rendering;
it does **not** send stereo scene views to the headset. Calibrate the world anchor
with `--anchor_pos X Y Z` before operating in the tabletop workspace. Wrist pose
drives the palm IK target relative to the first valid tracked sample, which is
recentered onto the measured simulated palm. Reset establishes that reference
again; initial connection therefore does not jump to an unrelated room position.
DexPilot retargets the raw hand skeleton to the named
Sharpa finger joints using the asset's calibrated hand basis; applying the world
anchor to those finger inputs a second time would mix coordinate frames. Tracking
loss holds the arm, and incomplete finger tracking retains the last complete hand
command. In the default mounted mode the spoon remains attached while the fingers move.
Contact mode allows the fingers to release it.

The retargeting follows the [NVIDIA IsaacTeleop Sharpa integration](https://github.com/NVIDIA/IsaacTeleop/releases)
and [DexPilot's geometric hand retargeting](https://arxiv.org/abs/1910.03135), with
the workstation's pinned
[IsaacTeleop revision](https://github.com/NVIDIA/IsaacTeleop/tree/c5fe6624cc4dff456485d2e786922c8e41100f83)
and [official Sharpa model](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/3e953f588ba9954cebaa720aaa4cee06a43a068e).
Use the canonical `r_palm_ctrl` frame, the named 22-joint output order, and the
calibration for this exact asset. The native hand mount frame is not an
interchangeable palm frame. The shipped DexPilot profile includes low-pass
filtering (`low_pass_alpha: 0.2`) and hand-size scaling (`scaling_factor: 1.2`),
followed by robot-joint limit clipping. Synthetic tracked-hand tests check bounded
motion and tracking-loss hold behavior; an actual headset session must still
validate operator alignment, tracking quality and latency.

`--grains_per_particle 5`, `10`, or `20` increases **visible grains**, without multiplying
the simulation particles. These decorative grains follow the simulated material;
they cannot create finer contacts, mass, or physical resolution. The physics voxel
size and particle spacing still determine whether material can enter the spoon and
remain inside its curved walls.

`--water` adds water seeded to 60 mm, which settles within the cup. At the
2 mm physics resolution this adds 192,920 native particles and 192.92 g of water.
The liquid surface and coffee tint require `--grains_per_particle 1`, `5`, `10`,
or `20`. The scripted controller adds a three-turn stir after the second pour:
65-degree spoon inclination, +Y heading, 9 mm circular radius, and an 8-second
profile with smooth acceleration and deceleration. It tilts above the rim,
lowers to a cavity height of 31 mm, then raises before returning level.

Coffee converts into liquid after 0.12 seconds of accumulated contact with
water or the already dissolved mixture in the same environment. This lets a
wetting front progress through a clump. `--coffee_wetting_time`
controls this interval. Contact uses a radius of 0.85 physics voxels (2.55 mm
with the 3 mm grid). The physics-owned conversion runs inside the captured MPM
step, including when presentation is disabled. It copies the contacted liquid's
material parameters, clears granular stress and sets reference volume to
coffee mass / water density. It preserves mass, position, velocity, particle
count and original coffee IDs used by delivery observations and rewards.
Reset restores the selected environments' dry material and liquid mask in place.
On the installed 2026-09-30 Warp build, wetting queries use a double-precision
hash grid to isolate the captured descriptor from the float grids rebuilt by
rendering. Native particle state and MPM arithmetic keep their existing precision.
The capture regression includes a rendering-grid rebuild between physics replays.

Converted coffee joins the liquid surface and stops rendering as grains. Its
mass initializes a finite passive colour tracer on the converted liquid carrier.
MPM motion advects the tracer; symmetric mass-weighted neighbour exchanges
diffuse it at 25 Hz over original water and converted coffee. Dry particles are
excluded. Separate input/output buffers preserve neighbour reads, and the
bounded mixing fraction keeps concentrations nonnegative. The default 18/s rate
is a visual mixing parameter, not a measured molecular diffusivity.
Water surface faces sample the local concentration: GL uses a face-colour
texture atlas, standalone RTX uses a colour primvar, and Kit receives expanded
vertices with the corresponding colour stream. The default display normalization
uses 1.5 g of uniformly mixed tracer for 63% of the final brown tint.
Recorded metrics expose physical conversion as `dissolved_coffee_g` and retain
`visual_tracer_g` for colour diagnostics; both use the converted coffee mass.

The liquid mesh uses a render grid at 0.3 times the physics voxel size, bounded
anisotropic kernels, particle position smoothing and Gaussian field smoothing.
These reconstruct a smoother free surface from water and converted coffee;
they do not change the MPM voxel size or particle motion.
The renderer compacts original water and converted coffee and sizes its sparse
surface buffers automatically for the fine grid, smoothing halo and stirring
splashes. This is separate from the bounded physics grid and its CUDA graph;
surface extraction and USD transfer run during rendering and synchronize to
the host. Reconstruction reads the coupled MPM entry's current particle radii,
including the smaller reference volume of converted coffee.

Dry coffee retains its size and granular behaviour. Wet conversion approximates
instant soluble coffee at fixed water density; it does not model ground-coffee
extraction, dissolution chemistry, thermal transfer or concentration-dependent
rheology. Colour transport and rendering remain optional and disabled for
batched RL. Reset clears the tracer and restarts both scoops and the stir.

```bash
uv run python scripts/demos/rizon_sharpa_coffee.py \
    --water --physics_hz 400 --mpm_grid_capacity 131072 \
    --visualizer newton_gl --grains_per_particle 5 --video --keep_open
```

The faster wet preview recording uses the following coarser configuration.
The default solver cap remains 100; reducing it to 50 trades accuracy for speed.
See Runtime validation for the measured water containment error.

```bash
uv run --no-sync python scripts/demos/rizon_sharpa_coffee.py \
    --water --voxel_size 0.003 --physics_hz 200 --mpm_iterations 50 \
    --mpm_grid_capacity 49152 --visualizer newton_gl \
    --grains_per_particle 5 --video --keep_open
```

## Asset selection and dimensions

The public [NVIDIA SimReady asset-pack catalog](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
includes kitchen props in its Furniture & Misc pack. The official
[NVIDIA VoMP asset metadata](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-PhysicalAssets-VoMP/blob/main/metadata.csv)
lists `spoon_small`, `spoon_big`, `serving_bowl`, and `blackandbrassbowl_small`.
The workstation's teapot-fill example also uses the Isaac SimReady `Bowl_G01` and
`Island_A01` models; their public Isaac 6.1 asset URLs were reachable when this
scene was implemented. The SimReady Central web host could not be accessed in
this session, so this is a public-catalog audit, not a claim of an authenticated
Central inventory search.

This task uses procedural assets to make the spoon cavity, cup wall thickness,
and particle clearances explicit, and to keep the scene independent of large asset
pack downloads. The design reference is the restrained glass, wood and ceramic
styling in [KINTO's gallery](https://kinto.co.jp/pages/styling-gallery) and its
[CAST glass photography](https://kinto-usa.com/collections/cast/products/21461).
No product photography or third-party mesh is redistributed by this task.

| Asset | Size / convention |
| --- | --- |
| Coffee bowl | 159.9 × 149.5 mm oval inner mouth, 40 mm height, 4 mm solid bottom |
| Ceramic cup | 78 mm inner diameter, 85 mm height, 4 mm solid bottom |
| Coffee scoop bowl | 50 × 32 mm, 10 mm cavity depth, 1.8 mm vertical shell thickness |
| Coffee scoop handle | 95 mm long, extends along local −X; center is 10 mm above the cavity floor; no neck |
| Spoon frame | Inner bowl bottom is the origin; local +Z exits the cavity |
| Table | Top at world Z = 0; robot base shares that height |

The bowl and cup are closed **wall volumes with open mouths**. The scoop
likewise has an open concave cavity. Their USD collision approximation is exact
triangles; a convex hull would seal these cavities and invalidate the task.
The source bowl scales its centered radial profile by 1.23 along X and 1.15
along Y. Its below-rim capacity is about 493.097 mL, 15.14% above the preceding
428.257 mL bowl. Its inner radial profile also bounds the circular powder seed,
including a tapered above-rim mound. `coffee_cavity_radius(z)` defines that
conservative seed region; the settled bed can spread across the container.
The original spoon and mount frame are restored. This geometry change does not
certify a loaded frictional grasp; current motion and clearance checks use the
mounted hand, and contact mode remains experimental.
Fill samples must retain a particle-radius-plus-contact-margin clearance from the
bottom and walls. The spoon's broad faces and handle use a collision margin of
1.2 voxel widths (2.4 mm on the default 2 mm grid); bowl and cup margins remain
0.3 voxel widths (0.6 mm). Over the outermost 15% of the spoon's elliptical radius,
native MPM face materials taper that margin to one-ninth of its broad-face value
in smoothstep bands. At 3 mm resolution, the broad margin is 3.6 mm and the
rim margin is 0.4 mm (0.267 mm on the default 2 mm grid). This reduces expansion
above the lip while retaining floor protection, without changing the original
spoon mesh or adding per-step kernels.
`--spoon_rim_margin_scale 1` restores the uniform margin for comparison.
The GL shape-margin overlay shows the broad value; the native MPM face margins
are configured separately before graph capture.
The vertical shell thickness is 1.8 mm; curved sections are thinner
along their surface normals, making grid resolution important even with a closed,
correctly wound mesh.
This separates thin-tool contact tuning from container contact tuning.

The original ideal paraboloid cavity holds approximately 6.283 mL before
particle and wall clearance, or 2.827 g at the configured 0.45 g/mL bulk density.
Actual heaping and delivered mass require a physical trial; the shallow spoon
geometry is retained independently of that measured dose.
Actual coffee bulk density varies with grind, roast and
packing; [a study of fine ground Arabica](https://www.scielo.br/j/aabc/a/F44XMrhpdQtMbjCySbg3pBM/?lang=en)
reported loose roasted-coffee densities of about 0.28–0.39 g/mL.
Nominal particle reference volume and configured density also do not establish
the settled packing density; that requires checking the actual material state
and numerical convergence.

Inspected keyframes from
[Cup of Couple's matcha-scooping clip](https://www.pexels.com/video/a-person-scooping-match-powder-using-a-wooden-spoon-8473982/)
show a short immersed sweep, leveling inside the powder, then lifting with the
opening upward. The single camera supports that qualitative
ordering rather than calibrated joint angles or a measured coffee mass.
[Taryn Elliott's coffee clip](https://www.pexels.com/video/close-up-view-of-person-scooping-black-coffee-powder-4786571/)
shows a heaped spoon lifting nearly level, but begins after entry.

Current contacts use Newton's native velocity-level constraints with `pic27`
particle sampling, `P1d` strain interpolation and `backward` collider velocities.
The task no longer calls `project_outside` after each MPM substep. In
[Newton issue 1055](https://github.com/newton-physics/newton/issues/1055#issuecomment-3616509791),
the maintainer explains that position projection can push particles together
without updating their stored deformation and recommends particle-sampled
contacts; projection is mainly a failsafe. The installed `backward` mode uses
previous and current collider poses, addressing differences between instantaneous
rigid velocity and actual displacement. Decorative grains follow native MPM
motion and do not change physical particle masses or positions.

`build_coffee_assets()` returns the `bowl`, `cup`, `spoon`, and `table` USD paths.
The assets and synthesized walnut texture are cached under
`~/.cache/isaaclab/rizon_sharpa_coffee/<source-hash>/` (or `$XDG_CACHE_HOME`).
Generated USD and textures stay in the cache; the generator and liquid MDL source
are versioned. Each change to the generator builds a new cache entry.
All assets use meters, Z up, and an explicit default prim. The spoon root is a
kinematic rigid body by default; contact mode overrides it to dynamic. The bowl
and cup are static.

## Rendering limits

The cup uses opaque ceramic USD Preview Surface materials in every renderer.
Kit's liquid uses a task-local MDL shader built on OmniSurfaceBase, with
refraction and concentration-dependent absorption. GL and standalone Newton
RTX retain the simpler coloured surface presentation with fractional opacity.
RTX provides soft studio lighting, reflections, polished steel, and textured
walnut. A GL viewer may show simpler opacity, color, or material behavior; it is
useful for fast motion inspection but does not certify the final appearance.
Inspect a Kit capture when judging water refraction and optical absorption.

The spoon and container collision shells share their visible mesh geometry.
Decorative cup handles, fabric, stitches, table trim and supports are visual
geometry without particle collision. The studio floor catches particles that
fall beyond the tabletop: its visible slab remains 6 mm thick with its top at
world Z = −0.746 m, while a hidden collider extends 50 mm downward from that same
surface to resist tunneling. The hollow-collision test imports the generated USD
into Newton and checks that downward rays reach solid bottoms and upward rays
leave the cavities. This validates the asset geometry; successful scooping,
transfer, teleoperation, and convincing final images require their own runtime
validation.

## Runtime validation

The 2026-10-02 0.4 mm-rim Kit recording completed both scoops and stirring in
78.37 simulated seconds. All 1,960 frames decoded at 1280 × 720 and 25 fps
(78.40 seconds). Native collider measurements confirmed the 0.4 mm rim and
unchanged 3.6 mm broad-face margin. The native clearance and loaded-spoon
hold, carry and tipped-release regressions passed with the one-ninth taper.
The recording retained the 3 mm grid, 200 Hz MPM and five rendering grains.

Carried cohorts of 1,529 and 1,492 particles reported zero native floor crossings,
pose-rebase crossings and ambiguous samples before tipping. Their 53 and 54
central watch exits remained unwatched thereafter. At 78 seconds, 5.21994 g of
coffee had converted to liquid, with 5.21387 g of converted coffee and no dry
coffee inside the cup-interior mask. Final task classification reported
4.94049 g delivered and 0.03189 g spilled. There were 529 original water particles
(1.785375 g, 0.96%) outside the cup mask, so this is not a spill-free profile.
Native mass stayed unchanged, colour checks passed, and the captured sparse
graph had zero physics and surface grid statuses at the checked steps.
Combined physics, Kit rendering, capture and diagnostics took 952.900 seconds
through the 78-second sample (0.0819 RTF), excluding initialization on the shared
GPU; this is not an isolated throughput comparison. The recording is
`~/Videos/rizon_sharpa_coffee_kit_rim04.mp4`, with measurements and validation under
`logs/coffee_rim04_20261002/video/`.

The preceding 2.4 mm-rim Kit recording completed both scoops and stirring in
78.37 simulated seconds. All 1,960 frames of its 1280 × 720 MP4 decoded at
25 fps (78.40 seconds). The original spoon mesh, 3 mm physics grid, 200 Hz MPM,
800 Hz rigid integration, 50 solver iterations and five rendering grains were
retained. Native contact queries confirmed a 2.4 mm rim margin while the floor
and handle kept their 3.6 mm margin. The loaded-spoon regression also passed
its hold, carry and tipped-release checks with the taper enabled.

Fixed carried cohorts of 1,331 and 1,284 particles reported zero native floor
crossings, pose-rebase crossings and ambiguous samples before tipping. Six and
three particles respectively left the central watch region and were not watched
thereafter; these exits do not establish that those particles left the spoon.
At 78 seconds, 5.11211 g of coffee had converted into liquid, with 5.09541 g of
converted coffee and no granular coffee inside the cup-interior mask. The
stricter task classification reported 4.86456 g delivered and 0.04101 g spilled
at completion. There were 225 original water particles (0.759375 g, 0.41%)
outside the cup containment mask. This remains scoped evidence rather than
exhaustive containment or spill-free operation.

Native mass remained unchanged, the colour tracer agreed with converted mass
within the checked 0.2% tolerance, and the sparse graph stayed captured with
zero physics and surface grid statuses at the checked steps. Combined physics,
Kit rendering, colour transport, capture and diagnostics took 1,504.961 seconds
through the 78-second sample (0.0518 RTF), excluding initialization on the shared
GPU. This is not an isolated throughput comparison. The recording is
`~/Videos/rizon_sharpa_coffee_kit_tapered_rim.mp4`; native margin measurements,
cohort audits, conversion traces and full-decode validation are under
`logs/coffee_rim_margin_20261002/video/`.

The preceding uniform-margin coffee-to-liquid Kit recording completed both scoops and stirring
in 78.37 simulated seconds. All 1,960 frames of the 1280 × 720 MP4 decoded
successfully at 25 fps (78.40 seconds), with no artificial tracer or diagnostic
frames in the movie. It retained the 3 mm physics grid, 200 Hz MPM, 800 Hz rigid
integration, 50 solver iterations, five rendering grains and four Kit samples.
The finer 0.9 mm liquid surface used compacted liquid carriers and automatic
render-buffer sizing. The physical sparse graph remained captured, with zero
physics and surface grid statuses at the checked steps.

At 78 seconds, 4.89037 g of coffee had physically converted into liquid. The
cup-interior mask contained 4.87063 g of converted coffee and no granular coffee;
the task's stricter delivery classification reported 4.67016 g at completion.
Native particle mass stayed unchanged at 0.384476621 kg. The colour tracer stayed
finite and nonnegative, and its sum agreed with converted mass within the checked
0.2% tolerance. Converted grains had zero display radius while dry grains remained
visible. At 78 seconds, 739 original water particles (2.494125 g, 1.35%) lay
outside the cup containment mask. This coarse profile does not establish
watertight contacts or calibrated fluid behaviour.

Combined physics, Kit rendering, colour transport, capture and diagnostics took
575.706 seconds through the 78-second sample (0.1355 RTF), excluding initialization
on the shared GPU. This is a capture measurement, not isolated physics throughput.
A separate captured wetting-only microbenchmark with the same 186,035 particles
measured a median 0.559 ms per update, excluding native MPM and rendering.
The recording is `~/Videos/rizon_sharpa_coffee_kit_dissolving.mp4`, with conversion
traces, decoded phase images and validation under
`logs/coffee_dissolution_20261002/video/`. Focused native tests cover conservation,
wetting through the converted mixture, world isolation, rendering-grid interference
during graph replay, and restoration of selected worlds' dry material on reset.

The current 60-degree entry and 15 mm forward-curl profile passed a complete
contact-disabled CPU rigid check at 400 Hz coupling and 800 Hz rigid integration
in 60.24 simulated seconds. Both cavity poses reached approximately 0.13-degree
tilt at 8.2 seconds while still below 36 mm height. Sampled mesh clearances were
3.956 mm for the tool/bowl and 6.667 mm for the mounted hand/bowl, with no sampled
mounted intrusion. This establishes motion and finite sampled geometry, not
particle retention. The alternative contact hand still intrudes and remains
uncertified. CPU evidence is in
`logs/coffee_ik_cpu_20261001/steep60_original_flat_400.json` and its
`_narrow1.23_1.15_geometry.json` companion.

The complete mounted 400 Hz RTX run lasted 60.25 simulated seconds. Cup-mass
increments after emptying were 2.78730 g at 25.38 s and 2.74275 g at 55.23 s,
including ongoing settling. Their sum is about 60% greater than the preceding
shallow-entry run's 3.45150 g. Final settled cup mass was 5.53680 g. The original
spoon, bowl, 456,836-particle seed and verified native masses remain unchanged;
these improved teaspoon doses still do not meet the earlier 8–10 g target.

Watched source-cleared cohorts of 5,131 and 4,853 particles reported zero native
floor crossings, pose-rebase crossings and ambiguous samples before tipping.
Each cohort had 21 radial watch exits, which remained unwatched thereafter;
top exits and continuous floor flags were zero. Classified final spill was
0.17505 g (389 particles), higher than the shallow-entry run's 0.06210 g, and
0.01080 g (24 particles) remained unclassified. This is scoped evidence, not
exhaustive containment or a claim of spill-free operation.

The 1280 × 720 MP4 has 1,507 frames at 25 fps and lasts 60.28 seconds; full
decoding and complete-sequence checks passed. It is saved as
`~/Videos/rizon_sharpa_coffee_steep_entry_scoop.mp4`. Motion, mass, video and
particle-audit records are under
`logs/coffee_steep_scoop_20261001/video_steep60_flat_400/`. The run retained the
2 mm grid, captured sparse physics graph, native `P1d`/`pic27`/`backward`
configuration, no hard projection and five decorative grains. Capture wall
time was 2,270.360 seconds including RTX and passive diagnostics; this is not
a pure environment throughput measurement.

The earlier colour-only water-and-stir sequence passed a complete mounted GL recording
at 3 mm, 200 Hz MPM, 800 Hz rigid integration, 50 solver iterations and five
decorative grains. It completed in 78.37 simulated seconds, with 4.65649 g of
coffee classified in the cup after stirring. This coarser seed contained
131,120 coffee particles (199.1385 g) and 54,915 water particles (185.338125 g).
The sparse physics graph stayed captured and its accumulated grid status was
zero throughout the checked steps. Native positions and tracer stayed finite,
the tracer stayed nonnegative, and its summed mass agreed with the cumulative
release counter within the checked 0.2% tolerance. Concentration standard
deviation declined from a stirring peak of 0.008865 to 0.003650 by 78 seconds,
including continued release and subsequent settling. Physical grounds remain
visible; this is an optical mixing demonstration.

The CPU stir-motion check completed all phases with a 65-degree inclination,
approximately 40 mm/s peak stirring speed, and a minimum sampled spoon/glass
mesh gap of 5.009 mm. At 78 seconds in the coarse GPU recording, 345 water
particles (1.164375 g, 0.63%) were outside the cup containment mask. This profile
does not establish watertight liquid contacts or calibrated fluid behaviour.
The MP4 contains 1,959 fully decoded frames at 25 fps (78.36 seconds), after
removing one initial scratch shader-diagnostic frame; the raw movie is retained.
It is saved as `~/Videos/rizon_sharpa_coffee_water_stir.mp4`, with evidence under
`logs/coffee_water_stir_20261001/video_coloured/`. Combined simulation, GL,
colour transport, video capture and diagnostics took 506.343 seconds through
the 78-second sample, giving 0.154 RTF, excluding initialization. This is not
a renderer-free environment benchmark.

The earlier colour-only Kit version completed the same wet motion in 78.37 simulated seconds. Its
1280 × 720 MP4 contains 1,960 fully decoded frames at 25 fps (78.40 seconds),
with two scoops, stirring, five decorative grains and no shader-diagnostic
frames in the movie. A local tint check confirmed that separate water regions
receive separate colours, then reset that artificial tracer before the first
physical step. Kit's glass is more reflective than GL's, making the liquid tint
less obvious behind the wall reflections. The final cup classification was
4.67471 g of coffee; the tracer field stayed finite and nonnegative, with
captured sparse physics and zero grid status at the checked steps. At 78 seconds,
429 water particles (1.447875 g, 0.78%) lay outside the cup mask. Combined physics,
Kit rendering, colour transport, capture and diagnostics took 931.919 seconds
through that sample (0.0837 RTF), excluding initialization and sharing the GPU
with another render job. This is a capture measurement, not an isolated physics
benchmark. The recording is `~/Videos/rizon_sharpa_coffee_water_stir_kit.mp4`;
validation and phase images are under `logs/coffee_kit_20261001/video/`.

The later colour-only Kit cup-view recording also passed full decoding of all 1,960 frames
at 25 fps (78.40 seconds). The native camera trace confirmed a smooth rise from
32.64 to 50.30 degrees after the first pour and a 55.00-degree view from the
cup's +Y side during stirring. The view was checked with the complete scoop
path to avoid forearm occlusion. The final cup held 4.59422 g of coffee, and the
passive tracer remained finite, nonnegative and conserved at the checked steps.
At 78 seconds, 509 water particles (1.717875 g, 0.93%) were outside the cup mask.
Combined physics, Kit rendering, colour transport, capture and diagnostics took
491.845 seconds through that sample (0.1586 RTF), excluding initialization and
including desktop/video-preview GPU activity. This is a capture measurement.
The recording is `~/Videos/rizon_sharpa_coffee_water_stir_kit_cup_view.mp4`, with
validation and decoded phase images under `logs/coffee_kit_cup_view_20261001/video/`.

The earlier colour-only 2026-10-02 ceramic close-up capture completed the full 78.37-second wet
motion. Its 78.40-second Kit MP4 passed decoding of all 1,960 frames at 25 fps.
Both pours use a close elevated view; the stirring camera stays 29.0 cm from
the opening at 55 degrees from -X/+Y. A separate half-tinted-water render
confirmed local optical absorption before recording, with no artificial tracer
or diagnostic frames in the movie. The opaque sage cup, ivory interior and
external loop handle replace the earlier glass appearance. Native water motion
is unchanged; the finer mesh and refracting MDL surface change presentation.
The first attempt exhausted a one-million-cell surface buffer late in stirring;
the completed recording uses four million render cells. Sparse physics and
surface statuses stayed zero, and the passive tracer stayed finite, nonnegative
and conserved at the checked steps. At 78 seconds, 232 water particles
(0.783 g, 0.42%) were outside the cup mask. The final cup classification was
4.64434 g of coffee. Combined physics, Kit rendering, colour transfer, capture
and diagnostics took 1,082.290 seconds through 78 seconds (0.0721 RTF), excluding
startup and sharing the GPU with another job; this is not an isolated physics
benchmark. The fine liquid reconstruction and transmission add render cost.
The recording is `~/Videos/rizon_sharpa_coffee_kit_closeup_ceramic.mp4`, with
validation and decoded phase images under `logs/coffee_kit_appearance_20261002/video/`.

### Previous shallow-entry profile

The restored original 50 × 32 × 10 mm mounted spoon completed both localized
scoops and pours in a 400 Hz RTX recording lasting 59.90 simulated seconds.
Glass-mass increments after emptying were 1.86120 g at 25.03 s and 1.59030 g
at 54.88 s, including ongoing settling. The final settled glass contained
3.45285 g. These teaspoon doses do not meet the earlier 8–10 g per-scoop target;
the original spoon shape, source seed, density and physical particle mass were
retained. Measurements used verified uniform native masses of about
0.00045 g across 456,836 particles, totaling 205.57619 g initially.

Fixed source-cleared central cohorts of 3,464 and 2,968 particles reported zero
native floor crossings, pose-rebase crossings or ambiguous samples before
deliberate tipping. One particle in the second cohort left the radial watch
region; it remained unwatched thereafter. Top watch exits were zero, and the
continuous observer reported zero unique floor flags. Classified scene spill
was 0.06210 g (138 particles), with 0.00540 g (12 particles) unclassified.
These scoped checks do not establish spill-free or exhaustive containment.

The 1280 × 720 MP4 contains 1,498 frames at 25 fps, lasting 59.92 seconds;
complete-sequence and full-decoding checks passed. The recording is
`~/Videos/rizon_sharpa_coffee_flat_short_scoop.mp4`, with final records under
`logs/coffee_local_scoop_20261001/video_local_flat_400/`, including
`physics_audit.json` and `video_validation.json`. It used the current 2 mm grid,
`P1d` strain, `pic27` contacts, `backward` collider velocities, no hard projection,
five decorative grains and a captured sparse-grid physics graph. Capture wall
time was 1,911.862 seconds including RTX and passive diagnostics; this is not a
pure environment performance measurement.

The restored original 50 × 32 × 10 mm spoon and 13.2-second localized scoop
have passed a contact-disabled CPU rigid check at 400 Hz coupling with 800 Hz
rigid integration, completing both scoops and pours in 59.88 simulated seconds.
Measured local draws were 19.81 and 19.96 mm. Both scoops leveled below one
degree at about 7.56 and 7.55 seconds while the cavity remained below 35 mm,
before the level lift. The action interface and measured arrival gates remain
unchanged; see `logs/coffee_ik_cpu_20261001/original_flat_localized_production_400.json`.

The matching dense CPU mesh audit found minimum tool/bowl and mounted-hand/bowl
gaps of 4.085 and 8.829 mm, and tool/glass and hand/glass gaps of 32.275 and
23.557 mm. These finite sampled checks establish modeled tracking and clearance,
not particle retention. The contact-hand alternative intruded at these poses,
so the loaded contact script is not certified. Geometry evidence is in
`logs/coffee_ik_cpu_20261001/localized_flat_core_curl30to365_lane4_400_narrow1.23_1.15_geometry.json`.
Its agreement with the production trace is recorded in
`logs/coffee_ik_cpu_20261001/original_flat_localized_production_400_geometry_match.json`.

The following 64 × 42 × 19 mm deep-scoop recording and long sideways motion
were rejected during shape review. They are historical evidence and do not
validate the restored original spoon or its localized motion.

That mounted deep scoop completed both sideways scoops and
pours in a 400 Hz RTX recording lasting 69.35 simulated seconds. It used a
2 mm grid, `P1d` strain, `pic27` contacts, `backward` collider velocities, no hard
projection and five decorative grains per physical particle. Glass-mass
increments after emptying were 10.52460 g at 31.63 s and 9.37710 g at 64.31 s,
including ongoing settling: 19.90170 g cumulatively at the second checkpoint.
The final settled glass contained 19.98090 g. Pre-tip cavity-only metrics were
8.35560 and 7.38765 g; their bounded mask excludes a higher heap.

The workstation wrapper's 8–10.5 g dose guardrail failed on the first scoop by
0.02460 g. That failed status is preserved in `video_validation.json`. The
10.52460 g dose was reviewed as approximately 10 g for the earlier approximate
8–10 g scoops, with its exact mass disclosed; it exceeds the nominal 10 g upper
target. The wrapper measures `delivered_per_scoop` from glass-mass increments
after each emptying phase and also checks watched carry-floor crossings.
Ordinary `--video` uses the configured final delivery-fraction threshold and
does not perform these per-scoop dose or floor-audit assertions.

Fixed source-cleared central cohorts of 17,882 and 15,390 particles reported
zero native and pose-rebase floor crossings before deliberate tipping, with
zero ambiguous samples. Radial watch exits were 75 and 54, and top exits were
3 and 2; exited particles remain unwatched. The continuous observer reported
zero unique floor flags. Classified scene spill was 7.87770 g, with another
3.98295 g unclassified. Visible source-side spillage remains; these scoped
checks do not establish spill-free or exhaustive containment, and still images
do not identify a rim-versus-wall escape path.

The 1280 × 720 MP4 contains 1,734 frames at 25 fps, lasting 69.36 seconds;
full decoding passed. The historical recording is
`~/Videos/rizon_sharpa_coffee_sideways_full_scoops.mp4`. Records are
under `logs/coffee_sideways_20261001/video_sideways_final64_400/`, including
`physics_audit.json` and `video_validation.json`. Capture wall time was
2,509.743 seconds including RTX, five-grain rendering and passive diagnostics;
this is not a pure environment benchmark.

The rejected deep scoop's 16-second side-entry motion completed both scoops and pours
in a contact-disabled CPU rigid check at 400 Hz coupling, lasting 69.34 simulated
seconds. It used the final centered bowl, 50 mm shaft,
mount compensation, X lanes and glass position, with 800 Hz rigid integration
and the unchanged 29-action interface and measured arrival gates. Actual
buried sweep pitch was about 10.11 and 10.09 degrees, with bank angles about
−7.94 and +8.00 degrees. Minimum sampled cavity heights over the full scoops were
13.726 and 14.045 mm. Peak transfer speeds were 0.10939 and 0.08669 m/s, with
maximum cavity-normal tilts of 2.982 and 0.558 degrees. This check validates
modeled reachability and tracking, with no MPM particles or spoon contacts;
it does not establish a retained dose. The trace is
`logs/coffee_ik_cpu_20261001/side_final_cavity64x42x19_bowl123x115_400.json`.
Earlier deep-scoop checks at both 400 and 800 Hz used the 72 × 50 × 23 mm scoop;
they do not validate the restored spoon's current localized profile at 800 Hz.

A CPU mesh audit of that historical 400 Hz trace sampled all phases of both scoops
at 10 ms intervals, using the three tool collision meshes, mounted hand vertices
and face centers, and the vessel profiles. Minimum scoop/bowl and hand/bowl gaps
were 6.680 and 4.755 mm; scoop/glass and hand/glass gaps were 27.745 and
40.857 mm. There was no sampled physical intrusion, and the scoop/bowl gap
exceeded its combined 3 mm collision margin. Finite pose and surface sampling
does not prove continuous contact freedom or loaded-grasp retention; see
`logs/coffee_ik_cpu_20261001/side_final_cavity64x42x19_bowl123x115_400_narrow1.23_1.15_geometry.json`.

The intermediate 66 × 44 × 20 mm native-contact calibration completed in
69.33 simulated seconds. Glass-mass increments after emptying were 11.21895 g
and 9.87660 g; the final settled glass contained 21.17835 g. Pre-tip cavity-only
metrics were 9.315 and 8.0613 g, excluding higher heaps and understating the
delivered dose. Fixed cohorts reported zero native and pose-rebase floor
crossings before tipping, and the continuous observer reported zero floor flags.
These are historical calibration results and do not validate the restored flat
spoon; see `logs/coffee_sideways_20261001/native_full66/`.

Earlier 60 × 40 × 18 mm trials with hard projection carried about 17–24 g,
with numerical clustering near the scoop floor reaching 417 particles in a
2 mm cell. Increasing the solve cap to 100 or 250 iterations did not establish
correct packing. The earlier 100-iteration settling screen's higher bed level
also did not isolate convergence from projection, duration or motion. Native
mass and radius checks matched 0.00045 g and 0.5 mm, so the excessive dose was
not a particle-unit correction. Those projected trials failed dose and packing
validation despite reporting zero floor crossings in some runs.

A subsequent native-contact first-lift screen with the 60 × 40 × 18 mm scoop,
`P1d` strain, `pic27` contacts, `backward` velocities and no hard projection carried
6.83595 g at 23.5 simulated seconds. Its loaded-particle diagnostic measured a
0.721 mm median nearest-neighbor distance and at most 15 particles per 2 mm cell.
The fixed cohort and continuous observer reported zero floor crossings during
that screen. It did not complete a carry and pour or validate the restored flat
spoon; see `logs/coffee_sideways_20261001/native_backward_p1d_thick/`.

The historical partial 400 Hz RTX trial of the 60 × 40 × 18 mm scoop used hard
projection, completed the first pour and stopped during the second entry at
about 45.68 simulated seconds. Its pre-tip dose was 20.5704 g and the glass
contained 20.60595 g afterward. Classified spill was 0.05445 g (121 particles),
with two final particles unclassified. Of 42,201 fixed central-cohort particles,
41,861 remained watched before tipping and 340 left the radial watch region;
native floor, pose-rebase, top and ambiguous flags were zero, as were continuous
floor flags. These counts do not establish correct packing or exhaustive
retention. The fully decoded MP4 is
`~/Videos/rizon_sharpa_coffee_sideways_trial_20g.mp4`; records are under
`logs/coffee_sideways_20261001/video_sideways_full_400/`. This partial trial is
historical diagnostic evidence, not target-dose or throughput validation.

The historical captures and
timings below used a 50 × 32 × 10 mm teaspoon with earlier handle variants,
127,512 particles and a 30 mm fill. They do not validate the current larger
source fill, localized motion or runtime cost.

The previous 18-second forward scoop completed both dry mounted scoops and pours in a
400 Hz RTX capture lasting 72.68 simulated seconds, with native particle
projection, one MPM substep, a 2 mm grid, 2.4 mm spoon margin and five decorative
grains per particle. The pours delivered 4,744 and 991 particles: 5,735 particles
or 2.58075 g, 16.68% more than the previous flat profile's 2.21175 g. Measured
spill was zero; one final particle was outside the source and glass classification
masks, so this is not an exhaustive containment result.

Fixed central cohorts of 4,505 and 991 particles retained every particle from
source clearance through carrying and lowering, ending before each deliberate
tip. The audit recorded zero native floor crossings, pose-rebase crossings,
radial or top escapes, or ambiguous samples over 29,072 native MPM ticks. The
separate end-pose-corrected continuous observer also reported zero floor flags.
Recorded submerged sweep pitch was 10.09–10.12 degrees; minimum leveling cavity
heights were 9.396 and 9.501 mm, with about 0.12-degree pitch at the level endpoint.

The 1280 × 720 movie contains 1,817 frames at 25 fps, matching 72.68 seconds of
simulation. RTX rendering and diagnostics took 529.56 seconds on the shared
workstation, a 0.137 real-time factor that includes rendering and diagnostics
rather than measuring pure physics throughput. The movie, measured poses,
metrics and audits are in
`logs/coffee_slow_flat_20260930/video_ten_degree_smooth_400/`.

The previous 18-second forward scoop's mounted-motion CPU check on 2026-09-30 completed
both scoops and pours in 72.67 simulated seconds at 400 Hz coupling with 800 Hz
rigid integration. The imported robot used native IK and motor settings, gravity
compensation, and spoon poses derived from the measured palm with its rigid tool
offset; particles and spoon contacts were absent. Both sequences passed every
configured arrival gate. Actual pitch during the 7.6–8.8-second submerged sweeps
was about 10.1 degrees. During the four-second leveling windows, the minimum
sampled cavity heights were 9.392 and 9.498 mm for the first and second scoop,
with no sampled backward X motion. Dense analytical evaluation of the commanded
quintics also confirmed forward motion from the shallow sweep through the lift,
constant commanded depth while leveling, zero pitch throughout the lift, and
continuous position, velocity and acceleration at the knots. These checks
establish modeled motion reachability and ordering; they do not establish
loaded-powder retention. The measured trace is
`logs/coffee_ik_cpu_20260930/ten_degree_level40_forward_400.json`.
A separate CPU spoon-mesh check found no source-floor intrusion: minimum
clearance was 1.468 mm during early entry and 3.469 mm during leveling. The
leveling clearance exceeded the combined 3 mm collision margin.

Earlier mounted-motion CPU checks of the previous flat profile completed both
scoops and pours in 64.69 simulated seconds at 400 and 800 Hz coupling rates.
The checks used the imported robot, native IK and motor settings, gravity
compensation, and spoon
poses derived from the measured palm with its rigid tool offset; particles and
spoon contacts were absent. Both sequences passed every configured arrival gate.
Peak carrying speeds were 0.154 and 0.122 m/s for the first and second scoop.
Measured pitch stayed within 2.32 and 0.93 degrees, while cavity-normal tilt
reached 5.89 and 2.14 degrees in the saved pose samples. Before tipping, the
cavity-normal tilt was below 0.53 degrees. These checks establish modeled motion
reachability; they do not establish loaded-powder retention.

The previous flat profile's complete dry mounted RTX capture passed both scoops
and pours in 64.70 simulated seconds at 400 Hz with native particle projection,
one MPM substep, a 2 mm grid, 2.4 mm spoon margin and five decorative grains per particle.
The first pour delivered 3,999 particles and the second delivered 916: 4,915
particles, or 2.21175 g, in the glass at completion. The measured spill fraction
was zero and all final particles were classified in the source or glass.

That capture's audit tracked fixed central cohorts of 3,759 and 913 particles
after source clearance through the remaining lift, carry and lowering, ending
before each deliberate tip. Both cohorts retained every particle, with zero native floor
crossings, pose-rebase crossings, radial or top escapes, or ambiguous samples.
The observer used the collider's predicted end pose, stopped each particle's
watch after its first escape, and recorded 25,880 native MPM ticks, exactly four
for each of the 6,470 control steps. The separate end-pose-corrected continuous
observer also reported zero floor flags. These results validate the recorded
previous flat mounted profile and the observed central cohorts. The previous
18-second forward profile's results are reported above; the enlarged scoop,
sideways profile, contact grasp, wet motion and
other rates or resolutions require their own particle checks.

Before native projection, the previous flat profile's 400 Hz motion with the
corrected fixed-cohort observer reported 11 native floor crossings in the first
carry and one in the second, with zero pose-rebase crossings. Earlier observers compared end-of-step
particles with beginning-of-step collider poses and allowed escaped particles
to be enrolled again; those earlier flags do not establish a physical crossing.

In that previous flat-profile recording, the submerged sweeps measured
0.09–0.12 degrees of pitch and cavity heights of 10.56–10.79 mm, advancing about
14.36 mm horizontally before lifting.
Both passes kept the spoon approximately level while submerged before lifting.

That flat-profile 1280 × 720 H.264 recording contains 1,618 frames at 25 fps, or
64.72 seconds of playback for 64.70 seconds of simulation. Full decoding and
sequence checks passed. RTX rendering and particle diagnostics together took 570.42 seconds
(0.113 real-time factor) on the shared workstation. The movie, settings, metrics
and both particle audits are in `logs/coffee_slow_flat_20260930/video_projected_400/`.

The following historical initial-motion timing measurements include native particle
projection. They use the previous smaller spoon in mounted mode, 127,512 physical
particles, five visual grains per particle, a captured 32,768-cell sparse grid,
and no rendering. The 400 Hz run excludes 100 warmup steps and times the next
300 at the fixed 100 Hz control rate. The 800 Hz comparison uses its final 200
timed steps, after competing GPU jobs finished:

| Coupled physics rate | Mean wall time per control step | Real-time factor |
| --- | --- | --- |
| 400 Hz | 36.028 ms | 0.278 |
| 800 Hz | 65.105 ms | 0.154 |

At 400 Hz, grain advection measured 0.2638 ms per MPM tick, or about 1.06 ms
across the four ticks per control step: approximately 3% of the measured step
cost. These are early-motion measurements on a shared workstation. Other GPU
jobs reduced a separate 400 Hz run to 0.044 RTF; the process samples preserve
that distinction. The previous forward profile's RTX capture time above includes rendering and diagnostics
and does not measure pure physics throughput.
Timing scripts, process samples and particle checks are in
`logs/coffee_slow_flat_20260930/`; CPU motion checks are in
`logs/coffee_ik_cpu_20260930/`.

The preceding 13-second raised-neck contact trial on 2026-09-30, with the glass
at its earlier position, used 400 Hz and proxy mass scale 40. It reached
0.121–0.177 degrees of actual pitch at a cavity height of 10.33–10.61 mm during
the buried constant-depth sweep. It advanced 15.05 mm
at that depth and 25.14 mm while remaining level through the beginning of the
lift. Peak carried coffee was 2,294 particles (1.03230 g); the first pour delivered
2,252 particles (1.01340 g). Its earlier observer reported no central-floor
crossings, including during digging, subject to the unresolved observer issues
above. The contact grasp nevertheless slipped during first recovery
at 26.03 s, before the second scoop, so this remains a partial contact-mode trial.

Earlier ten-second contact holds, before native particle projection, started from
the derived reset pose and loaded 4,909 particles through the public particle-state
API. At 400 Hz with proxy mass scale 40, 4,907 remained; at 800 Hz with scale 20,
all 4,909 remained.
Both had zero observed central-floor crossings, with observer counters confirming
all 4,000 and 8,000 MPM ticks respectively ran inside captured graphs. These
stationary tests use the reset orientation and do not establish grasp retention
through the complete moving sequence. The detector covers the approximate
central shell in the MPM proxy frame and stops watching particles that exit
over the rim; it does not certify every possible leak path.

The 2026-09-30 workspace update pins Newton main at
`1aca1a72dbeedc7cadc05d28f44575a2bbab9003` and Warp
`1.19.0.dev20260930` (commit `42a409a9c1c05ae23de1e184dafb71873d5fb5a4`).
That official Warp nightly includes every runtime change at main
`44bd2379dac607fec7cfbaace4e2229bfb1d8a8d`; the remaining commit changes only
documentation. Both pins are recorded in the workspace lockfile.
The robot loader references the original licensed USD through a cached override
that blocks malformed corner-normal arrays on two visual meshes. This lets the
updated importer recompute normals without changing geometry, UVs or collisions.

On the workstation's RTX 4090, the dependency-only comparison improved from
241.71 to 149.18 ms per control step (1.62× throughput, 38.28% less wall time).
Both runs used 127,512 particles, five passive grains per particle, the original
1.8 mm margin and unit proxy mass scale, and no renderer. The comparison uses
the same 400 control steps from simulation time 1–5 s; the first 100 steps,
compilation and capture are excluded. A longer updated run overlapped another
GPU job and is excluded from the speed claim. Timings, process samples, scripts
and dependency fingerprints are in `logs/coffee_mpm_update_20260930/`.
This measures a matched initial-motion window, not full-cycle or video throughput.

The earlier 800 Hz timing configuration used a rebuildable sparse grid at 2 mm,
1 mm physical particle spacing, S2 collider sampling, 25 solver iterations, and
one MPM step per physics tick. A CUDA graph was captured for physics; the Python
controller and complete eight-tick control loop were not captured together.
That 127,512-particle fill used an automatic RL budget of 128,000 active cells;
the scripted demo used 32,768. The current larger fill uses the budgets described
in Run above. Capacity sizes padded GPU work as well as storage.
Reducing it does not coarsen the grid, but requires validating active-cell demand
over the intended trajectory. Coarser grids failed earlier thin-spoon containment
checks, so the demonstration retains the 2 mm resolution.

With those previous supported-spoon contact settings, a clean RTX 4090 comparison
measured 145.19 ms per control step at 128,000 cells versus 73.75 ms at 32,768
cells: 1.97× throughput, with five visual grains enabled in both cases. Disabling
those grains at 32,768 cells reduced this to 71.67 ms, about a 3% saving.
These are synchronized physics/controller timings without a renderer, using
the same 200 steps from simulation time 1–3 s after compilation, capture, and
warmup. Reported process GPU memory fell from 4,018 to 2,034 MiB. The task is
still slower than real time; these timings are not video-rendering throughput.

The preceding 32,768-cell contact rollout detected no floor crossings through
its first scoop and pour; peak sampled occupancy was 18,147 cells. Its second
grasp still slipped at 26.73 s. Measurements and scripts are in
`logs/coffee_performance_20260930/`; grid capacity does not fix finger retention.

Drawing cost is separate from grain advection. Before reusing static instance
attributes, the initial 1280×720 RTX shot at four samples measured 49.94 ms/frame
for physical particles and 157.10 ms/frame for five grains, including RGB
readback. Dry grains now upload radii and colors on first draw and after reset,
then use Newton's positions-only update; wet grains still update their shrinking
radii. Live previews default to GL; recordings retain RTX.
The earlier complete 32,768-cell mounted rollout also passed both scoops in 48.83 s,
with no observed central-floor crossings and 6,799 particles (3.05955 g) in the cup.

Before that dependency and contact update, the complete dry two-scoop recording
passed 4,883 control steps over 48.83
simulated seconds without a reset or refill. It used 127,512 physical particles,
2 mm voxels, 30 mm source fill, 1.8 mm spoon margin, one MPM substep, 25 solver
iterations, native background-air regularization, and five visible grains per
physical particle. The glass ended with 6,784 particles (3.05280 g), and the spoon
emptied. Mean delivered mass was 1.52640 g per scoop, compared with 1.05735 g in
the previous three-scoop run. The two-scoop total remains slightly below that
earlier three-scoop total of 3.17205 g.

The audit tracked the exact physical-particle identities inside the spoon
immediately before each tip, then classified them at the recording's end:

| Scoop | Pre-tip dose | Dose in glass at recording end | Delivery |
| --- | --- | --- | --- |
| First, Y −16 mm | 4,545 particles / 2.04525 g | 4,542 particles / 2.04390 g | 99.9340% |
| Second, Y +16 mm | 2,237 particles / 1.00665 g | 2,237 particles / 1.00665 g | 100% |

The first dose had one classified spill particle and two unclassified particles.
Total scene spill was 625 particles (0.28125 g), including source-bed and carry
losses outside these pre-tip cohorts. Settings, metrics, poses, and audits are
saved under `logs/coffee_two_deep_scoop_final/` on the workstation.

No central spoon-floor crossings were detected across the two carries,
audited at every native 800 Hz physics step: 6,594 and 6,580 samples.
This detector covers particles that remain in the spoon's central region after
bowl clearance and stops before deliberate tipping; submerged scooping and
outer-rim paths are outside its scope. No particles escaped through the physical
studio floor, sampled states stayed finite, and all 65 recorded joint checks
passed. See `physics_validation.json` and `floor_crossing_audit.json`.

The 1920 × 1080 H.264 movie contains 1,221 frames at 25 fps: 48.84 seconds of
playback for 48.83 seconds of simulation, within one video frame. Frame-count,
playback-timing, and full H.264 decoding checks passed; results are in
`video_validation.json`.

Sampled recorded spoon poses reached nose-down entry angles of 61.85 and
61.92 degrees. Minimum tool/bowl gaps were 3.079 and 3.213 mm, above the combined
2.4 mm margins; tool/cup gaps exceeded 31.6 mm. The spoon's lowest outer surface
was about 7.1 mm above the tabletop, retaining the established near-floor
undercut depth. These tool-only checks are in `tool_clearance_audit.json`;
finite pose and surface sampling do not prove continuous collision freedom.

The preceding isolated native-MPM containment regression passed with its recorded settings.
The exact imported-spoon fixture retained 6,245 of 6,284 particles (99.38%) after
a three-second hold and 26 cm translation with 8-degree pitch, with no detected
central floor crossings; particles can still leave over the rim. A 90-degree tip
released all particles. Those historical checks used no task-level physical-particle
clamping or projection and predate the historical native projection stage; see
`logs/coffee_spoon_containment/containment_evidence.json`.
The studio floor separately passed 4.4 and 6 m/s particle drops onto its
50 mm collision slab with zero crossings; see
`logs/coffee_three_floor_validation/floor_drop_50mm.json`.

These historical results validated the recorded dry scripted rollout with a
mounted spoon. They do not establish the current motion, free grasp retention
or an RL-trained policy. Short fine-grid wet stability checks have passed;
complete wet transfer and live headset operation still require validation.

## Grasp reset authoring

[Isaac Lab PR 7735](https://github.com/isaac-sim/IsaacLab/pull/7735) provides
`KukaAllegroResetStateTable`: an authored reset bank with arm IK, separate
open/contact/preload hand postures and physical grasp screening. Its Allegro
joint values and cube geometry do not transfer to this 22-joint Sharpa hand.
This task reuses that reset contract through a small named-joint JSON adapter.

```bash
uv run python scripts/tools/create_coffee_grasp_reset.py \
  --output logs/coffee_grasp/reset.json

# Edit named joint_positions and joint_targets independently, then screen it.
uv run python scripts/tools/create_coffee_grasp_reset.py \
  --input logs/coffee_grasp/reset.json --validate

uv run --extra ovrtx python scripts/demos/rizon_sharpa_coffee.py \
  --grasp_reset logs/coffee_grasp/reset.json --keep_open
```

The artifact pins the robot asset fingerprint, all 29 joint names, zero-velocity
reset positions, separate motor preload targets and the palm-to-spoon transform.
`capture_grasp_reset(env)` can also save a measured simulator pose. Loading a
candidate preserves its preload targets through reset and scripted operation.

The optional validator uses the imported robot meshes and separate spoon
handle/bowl colliders in Newton's MuJoCo CPU backend. It disables the utensil
mount, uses a free 25 g spoon under gravity, checks finger/tool clearance,
screens cross-finger intersections throughout the test, and requires loaded
thumb/finger contact normals to oppose. It saves a failure report and exits
nonzero when retention fails. The present mounted candidate is not certified as
a frictional grasp. The preceding contact reset passed a stationary screen;
that historical result does not certify the current localized motion, acquisition,
disturbances or loaded MPM tool use.

## Experimental contact grasp

```bash
uv run python scripts/demos/rizon_sharpa_coffee.py \
  --grasp_mode contact --visualizer newton_gl --grains_per_particle 0
```

This selects the bundled `assets/contact_grasp.json`. Use `--grasp_reset PATH`
to override it. The 25 g spoon is a free MJWarp body supported by finger contact
forces; no attachment callback runs in this mode. Its MPM proxy returns reaction
impulses to the rigid solver with finite virtual mass scale 20. The action and observation
interfaces are unchanged. Both modes start with an already-held spoon.

`--spoon_proxy_mass_scale` changes only the spoon's effective mass and inertia
in MPM; its 25 g MJWarp mass and contact-release behavior remain unchanged.
The value must be finite and positive. A larger multiplier approximates the
support missing from the MPM proxy; it is not a measured hand inertia and can
suppress legitimate free-tool motion. Rigid substeps preserve contact integration
accuracy at lower physics rates, but coupling reactions are exchanged only at
the selected outer rate. Additional MPM substeps do not increase that exchange rate.

The particle solve sees a free-body proxy without the support of the fingers.
With unit proxy mass scale, a stationary, contact-held spoon lost roughly 65%
of a prefilled dose over five seconds, including tracked crossings through the
central floor. Increasing only its collision margin delayed the leak. The
larger virtual inertia compensates for the missing contact support while leaving
the rigid spoon's mass at 25 g and preserving two-way reaction impulses. This is
a coupling approximation, not a measured effective inertia of the hand. It
needs reassessment for unsupported, freely moving utensils or different grasps.
On the updated stack, the preceding 2.4 mm margin and mass scale 20 retained all
4,909 prefilled particles for five seconds, with no central-floor crossings
observed at 800 Hz. Scale 10 retained 4,908 particles but had one crossing in
that same diagnostic. The batched regression also checks contact release and
partial reset; a stationary hold does not certify a moving scoop or grasp acquisition.

The preceding updated full-scene trial carried 1,808 particles unchanged through its first
transfer and delivered all 1,808 (0.81360 g) into the cup. Substep tracking found
no central-floor crossings from the first lift through the pour. The spoon then
slipped during the second scoop, between 28 and 29 s, and the existing grasp
guard stopped the run at 36.3 mm relative displacement and 37.8 degrees of
rotation. Thus the containment improvement does not establish a reliable
two-scoop contact grasp. This trial and the separate entry audit are in
`logs/coffee_mpm_update_20260930/`; the earlier videos below predate these changes.
The entry audit continuously enrolled particles above the central cavity floor
at every MPM step throughout the first 15 s, including digging, and likewise
detected no floor crossings.

The reset holds the handle about 35 mm closer to the cavity than the mounted
pose. Native MuJoCo convex contacts use a 0.5 mm search gap, two rigid substeps
per default 800 Hz tick, a 2 ms contact-response time, an elliptic friction cone, and
six-dimensional contact friction.
The 5 mm torsional/rolling friction coefficients approximate a finite contact
patch; they are tuning values, not measured Sharpa pad properties. Hand damping
is 0.2 in contact mode. The reset validator and runtime share this profile.

No additional SDF is needed for the convex handle/finger pinch. MPM already uses
its own implicit representation of the concave spoon and vessel surfaces.
Changing that representation would not fix an unsuitable rigid contact gap or
insufficient preload. See [Newton's collision documentation](https://newton-physics.github.io/newton/latest/concepts/collisions.html).

The preceding bundled reset passed a ten-second, gravity-only CPU screen with 1.99 mm
maximum relative movement, 2.23 degrees of rotation, opposing finger contacts in
all sampled frames, and at most 0.087 mm tool penetration. The earlier, softer
profile passed a separate two-world MJWarp/MPM hold-and-release probe: the closed
hand retained its spoon for 3.4 seconds;
opening the other hand released its spoon. A CPU replay with the firmer profile
retained the spoon through the first pour and a three-second tilted hold, with
8.46 mm maximum movement and 6.66 degrees of rotation. These results establish
contact retention and release, not a complete two-scoop demonstration.

The full 2 mm MPM trial completed its first scoop and pour, delivering 857
particles (0.38565 g). It then lost the grasp during `recover_1`, around 22.1 s,
with 29 mm relative displacement and 44.8 degrees of rotation at the stop
condition. The second scoop did not run. Sampled cross-finger clearance stayed
above 0.66 mm; the failure was loss of retention, not fingers passing through
one another. The run also lost substantial coffee during transport. Its
metrics and sampled geometry audits are in `logs/coffee_contact_grasp/loaded_run/`.
A subsequent 30-second RTX recording retained the spoon through that first
recovery and into the second scoop. The retention failure is therefore
intermittent; this partial recording does not establish complete two-scoop
success. A labeled, quarter-speed replay of the earlier measured hand and spoon
poses covers 20.00–22.10 s, ending just before the stop. It hides particles because
their trajectories were not recorded in that run. Both videos and capture
metadata are in `logs/coffee_contact_grasp/failure_video/`.

Contact-mode IK measures the palm-to-spoon transform every control step, using
simulator ground truth to compensate for compliance. It stops if displacement
exceeds 30 mm or rotation exceeds 35 degrees relative to the reset grasp.
Both grasp modes command the current shallow stroke, but its motion and
clearance checks cover the mounted grip; the contact-hand alternative intruded.

Hand self-collisions and rigid contacts against the table and vessels remain
disabled; those props currently constrain MPM particles. Rigid-body clearance
must therefore be checked independently. Imported fingertip marker bodies also
receive Newton's minimum-inertia correction. These settings and the approximate
contact-patch model are limitations for physical fidelity and later RL transfer.
The previous dry mounted 18-second forward profile passed the complete 400 Hz
capture and scoped particle audits described above, alongside its CPU motion and
mesh-clearance checks. Current CPU motion and sampled mesh checks are reported
above, alongside the restored original spoon's completed mounted recording and
scoped particle audit. Contact mode remains
experimental, with observed grasp-retention failures in preceding full-scene trials.

For video-derived motion, [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X) estimates
SOMA body and hand motion; [SOMA Hand](https://github.com/NVlabs/SOMA-X#soma-hand)
provides a detailed hand representation. Kimodo is a motion generator, and its
[internal skeleton removes most finger detail](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/skeleton.html).
The [TACO dataset](https://github.com/leolyliu/TACO-Instructions) includes human
spoon/bowl actions with hand and object poses. Its annotations provide a starting
point for metric retargeting, but still need adaptation to the Sharpa hand and
this coffee scoop, followed by contact and retention checks. This environment does
not claim automatic video-to-grasp recovery.
