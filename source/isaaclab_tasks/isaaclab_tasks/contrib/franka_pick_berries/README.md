# Berry picking

The implementation, CLI and installer are entirely self-contained in this task
directory; no external task checkout, Carsten source checkout, or sibling contrib task
is required. Berry USDZ packages and the EBC background are fetched on demand from
Nucleus through Isaac Lab's own asset cache (`isaaclab.utils.assets.retrieve_file_path`);
there is no separate asset-download step.

## Layout

```text
franka_pick_berries/
  pick_berries_env.py, pick_berries_env_cfg.py   environment class and configuration
  config/franka/                                 gym registration
  mdp/                                           action terms
  physics/                                       Newton solver manager, materials, contact, per-berry runtime, coupled tissues
  physics/mpm/                                   explicit MPM solver and Gaussian binding
  rendering/                                     interactive viewer, per-berry Gaussian streams, sampling settings
  scene/                                         tableware props and the EBC room background
  assets/                                        asset locations, USD/USDZ loading, spherical-harmonics transport
  control/                                       gamepad devices and scripted arm motion / sorting sequences
  scripts/                                       teleop / scripted-demo CLI and physics benchmarks
  offline/                                       asset inspection and appearance-repair tools (not needed to run the task)
  setup/                                         runtime installer (setup.sh)
```

## Follow-up work

This task was imported as a working, self-contained first version. Planned next steps:

1. **Replace the explicit MPM solver with the Newton manager.** `physics/mpm/` is a standalone
   solver adapted from an internal proof of concept and a third-party MPM sample, with its own
   licensing and provenance still to be settled. Moving tissue simulation onto Newton removes it
   together with the unused cube-tool and adhesion parameters, the CUDA-graph capture code and the
   separate `mpm_device`.
2. **Publish the assets publicly.** The berry USDZ packages and the EBC background are still read
   from a personal Nucleus folder (`assets/asset_root.py`). Publish them in the public repository
   that hosts the other assets used by the codebase and update the default URL.
3. **Use the codebase's standard Gaussian binding.** `physics/mpm/binding.py` carries four custom
   Gaussian-to-particle binding modes, and only the MLS modes work with `BerryGaussianStream`
   (`affine4` does not). Replace it with the standard binding.

## 1. Install the task runtime

Supported deployment: Linux x86-64, Python 3.12, NVIDIA RTX GPU with a compatible
CUDA/RTX driver, `git`, and `uv`. Interactive commands need a graphical display;
gamepad commands need a readable Linux `/dev/input/js*` device.

From the IsaacLab root:

```bash
bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh --renderer-internal
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"
```

Live rendering requires OVRTX 0.6 / OVStage 0.3; `--renderer-internal` installs the
pinned internal builds from NVIDIA Artifactory (NVIDIA network or VPN access required).
The public 0.5 renderer, installed when `setup.sh` is run without a `--renderer-*` flag,
is rejected for this task because live Gaussian updates can become invisible despite
correct array readback. Physics-only `--no_render` remains supported with the public
runtime. For an existing authorized offline distribution, use
`setup.sh --renderer-wheels DIR` with
`ovrtx-0.6.0-py3-none-manylinux_2_35_x86_64.whl` and
`ovstage-0.3.0.0-py3-none-manylinux_2_35_x86_64.whl` in that directory.

`setup.sh` creates an isolated `.venv-tasks`; it does not modify `.venv-kitless` or global
Python. It installs this branch's frozen environment (Newton, Warp and MuJoCo-Warp at the
versions pinned in `uv.lock`, plus the `ovrtx` extra), `isaaclab_teleop`, and, when
requested, the internal OVRTX / OVStage builds. No separate Newton checkout is needed.
**Use `uv run --no-sync` for this task.** A normal `uv sync` removes the internal renderer
builds and `isaaclab_teleop`; rerun `setup.sh` to restore them.

## 2. Assets

No manual download step is needed. The first time a berry USDZ package or the EBC
background is opened, `retrieve_file_path` fetches it from Nucleus and caches it
locally (default source:
`omniverse://content.ov.nvidia.com/Users/nicolasm@nvidia.com/isaaclab_assets/tasks`,
currently a bundle shared with the tomato and gaussian_twin tasks — berry reads only
its own `berry/` USDZ packages and the EBC background under
`gaussian_twin/background/ebc/`). Every subsequent run reuses the cached copy after a
cheap server freshness check, so access to the Nucleus folder is only required while
online; once cached, the task runs fully offline.

`ISAACLAB_BERRY_ASSET_ROOT` overrides the source root, for example to point at a local
staged copy for testing, or a future berry-only republish with a flat `<berry>/...`
and `background/ebc/...` layout at that root.

## 3. Run the task

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py --mode gamepad
```

Use `--mode keyboard` if no joystick is connected. Raspberry is default;
`--berry` also accepts blackberry, blueberry, strawberry and `all`. `--help` lists all
CLI options. Robot state uses the native CPU MuJoCo backend; MPM and Gaussians
run on CUDA device 0. An RTX GPU is required for rendering.
Per-step JSON metrics are quiet by default. Add `--print_metrics` to show them
in the shell, or use `--output` to save them in `report.json` without shell spam.

Add `--hide_interior` to render only exterior Gaussians when checking appearance
artifacts, for example `--mode keyboard --berry strawberry --hide_interior`.
This leaves tissue physics and the source USDZ unchanged. Omit the flag to show
all Gaussians again.

Use `--sh_rotation off` to disable the live material-frame SH rotation in the MDL
shading (default: `--sh_rotation on`). You can also toggle **Rotate SH with material**
in the viewer for a live comparison. This keeps SH view dependence, bruise tint,
Gaussian geometry deformation and physics unchanged; it does not undo the initial
orientation correction baked into the asset. Both asset versions support the toggle.

Use `--asset_version v2` for the repaired assets, stored alongside the originals
as `<berry>/<berry>_v2.usdz` in the configured berry asset directory. Omitting
the option (or using `--asset_version v1`) loads the original `<berry>.usdz`.

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode keyboard --berry strawberry --asset_version v2
```

Alternatively, use `--asset /path/to/strawberry_simready.usdz --berry strawberry`
to test a specific self-contained package. `--asset` and `--asset_version` are
mutually exclusive. The selected berry must match the package.

## All four berries

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --berry all --asset_version v2 --background ebc
```

This places raspberry, blackberry, blueberry and strawberry at separate positions
in the starting punnet (or on the studio surface). Each retains its own material, deformation,
damage and Gaussian shading. Gripper reactions are summed across all four berries.
The camera defaults to a fixed overview; **View raspberry/blackberry/blueberry/strawberry**
buttons switch the close-up without changing the scripted target. Reset restores
all four berries. `--output` includes per-berry states in each metric sample.

Teleop can handle any berry. For scripted handling, add `--target_berry strawberry`
(default: raspberry), for example `--mode place --berry all --background ebc
--target_berry strawberry`. Scripts handle that one berry per cycle; they do not
automatically collect all four. `--asset_version` applies to every berry; a custom
`--asset` is supported only in single-berry mode. No new asset packages are needed.
Scripts pre-shape the opening above the punnet to avoid neighboring fruit. In
teleop, similarly narrow the opening before descending between nearby berries.

**Limitations:** these are independent MPM systems. They collide with the gripper
and support surfaces, **not with one another**; do not use this mode for piles or
berry-to-berry contact demonstrations. Four active solvers and Gaussian streams
cost more GPU time and memory than a single berry. Use studio background for a
lighter scene; the existing solver rates and physical resolution are preserved.

### Lower-resolution physics

Add `--physics_resolution half` to keep roughly half the tissue particles while
retaining every visual Gaussian, including repaired skin and interior:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --berry all --asset_version v2 --background ebc --physics_resolution half
```

The default `--physics_resolution full` preserves the original solver input.
The half preset selects alternating sites of the tissue lattice, compensates
particle volume to preserve total mass, derives the finite contact spacing from
that volume, and rebuilds Gaussian bindings against the reduced tissue. Material
parameters, MPM grid and timestep stay unchanged. The USDZ files are not modified,
and no new downloads or Nucleus assets are required.

Use v2 assets: all four contain supported regular-lattice tissue. Irregular or
multi-region custom proxies are rejected in half mode rather than silently
changing their material distribution. The older raspberry/strawberry v1 proxies
are irregular and require full mode. Coarser physics can change local deformation,
contact and tearing even though the visual Gaussian count is unchanged.

`report.json` records source/reduced particle counts, contact spacing and mass
for each berry. `benchmark_physics.py` accepts the same resolution option for
physics-only pick, squash and compression comparisons.

Measured on an RTX 6000 Ada, with all four v2 berries, EBC, 1280×720 rendering
and the default 2× scripted place/loop sequence (721 steps, excluding the first
10 warm-up frames):

| Resolution | Physics particles | Physics step time per frame | Compute FPS |
| --- | ---: | ---: | ---: |
| full | 11,052 | 59.1 ms | 9.38 |
| half | 5,539 | 47.0 ms | 10.91 |

Both retained 582,187 berry Gaussians and the same per-berry mass to floating-point
precision, placed the raspberry fully inside the bowl, and passed native render
readback and all-berry reset checks. This was a roughly 16% overall speedup, not
2×: grid, shading and rendering work remain. Timings exclude capture overhead and
interactive pacing; physics and rendering overlap, so these are loop timings,
not isolated GPU-kernel timings. Treat this as a local benchmark, not a guaranteed
frame rate or proof of identical fine-scale damage behavior.

The isolated raspberry pick benchmark lifted 49.4 mm at full resolution and
48.7 mm at half resolution, with zero mean damage during the hold in both cases.
The deliberate squash benchmark reached mean damage 0.164 versus 0.145;
half resolution still supported damaging compression but did not reproduce the
same quantitative damage/tear response. These are qualitative handling presets,
not resolution-independent calibrated material measurements.

To reproduce the comparison, run each preset sequentially with fresh output paths:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode place --berry all --asset_version v2 --background ebc --physics_resolution half \
  --steps 721 --loop --capture_every 660 --verify_render --verify_reset --output /tmp/berry-half-comparison
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/benchmark_physics.py \
  --berry raspberry --asset_version v2 --physics_resolution half --mode pick --output /tmp/berry-half-pick
```

Repeat with `--physics_resolution full` and different output paths; replace
`--mode pick` with `--mode squash` for the damaging-contact comparison.

## Two or three interacting raspberries (experimental)

Use `--count 2` (or its alias `--pair`) for two raspberries with mutual contact, unlike the independent
`--berry all` mode. Both tissues share one MPM solver with separate velocity
fields, equal-and-opposite contact impulses and Coulomb friction. Each retains
its own deforming Gaussian appearance. No additional assets are required.

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --berry raspberry --pair --asset_version v2 \
  --physics_resolution half --background ebc
```

The default `--pair_layout plate` places both in the starting punnet. The legacy
layout name is retained for CLI compatibility. Teleop can manipulate
either; scripted modes target `raspberry_1`. Use `--pair_layout bowl` to start
the second raspberry inside the receiving bowl, then place the first onto it:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode place --pair --pair_layout bowl --berry raspberry --asset_version v2 \
  --physics_resolution half --background ebc --window --loop
```

`bowl` requires the EBC background. `--pair_layout drop --mode idle --window`
instead starts one raspberry 4 cm above the other. `--berry_friction 0.4`
controls berry-to-berry friction only, not finger friction; this is an empirical
demo parameter, not a measured fruit coefficient. Reset restores both bodies.
Reports contain per-instance states; their `finger_force_n` is explicitly marked
as the **coupled pair total**, applied to the robot only once.

Validation on RTX 6000 Ada with the commands above (721 steps, loop reset and
native Gaussian readback enabled) retained 367,206 visual Gaussians and 2,946
half-resolution physics particles, at approximately 17.4 compute frames/s.
The released berry displaced the second berry; both settled entirely inside
the bowl with zero mean damage. This is a different workload from the four-berry
benchmark, not a direct speed comparison.

Standalone contact/control probes:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/benchmark_interactions.py \
  --case push --contact on --physics_resolution half --output /tmp/berry-pair-push
```

Repeat with `--contact off` and a fresh output directory, or use `--case separate`
and `--case drop`. The push probe transferred motion to the stationary berry;
the disabled control passed through it. Separation probes and contact unit tests
checked non-attraction, momentum conservation and no contact-induced energy gain.

### Three in the punnet: damage, discard, then gentle handling

Use `--count 3` to start three interacting raspberries in the punnet:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --count 3 --berry raspberry --asset_version v2 \
  --physics_resolution half --background ebc
```

The scripted `sort` mode defaults to three raspberries and requires EBC:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode sort --asset_version v2 --physics_resolution half --background ebc --window --loop
```

All three start in the punnet, now with its long axis along Y (rotated 90°).
Coupled two/three-raspberry EBC punnet layouts use separated random positions and
full 3D orientations by default. `--layout_seed 0` is reproducible; choose another
nonnegative seed for a different arrangement, or `--fixed_layout` for the ordered,
unrotated fruit layout. Reset/loop replays the same arrangement. Tissue, Gaussian
orientations and SH coefficients receive the same spawn rotation.

The robot deliberately compresses the first to a
1 mm commanded aperture, carries the damaged tissue to the shallow metal reject
dish, and releases it. It then grasps the second and third gently
and deposits them in the glass bowl. The actual pad gap can differ under load.
The discarded tissue remains in the simulation throughout; nothing is deleted,
teleported, attached to the gripper, or assigned artificial damage. The reject
dish has a real supporting floor and open collision walls, not a target marker.

The arm waits for measured TCP arrival at approach, lift, transfer and lowering
boundaries before proceeding; in particular, it does not open on a timer while
still travelling. Smooth arm ramps reduce abrupt starts/stops. For accepted berries,
early downward slip triggers bounded extra closure (1 mm/s, at most 18% of the
initial grasp aperture); `--pick_gap` disables this automatic closure. Larger slip
pauses travel, and a lost grasp pauses the sequence with an `R` retry prompt.
This is physical contact feedback, not an attachment or a guarantee for every pose.
Timing therefore includes travel-dependent waits in addition
to the nominal 69 seconds at default speed 2. The viewer shows the current phase;
reports include per-instance states and a `sorting_result` after a completed cycle.
This result checks discard containment and damage, plus bowl containment and low
damage for the accepted berries; an incomplete run reports `null`, not success.
`--loop` resets all three only after the entire sequence. Without it, the final
scene is held for inspection. `R` restarts the sequence.

The earlier ordered-layout punnet/reject-dish validation used 4,419 tissue particles and retained all
550,809 visual Gaussians, at approximately 13.4 compute frames/s on RTX 6000 Ada.
Both accepted berries settled entirely inside the bowl with mean damage below
0.0001; all of the damaged berry's particles settled inside the reject dish,
with mean damage about 0.074. These dimensionless damage values are model state,
not a calibrated food-quality measurement. Some damaged particles can scatter;
this is not a clean, fragment-free sorting guarantee.

Randomized layouts with seeds 0, 1 and 2 were also tested at default speed 2,
v2 assets and half physics resolution: both accepted berries ended entirely in
the bowl (mean damage below 0.00061), with over 99.5% of rejected particles in the
reject dish. Placement-only checks covered 100 seeds; this does not imply that
every random orientation is a reliable grasp.
Seed 0 also passed a rendered speed-5 run with render/reset verification: both
accepted berries fully in the bowl, all rejected particles in the reject dish,
no lost-grasp pause, and mean accepted damage below 0.000007. At 960 × 720 this
run averaged 13.4 compute frames/s on RTX 6000 Ada, excluding screenshot cost.

“Broken” here means physically crushed/damaged tissue with permanent deformation;
it is not a guarantee of clean separation into independently colliding fragments.

**Limitations:** coupled modes support two or three copies of raspberry only;
three requires the legacy-named `plate` (now punnet) layout. `--berry all` still means four different species
in independent, non-interacting solvers, not this coupled group.
Contact is approximate, grid-based tissue contact, not Gaussian collision;
contact tolerances and results depend on resolution. Drop tests produced contact
and rolling apart, not a stable stack. Dense piles, sustained stacking and
fractured-fragment contact have not been validated.

## EBC tabletop showcase

Add `--background ebc` for a freestanding lab table near the EBC coffee counter,
with a clear ribbed starting punnet, an open glass receiving bowl and a small
metal reject dish. The glass bowl is unchanged. The robot no longer sits
on the scanned counter.

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --background ebc --asset_version v2
```

This uses `background/ebc/aligned.usda` and `point_cloud.usd`, fetched and cached the
same way as the berry USDZ packages (see step 2); no separate download or Isaac Sim
installation is needed. The task references the original room Gaussian arrays,
normalizes their render transform/hints and binds the SH shader already packaged in
the berry USDZ. Source assets are not modified. The room is static, with no per-frame
Gaussian uploads; its approximately 7.2 million Gaussians still add GPU memory/render
cost.
`--background studio` remains the lightweight default.

EBC starts with a fixed **punnet and bowl** view, so a 2 cm berry remains legible.
Use `--view scene` for the whole table/robot/room, `--view workcell` for the
tabletop, or `--view berry` for a following macro view. The viewer has buttons
for all three. **Follow berry** can still be toggled independently.

The scan is positioned around the workcell: robot base and table top
stay at task `z=0`, with no change to robot scale, berry size, IK controls or
finger-contact parameters. The berry starts on the punnet's 4 mm-high inner floor.
The punnet is 14 × 11 cm and 2.8 cm tall, with rounded corners, external molded
ribs and a rolled flange. Its thin-plastic appearance uses an opacity-based
real-time approximation rather than solid-glass refraction. Decorative ribs and
the flange are omitted from collision; smooth rounded walls contain the tissue.
The glass bowl remains 12 cm across and 4.5 cm tall. The metal reject dish is
5.6 cm across and 1.8 cm tall. These are fixed props, not movable rigid objects.
Smooth visual meshes are
authored by task code; no extra asset files or uploads are required.
EBC uses softer task lighting and eight total/specular-transmission bounces for
the solid glass. The bowl still reads dark against the table in this real-time
renderer; its appearance needs another art-direction pass before a final GTC
capture. These changes do not affect studio lighting or fruit physics.

Robot contact uses the table's authored collider and separate base/wall proxies
for the props. Berry contact uses analytic disks, annular walls and rounded-rectangle shells at
both the MPM grid and particle levels, with friction and no adhesion. There is
no convex hull or invisible lid across the bowl opening. The MPM domain and IK
workspace extend sideways to cover the transfer; reports include
`fraction_in_bowl`. Room furniture remains visual-only. The local MPM tabletop
plane does not model falls off the table edge.
The EBC grid includes an additional 4 cm on its negative-X/Y (discard) sides for
crushed-tissue spillover. This increases grid allocation, not particle count or
grid spacing; active-node stepping and integration rates are unchanged. The
domain is still finite. On a solver failure, the exception includes particle
bounds, safe grid bounds and the world offset instead of only an error counter.
The default-seed, v2/half-resolution sort loop was checked for 9,001 headless
steps (three complete cycles plus part of a fourth), including reset verification.

For a GTC presentation, a useful sequence is **gentle pick → lift → place**, then
a reset and **deliberate squash**. Use the room view to establish the setting and
the macro view to show deformation; a full-room shot alone hides berry detail.
Keep the distinction between this qualitative material demo and calibrated fruit
properties—these are empirical handling presets, not measured force limits.

```bash
# Repeatable gentle transfer, with an interactive window:
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --background ebc --asset_version v2 --mode place --window --loop

# Record one transfer, including release and gripper retreat (fresh output path):
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --background ebc --asset_version v2 --mode place --steps 931 \
  --video --verify_render --verify_reset --output /tmp/berry-place-capture

# Contrast with deliberate compression in close-up:
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --background ebc --asset_version v2 --mode squash --view berry --window --loop
```

Arm travel is 2× faster by default. `--motion_speed 1` restores the previous
speed; `--motion_speed 3`, `4` or `5` requests the corresponding arm speed multiplier.
Any finite positive multiplier is accepted. This applies to gamepad/keyboard
translation and rotation and scripted arm travel, not gripper opening/closing,
settling waits, or the physics timestep. Actual speed depends on robot tracking
and simulation performance; higher speeds can disturb a grasp.

At the default speed, `--loop` resets pick every 15 simulated seconds, squash
every 16.5 seconds, and place every 23 seconds. With `--motion_speed 1`, these
remain 18, 18 and 32 seconds respectively.
The place sequence lifts clear of the bowl rim, translates, lowers, releases and
retreats. Reset restores the berry in the punnet and clears damage/contact history.
Gamepad mode remains fully operator-controlled, with the same grasp mapping.

If other Newton tests are running, use a dedicated `WARP_CACHE_PATH` to prevent
their cache cleanup from deleting in-progress teleop compilations. On machines
with a full home cache, `UV_CACHE_DIR` can also point to another filesystem.

## Controls

- Hold **LB/L1** to enable gamepad motion and grasp commands.
- Left stick: XY translation. Right stick: Z translation and yaw. D-pad: roll/pitch.
- Default full-axis translation command: 90 mm/s; stick yaw/keyboard rotation:
  1.2 rad/s (D-pad: 0.96 rad/s). `--motion_speed 1` halves these commands.
- **RT/R2 closes**, **LT/L2 opens** continuously. Trigger pressure controls speed,
  up to 12 mm/s total aperture change. Release triggers to hold the commanded opening.
- Menu/Start resets robot and tissue. Release and press LB again after reset.
- Keyboard: **W/S, A/D, Q/E** translate; **Z/X, T/G, C/V** rotate; **K** closes,
  **J** opens, **R** resets. Focus the render window.

This is **position control, not force control**: holding a compressed opening can
continue deforming the berry. Opening the fingers relieves compression. There is
no automatic stop-at-contact behavior. The viewer has a follow-berry camera toggle.

## Scripted checks

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode idle --no_render --steps 30 --output /tmp/berry-smoke
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode pick --steps 540 --verify_reset --output /tmp/berry-pick
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode squash --steps 480 --output /tmp/berry-squash
```

Choose fresh output directories. Add `--window` for interactive scripted viewing,
or `--video` to record (requires ffmpeg). Scripts without `--window` render offscreen.
`--verify_render` checks native Gaussian publication; `--render_aa dlaa --rtpt_spp 4`
are optional sampling overrides, not guaranteed appearance fixes.

## Handling physics and validation

The default `--physics_profile handling` uses firmer tissue and continuous contact
against the analytic pad surfaces. `--physics_profile legacy` restores the previous
material and grid-node contact for comparison. Neither option modifies the USDZ;
both work with `--asset_version v1` and `v2`.

Handling Young's moduli are 12 kPa (raspberry), 13.5 kPa (blackberry), 15 kPa
(blueberry), and 18 kPa (strawberry). Poisson's ratio is 0.4. Damage-induced
softening is reduced, and raspberry tearing starts later in accumulated plastic
strain. Deliberate closing can still bruise, permanently deform, and tear tissue.
The integration rate is increased only as needed for CFL stability (raspberry:
6120 Hz versus the previous 5040 Hz). Explicit `--mpm_hz` overrides must still
satisfy the solver's stability check.

The new contact uses a finite particle radius of half the tissue spacing,
compressive spring/damper normal forces, and tangential spring history capped by
Coulomb friction. This supports static friction without requiring sustained slip
or depending on whether an MPM grid node lies inside a finger. The pad coefficient
remains 1.2. Contact history clears on separation; normal contact cannot pull the
berry toward a finger. The separate, damage-activated wet adhesion from the POC
remains available for crushed tissue. Reaction impulses still feed back into the
robot. Opening the fingers releases an undamaged berry.

The scripted `--mode pick` centers on the settled tissue and defaults to an opening
of 85% of its width. Override with e.g. `--pick_gap 0.022` (metres). Teleoperation
remains fully operator-controlled: triggers command opening, not an automatic
force clamp. Reports include actual pad separation, tissue extent, mean tearing,
and the effective material settings.

For reproducible contact-only tests without a robot or renderer:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/benchmark_physics.py \
  --berry raspberry --asset_version v2 --gap 0.022 --output /tmp/berry-contact-test
```

Repeat in new output directories with `--grid_shift 0.001` (grid-only alignment
control), `--physics_profile legacy`, or `--friction 0` (no dry grip). `--mode squash`
closes to 1 mm and reopens; `--mode compression` presses an 18 mm square tool down
to 20% of the initial height. The benchmark checks finite state, inversions, and
domain bounds, and records force, deformation, damage, lift, hold drift, release,
and compute time in `report.json`. It does not automatically certify grasp success.

Validation with the local v2 assets lifted all four berries approximately 49 mm
with the Franka, held them, released them, and passed exact reset checks. The v2
raspberry used a 20.04 mm commanded opening with zero modeled damage during the
grasp; the legacy physics slipped at a comparable 20.16 mm command. The original
v1 raspberry also lifted at a 22 mm command. Separate pad tests covered multiple
sub-cell grid shifts, and zero-friction control trials did not carry the raspberry.
Deep closure still produced permanent deformation and damage. These checks do not
establish robustness across arbitrary approach poses or validate real-world forces.
After adding a conservative distant-pad rejection check, a sequential 451-step
raspberry v2 pick comparison at 1280x720 on an RTX 6000 Ada averaged 31.1 compute
FPS for handling versus 31.4 for legacy (capture/pacing excluded). Timing varies
with machine load, assets, and renderer settings; this is not a universal FPS guarantee.

Contact covers the inner fingertip pads, not the entire arm. These are empirical
handling presets, informed qualitatively by the supplied raspberry compression
video, not a calibrated force–displacement fit or measured fruit properties.
Irregular shape, approach alignment and narrow contact area can still cause slip.
The video used a rounded wooden indenter; the square-tool benchmark is not an
exact reproduction of that experiment.

All prepared exterior/interior Gaussians and degree-three SH data are retained.
The runtime reads Gaussian appearance, tissue arrays, and base simulation settings
directly from `<berry>/<berry>.usdz`, then applies the task's selected physics preset.
No NPZ or JSON asset files are needed. The USDZ
contains a USD layer with `/Berry/Gaussians`, `/Berry/Tissue`, typed `/Berry/TaskData`
metadata, and the MDL shader. Renderer resources resolve inside the USDZ archive.
Importing the USD in a generic Gaussian-capable viewer displays the
appearance; the task's MPM adapter instantiates the tissue simulation.

The raspberry SH coefficients were rebuilt from the original scan and rotated
through the settled material frame (about 59 degrees), a step missing from the
original `raspberry-squash` preparation. Geometry, opacity, interiors and physics
were preserved. `assets/sh_rotation.py` and `offline/rebuild_appearance.py` contain the offline
repair; the runtime performs no coefficient fitting or additional per-frame work.
The asset-repair/packaging pipeline itself (rebuilding a `*_simready.usdz` from raw
scans) is out of scope for this standalone task and is not included here; publish a
replacement bundle with the layout and manifest tooling described in step 2 above.

## Inspecting simready assets

Inspect six identical camera views of a prepared berry asset with the standalone
Newton RTX viewer:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/offline/inspect_simready.py \
  --asset /path/to/new/berry/strawberry/strawberry_simready.usdz \
  --output /tmp/strawberry-full --variant full
```

Use fresh output directories with `--variant skin` (repaired exterior only) or
`--variant scan` (retained scan Gaussians only) to compare contributions. The
same prepared pose, lighting and cameras are used for every variant.
