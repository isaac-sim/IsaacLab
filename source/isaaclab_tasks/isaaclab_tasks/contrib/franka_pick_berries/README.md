# Berry picking

The implementation, CLI and installer are entirely self-contained in this task
directory; no external task checkout or sibling contrib task is required. Berry
USDZ packages and the EBC background are fetched on demand from Nucleus through
Isaac Lab's own asset cache (`isaaclab.utils.assets.retrieve_file_path`); there is
no separate asset-download step.

## Layout

```text
franka_pick_berries/
  pick_berries_env.py, pick_berries_env_cfg.py   environment class and configuration
  config/franka/                                 gym registration
  mdp/                                           action terms
  physics/                                       Newton coupled physics, implicit and explicit tissue solvers, tissue
  rendering/                                     interactive viewer, per-berry Gaussian streams and binding
  scene/                                         table, tableware props and the EBC room background
  assets/                                        asset locations and USD/USDZ loading
  control/                                       gamepad devices and scripted arm motion
  scripts/                                       teleop / scripted-demo CLI
  profiling.py                                   optional Tracy zones and renderer profiling
  offline/                                       asset inspection and appearance tools (not needed to run the task)
  setup/                                         runtime installer (setup.sh)
```

## Physics

The task runs on Isaac Lab's Newton backend through the coupler in `isaaclab_contrib.coupling`
(`physics/coupling.py`): the Franka on MuJoCo-Warp, the berry tissue on one of two MPM solvers selected with
`--tissue_solver`. Each berry is an `MPMObject` spawned from the tissue particles in its USDZ, with the stiffness of
its species (`physics/materials.py`): Young's moduli of 12 kPa (raspberry), 13.5 kPa (blackberry), 15 kPa
(blueberry) and 18 kPa (strawberry), and a Poisson's ratio of 0.4. These are empirical handling presets, not
calibrated fruit properties. Each solver module contains all of its physics, including the tissue's yield limits,
plasticity and damage (plastic-strain damage, tearing and bruising), and exposes the same interface to the task.

### Explicit solver (default)

By default (`--tissue_solver explicit`) the tissue runs on `SolverGraspExplicitMPM` (`physics/grasp_explicit_mpm.py`),
an explicit MPM entry of the Newton coupler: MLS-MPM transfers with APIC, fixed-corotated elasticity with a
log-strain return mapping, and penalty contact with the finger pads whose tangential springs are capped by Coulomb
friction. It grips by friction, so a held berry can slip and the robot feels its reaction. Each berry is its own
velocity field, in frictional contact with the others. It is faster than the implicit solver for this scene
(rendered three-raspberry place on the EBC background: about 14 frames/s, against 9.5). The module docstring lists
the methods and their references.

### Implicit solver (opt-in)

`--tissue_solver implicit` runs the tissue on `SolverGraspImplicitMPM` (`physics/grasp_implicit_mpm.py`) instead,
a subclass of Newton's implicit MPM selected through the MPM manager's `solver_class`, with the fingers and the table
exposed to the tissue as proxy colliders. In Newton's implicit MPM the stress with which a pinched solid pushes back on two
fingers drains away while they hold it, even though its elastic deformation is kept, so a berry held only by
friction slips out when the hand carries it. While the gripper holds an aperture (it is commanded close to its
current opening, short of fully open, and its fingers are still), tissue particles pressed against a finger are
therefore clamped to it as Newton kinematic particles, as in Newton's `example_mpm_beam_twist`; the rest of the berry
hangs from them and deforms as MPM tissue. Moving the fingers releases them: closing compresses, and can squash and
damage, the berry through contact alone, and opening lets go of it. A held berry therefore does not slip, and the
robot does not feel its weight. Its damage is a weak, qualitative signal: the velocity gradients it integrates are
noisy, and damage must grow slowly (`DamageConfig.max_rate`) for its softening not to make the solve diverge, so a
full squash crushes the berry but marks it far less than the explicit solver does. The tissue uses the static
tableware colliders, which a shape can belong to one coupled entry only, so with this solver the arm itself does not
collide with the punnet, bowl and reject dish; with the explicit solver it does.

## Follow-up work

1. **Publish the assets publicly.** The berry USDZ packages and the EBC background are still read
   from a personal Nucleus folder (`assets/asset_root.py`). Publish them in the public repository
   that hosts the other assets used by the codebase and update the default URL.
2. **Use the codebase's standard Gaussian binding.** `rendering/binding.py` carries four custom
   Gaussian-to-particle binding modes, and only the MLS modes work with `BerryGaussianStream`
   (`affine4` does not). Replace it with the standard binding.
3. **Implicit grip by friction.** Once Newton's implicit MPM keeps the grip stress of a pinched solid,
   replace the implicit solver's clamped grasp with frictional contact, as in the explicit solver.
4. **Interacting berries.** With the implicit solver, berries sharing the MPM grid interact as sticky
   contact, unvalidated; the explicit solver gives them frictional contact. The previous task's scripted
   sort into the reject dish was not ported.
5. **OVRTX 0.5.** Find why live Gaussian updates can become invisible with the public 0.5 renderer.

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
CLI options. Physics and Gaussians run on CUDA device 0 (`--device` selects another);
an RTX GPU is required for rendering.
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
damage and Gaussian shading.
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

**Limitations:** the berries share one MPM grid, so tissue of two berries that touch interacts
through the grid, as sticky rather than frictional contact; berry-to-berry contact has not been
validated, so do not use this mode for piles. Four berries cost more GPU time and memory than one;
use the studio background for a lighter scene.

## Two or three raspberries

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --count 3 --berry raspberry --asset_version v2 --physics_resolution half --background ebc
```

`--count 2` or `3` places that many berries of the selected species in the punnet, as `raspberry_1`,
`raspberry_2`, ... With the EBC background they are scattered at separated random positions with random
3D orientations; tissue, Gaussians and their spherical harmonics turn together. `--layout_seed 0` (the
default) is reproducible; choose another nonnegative seed for another arrangement, or `--fixed_layout` for
an ordered, unrotated layout. Reset replays the same arrangement.

Teleop can handle any of the berries; scripted modes handle `raspberry_1`, for example `--mode place
--count 3` carries it to the bowl and leaves the others in the punnet, though the gripper can nudge a close
neighbor. The same limitations as for all four berries apply.

`--mode sort` handles all three in sequence on the EBC table: it crushes the first berry and drops it in the
reject dish, then picks the other two gently, one after the other, and places them in the glass bowl. A grasp that
slips is tightened slightly; one that loses the berry stops the sequence until reset (R). The report's
`sorting_result` checks the outcome: the first berry damaged and in the reject dish, the others in the bowl and
barely damaged.

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode sort --count 3 --berry raspberry --asset_version v2 --physics_resolution half --background ebc --window --loop
```

### Lower-resolution physics

Add `--physics_resolution half` to keep roughly half the tissue particles while
retaining every visual Gaussian, including repaired skin and interior:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --berry all --asset_version v2 --background ebc --physics_resolution half
```

The default `--physics_resolution full` uses every tissue particle in the asset.
The half preset selects alternating sites of the tissue lattice and compensates
particle volume to preserve total mass; Gaussian bindings are rebuilt against the
reduced tissue. The USDZ files are not modified.

Use v2 assets: all four contain supported regular-lattice tissue. Irregular or
multi-region custom proxies are rejected in half mode rather than silently
changing their material distribution. The older raspberry/strawberry v1 proxies
are irregular and require full mode. Coarser physics changes local deformation,
contact and damage even though the visual Gaussian count is unchanged.

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
the solid glass. The bowl's normals are smooth around its axis, so its 128-sided
mesh refracts as a round bowl rather than as facets; along its outline they stay
sharp except at the rounded foot and lip. The bowl stands just above the table's visible top, so no glass
face is buried in or coincident with it, and casts no shadow: RTX shadow rays do
not refract, so solid glass would otherwise leave the table under it black.
These changes do not affect studio lighting or fruit physics.

The table, punnet, bowl and reject dish are static colliders for both the robot and the
tissue; decorative ribs and the punnet flange are visual only, and there is no lid across
the bowl opening. Room furniture remains visual-only. Reports include `fraction_in_bowl`.

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
retreats. Reset restores the berry in the punnet and clears its damage and grasp.
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

This is **position control, not force control**: closing further compresses and can
damage the berry. The explicit solver grips by friction; with `--tissue_solver implicit`, holding an
aperture clamps the tissue the fingers press instead, and closing further or opening releases it (see
[Physics](#physics)). There is no automatic stop-at-contact behavior. The viewer
has a follow-berry camera toggle.

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
`--verify_render` checks native Gaussian publication; `--rtpt_spp 4` is an optional
sampling override, not a guaranteed appearance fix. Antialiasing keeps the renderer's default.

## Profiling with Tracy

`--tracy` profiles a rendered run with [Tracy](https://github.com/wolfpld/tracy). OVRTX ships
Carbonite's profiler with a Tracy backend, and the option starts it with the renderer through
Carbonite settings (`profiling.PROFILER_SETTINGS`). It records the renderer's CPU zones (stepping,
Hydra and RTX rendering, scene population, MDL loading) and GPU zones for each RTX render pass,
from Vulkan timestamps. Its capture mask is 3, which adds detail zones to OVRTX's default of 1.
`--tracy_setting` passes more settings, for example `--tracy_setting=--/app/profilerMask=7`.
The task adds its own CPU zones through the same client:

| Zone | Measures |
|---|---|
| `env: step` | the environment step, including its four physics steps |
| `physics` | one 120 Hz physics step (a CUDA graph launch, unless `--tracy_sync`) |
| `coupled solver`, `arm: SolverMuJoCo`, `tissue: SolverGrasp...MPM` | the coupled step and its entry solvers (`--tracy_sync` only) |
| `viewer: draw` | the frame's rendering work, including the zones below |
| `gaussians: deform and shade`, `gaussians: publish` | Gaussian deformation, shading, gather and upload |
| `ovrtx: wait for frame`, `ovrtx: step_async` | waiting for the previous frame, and submitting the next |
| `viewer: log state`, `viewer: end frame` | Newton's viewer updates of the renderer's transforms |

Frames are marked once per teleop loop iteration. Zones that open before the renderer starts
(environment creation and the CUDA graph capture) are not recorded.

The task zones time CPU work. Physics and rendering run asynchronously on the GPU, so a zone
measures the launch of its GPU work, not its execution. Unlike the RTX passes, the physics and
Gaussian CUDA kernels have no GPU zones: Carbonite times only graphics work. `--tracy_sync` synchronizes the device
at the end of each task zone and turns off the physics CUDA graph. With it, the solver zones appear
on every step and measure their GPU time, but the run is slower and the overlap between physics and
rendering is lost.

Use a Tracy viewer or capture tool whose protocol matches the bundled client: the `Tracy` or
`capture` binaries of Kit's `omni.kit.profiler.tracy` 1.2 extension. Tracy 0.12 cannot connect to
it. The client listens on port 8086:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/teleop_berry.py \
  --mode gamepad --count 3 --background ebc --tracy
# in another terminal, live:      <omni.kit.profiler.tracy>/bin/Tracy  (connect to 127.0.0.1)
# or record to a file to open later: <omni.kit.profiler.tracy>/bin/capture -o berries.tracy -f
```

## Asset format

All prepared exterior/interior Gaussians and degree-three SH data are retained.
The runtime reads Gaussian appearance, tissue arrays, and base simulation settings
directly from `<berry>/<berry>.usdz`, then applies the task's tissue material.
No NPZ or JSON asset files are needed. The USDZ
contains a USD layer with `/Berry/Gaussians`, `/Berry/Tissue`, typed `/Berry/TaskData`
metadata, and the MDL shader. Renderer resources resolve inside the USDZ archive.
Importing the USD in a generic Gaussian-capable viewer displays the
appearance; the task's MPM adapter instantiates the tissue simulation.

The raspberry SH coefficients were rebuilt from the original scan and rotated
through the settled material frame (about 59 degrees), a step missing from the
original `raspberry-squash` preparation. Geometry, opacity, interiors and physics
were preserved; the runtime performs no coefficient fitting or additional per-frame work.
The asset-repair/packaging pipeline (rebuilding a `*_simready.usdz` from raw
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
