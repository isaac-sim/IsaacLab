# Rizon–Sharpa coffee IK demo

A Rizon4s arm and Sharpa Wave hand scoop coffee with a shallow spoon, pour it
into a cup, and optionally stir using scripted IK.

```bash
uv run isaaclab demo rizon-sharpa-coffee
```

The default is a dry Newton GL scene with a 4 mm grid and five visual grains
per physical particle. Add `--water` for a filled cup, physical coffee-to-liquid
conversion, conservative colour diffusion and stirring. Kit and Newton RTX
load the same full kitchen presentation; use `--water --visualizer kit` or
`--water --visualizer newton_rtx`. Both support `--backdrop studio` for the
plain studio. Newton GL keeps its tabletop scene. `--visualizer none --headless`
runs physics without presentation.
Use `--help` for voxel, solver-rate and rendering options. Material properties,
tool margins and solver iteration limits belong to `CoffeeDemoEnvCfg`.

This is a demo: a complete visual showcase, like `teapot-fill`. Focused examples
show an individual API or tuning method. This demo has no learning task or
policy registration.

## Code layout

| Files | Responsibility |
| --- | --- |
| `demo_env_cfg.py`, `demo_env.py` | Scene, solver/render configuration, IK actions and resets |
| `assets.py`, `robot.py` | Asset dimensions, robot joint contract and measured reset pose |
| `physics.py`, `dissolution.py`, `attachment.py` | Coupled physics, soluble coffee and tool mounting |
| `scripted.py` | Scoop, pour and stir state machine |
| `rendering.py` | Shared grain, colour and native liquid-surface state for Newton viewers |
| `kit_rendering.py` | Optional Kit USD publication, loaded after app startup |

Native Newton reconstructs and smooths the surface using bounded reusable
storage. The rendering code adds the local coffee colour. Kit stays separate
because its runtime cannot be imported during configuration. Its camera
session-layer handling is local to this demo. The shared Newton visualizer only
provides a viewer-subclass hook for publishing liquid colours during the native
RTX update.

The custom MPM solver retains its manager's USD configuration preparation and
only specializes solver construction. Native RTX material paths and updates
belong to the coffee viewer; its adapter still depends on the pinned Newton
runtime API.
Kit and RTX reuse one render-only backdrop through the simulation's existing
resource registry. It remains alive until the viewers close. Both use the
authored refracting water material, with local coffee colour driving optical
absorption. GL uses its native textured surface.

## Assets and runtime

Use Isaac Lab's uv environment, Newton and a CUDA-capable NVIDIA GPU. Standalone
RTX needs the `ovrtx` extra; Kit needs an Isaac Sim runtime. Assets are loaded
through the existing USD spawner and asset resolver.

The fixed bowl, cup, spoon, table and prepared kitchen are supplied at
`{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaCoffee`. The bundle includes
`walnut.png`, `CoffeeWater.mdl`, `kitchen.usda`, the collected kitchen geometry,
materials and textures in `Kitchen/`, the prepared robot in `Robot/`, licenses
and checksum manifests.
Props are authored in metres with Z up; their default
prim is `/Asset`. The exact concave spoon triangles and material groups are
required for MPM contact. Assets load without runtime generation.

For the internal development bundle, override the asset root with:

```bash
export ISAACLAB_COFFEE_ASSET_ROOT=omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Environments/RizonSharpaCoffee
```

The variable also accepts a local copy of the complete bundle. Preserve relative
paths when copying it. A missing or inaccessible bundle fails through the
standard asset loader; the demo does not generate replacements.

The prepared Rizon/Sharpa robot reuses the licensed Fabrics-Sim USD and textures
from the RJ45 insertion scene, with two malformed visual normals arrays blocked
in a separate USD layer. It loads from `Robot/` under the selected bundle root.
`ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` can select a separate robot folder.
That folder must include `rizon4s_sharpa.usda`, its source USD and `textures/`.
The source revision is `d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`; its manifest
records the source fingerprint and file checksums. The robot retains its original
licensing and access restrictions and is not part of the BSD-licensed coffee
props. Hosting does not change the source asset's licensing terms.

The optional kitchen is prepared from NVIDIA's public
`Isaac/Environments/replicator_kitchen/kitchen_l_island.usda`. Its placement,
lighting, materials and exclusions are baked into the companion USD, with
textures and MDL dependencies collected by Kit's native asset collector.
It loads after physics construction. The sample island, apple and decorative
coffee cups remain hidden, and collisions are disabled. The backdrop adds
rendering memory but no solver bodies, contacts or particles. The demo does
not generate or modify the kitchen asset at runtime.

## Control and recording

The spoon follows the measured palm through a kinematic mount, with the tool
offset configured in `CoffeeDemoEnvCfg`. Physical grasp acquisition and
retention are separate tasks.

The controller commands a 29-value action: robot-base palm XYZ, XYZW quaternion,
then 22 finger angles in the named order declared by `robot.py`. IK uses Isaac
Lab's differential IK action and DLS controller. Finger commands use its normal
joint-position action. Scripted waypoints enter steeply, flatten inside the
coffee, lift, pour and optionally stir. The camera follows smoothly.

`--video` uses Isaac Lab's `VideoRecorderCfg` and defaults to Newton RTX.
Install the recording and renderer extras for the selected viewer:

```bash
uv run --extra ovrtx --extra video isaaclab demo rizon-sharpa-coffee --water --video
uv run --extra video isaaclab demo rizon-sharpa-coffee --water --video --visualizer newton_gl
uv run --extra isaacsim --extra video isaaclab demo rizon-sharpa-coffee --water --video --visualizer kit
```

The selected visualizer supplies the camera and resolution, including headless
capture. `--video_fps` controls cadence and `--output_dir` selects the destination
(default `logs/coffee_video`). Clips are named `coffee_0000.mp4`, incrementing
on later runs. The native recorder buffers frames until encoding, so long or
4K captures require sufficient host memory. This demo adds no encoder,
snapshots, overlays or rollout-report files.

## Reusing the simulation

`CoffeeDemoEnv` derives from the non-learning `ManagerBasedEnv`. Its configuration
separates `CoffeeSceneCfg`, native action terms, reset events and coupled Newton
physics. There is no Gym registration, policy observation, reward or PPO setup.
MPM owns coffee-to-liquid conversion independently of presentation. Rendering
grains are passive and do not increase physical resolution; surface filtering
changes appearance without changing particle mass. Liquid-level fidelity still
depends on grid resolution and pressure convergence.

Newton's `ParticleSurface` owns reconstruction, anisotropy and spatial smoothing.
Its native active-particle mask excludes dry coffee. The demo adds coffee colour
transport and renderer publication of those colours.
Materials and recording use Isaac Lab's existing spawners and recorder.

To create an RL task, copy `demo_env.py` and `demo_env_cfg.py` and change their
bases to `ManagerBasedRLEnv` and `ManagerBasedRLEnvCfg`. Adapt the constructor to
accept and forward `render_mode` and `**kwargs`, update the copied package paths
and `class_type`, and remove the explicit startup dispatch from `load_managers()`:
the RL base owns that dispatch. Add task-specific observations, rewards,
terminations, horizon, registration and an agent configuration separately.
Reuse the scene, physics, reset logic and measured pose/delivery queries.
Presentation supports one demo world and should be disabled for batched learning.
The demo uses argparse and builds robot and prop configurations from its dimensions.
Nested scene asset overrides are rebuilt by `configure_scene`; preserve those
overrides explicitly when adapting it into a Hydra-configured learning task.
