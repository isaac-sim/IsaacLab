# Rizon–Sharpa coffee IK demo

A Rizon4s arm and Sharpa Wave hand scoop coffee with a shallow spoon, pour it
into a cup, and optionally stir. Scripted IK and Apple Vision Pro control use
one absolute palm-pose and finger-angle interface.

```bash
uv run isaaclab demo rizon-sharpa-coffee
```

The default is a dry Newton GL scene with a 4 mm grid and five visual grains
per physical particle. Add `--water` for a filled cup, physical coffee-to-liquid
conversion, conservative colour diffusion and stirring. For the kitchen
presentation, use `--water --visualizer kit`; `--backdrop studio` selects the
plain studio. `--visualizer none --headless` runs physics without presentation.
Use `--help` for voxel, solver-rate, tool-margin and rendering options.

## Assets and runtime

Use Isaac Lab's uv environment, Newton and a CUDA-capable NVIDIA GPU. Standalone
RTX needs the `ovrtx` extra; Kit needs an Isaac Sim runtime. Assets are loaded
through the existing USD spawner and asset resolver.

The fixed bowl, cup, spoon and table are supplied as a companion USD bundle at
`{ISAACLAB_NUCLEUS_DIR}/Objects/CoffeeScooping/v1`. The bundle includes
`walnut.png`, `CoffeeWater.mdl`, its license and a checksum manifest. The props
are authored in metres with Z up; their default prim is `/Asset`. The exact
concave spoon triangles and material groups are required for MPM contact.
Props and the prepared robot are loaded without runtime asset generation.

Until the development bundle is promoted to the public asset library, select
the internal release with:

```bash
export ISAACLAB_COFFEE_ASSET_ROOT=omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Objects/CoffeeScooping/v1
```

The same variable can point to a local copy of that bundle. Preserve relative
texture paths when copying it. A missing or inaccessible bundle fails through
the standard asset loader; the demo does not silently generate replacements.

The prepared Rizon/Sharpa robot reuses the licensed Fabrics-Sim USD and textures
from the RJ45 insertion scene, with the two malformed visual normals arrays
blocked in a separate USD layer. It is hosted separately for internal development:

```bash
export ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT=omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Contrib/RizonSharpaCoffee/Robot/v1
```

A local copy must include `rizon4s_sharpa.usda`, its source USD and `textures/`.
The source revision is `d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`; the hosted
manifest records source fingerprint and file checksums. The native spawner
loads the prepared asset without runtime USD authoring. This robot retains its
original licensing and access restrictions; it is not part of the BSD-licensed
coffee props. Public redistribution requires asset-owner approval. The public
Rizon arm asset does not include the Sharpa hand.

The optional kitchen references NVIDIA's public
`Isaac/Environments/replicator_kitchen/kitchen_l_island.usda` after the physics
model is constructed. Its collisions are disabled. The sample island, apple
and decorative coffee cups are hidden. The backdrop adds rendering memory but
no solver bodies, contacts or particles.

## Control and recording

The supported default mounts the spoon to the measured palm pose. The optional
`--grasp_mode contact` uses a free spoon, authored finger preload and Newton
proxy coupling; complete contact-mode manipulation remains experimental.
`--grasp_reset` accepts a named-joint reset JSON; reset-authoring tools are not
part of this demo.

Both controllers command a 29-value action: robot-base palm XYZ, XYZW quaternion,
then 22 finger angles in the named order declared by `robot.py`. IK uses Isaac
Lab's differential IK action and DLS controller. Finger commands use its normal
joint-position action. Scripted waypoints enter steeply, flatten inside the
coffee, lift, pour and optionally stir. The camera follows smoothly.

For live AVP input, install the `teleop` extra and configure the headset using
the [CloudXR guide](../../../../../docs/source/how-to/cloudxr_teleoperation.rst).
The pinned Sharpa kinematic model is external; follow
[IsaacTeleop's model setup](https://github.com/NVIDIA/IsaacTeleop/blob/c5fe6624cc4dff456485d2e786922c8e41100f83/examples/teleop_ros2/assets/urdf/sharpa_standalone/README.md).
Place `right_sharpa_wave.urdf` under
`~/.cache/isaaclab/sharpa/3e953f588ba9954cebaa720aaa4cee06a43a068e/`, or set
`ISAACLAB_SHARPA_RIGHT_URDF`. Run with `--mode avp`. Acquisition recenters to the
measured palm; tracking loss holds the last valid command. Live headset behavior
has not been validated. Retargeting provenance and licenses are in
[assets/NOTICE](assets/NOTICE).

`--video` uses Isaac Lab's `VideoRecorderCfg`; install the `video` extra. The
selected visualizer supplies the camera and resolution, including headless
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

To create an RL task, copy `demo_env.py` and `demo_env_cfg.py` and change their
bases to `ManagerBasedRLEnv` and `ManagerBasedRLEnvCfg`. Adapt the constructor to
accept and forward `render_mode` and `**kwargs`, update the copied package paths
and `class_type`, and remove the explicit startup dispatch from `load_managers()`:
the RL base owns that dispatch. Add task-specific observations, rewards,
terminations, horizon, registration and an agent configuration separately.
Reuse the scene, physics, reset logic and measured pose/delivery queries.
Presentation supports one demo world and should be disabled for batched learning.
