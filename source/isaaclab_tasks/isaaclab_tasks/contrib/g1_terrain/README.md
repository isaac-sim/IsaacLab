# G1 terrain traversals

Three single-robot examples run pretrained locomotion policies over uneven white
platforms (Block), wooden ramps (Ramp), and gravel followed by sand (SONIC).
They support Isaac Sim PhysX and Newton MJWarp, with RTX rendering and fixed cameras.

| Task | CLI name | Course |
| --- | --- | --- |
| `IsaacContrib-G1-Terrain-Platforms` | `platforms` | White platforms with 3–8 cm height differences |
| `IsaacContrib-G1-Terrain-Ramps` | `ramps` | Wooden modules with approximately 10° slopes |
| `IsaacContrib-G1-Terrain-Granular` | `granular` | Deformable gravel and sand beds, controlled by SONIC |

[EXAMPLE.md](EXAMPLE.md) describes the example script, checkpoint downloads, and commands.

## Run

From the Isaac Lab repository root:

```bash
uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task platforms physics=newton_mjwarp

uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task ramps physics=newton_mjwarp
```

Replace `physics=newton_mjwarp` with `physics=isaacsim_physx` for PhysX.
Platforms and ramps open a viewer without recording. They accelerate from 0 to 0.6 m/s
in 0.3 seconds and maintain the initial heading, with no lateral command or
keyboard input. The platform traversal runs for 7.76 simulation seconds; ramps
run for 8.68 seconds. `--seconds` overrides the duration.

The depth checkpoint and robot are fetched from `isaac-dev.ov.nvidia.com`; access to
that Nucleus server is required. No resource-directory argument is needed.
The environment configuration contains the default URLs:

- Checkpoint: `omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/IsaacContrib-Velocity-Rough-G1_Terrain_newtonmjwarp_none_rsl_rl.pt`
- Robot: `omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof.usd`

Use `--checkpoint /path/to/policy.pt` to select a local TorchScript export. The
robot path can be overridden with `env.scene.robot.spawn.usd_path=/path/to/g1.usd`.

Optional recording requires the existing `video` extra:

```bash
uv run --extra isaacsim --extra video python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task ramps --video --output logs/g1_terrain/ramps_newton physics=newton_mjwarp
```

Platforms and ramps require a window. `--width` and `--height` set the viewport
and capture resolution (default 1920 × 1080). Recording produces `video.mp4` at 50 fps.

SONIC records the granular traversal by default:

```bash
uv run --extra isaacsim --extra video --extra g1-terrain python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task granular --output logs/g1_terrain/granular_newton physics=newton_mjwarp
```

Newton records offscreen. Use `physics=isaacsim_physx` for the PBD version, which
requires a window to synchronize the robot and particle visuals.
`g1-terrain` installs ONNX Runtime GPU
and the Hugging Face downloader; `video` installs recording dependencies. The first
run downloads approximately 865 MB of SONIC models from
[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) into the Hugging Face cache.
The revision is pinned in `G1GranularEnvCfg`. No SONIC repository checkout is required.
The models retain their [upstream license](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/LICENSE).

The granular command holds at zero for 3 seconds, increases from 0.1 to 0.6 m/s in
0.5 seconds, and slows to zero between 12.6 and 17.1 seconds. The run lasts 23.04
simulation seconds. Particle simulation can be slower than real time; console
progress reports show both simulation and elapsed wall time. All commands and
model updates use simulation time, so slow rendering does not advance the planner.

## Configuration

`terrain.py` generates the course meshes and textures in a temporary directory.
Both platform and ramp lanes are present because they contribute to depth observations.
Dimensions and camera poses are estimates from the reference videos, not surveyed
measurements. There is no training entry point, observation noise, external push,
or automatic reset.

`depth_parameters.py` defines the joint order, default pose, action scales, history,
and depth normalization. The exported policy takes 690 proprioceptive values and
three 38 × 64 depth frames. Its 43 joint slots include 29 body motors and 14 finger
slots; finger feedback is held at the deployment default. No external observation
contract or ONNX dependency is required for these two tasks.

`terrain_env_cfg.py` configures the depth camera, PD gains, effort limits, spawn
poses, and backends. Control runs at 50 Hz, with a 2.5 ms solver cadence. Both
backends use the same robot asset and rigid meshes for platforms and ramps. The sole colliders
remain active but are hidden from the spectator view. Ramp-base visual surfaces
are separated after physics import to avoid flickering without changing contact
or depth geometry.
The policy's depth camera uses Warp ray tracing on both backends; the spectator
view uses RTX.

`sonic_policy.py` runs the released planner, encoder, and decoder. It resamples
30 Hz reference motion for the 50 Hz tracker, assembles observation history, and
maps the 29 outputs to joint targets. Replanning checks occur every 100 ms; a new
plan is requested on speed changes and periodically while walking. SONIC has no
depth input. `sonic_parameters.py` defines its joint order, action scaling, and PD gains.
ONNX Runtime inference is not bitwise equivalent to the upstream TensorRT deployment.

`granular.py` configures the two particle backends:

| Backend | Gravel | Sand |
| --- | --- | --- |
| Newton MJWarp + MPM | 3 cm bed, density 1800 kg/m³, friction 0.85, yield pressure 1 MPa | 3 cm bed, density 1600 kg/m³, friction 0.6, yield pressure 0.1 MPa |
| PhysX + PBD | 2 cm bed, 7.5 mm solid particles | 3 cm bed, 5 mm solid particles |

MPM uses a 1 cm grid and two material points per cell axis (5 mm nominal spacing).
Material points represent a bulk material, not individual stones. PBD uses particle
friction 0.8, damping 0.1, and zero cohesion; the two particle systems do not collide
with each other. MPM and PBD are different approximations, and their materials have
not been calibrated to measured sand or gravel properties.
PhysX particles use GPU dynamics with CPU state buffers for USD rendering readback.
The granular example uses SONIC PD gains with the simulation's configured effort
limits and armature; it is not a hardware deployment configuration.
