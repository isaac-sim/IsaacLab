# G1 terrain traversals

Two single-robot examples run a pretrained depth-based locomotion policy over
uneven white platforms (Block) and wooden ramps (Ramp). Both support
Isaac Sim PhysX and Newton MJWarp, with RTX rendering and fixed camera poses.

| Task | CLI name | Course |
| --- | --- | --- |
| `IsaacContrib-G1-Terrain-Platforms` | `platforms` | White platforms with 3–8 cm height differences |
| `IsaacContrib-G1-Terrain-Ramps` | `ramps` | Wooden modules with approximately 10° slopes |

## Run

From the Isaac Lab repository root:

```bash
uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task platforms physics=newton_mjwarp

uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task ramps physics=newton_mjwarp
```

Replace `physics=newton_mjwarp` with `physics=isaacsim_physx` for PhysX.
The examples open a viewer without recording. They accelerate from 0 to 0.6 m/s
in 0.3 seconds and maintain the initial heading, with no lateral command or
keyboard input. The platform traversal runs for 7.76 simulation seconds; ramps
run for 8.68 seconds. `--seconds` overrides the duration.

The checkpoint and robot are fetched from `isaac-dev.ov.nvidia.com`; access to
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

Playback and recording require a window. `--width` and `--height` set the viewport
and capture resolution (default 1920 × 1080). Recording produces `video.mp4` at 50 fps.

## Configuration

`terrain.py` generates the course meshes and textures in a temporary directory.
Both lanes are present because they contribute to the robot's depth observations.
Dimensions and camera poses are estimates from the reference videos, not surveyed
measurements. There is no training entry point, observation noise, external push,
or automatic reset.

`depth_parameters.py` defines the joint order, default pose, action scales, history,
and depth normalization. The exported policy takes 690 proprioceptive values and
three 38 × 64 depth frames. Its 43 joint slots include 29 body motors and 14 finger
slots; finger feedback is held at the deployment default. No external observation
contract or ONNX dependency is required.

`terrain_env_cfg.py` configures the depth camera, PD gains, effort limits, spawn
poses, and backends. Control runs at 50 Hz, with a 2.5 ms solver cadence. Both
backends use rigid terrain meshes and the same robot asset. The sole colliders
remain active but are hidden from the spectator view. Ramp-base visual surfaces
are separated after physics import to avoid flickering without changing contact
or depth geometry.
The policy's depth camera uses Warp ray tracing on both backends; the spectator
view uses RTX.
