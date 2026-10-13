# Download a policy and run a G1 terrain demo

The runnable example is [play.py](play.py). It downloads the pretrained policy,
launches the selected Isaac Lab environment, and runs a scripted traversal.
Users do not need to train a policy, clone SONIC, or assemble a resource directory.

## Run the example

From the Isaac Lab repository root, choose one command:

```bash
# Block: open the simulation viewer.
uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task platforms physics=newton_mjwarp

# Ramp: open the simulation viewer.
uv run --extra isaacsim python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task ramps physics=newton_mjwarp

# SONIC: record the gravel and sand traversal.
uv run --extra isaacsim --extra video --extra g1-terrain python -m isaaclab_tasks.contrib.g1_terrain.play \
  --task granular --output logs/g1_terrain/granular_newton physics=newton_mjwarp
```

Replace `physics=newton_mjwarp` with `physics=isaacsim_physx` to use PhysX.
For SONIC, Newton couples MJWarp rigid bodies with MPM; PhysX uses PBD particles.
An NVIDIA GPU is required. The default robot and depth checkpoint require access
to `isaac-dev.ov.nvidia.com`; SONIC weights are downloaded from Hugging Face.
The full MPM course with RTX rendering was tested on a 48 GB NVIDIA GPU; it has
substantially higher memory requirements than the rigid platform and ramp demos.

Block and Ramp open a window and do not record by default. SONIC saves `video.mp4`
in the output directory; Newton records offscreen, while PhysX requires a window
to synchronize robot and particle visuals. The initial stationary interval is
3 simulation seconds; it can take longer in wall time when simulating particles.
The console reports both clocks. No keyboard input is required.

## What the script does

1. Parse the task and backend, then resolve the registered environment configuration.
2. Resolve model resources. SONIC uses Hugging Face's `snapshot_download`
   with a pinned revision and a file allowlist: planner, encoder, decoder,
   observation configuration, and license. Downloads are cached for subsequent runs.
3. Launch the simulator and create the environment with `gym.make`. Terrain meshes,
   textures, and particle beds are generated in code; the robot comes from the
   configured USD URL. Block and Ramp fetch their TorchScript checkpoint through
   Isaac Lab's `retrieve_file_path` when initializing the policy.
4. At each 20 ms control step, read robot state, evaluate the velocity command,
   run policy inference, and pass joint-position targets to `env.step`.
   Block and Ramp also read depth images; SONIC generates reference motion with
   its planner and tracks it with its encoder and decoder.
5. Capture RTX frames when recording, then close the video writer and environment.

The script owns downloads, playback, and recording. Environment definitions live
in [terrain_env_cfg.py](terrain_env_cfg.py); policy adapters live in
[depth_policy.py](depth_policy.py) and [sonic_policy.py](sonic_policy.py).
The task provides demonstration playback only, with no training entry point.

## Options

- `--seconds 6`: run six simulation seconds, including SONIC's first steps.
- `--width 1280 --height 720`: change viewer and recording resolution.
- `--checkpoint /path/to/policy.pt`: override the Block/Ramp TorchScript checkpoint.
- Add `--extra video` and `--video --output logs/g1_terrain/run` to record Block/Ramp.

Resource defaults and particle parameters are documented in [README.md](README.md).
