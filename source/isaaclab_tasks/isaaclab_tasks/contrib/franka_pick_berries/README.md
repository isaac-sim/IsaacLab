# Picking raspberries: MPM tissue rendered as Gaussians

A Franka picks soft raspberries. Their tissue deforms, bruises and tears under the gripper, and they look like the
scanned fruit they come from. This task shows how to simulate a complex, deformable and visually rich object with
Isaac Lab and Newton:

- **Physics: the material point method (MPM).** Each berry is a few thousand tissue particles, simulated on Newton
  with an MPM solver coupled to the MuJoCo-Warp robot. The finger pads grip the tissue by friction, so a berry can be
  held gently, slip, or be crushed.
- **Appearance: 3D Gaussians.** Each berry is a few hundred thousand Gaussians from a scan. Every frame they follow the
  tissue particles, stretching, turning and darkening with bruises, and are streamed to the RTX renderer without
  leaving the GPU.

```text
 gamepad / keyboard / script ─▶ Franka arm (MuJoCo-Warp) ◀─ contact ─▶ berry tissue (MPM particles)
                                                                              │
                                    RTX renderer ◀── Gaussians follow the particles (gaussian_splats/mpm_binding.py)
```

## Install

Linux or Windows with an NVIDIA RTX GPU, and [uv](https://docs.astral.sh/uv/). Nothing else: from the Isaac Lab root,
each command below runs with `uv run --extra ovrtx`, which creates and syncs the `.venv` with Isaac Lab and its OVRTX
renderer before running. Keep `--extra ovrtx` on every run; without it, uv syncs the renderer out of the environment.

Live Gaussian deformation needs OVRTX 0.6, which Isaac Lab pins from the NVIDIA Omniverse package index (NVIDIA
network access required until it is published on PyPI). The first run takes several minutes more: it downloads the
packages and the assets, which are then cached, and compiles the renderer's shaders.

## Run the demo

Pick a raspberry with the keyboard (use `--mode teleop_gamepad` with a controller):

```bash
uv run --extra ovrtx python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/pick_berries.py \
  --mode teleop_keyboard
```

It runs at about 17 frames/s while grasping (1280 × 720, RTX 6000 Ada); the side panel shows the frame rate.

| | Keyboard (focus the window) | Gamepad |
|---|---|---|
| Move the hand: forward / left in the view, up | W/S, A/D, Q/E | hold LB; left stick, right stick up/down |
| Turn the hand | Z/X, T/G, C/V | hold LB; D-pad (tilt), right stick left/right (yaw) |
| Close / open the gripper | K / J | RT / LT; release to hold |
| Reset | R | Menu |

The hand moves relative to the camera: forward is away from it, whichever view is shown. The gripper is
position-controlled and closes slowly: closing further squeezes, and can crush, the berry. It opens faster, so a
crushed berry, whose damaged tissue sticks briefly to the fingers, drops off.

## The showcase video

In the scripted demo, the robot crushes the first of three raspberries and drops it in the reject dish, then gently
sets the other two down in the bowl. A scripted camera films it shot by shot. Render its frames, then encode them
with ffmpeg:

```bash
uv run --extra ovrtx python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/pick_berries.py \
  --mode scripted_demo --camera scripted_camera --arm_speed 6 --bowl_material porcelain --f_stop 64 \
  --width 1920 --height 1080 --samples_per_pixel 64 --save_frames berry_demo_frames
ffmpeg -framerate 30 -i berry_demo_frames/%05d.png -c:v libx264 -profile:v main -bf 0 -crf 16 -pix_fmt yuv420p \
  -movflags +faststart berry_demo.mp4
```

Each frame is 1/30 s of simulated time, so the video plays at **30 frames/s** (about 50 s). Rendering the frames at
64 samples per pixel runs at about **0.4 frames/s**, about an hour. Without `--save_frames`, the same sequence plays
live in a window.

## Reference

### Options of `scripts/pick_berries.py`

| Option | Default | |
|---|---|---|
| `--mode {teleop_gamepad,teleop_keyboard,scripted_demo}` | `teleop_keyboard` | Teleoperate the robot, or watch the scripted demo |
| `--num_berries {1,2,3}` | 1 (scripted_demo: 3) | Raspberries in the punnet |
| `--layout_seed N`, `--fixed_layout` | 0 | Random punnet layout (reset replays it), or side by side |
| `--tissue_resolution {full,half}` | `full` | Half the tissue particles, for speed; the Gaussians are unchanged |
| `--tissue_solver {explicit,implicit}` | `explicit` | Tissue solver (see below) |
| `--arm_speed X` | 2 | Arm speed multiplier; the gripper keeps its gentle closing pace |
| `--camera {punnet_and_bowl,berry_closeup,room,scripted_camera}` | `punnet_and_bowl` | Fixed views, a close-up following the handled berry, or a shot-by-shot film (scripted demo) |
| `--bowl_material {glass,porcelain}` | `glass` | Material of the receiving bowl |
| `--repeat` | | Scripted demo: start again when it ends |
| `--save_frames DIR` | | Scripted demo: render offscreen, save one PNG per 1/30 s, then exit |
| `--width`, `--height`, `--samples_per_pixel`, `--f_stop` | 1280 × 720 | Image size, path-tracing samples, depth of field (film) |

Frame rates while grasping, at 1280 × 720 on an RTX 6000 Ada: about 17 frames/s with one berry, 9 with three
(`--num_berries 3`). Rendering the berries' Gaussians dominates, so `--tissue_resolution half` hardly changes them.

### Code layout

| Path | |
|---|---|
| `pick_berries_env_cfg.py`, `pick_berries_env.py` | The scene (robot, table, tableware, berries) and the environment |
| `physics/` | **MPM tissue.** `tissue.py` turns the asset into Newton MPM particles; `grasp_explicit_mpm.py` simulates them; `coupling.py` couples them to the arm |
| `gaussian_splats/` | **Gaussian appearance.** `mpm_binding.py` makes the Gaussians follow the particles; `render_delegate.py` publishes them to the renderer |
| `mdp/actions.py` | Arm and gripper actions |
| `control/` | `teleop_devices.py` (keyboard and gamepad) and `scripted_demo.py` |
| `rendering/` | The viewer, and the scripted camera that films the demo |
| `scene/` | Table and tableware, the scanned room, and the robot's and table's materials |
| `assets/` | Where the assets live (`asset_paths.py`) and how the berry's is read (`berry_asset.py`) |
| `scripts/pick_berries.py` | The demo |

Each module's docstring explains its part; the solvers' docstrings list their methods and references.

### Tissue solvers

The default **explicit** solver (`physics/grasp_explicit_mpm.py`) is an MLS-MPM with fixed-corotated elasticity,
plasticity, damage (bruising and tearing) and frictional finger-pad contact; each berry is its own velocity field, in
frictional contact with the others. The **implicit** solver (`physics/grasp_implicit_mpm.py`) builds on Newton's
implicit MPM. It cannot yet hold a berry by friction, so while the fingers hold still the tissue they press is clamped
to them. It is slower, and the arm passes through the tableware. Stiffnesses are empirical handling presets, not
measured fruit properties.

### Assets

The raspberry is one USDZ package: its Gaussians with degree-three spherical harmonics, its tissue particles,
simulation metadata and MDL shader. The room is a Gaussian scan. Both are fetched from Nucleus on first use and cached
(`assets/asset_paths.py`); `ISAACLAB_BERRY_ASSET_ROOT` points to another root, holding `raspberry/raspberry_v3.usdz`
and `background/ebc/`.

### Known limitations

- The assets still live in a personal Nucleus folder; they should move to the public asset repository.
- OVRTX 0.6, which live Gaussian deformation needs (0.5 can lose the updates), is not yet on PyPI: Isaac Lab pins an
  internal build.
- The Gaussians are stochastically composited, so the berries show slight sampling noise between frames.
