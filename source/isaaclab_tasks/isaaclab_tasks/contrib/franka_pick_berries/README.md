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
                                    RTX renderer ◀── Gaussians follow the particles (gaussians/binding.py)
```

## Install

Linux x86-64 with an NVIDIA RTX GPU. From the Isaac Lab root:

```bash
bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh --renderer-internal
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"
```

This creates an isolated `.venv-tasks` with the OVRTX 0.6 renderer, which live Gaussian deformation needs
(`--renderer-internal` requires NVIDIA network access). Use `uv run --no-sync` for this task: `uv sync` would remove
the renderer. The assets are downloaded and cached on first use.

## Run the demo

Pick a raspberry with the keyboard (use `--mode teleop_gamepad` with a controller):

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/pick_berries.py \
  --mode teleop_keyboard
```

It runs at about 17 frames/s while grasping (1280 × 720, RTX 6000 Ada); the side panel shows the frame rate.

| | Keyboard (focus the window) | Gamepad |
|---|---|---|
| Move the hand | W/S, A/D, Q/E | hold LB; left stick (x, y), right stick (z) |
| Turn the hand | Z/X, T/G, C/V | hold LB; right stick (yaw), D-pad (roll, pitch) |
| Close / open the gripper | K / J | RT / LT; release to hold |
| Reset | R | Menu |

The gripper is position-controlled: closing further squeezes, and can crush, the berry.

## The showcase video

The robot sorts three raspberries: it crushes the first and drops it in the reject dish, then gently sets the other
two down in the bowl. A camera director films it shot by shot:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/scripts/pick_berries.py \
  --mode scripted_sorting --camera film_director --arm_speed 6 --bowl_material porcelain --f_stop 64 \
  --width 1920 --height 1080 --samples_per_pixel 64 --video berry_sorting.mp4
```

The video plays at **30 frames/s** of simulated time (about 50 s); rendering it at 64 samples per pixel runs at about
**0.4 frames/s**, about an hour. Without `--video`, the same sequence plays live in a window.

## Reference

### Options of `scripts/pick_berries.py`

| Option | Default | |
|---|---|---|
| `--mode {teleop_gamepad,teleop_keyboard,scripted_sorting}` | `teleop_keyboard` | Teleoperate, or watch the robot sort three berries |
| `--num_berries {1,2,3}` | 1 (sorting: 3) | Raspberries in the punnet |
| `--layout_seed N`, `--fixed_layout` | 0 | Random punnet layout (reset replays it), or side by side |
| `--tissue_resolution {full,half}` | `full` | Half the tissue particles, for speed; the Gaussians are unchanged |
| `--tissue_solver {explicit,implicit}` | `explicit` | Tissue solver (see below) |
| `--arm_speed X` | 2 | Arm speed multiplier; the gripper keeps its gentle closing pace |
| `--camera {punnet_and_bowl,berry_closeup,room,film_director}` | `punnet_and_bowl` | Fixed views, a close-up following the handled berry, or a shot-by-shot film (sorting) |
| `--bowl_material {glass,porcelain}` | `glass` | Material of the receiving bowl |
| `--repeat` | | Sorting: start again when it ends |
| `--video FILE.mp4` | | Sorting: record offscreen at 30 frames/s, then exit |
| `--width`, `--height`, `--samples_per_pixel`, `--f_stop` | 1280 × 720 | Image size, path-tracing samples, depth of field (film) |

Frame rates while grasping, at 1280 × 720 on an RTX 6000 Ada: about 17 frames/s with one berry, 9 with three
(`--num_berries 3`). Rendering the berries' Gaussians dominates, so `--tissue_resolution half` hardly changes them.

### Code layout

| Path | |
|---|---|
| `pick_berries_env_cfg.py`, `pick_berries_env.py` | The scene (robot, table, tableware, berries) and the environment |
| `physics/` | **MPM tissue.** `tissue.py` turns the asset into Newton MPM particles; `grasp_explicit_mpm.py` simulates them; `coupling.py` couples them to the arm |
| `gaussians/` | **Gaussian appearance.** `binding.py` makes the Gaussians follow the particles; `publisher.py` publishes them to the renderer |
| `mdp/`, `control/` | Arm and gripper actions; gamepad, keyboard and the scripted sorting |
| `rendering/`, `scene/`, `assets/` | Viewer, camera director and video; table, tableware and room; asset loading |
| `scripts/pick_berries.py`, `setup/setup.sh` | The demo, and its installer |

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
- The public OVRTX 0.5 renderer can lose live Gaussian updates; the task requires OVRTX 0.6.
- The Gaussians are stochastically composited, so the berries show slight sampling noise between frames.
