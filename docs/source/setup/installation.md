---
orphan: true
---

# Isaac Lab Installation Guide

## Quick Start

This is it. `./isaaclab.sh -i` installs everything needed to run Newton tasks
out of the box.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create environment and install
uv venv .venv --python 3.12
source .venv/bin/activate
./isaaclab.sh -i

# Run a benchmark
./isaaclab.sh -p scripts/benchmarks/benchmark_rlgames.py \
  --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
  --headless --enable_cameras --num_envs=1225 --max_iterations=10 \
  presets=newton,newton_renderer,depth
```


---

## Running Tasks

The `presets=` Hydra override selects the physics backend and renderer at
runtime:

```bash
# Newton (Kit-less)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 4096 \
  presets=newton \
  --visualizer newton

# PhysX (Kit — requires Isaac Sim)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 4096 \
  presets=physx

# Newton with a specific visualizer
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 4096 \
  presets=newton \
  --visualizer viser
```

Kit-less visualizer options: `newton`, `rerun`, `viser`. Multiple can be
combined: `--visualizer newton,rerun`.

### Available Presets

Presets are combined with commas: `presets=newton,newton_renderer,depth`.

```bash
presets=newton,newton_renderer,rgb  # presets=physics,renderer,render mode
presets=newton,newton_renderer,depth
presets=physx,isaacsim_rtx_renderer,rgb
presets=physx,isaacsim_rtx_renderer,depth
presets=physx,isaacsim_rtx_renderer,albedo
presets=physx,isaacsim_rtx_renderer,simple_shading_constant_diffuse
presets=physx,isaacsim_rtx_renderer,simple_shading_diffuse_mdl
presets=physx,isaacsim_rtx_renderer,simple_shading_full_mdl
presets=newton,ovrtx_renderer,rgb
presets=newton,ovrtx_renderer,depth
presets=newton,ovrtx_renderer,albedo
presets=newton,ovrtx_renderer,simple_shading_constant_diffuse
presets=newton,ovrtx_renderer,simple_shading_diffuse_mdl
presets=newton,ovrtx_renderer,simple_shading_full_mdl
```

---

## Optional Add-ons

### Isaac Sim (Kit/RTX rendering)

```bash
./isaaclab.sh -i isaacsim
```

### Isaac Sim from source (PhysX + RTX rendering)

If you want to install a source build of Isaac Sim (for PhysX or IsaacSimRTX rendering):

```bash
git clone https://github.com/isaac-sim/IsaacSim.git  # clone isaacsim
cd isaacsim
./build.sh

ln -s <path/to/isaacsim>/_build/linux-x86_64/release _isaac_sim
./isaaclab.sh -uv             # injects activation hooks that source isaac sim paths
```
The activation hooks automatically set `CARB_APP_PATH`, `EXP_PATH`,
`ISAAC_PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH` for Isaac Sim.

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 --headless --enable_cameras --num_envs=1225 --max_iterations=10 presets=physx,isaacsim_rtx_renderer,depth
```

### OVRTX Rendering

OVRTX provides GPU-accelerated rendering for vision tasks without Kit.

```bash
./isaaclab.sh -i ovrtx

export LD_PRELOAD=$(python -c "import ovrtx, pathlib; print(pathlib.Path(ovrtx.__file__).parent / 'bin/plugins/libcarb.so')") # Set LD_PRELOAD

# Then use `presets=newton,ovrtx_renderer`

./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
  --task Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
  --headless --enable_cameras --num_envs 1225 --max_iterations 10 \
  presets=newton,ovrtx_renderer,simple_shading_diffuse_mdl
```



---

## Advanced: Selective Install

If you want a minimal environment, `./isaaclab.sh -i` accepts comma-separated
sub-package names:

| Option | What it does |
|--------|-------------|
| `isaacsim` | Install Isaac Sim pip package |
| `newton` | Install Newton physics + Newton visualizer |
| `physx` | Install PhysX physics runtime |
| `ovrtx` | Install OVRTX renderer runtime |
| `tasks` | Install built-in task environments |
| `assets` | Install robot/object configurations |
| `visualizers` | Install all visualizer backends |
| `rsl_rl` | Install RSL-RL framework |
| `skrl` | Install skrl framework |
| `sb3` | Install Stable Baselines3 framework |
| `rl_games` | Install rl_games framework |
| `robomimic` | Install robomimic framework |
| `none` | Install only core `isaaclab` package |

Examples:

```bash
# Minimal Newton setup
./isaaclab.sh -i newton,tasks,assets

# Newton with OVRTX and RSL-RL only
./isaaclab.sh -i newton,tasks,assets,ovrtx,rsl_rl

# Full Kit install with skrl
./isaaclab.sh -i isaacsim,skrl
```

---

### Conda Alternative (Kit Path Only)

```bash
conda create -n env_isaaclab python=3.12
conda activate env_isaaclab
pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
./isaaclab.sh -i
```

---

## Running Installation Tests

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_cli_utils.py -v
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pip'`

Your venv was created without pip. The install system auto-detects and uses
`uv pip` when pip is absent, so this error should no longer occur with the
latest Isaac Lab.

### `ModuleNotFoundError: No module named 'isaacsim'`

You are running a script that requires Isaac Sim, but it is not installed.
Either:
- Install Isaac Sim: `./isaaclab.sh -i isaacsim`, or
- Use a Newton-based task with `presets=newton --visualizer newton` (Kit-less path)

### `ModuleNotFoundError: No module named 'isaaclab_physx'` or `'isaaclab_ov'`

These config packages are auto-installed by `./isaaclab.sh -i`. If using a
selective install, re-run with the default `./isaaclab.sh -i` to get all
packages.

### `ModuleNotFoundError: No module named 'isaaclab_assets'`

Include `assets` in your install command, or use `./isaaclab.sh -i` to install
everything.

### `ModuleNotFoundError: No module named 'rsl_rl'`

Include the RL framework: `./isaaclab.sh -i rsl_rl`, or use
`./isaaclab.sh -i` to install all frameworks.

### Crash in `libusd_tf` / USD Symbol Collision with OVRTX

If you see a crash involving `libusd_tf-*.so` and conflicting USD versions
(e.g. `pxrInternal_v0_25_5` vs `pxrInternal_v0_25_11`):

1. Ensure `LD_PRELOAD` is set to ovrtx's `libcarb.so` (see OVRTX section)
2. Ensure `isaacsim` / `omniverse-kit` is **not** installed in the same
   environment — their bundled USD libraries conflict with ovrtx's

### Visualizer Not Appearing

If `--visualizer newton` shows no window, you may be missing `imgui-bundle`:

```bash
uv pip install imgui-bundle
```

For `viser`, check the terminal for a URL (e.g. `http://localhost:8012`).

### `GLIBC Version Too Old`

Isaac Sim pip packages require GLIBC 2.35+. Check with `ldd --version`.
Ubuntu 22.04+ satisfies this. For older distributions, use the
[binary installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
method for Isaac Sim.

---

## System Requirements

- **OS:** Ubuntu 22.04+ (Linux x86_64) or Windows 11 (x86_64)
- **RAM:** 32 GB or more
- **GPU:** NVIDIA GPU with 16 GB+ VRAM (for Kit/RTX rendering)
- **Drivers:** NVIDIA driver 580.65.06+ (Linux), 580.88+ (Windows)
- **Python:** 3.11 (Isaac Sim 5.x) or 3.12 (Isaac Sim 6.x, when available)

For Kit-less installs, any CUDA-capable GPU will work.
