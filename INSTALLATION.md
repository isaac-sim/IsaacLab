# Isaac Lab Installation Guide

Isaac Lab supports multiple installation paths depending on whether you need
the full Isaac Sim/Kit rendering stack or just the Kit-less Newton physics
backend.

## System Requirements

- **OS:** Ubuntu 22.04+ (Linux x86_64) or Windows 11 (x86_64)
- **RAM:** 32 GB or more
- **GPU:** NVIDIA GPU with 16 GB+ VRAM (for Kit/RTX rendering)
- **Drivers:** NVIDIA driver 580.65.06+ (Linux), 580.88+ (Windows)
- **Python:** 3.12 (Isaac Sim 6.x) or 3.11 (Isaac Sim 5.x)

For Kit-less Newton, any CUDA-capable GPU will work.

---

## Choose Your Path

| Path | Physics | Rendering | Requires Isaac Sim? | Best For |
|------|---------|-----------|---------------------|----------|
| **Full (Kit)** | PhysX or Newton | RTX / Kit | Yes | Full sim-to-real, camera sensors, RTX rendering |
| **Kit-less (Newton)** | Newton (MuJoCo/Warp) | Newton / Rerun / Viser | No | Fast RL training, headless clusters, lightweight setups |

---

## Path 1: Kit-less Newton (uv)

No Isaac Sim required. Uses `uv` for fast, reproducible installs.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate a virtual environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3. Clone Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 4. Install Isaac Lab

```bash
./isaaclab.sh -i newton,physx,tasks,assets,rsl_rl
```

This installs the core `isaaclab` package plus the `newton`, `physx`, `rl`,
`tasks`, and `assets` sub-packages, along with the `rsl_rl` RL framework.
The `physx` sub-package is needed because task configs define presets for both
physics backends. The `assets` sub-package provides robot and object
configurations used by tasks. It does **not** pull in Isaac Sim.

The install system auto-detects `uv` and uses `uv pip` when `pip` is not
present in the venv.

### 5. (Optional) Install additional visualizers

The Newton visualizer is automatically installed with the `newton` sub-package.
To add other visualizers:

```bash
uv pip install -e "source/isaaclab_visualizers[rerun]"
uv pip install -e "source/isaaclab_visualizers[viser]"
```

Visualizer extras: `newton`, `rerun`, `viser`, `all`.

### 6. Verify the installation

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Cartpole-Direct-v0 --num_envs 128 presets=newton
```

### 7. Train

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 4096 \
  --visualizer newton \
  presets=newton
```

The `presets=newton` Hydra override selects the Newton physics backend. Without
it, the task defaults to PhysX (which requires Isaac Sim).

Kit-less visualizer options: `newton`, `rerun`, `viser`. Multiple can be
combined: `--visualizer newton,rerun`.

---

## Path 2: Full Install with Isaac Sim (uv)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate a virtual environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3. Clone Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 4. Install Isaac Lab with Isaac Sim

```bash
./isaaclab.sh -i isaacsim
```

This installs Isaac Sim, PyTorch with CUDA, all sub-packages, and all RL
frameworks.

Or install a specific RL framework only:

```bash
./isaaclab.sh -i isaacsim,rsl_rl
```

### 7. Verify the installation

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

This should launch the simulator and display a window with a black viewport.
Exit with `Ctrl+C`.

### 8. Train

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-v0 --headless
```

---

## Path 3: Full Install with Isaac Sim (conda)

### 1. Create and activate a conda environment

```bash
conda create -n env_isaaclab python=3.12
conda activate env_isaaclab
```

### 2. Install Isaac Sim

```bash
pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
```

### 3. Install PyTorch (CUDA)

```bash
pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Clone Isaac Lab and install

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i
```

### 5. Verify and train

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-v0 --headless
```

---

## Install Options Reference

`./isaaclab.sh -i <option>` accepts the following:

| Option | Behavior |
|--------|----------|
| *(empty)* or `all` | Install all sub-packages + all RL frameworks |
| `none` | Install only core `isaaclab` package |
| `isaacsim` | Install Isaac Sim pip package before installing sub-packages |
| `rsl_rl`, `skrl`, `sb3`, `rl_games`, `robomimic` | Install all sub-packages + that RL framework |
| `newton,physx,tasks,assets,rsl_rl` | Install core + named sub-packages + RL framework (comma-separated). Including `newton` also installs `isaaclab_visualizers[newton]`. |
| `visualizers` or `visualizers[newton]` | Install visualizer sub-package with extras |

Options can be combined: `./isaaclab.sh -i isaacsim,rsl_rl` installs Isaac Sim +
all sub-packages + the rsl_rl framework.

**Sub-packages:** `assets`, `physx`, `contrib`, `mimic`, `newton`, `rl`,
`tasks`, `teleop`, `visualizers`.

**Visualizer extras:** `all`, `kit`, `newton`, `rerun`, `viser`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pip'`

Your venv was created without pip. Either:
- Recreate with `uv venv --seed .venv`, or
- Install pip into the existing venv: `uv pip install pip`

The install system auto-detects and uses `uv pip` when pip is absent, so
this error should no longer occur with the latest Isaac Lab.

### `ModuleNotFoundError: No module named 'isaacsim'`

You are running a script that requires Isaac Sim, but it is not installed.
Either:
- Install Isaac Sim: `./isaaclab.sh -i isaacsim`, or
- Use a Newton-based task with `presets=newton --visualizer newton` (Kit-less path)

### `ModuleNotFoundError: No module named 'isaaclab_physx'`

Task configs define presets for multiple physics backends and import from
`isaaclab_physx` at module level. Install it even for Newton-only use:

```bash
uv pip install -e source/isaaclab_physx
```

Or include `physx` in your install command: `./isaaclab.sh -i newton,physx,...`

### `ModuleNotFoundError: No module named 'isaaclab_assets'`

Task configs reference robot/object configurations from `isaaclab_assets`.
Include `assets` in your install command:

```bash
./isaaclab.sh -i newton,physx,tasks,assets
```

Or install it directly: `uv pip install -e source/isaaclab_assets`

### `ModuleNotFoundError: No module named 'isaaclab_contrib'`

The core `isaaclab` package has an optional dependency on `isaaclab_contrib`
for tactile sensors. This is handled gracefully and should not block Newton
tasks. If you see this error, update to the latest Isaac Lab.

### `ModuleNotFoundError: No module named 'rsl_rl'`

RL frameworks are not installed by the comma-separated sub-package mode.
Install separately:

```bash
uv pip install -e "source/isaaclab_rl[rsl_rl]"
```

### Visualizer not appearing

If you pass `--visualizer newton` but no window appears, you may be missing
`imgui-bundle`:

```bash
uv pip install imgui-bundle
```

For `viser`, check the terminal for a URL (e.g. `http://localhost:8012`) to
open in your browser.

### `GLIBC version too old`

Isaac Sim pip packages require GLIBC 2.35+. Check with `ldd --version`.
Ubuntu 22.04+ satisfies this. For older distributions, use the
[binary installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
method for Isaac Sim.
