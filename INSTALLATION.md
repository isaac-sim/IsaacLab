# Isaac Lab Installation Guide

## System Requirements

- **OS:** Ubuntu 22.04+ (Linux x86_64) or Windows 11 (x86_64)
- **RAM:** 32 GB or more
- **GPU:** NVIDIA GPU with 16 GB+ VRAM (for Kit/RTX rendering)
- **Drivers:** NVIDIA driver 580.65.06+ (Linux), 580.88+ (Windows)
- **Python:** 3.11 (Isaac Sim 5.x) or 3.12 (Isaac Sim 6.x, when available)

For Kit-less installs, any CUDA-capable GPU will work.

---

## Prerequisites

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone Isaac Lab and create a virtual environment
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
uv venv --python 3.11 .venv
source .venv/bin/activate
```

---

## Install

Isaac Lab has two axes of configuration:

- **Runtime**: Kit (Isaac Sim) or Kit-less (no Isaac Sim)
- **Physics backend**: Newton (MuJoCo/Warp) or PhysX

These combine into the following install paths:

| Install path | Command | Physics | Rendering | Isaac Sim? |
|-------------|---------|---------|-----------|------------|
| Newton, Kit-less | `./isaaclab.sh -i newton,physx,tasks,assets,rsl_rl` | Newton | Newton / Viser / Rerun | No |
| Newton + Kit | `./isaaclab.sh -i isaacsim,newton,rsl_rl` | Newton | RTX + Newton | Yes |
| PhysX + Kit | `./isaaclab.sh -i isaacsim,rsl_rl` | PhysX | RTX | Yes |

The `physx` sub-package is always needed — task configs define presets for both
backends and import from `isaaclab_physx` at module level.

Replace `rsl_rl` with any supported RL framework: `skrl`, `sb3`, `rl_games`,
`robomimic`, or omit it to install without an RL framework.

---

## Verify and Train

The `presets=` Hydra override selects the physics backend at runtime:

```bash
# Newton (Kit-less)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 4096 \
  presets=newton \
  --visualizer newton

# PhysX (Kit)
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

---

## (Optional) Additional visualizers

The Newton visualizer is automatically installed with the `newton` sub-package.
To add other Kit-less visualizers:

```bash
uv pip install -e "source/isaaclab_visualizers[rerun]"
uv pip install -e "source/isaaclab_visualizers[viser]"
```

Visualizer extras: `newton`, `rerun`, `viser`, `all`.

---

## Install Options Reference

`./isaaclab.sh -i <option>` accepts comma-separated values:

| Option | What it does |
|--------|-------------|
| `isaacsim` | Install Isaac Sim pip package |
| `newton` | Install Newton physics + Newton visualizer |
| `physx` | Install PhysX physics |
| `tasks` | Install built-in task environments |
| `assets` | Install robot/object configurations |
| `visualizers` | Install all visualizer backends |
| `rl` | Install `isaaclab_rl` (no framework) |
| `rsl_rl` | Install RSL-RL framework (auto-includes `isaaclab_rl`) |
| `skrl` | Install skrl framework |
| `sb3` | Install Stable Baselines3 framework |
| `rl_games` | Install rl_games framework |
| `robomimic` | Install robomimic framework |
| *(empty)* or `all` | Install all sub-packages + all RL frameworks |
| `none` | Install only core `isaaclab` package |

Examples:

```bash
# Minimal Newton setup for RL training
./isaaclab.sh -i newton,physx,tasks,assets,rsl_rl

# Full Kit install with skrl instead of RSL-RL
./isaaclab.sh -i isaacsim,skrl

# Everything
./isaaclab.sh -i all
```

---

### Conda alternative (Kit path only)

```bash
conda create -n env_isaaclab python=3.11
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

This runs unit tests for the CLI utilities and integration tests that create a
fresh uv venv, install all Newton-path sub-packages, and verify they are
importable and that tasks register correctly.

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

### `ModuleNotFoundError: No module named 'isaaclab_physx'`

Task configs define presets for multiple physics backends and import from
`isaaclab_physx` at module level. Include `physx` in your install command:
`./isaaclab.sh -i newton,physx,...`

### `ModuleNotFoundError: No module named 'isaaclab_assets'`

Task configs reference robot/object configurations from `isaaclab_assets`.
Include `assets` in your install command:
`./isaaclab.sh -i newton,physx,tasks,assets`

### `ModuleNotFoundError: No module named 'isaaclab_contrib'`

The core `isaaclab` package has an optional dependency on `isaaclab_contrib`
for tactile sensors. This is handled gracefully and should not block Newton
tasks. If you see this error, update to the latest Isaac Lab.

### `ModuleNotFoundError: No module named 'rsl_rl'`

Include the RL framework in your install command:
`./isaaclab.sh -i newton,physx,tasks,assets,rsl_rl`

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
