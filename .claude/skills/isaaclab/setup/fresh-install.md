# Isaac Lab Fresh Machine Installation

## Prerequisites

- Ubuntu 22.04+ (or compatible Linux, tested on 24.04 aarch64 and x86_64)
- NVIDIA GPU with driver 535+ (`nvidia-smi` to check)
- CUDA 12.0+ toolkit (`nvcc --version` to check)
- Python 3.10-3.12

### Installing CUDA Toolkit (if missing)
```bash
# Check if already installed
nvcc --version

# Ubuntu — install via apt
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Or use NVIDIA's official repository for latest version
# See: https://developer.nvidia.com/cuda-downloads
```

### Installing NVIDIA Drivers (if missing)
```bash
# Check current driver
nvidia-smi

# Ubuntu — install recommended driver
sudo apt install nvidia-driver-535
sudo reboot
```

## Step 1: Install Isaac Sim

Two options:

**Option A — Pip install (simpler, recommended for first-time users)**:
```bash
pip install isaacsim
```
This installs a pre-built Isaac Sim. The `_isaac_sim` symlink is auto-configured.

**Option B — Build from source (for development / latest features)**:
```bash
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
# Follow build instructions in IsaacSim README
# Build output goes to _build/linux-{arch}/release/
```

## Step 2: Clone Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout main      # Stable (Isaac Sim 5.1)
# OR
git checkout develop   # Development (Isaac Sim 6.0)
```

## Step 3: Create Symlink to Isaac Sim

```bash
# Auto-detect (pip-installed):
./isaaclab.sh --install

# Manual (source build, aarch64):
ln -s /path/to/IsaacSim/_build/linux-aarch64/release _isaac_sim
# Manual (source build, x86_64):
ln -s /path/to/IsaacSim/_build/linux-x86_64/release _isaac_sim
```

Verify:
```bash
ls -la _isaac_sim             # Should be a valid symlink
ls _isaac_sim/python.sh       # Must exist
ls _isaac_sim/setup_python_env.sh  # Must exist
```

## Step 4: Install Isaac Lab Extensions

```bash
./isaaclab.sh -i all
```

This installs all packages in editable mode:
- isaaclab, isaaclab_assets, isaaclab_tasks, isaaclab_rl
- isaaclab_physx, isaaclab_newton, isaaclab_contrib
- isaaclab_mimic, isaaclab_teleop

## Step 5: Install RL Frameworks

```bash
./isaaclab.sh -i rsl_rl
./isaaclab.sh -i sb3
./isaaclab.sh -i skrl
./isaaclab.sh -i rl_games
```

## Step 6: Verify

```bash
PYTHON="./_isaac_sim/python.sh"

# Check Python + CUDA
$PYTHON -c "
from isaaclab.app import AppLauncher; print('AppLauncher: OK')
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"

# Check all RL frameworks
$PYTHON -c "
import importlib.metadata
for pkg in ['stable-baselines3', 'skrl', 'rsl-rl-lib', 'rl-games', 'tensorboard']:
    try: print(f'{pkg}: {importlib.metadata.version(pkg)}')
    except: print(f'{pkg}: NOT FOUND')
"

# Check training scripts
$PYTHON scripts/reinforcement_learning/sb3/train.py --help 2>&1 | head -3
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py --help 2>&1 | head -3
```

**Note**: `pip list` returns empty results under Kit Python — this is a known quirk.
Use `importlib.metadata.version('package-name')` instead.

## Step 7: TensorBoard

TensorBoard is installed as a dependency of the RL frameworks. Verify:
```bash
$PYTHON -c "import tensorboard; print(f'TensorBoard: {tensorboard.__version__}')"
```

**Important**: TensorBoard is installed inside Kit Python but NOT on the system PATH.
To launch it:
```bash
# Option 1 — Via Kit Python (always works)
$PYTHON -m tensorboard.main --logdir=logs/

# Option 2 — Install globally (adds tensorboard to PATH)
pip install tensorboard
tensorboard --logdir=logs/
```

Then open http://localhost:6006 in your browser.

The "TensorFlow installation not found" warning is safe to ignore.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ISAACLAB_NUCLEUS_DIR` | Path to Nucleus asset server for USD files |
| `ISAACSIM_PATH` | Isaac Sim installation (auto-detected from `_isaac_sim`) |

## Key Insight: python.sh is Required

The `_isaac_sim/python.sh` wrapper sources `setup_python_env.sh` which adds all Kit
extension paths to PYTHONPATH and LD_LIBRARY_PATH. Without it, `import omni.kit.app`
fails and no Isaac Lab modules can be loaded.

**Always use**: `_isaac_sim/python.sh` or `./isaaclab.sh -p`
**Never use**: bare `_isaac_sim/kit/python/bin/python3` or system `python3`
