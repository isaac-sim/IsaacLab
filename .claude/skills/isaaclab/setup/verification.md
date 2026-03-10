# Environment Verification

## Step 1: Detect Environment

```bash
# Find Isaac Lab root
ISAACLAB_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")
echo "Isaac Lab root: $ISAACLAB_ROOT"
cat "$ISAACLAB_ROOT/VERSION" 2>/dev/null
git -C "$ISAACLAB_ROOT" branch --show-current

# Find the python.sh wrapper (REQUIRED)
PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"
if [ -f "$PYTHON" ]; then
  echo "python.sh: FOUND at $PYTHON"
else
  echo "ERROR: python.sh not found. Check _isaac_sim symlink."
  ls -la "$ISAACLAB_ROOT/_isaac_sim" 2>/dev/null
fi

# GPU
nvidia-smi 2>&1 | head -5
```

## Step 2: Verify Core Imports

```bash
$PYTHON -c "
from isaaclab.app import AppLauncher
print('AppLauncher: OK')
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**Note**: Many Isaac Lab modules (`isaaclab_tasks`, `isaaclab_physx.assets.*`,
`isaaclab.sensors.*`, `isaaclab.sim.*`) will FAIL to import outside the Isaac Sim
runtime. This is by design — training scripts call `AppLauncher` first, which boots
the Omniverse Kit runtime. Only `isaaclab.app` and `isaaclab.utils` can be imported
without it.

## Step 3: Verify All RL Frameworks

```bash
$PYTHON -c "
import importlib.metadata

frameworks = {
    'stable-baselines3': 'SB3',
    'skrl': 'SKRL',
    'rsl-rl-lib': 'RSL-RL',
    'rl-games': 'RL-Games',
    'tensorboard': 'TensorBoard',
}
for pkg, label in frameworks.items():
    try:
        v = importlib.metadata.version(pkg)
        print(f'{label}: {v}')
    except importlib.metadata.PackageNotFoundError:
        print(f'{label}: NOT FOUND')

print()
print('All RL frameworks checked')
"
```

**Important**: Use `importlib.metadata.version()` instead of module `__version__`
attributes. Some packages (notably `rsl_rl` and `rl_games`) do not expose a
`__version__` attribute. Also, `pip list` returns empty results under Kit Python —
this is a known quirk of the Kit Python packaging. Always use `importlib.metadata`
to check installed packages.

## Step 4: Verify Training Scripts

```bash
$PYTHON scripts/reinforcement_learning/sb3/train.py --help 2>&1 | head -3
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py --help 2>&1 | head -3
$PYTHON scripts/reinforcement_learning/skrl/train.py --help 2>&1 | head -3
$PYTHON scripts/reinforcement_learning/rl_games/train.py --help 2>&1 | head -3
```

If all four show usage info, the environment is ready for training.

## Step 5: List Available Environments

Environments are registered via `gym.register()` in each task's `__init__.py` and
are only visible after the Kit runtime boots. To list them without running Isaac Sim:

```bash
grep -rh 'id="Isaac' source/isaaclab_tasks/ --include="*.py" \
  | sed 's/.*id="\([^"]*\)".*/\1/' | sort -u
```

This lists all 179 registered environment IDs (as of v4.0.0).
