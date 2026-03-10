# Isaac Lab Troubleshooting Guide

## "ModuleNotFoundError: No module named 'omni'"

**Cause**: Using bare Kit Python instead of `python.sh` wrapper.
**Fix**: Use `_isaac_sim/python.sh <script>` or `./isaaclab.sh -p <script>`.

The `python.sh` wrapper sources `setup_python_env.sh` which adds all Kit extension
paths to PYTHONPATH. Without it, omni.*, isaacsim.*, carb, and pxr modules are invisible.

## "ModuleNotFoundError: No module named 'isaacsim'"

**Cause**: Same as above — PYTHONPATH not configured.
**Fix**: Use `_isaac_sim/python.sh` or `./isaaclab.sh -p`.

## "ModuleNotFoundError: No module named 'toml'" / 'numpy' / 'yaml' / 'trimesh'

**Cause**: Missing pip packages, common after Isaac Sim rebuild.
**Fix**:
```bash
PYTHON="$(git rev-parse --show-toplevel)/_isaac_sim/python.sh"
$PYTHON -m pip install toml numpy pyyaml trimesh packaging pillow==12.0.0
```
Or reinstall everything: `./isaaclab.sh -i all`

## Cannot import Isaac Lab modules in a plain Python script

**Cause**: Most Isaac Lab modules transitively import `omni.kit.app` which requires
the Omniverse Kit runtime to be running.

**Affected modules** (not just `isaaclab_tasks`):
- `isaaclab_tasks.*` (all task environments)
- `isaaclab_physx.assets.*` (Articulation, RigidObject, etc.)
- `isaaclab.sensors.*` (Camera, ContactSensor, IMU, etc.)
- `isaaclab.sim.*` (SimulationContext, spawners, etc.)
- `isaaclab.actuators.*` (imports `omni.client` via `isaaclab.utils.assets`)

**Modules that CAN be imported without Kit runtime**:
- `isaaclab.app` (AppLauncher itself)
- `isaaclab.utils` (configclass, math, etc.)
- Top-level `import isaaclab_physx` (but not sub-modules like `.assets`)

**This is by design.** Training scripts handle it by calling `AppLauncher` first:
```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# NOW deep imports work
import isaaclab_tasks
from isaaclab_physx.assets.articulation import Articulation
```

## `pip list` returns empty results

**Cause**: Kit Python's pip doesn't enumerate packages the standard way.
**Fix**: Use `importlib.metadata` instead:
```bash
$PYTHON -c "
import importlib.metadata
v = importlib.metadata.version('isaaclab')
print(f'isaaclab: {v}')
"
```

This is a known quirk — all packages ARE installed, they just don't show in `pip list`.

## GPU Out of Memory

**Symptoms**: CUDA OOM error during training.
**Fix**:
- Reduce `--num_envs` (try 64 for small GPUs, 256-1024 for larger)
- Use `--headless` to skip viewport rendering
- CartPole on NVIDIA GB10: `--num_envs 64` for SB3, `--num_envs 4096` for RSL-RL
- CartPole on RTX 3090+: `--num_envs 4096` works headless

## Isaac Sim Crashes on Startup

- Check GPU driver: `nvidia-smi` (need driver 535+)
- Check CUDA: `nvcc --version` (need 12.0+)
- Verify _isaac_sim symlink: `ls -la _isaac_sim`
- Check kit logs: `cat ~/.local/share/ov/logs/Kit/` (look for crash reports)

## Training Doesn't Converge

- Check reward signs (positive = encourage, negative = penalize)
- Increase `max_iterations` or `n_timesteps`
- Reduce learning rate by 3-10x
- Monitor TensorBoard (see command below) for reward trends
- Verify environment resets correctly (observation space should be bounded)

## PyTorch Compute Capability Warning

**Message**: "PyTorch max supported compute capability is 12.0 but your GPU is 12.1"
**Status**: Safe to ignore. GPU works correctly despite this warning.

## Gym Deprecation Warning from SB3

**Message**: "Gym has been unmaintained since 2022 and does not support NumPy 2.0..."
**Status**: Safe to ignore. This warning appears when importing `stable_baselines3`
because it still references the old `gym` package internally. Isaac Lab uses
`gymnasium` (the maintained fork) and everything works correctly.

## "TensorFlow installation not found - running with reduced feature set"

**Message**: Appears when launching TensorBoard.
**Status**: Safe to ignore. TensorBoard works fully for scalar metrics, histograms,
and graphs without TensorFlow. The "reduced feature set" only affects TF-specific
profiling features that are not needed for RL training monitoring.

## numpy Import Warning with PyTorch

**Status**: Safe to ignore. Functionally works correctly.

## Direct CartPole Crash (develop branch)

**Message**: `RuntimeError: Cannot cast dtypes of unequal byte size`
**Location**: `cartpole_env.py:152` in `_reset_idx()` → `write_root_pose_to_sim()`
**Status**: Known bug on the develop branch (Isaac Sim 6.0). The Direct CartPole
environment (`Isaac-Cartpole-Direct-v0`) crashes with a warp dtype mismatch. This
affects ALL RL frameworks. The manager-based `Isaac-Cartpole-v0` works fine.
Use `Isaac-Cartpole-v0` instead until this is fixed.

## SB3 Batch Size Warning

**Message**: "mini-batch size of 4096, but RolloutBuffer is of size n_steps * n_envs = 1024"
**Cause**: CartPole's SB3 config has `batch_size: 4096` but with `--num_envs 64` and
`n_steps=16`, the actual buffer is only 1024. SB3 uses the full buffer as one batch.
**Status**: Safe to ignore. Training works correctly — the effective batch size is 1024.
To suppress, either increase `--num_envs` or reduce `batch_size` in the YAML config.

## SB3 `constant_fn()` Deprecation Warning

**Message**: "constant_fn() is deprecated, please use ConstantSchedule() instead"
**Cause**: Isaac Lab's SB3 wrapper uses the older `constant_fn()` API.
**Status**: Safe to ignore. No functional impact. Will be fixed in a future Isaac Lab release.

## SKRL Shows No Training Metrics in Terminal

**Symptom**: SKRL training shows only a progress bar with no reward/loss information.
**Status**: By design — SKRL logs metrics only to TensorBoard, not to stdout.
**Workaround**: Launch TensorBoard in a separate terminal:
```bash
$PYTHON -m tensorboard.main --logdir=logs/skrl/
```

## `--visualizer omniverse` Not Recognized

**Message**: `Unknown visualizer type 'omniverse' requested. Valid types: 'newton', 'rerun', 'kit'. Skipping.`
**Cause**: `omniverse` is not a valid visualizer name. The three valid options are:
- `kit` — Isaac Sim viewport (RTX rendering)
- `newton` — OpenGL standalone window
- `rerun` — Browser-based (WebGL via gRPC)

**Fix**: Use `--visualizer kit` instead of `--visualizer omniverse`.

## `--visualizer newton` or `--visualizer rerun` Fails Silently

**Error**: `Failed to build Newton model from USD: cannot import name 'DeviceLike' from 'warp'`
**Cause**: Warp version mismatch. The Newton Python module (`newton._src.core.types`)
imports `warp.DeviceLike` which exists in warp 1.11+ (site-packages) but NOT in the
Kit extscache warp 1.10.1. At runtime, `PhysxSceneDataProvider` loads the extscache
warp (1.10.1) which lacks `DeviceLike`, causing the Newton model build to fail.
Both `newton` and `rerun` visualizers depend on the Newton model, so both fail.

**Impact**: Training continues normally — the visualizer silently degrades to headless.
No visualization is produced, but no crash occurs.

**Workaround**: Use `--headless` (no visualizer) or omit `--visualizer` entirely.
The Kit visualizer (`--visualizer kit`) does NOT depend on Newton and would work
in a GUI environment, but requires a display.

**Root cause**: `omni.warp.core-1.10.1` in Isaac Sim 6.0 extscache predates
`warp.DeviceLike` (added in warp 1.11). The Newton package in site-packages was
built against warp 1.11+ but the runtime loads the older extscache version first.

## TensorBoard: "command not found"

**Cause**: TensorBoard is installed inside Kit Python but NOT on the system PATH.
**Fix**: Launch via Kit Python's module runner:
```bash
PYTHON="$(git rev-parse --show-toplevel)/_isaac_sim/python.sh"
$PYTHON -m tensorboard.main --logdir=logs/
```
Or install TensorBoard globally: `pip install tensorboard`
