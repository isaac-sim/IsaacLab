# Isaac Lab Troubleshooting Guide

## "ModuleNotFoundError: No module named 'omni'" or 'isaacsim'

**Cause**: Using bare Python instead of the Isaac Lab wrapper.
**Fix**: Use `./isaaclab.sh -p <script>` instead of `python3 <script>`.

## Cannot import Isaac Lab modules in a plain Python script

**By design.** Most Isaac Lab modules require the simulation runtime to be running.

**`isaaclab.sim` and modules depending on Kit** require AppLauncher to be called first.

**Non-exhaustive list of modules that CAN be imported without Kit runtime**:
- `isaaclab.app` (AppLauncher itself)
- `isaaclab.utils` (configclass, math, etc.)

Training scripts handle this by calling `AppLauncher` first:
```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# NOW deep imports work
```

Note: Lazy module imports have improved the situation — most import traversal issues
from earlier versions are resolved.

## GPU Out of Memory

- Reduce `--num_envs` (try 256+ for large GPUs — there is no hard upper limit)
- Use `--headless` to skip viewport rendering
- JAX users: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid pre-allocating all GPU memory

## Training Doesn't Converge

- Check reward signs (positive = encourage, negative = penalize)
- Increase `max_iterations` or `n_timesteps`
- Reduce learning rate by 3-10x
- Monitor TensorBoard for reward trends
- Verify environment resets correctly (observation space should be bounded)

## TensorBoard

```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs/
```

The "TensorFlow installation not found" warning is safe to ignore.

## Safe-to-Ignore Warnings

- **"PyTorch max supported compute capability"**: GPU works correctly despite this.
- **"Gym has been unmaintained since 2022"**: SB3 internal warning. Isaac Lab uses gymnasium.
- **"TensorFlow installation not found"**: TensorBoard works without TensorFlow.
- **numpy import warning with PyTorch**: No functional impact.
