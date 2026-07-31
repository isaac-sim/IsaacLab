# Newton Neural Checkpoint Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve remote neural-actuator checkpoints through Isaac Lab's shared temporary-file cache before Newton metadata authoring loads them with PyTorch.

**Architecture:** Keep path resolution inside `_resave_checkpoint_with_metadata()` so every current and future caller receives a local checkpoint path. Reuse `retrieve_file_path()` without changing the shared cache implementation, then preserve the existing TorchScript and dictionary-checkpoint loading paths.

**Tech Stack:** Python 3.12, PyTorch, Isaac Lab asset utilities, USD/Newton actuator schema authoring.

## Global Constraints

- Do not add a regression test, per maintainer direction.
- Do not add dependencies.
- Preserve local checkpoint behavior and existing error messages that identify the configured path.
- Validate remote MLP and LSTM checkpoints through real 4096-environment PhysX training runs with Newton actuators.
- Run `./isaaclab.sh -f` before committing.

---

### Task 1: Resolve neural checkpoints through the shared asset cache

**Files:**
- Modify: `source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py:340`
- Create: `source/isaaclab/changelog.d/newton-neural-checkpoint-resolution.rst`

**Interfaces:**
- Consumes: `isaaclab.utils.assets.retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = False) -> str`
- Produces: `_resave_checkpoint_with_metadata(original_path: str, metadata: dict[str, Any]) -> str` with local and remote configured-path support.

- [ ] **Step 1: Confirm the existing failure**

Use the already captured ANYmal-D startup result. The command reached `_resave_checkpoint_with_metadata()` and failed because `torch.jit.load()` and `torch.load()` received the HTTPS checkpoint URL directly.

- [ ] **Step 2: Resolve the configured path before loading**

Add a lazy import and resolve the path once inside `_resave_checkpoint_with_metadata()`:

```python
from isaaclab.utils.assets import retrieve_file_path  # noqa: PLC0415

local_path = retrieve_file_path(original_path)
```

Pass `local_path` to both PyTorch load operations while retaining `original_path` in the unsupported-checkpoint error message:

```python
net = torch.jit.load(local_path, map_location="cpu", _extra_files=extra_files)
checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
```

- [ ] **Step 3: Document the cache behavior**

Update `_resave_checkpoint_with_metadata()` to state that configured paths are resolved through the shared asset cache before loading.

- [ ] **Step 4: Add the package changelog fragment**

Create the following patch-level fragment:

```rst
Fixed
^^^^^

* Fixed Newton neural actuators failing to load actuator-network checkpoints
  from remote paths.
```

- [ ] **Step 5: Check the focused diff**

Run:

```bash
git diff --check
git diff -- source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py \
  source/isaaclab/changelog.d/newton-neural-checkpoint-resolution.rst
```

Expected: no whitespace errors; only path resolution, its docstring, and the changelog fragment are changed.

### Task 2: Validate remote MLP and LSTM training

**Files:**
- Verify: `logs/rsl_rl/train_newton_go1_flat_4096_500/<run>/`
- Verify: `logs/rsl_rl/train_newton_anymald_flat_4096_500/<run>/`

**Interfaces:**
- Consumes: the updated Newton neural-checkpoint authoring path from Task 1.
- Produces: completed Go1 and ANYmal-D training artifacts and TensorBoard metrics.

- [ ] **Step 1: Run Go1 with its remote MLP checkpoint**

Run 500 deterministic iterations with seed 42, 4096 environments, regular PhysX, and `env.sim.use_newton_actuators=true`.

Expected: the remote MLP checkpoint resolves to a cached local file, Newton loads the patched checkpoint, and training completes with `model_499.pt`.

- [ ] **Step 2: Run ANYmal-D with its remote LSTM checkpoint**

Run 500 deterministic iterations with seed 42, 4096 environments, regular PhysX, and `env.sim.use_newton_actuators=true`.

Expected: the remote LSTM checkpoint resolves to the cached local file, Newton loads the patched checkpoint, and training completes with `model_499.pt`.

- [ ] **Step 3: Extract comparable training statistics**

For each TensorBoard event file, report the final and last-50 mean values for:

```text
Train/mean_reward
Train/mean_episode_length
Episode_Termination/time_out
Metrics/success_rate
Metrics/base_velocity/error_vel_xy
Perf/total_fps
```

- [ ] **Step 4: Run the mandatory repository checks**

Run:

```bash
./isaaclab.sh -f
```

If hooks modify files, inspect and stage those changes, then rerun the command. Expected: every hook passes.

- [ ] **Step 5: Commit the implementation**

Stage only the production file and changelog fragment, then commit:

```bash
git add source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py \
  source/isaaclab/changelog.d/newton-neural-checkpoint-resolution.rst
git commit -m "Fix remote Newton neural checkpoints"
```
