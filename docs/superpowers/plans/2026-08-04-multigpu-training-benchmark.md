# Multi-GPU Training Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate experimental multi-GPU training benchmark that supports RSL-RL, RL-Games, and skrl Torch on one or more nodes and emits one rank-0 aggregate benchmark bundle.

**Architecture:** Extract the existing multi-GPU launcher mechanics into a private reusable CLI helper, then point a new `benchmark training_multigpu` launcher at the existing training benchmark dispatcher. Each backend contributes numeric rank-local timing data to a shared Torch-distributed reducer, and rank 0 builds and formats the sole aggregate bundle.

**Tech Stack:** Python 3.12, argparse, PyTorch distributed/NCCL and Gloo, Isaac Lab benchmark schema/builders, pytest, Sphinx/RST.

## Global Constraints

- Keep `isaaclab benchmark training` single-process and keep the typed `BenchmarkTrainingRequest` API unchanged.
- Support only RSL-RL, RL-Games, and skrl with `--ml_framework torch`; reject SB3 and skrl JAX.
- Support single-node and multi-node Torch launcher arguments; multi-node execution cannot be hardware-validated locally.
- Interpret `--num_envs` as environments per rank and report global environment and step totals.
- Use Torch distributed collectives only; do not require a shared filesystem or exchange arbitrary Python objects.
- Emit formatters, checkpoint metadata, learning curves, hardware, and resources only from global rank 0.
- Reject video, positive environment-sensor capture, and success-based early stopping in multi-GPU benchmark mode.
- Preserve the existing `train_multigpu` CLI, skrl JAX behavior, dry-run output, and diagnostics.
- Add no dependencies. Follow PEP 8, modern hints, Google docstrings, snake_case CLI names, and 2026 headers.
- Add one `source/isaaclab/changelog.d/` fragment; do not edit compiled changelogs or version files.
- Run commands through `./isaaclab.sh`; run `./isaaclab.sh -f` before every commit and again before any push.

---

### Task 1: Extract the reusable multi-GPU launcher

**Files:**
- Create: `source/isaaclab/isaaclab/cli/_multigpu.py`
- Modify: `scripts/reinforcement_learning/train_multigpu.py`
- Modify: `source/isaaclab/test/cli/test_train_multigpu_command_building.py`

**Interfaces:**
- Produces: `MultiGpuLauncherSpec`, `parse_multigpu_args()`, `build_distributed_command()`, and `run_multigpu()`.
- Preserves: the regular script's constants, private command-building functions, normal Torch behavior, and JAX behavior.

- [ ] **Step 1: Add failing tests for shared-helper delegation**

```python
def test_regular_launcher_spec_preserves_skrl_jax():
    assert train_multigpu.LAUNCHER_SPEC.allow_skrl_jax is True
    assert train_multigpu.LAUNCHER_SPEC.forwarded_args == ("--distributed",)


@pytest.mark.parametrize(
    "argv,launcher_module",
    [
        (["--num_gpus", "2", "--task", "X"], "torch.distributed.run"),
        (["--rl_library", "skrl", "--ml_framework", "jax", "--num_gpus", "2", "--task", "X"],
         "skrl.utils.distributed.jax"),
    ],
)
def test_regular_launcher_still_selects_expected_module(argv, launcher_module, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert launcher_module in _build_command(argv)
```

- [ ] **Step 2: Run the test and verify it fails because `LAUNCHER_SPEC` is absent**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_train_multigpu_command_building.py -q
```

- [ ] **Step 3: Implement the private helper with these exact interfaces**

```python
ForwardedArgsValidator = Callable[[argparse.ArgumentParser, argparse.Namespace, list[str]], None]


@dataclass(frozen=True)
class MultiGpuLauncherSpec:
    target_script: Path
    description: str
    supported_libraries: tuple[str, ...]
    default_library: str = "rsl_rl"
    allow_skrl_jax: bool = True
    forwarded_args: tuple[str, ...] = ("--distributed",)
    validate_forwarded_args: ForwardedArgsValidator | None = None


```

Expose `parse_multigpu_args(argv: list[str], spec: MultiGpuLauncherSpec) -> tuple[argparse.Namespace, list[str]]`, `build_distributed_command(args_cli: argparse.Namespace, forwarded_args: list[str], spec: MultiGpuLauncherSpec) -> list[str]`, and `run_multigpu(argv: list[str] | None, spec: MultiGpuLauncherSpec) -> int`.

Move the existing parser, visible-GPU checks, Torch/JAX selection, command construction, signal forwarding, and dry-run behavior without changing their diagnostics. Append every missing `spec.forwarded_args` token exactly once. Make the regular script a delegating compatibility wrapper with `allow_skrl_jax=True`.

- [ ] **Step 4: Run the focused tests and require all existing and new cases to pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_train_multigpu_command_building.py -q
```

- [ ] **Step 5: Format, stage, format again, and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/cli/_multigpu.py scripts/reinforcement_learning/train_multigpu.py \
    source/isaaclab/test/cli/test_train_multigpu_command_building.py
./isaaclab.sh -f
git commit -m "Refactor multi-GPU launcher"
```

### Task 2: Add the separate benchmark launcher workflow

**Files:**
- Create: `source/isaaclab/isaaclab/benchmark/entrypoints/training_multigpu.py`
- Create: `scripts/benchmarks/training_multigpu.py`
- Create: `source/isaaclab/test/cli/test_training_multigpu_command_building.py`
- Modify: `source/isaaclab/isaaclab/benchmark/dispatch.py`
- Modify: `source/isaaclab/test/cli/test_benchmark_entrypoint.py`

**Interfaces:**
- Consumes: Task 1 launcher helper.
- Produces: CLI-only `isaaclab benchmark training_multigpu`, targeting `scripts/benchmarks/training.py` and forwarding `--distributed --benchmark_multigpu`.

- [ ] **Step 1: Write failing launcher and dispatch tests**

```python
def test_benchmark_launcher_targets_training_dispatcher(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    command = _build_command(["--rl_library", "rsl_rl", "--num_gpus", "2", "--task", "X"])
    assert "torch.distributed.run" in command
    assert str(ROOT / "scripts" / "benchmarks" / "training.py") in command
    assert command[-2:] == ["--distributed", "--benchmark_multigpu"]


def test_benchmark_launcher_rejects_skrl_jax(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SystemExit):
        _build_command(["--rl_library", "skrl", "--ml_framework", "jax", "--num_gpus", "2", "--task", "X"])
```

Also cover SB3 rejection; `--nnodes`, `--node_rank`, and rendezvous forwarding; Kit/Hydra passthrough; forbidden video/sensor/early-stop options; dry-run quoting; workflow help; and launcher exit-status propagation.

- [ ] **Step 2: Run the tests and verify they fail because the workflow is absent**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_training_multigpu_command_building.py \
    source/isaaclab/test/cli/test_benchmark_entrypoint.py -q
```

- [ ] **Step 3: Implement the benchmark launcher configuration and validation**

```python
LAUNCHER_SPEC = MultiGpuLauncherSpec(
    target_script=ISAACLAB_ROOT / "scripts" / "benchmarks" / "training.py",
    description="Launch a multi-GPU Isaac Lab training benchmark.",
    supported_libraries=("rl_games", "rsl_rl", "skrl"),
    allow_skrl_jax=False,
    forwarded_args=("--distributed", "--benchmark_multigpu"),
    validate_forwarded_args=_validate_benchmark_args,
)
```

Make `_validate_benchmark_args()` understand both `--name value` and `--name=value`, while treating the token following `--kit_args` as opaque Kit input. Reject `--video`, positive `--capture_env_sensors`, and `--check_success`; allow sensor count zero. Add the standard script wrapper.

- [ ] **Step 4: Register a CLI-only workflow map**

```python
_CLI_WORKFLOW_MODULES = {
    "training_multigpu": "isaaclab.benchmark.entrypoints.training_multigpu",
}
```

Include it in command choices and propagate its integer return code. Do not add it to `_RL_WORKFLOW_MODULES`, `_workflow_module()`, or the typed request union.

- [ ] **Step 5: Run launcher and CLI tests and require them to pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_train_multigpu_command_building.py \
    source/isaaclab/test/cli/test_training_multigpu_command_building.py \
    source/isaaclab/test/cli/test_benchmark_entrypoint.py -q
```

- [ ] **Step 6: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/benchmark/entrypoints/training_multigpu.py \
    scripts/benchmarks/training_multigpu.py source/isaaclab/isaaclab/benchmark/dispatch.py \
    source/isaaclab/test/cli/test_training_multigpu_command_building.py \
    source/isaaclab/test/cli/test_benchmark_entrypoint.py
./isaaclab.sh -f
git commit -m "Add multi-GPU benchmark launcher"
```

### Task 3: Build numeric distributed metric aggregation

**Files:**
- Create: `source/isaaclab/isaaclab/benchmark/_distributed.py`
- Create: `source/isaaclab/test/benchmark/test_distributed.py`

**Interfaces:**
- Produces: `DistributedContext.from_env(enabled: bool)`, `LocalTrainingTiming`, `AggregatedTrainingTiming`, `aggregate_training_timing()`, `add_multigpu_benchmark_args()`, and `validate_multigpu_benchmark_args()`.

- [ ] **Step 1: Write failing pure and two-process Gloo tests**

Spawn two ranks with a `file://` rendezvous. Use local environment counts 16 and 32, local steps 64 and 128, iteration series `(2, 3)` and `(1, 4)`, and collection series `(1.5, 2)` and `(1, 3)`. Rank 0 must receive:

```python
assert aggregate.num_envs == 48
assert aggregate.steps_per_iteration == 192
assert aggregate.iteration_times_s == pytest.approx((2.0, 4.0))
assert aggregate.collection_times_s == pytest.approx((1.5, 3.0))
assert aggregate.total_fps == pytest.approx((96.0, 48.0))
assert aggregate.collection_fps == pytest.approx((128.0, 64.0))
```

Add errors for unequal series lengths, inconsistent optional simulation timing, unequal simulation-call counts, invalid world/local sizes, and enabled aggregation without an initialized group.

- [ ] **Step 2: Run the test and verify the module-not-found failure**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_distributed.py -q
```

- [ ] **Step 3: Implement these immutable payloads**

```python
@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.local_world_size


@dataclass(frozen=True)
class LocalTrainingTiming:
    startup_time_s: StartupTime
    iteration_times_s: tuple[float, ...]
    collection_times_s: tuple[float, ...]
    environment_step_times_s: tuple[float, ...]
    simulation_step_times_s: tuple[float, ...] | None
    simulation_step_calls: int | None
    num_envs: int
    steps_per_iteration: int


@dataclass(frozen=True)
class AggregatedTrainingTiming(LocalTrainingTiming):
    collection_fps: tuple[float, ...]
    total_fps: tuple[float, ...]
```

`from_env(False)` returns rank 0/world size 1. The enabled form reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and `LOCAL_WORLD_SIZE` and validates positive, divisible sizes.

- [ ] **Step 4: Implement tensor-only reductions**

Use CPU tensors for Gloo and `cuda:{local_rank}` tensors for NCCL. Validate each series length via MIN/MAX reductions, reduce timing values and startup fields with MAX, sum environments and steps, and require equal simulation-call counts. Compute FPS as global steps divided by the reduced duration. Never call `all_gather_object` or write rank exchange files.

- [ ] **Step 5: Implement parser gating with this truth table**

`add_multigpu_benchmark_args(parser)` adds only the hidden `--benchmark_multigpu` marker; each backend receives `--distributed` from `add_common_train_args(..., include_distributed=True)`.

| `distributed` | `benchmark_multigpu` | Result |
|---|---|---|
| false | false | regular benchmark |
| true | false | error directing the user to `benchmark training_multigpu` |
| false | true | error because the private marker requires distributed mode |
| true | true | multi-GPU benchmark; reject video, positive sensor capture, and early stop |

- [ ] **Step 6: Run aggregation tests and require them to pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_distributed.py -q
```

- [ ] **Step 7: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/benchmark/_distributed.py \
    source/isaaclab/test/benchmark/test_distributed.py
./isaaclab.sh -f
git commit -m "Add distributed benchmark aggregation"
```

### Task 4: Integrate RSL-RL distributed benchmark timing

**Files:**
- Modify: `source/isaaclab/isaaclab/benchmark/entrypoints/backends/rsl_rl/benchmark_train_rsl_rl.py`
- Modify: `scripts/benchmarks/test/test_training_adapters.py`

**Interfaces:**
- Consumes: Task 3 parser helpers and aggregation types.
- Produces: `_RslRlTimingRecorder` with `collection_times_s` and `iteration_times_s`.
- Produces: rank-0 `TrainingBundle.extra` distributed metadata.

- [ ] **Step 1: Write failing parser and logger-recorder tests**

Use a fake logger whose `log()` receives `collect_time` and `learn_time`, then assert:

```python
assert recorder.collection_times_s == [1.25, 1.5]
assert recorder.iteration_times_s == [2.0, 2.5]
assert fake_logger.calls == 2
```

Also assert plain `--distributed` is rejected while `--distributed --benchmark_multigpu` parses successfully.

- [ ] **Step 2: Run the selected tests and verify they fail**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    -k 'rsl_rl and (distributed or timing)' -q
```

- [ ] **Step 3: Implement `_RslRlTimingRecorder`**

On entry, wrap `runner.logger.log`; append `collect_time` and `collect_time + learn_time`; delegate the original call; and restore the original method on every exit path. This captures timings even though RSL-RL disables non-rank-0 TensorBoard logs.

- [ ] **Step 4: Match normal RSL-RL distributed setup**

Enable common distributed args, add/validate the private marker after preset parsing, and apply:

```python
_common.validate_distributed_device(args_cli)
context = DistributedContext.from_env(enabled=args_cli.distributed)
if context.enabled:
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed += context.rank
    env_cfg.seed = agent_cfg.seed
```

Write the run manifest only from rank 0 in distributed mode. Wrap `runner.learn()` with the existing environment timer and the new recorder.

- [ ] **Step 5: Aggregate on all ranks and emit only on rank 0**

Build `LocalTrainingTiming`, call `aggregate_training_timing()` on every rank, and return `None` from non-main ranks after the collective. Update the internal `run()` return annotation to `BenchmarkResult | None`. Rank 0 builds runtime with global counts and attaches:

```python
extra = {
    "distributed": True,
    "world_size": context.world_size,
    "local_world_size": context.local_world_size,
    "num_nodes": context.num_nodes,
    "num_envs_per_rank": local_num_envs,
    "learning_scope": "rank0",
    "resource_scope": "rank0_node",
}
```

Use prefix `benchmark_training_multigpu_<task>`, rank-0 TensorBoard learning/checkpoint data, and `video_path=None`. Preserve the regular TensorBoard timing path and prefix unchanged.

- [ ] **Step 6: Run RSL-RL and aggregation tests**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    source/isaaclab/test/benchmark/test_distributed.py -k 'rsl_rl or distributed' -q
```

- [ ] **Step 7: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/benchmark/entrypoints/backends/rsl_rl/benchmark_train_rsl_rl.py \
    scripts/benchmarks/test/test_training_adapters.py
./isaaclab.sh -f
git commit -m "Benchmark distributed RSL-RL training"
```

### Task 5: Integrate RL-Games distributed benchmark timing

**Files:**
- Modify: `source/isaaclab/isaaclab/benchmark/entrypoints/backends/rl_games/benchmark_train_rl_games.py`
- Modify: `scripts/benchmarks/test/test_training_adapters.py`

**Interfaces:**
- Consumes: Task 3 aggregation and parser helpers.
- Produces: `_RlGamesTimingObserver`, delegating the current observer API while recording collection and epoch duration.

- [ ] **Step 1: Write failing observer lifecycle tests without importing RL-Games**

Drive a fake delegated observer through `after_init()`, `after_steps()`, and `after_print_stats()` for two epochs with patched timestamps:

```python
assert observer.collection_times_s == [1.0, 1.5]
assert observer.iteration_times_s == [2.0, 3.0]
assert delegated.after_steps_calls == 2
assert delegated.after_print_stats_calls == 2
```

Add RL-Games parser cases to the distributed truth-table tests.

- [ ] **Step 2: Run selected tests and verify they fail**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    -k 'rl_games and (distributed or timing)' -q
```

- [ ] **Step 3: Implement the timing observer**

Wrap `RlGamesEarlyStopObserver`. Start timing after initialization, capture collection at `after_steps()`, capture total iteration at `after_print_stats()`, delegate every callback, and start the next iteration only after the complete epoch. Raise a descriptive error if print-stats arrives without a collection boundary.

- [ ] **Step 4: Match normal RL-Games distributed configuration**

```python
agent_cfg["params"]["seed"] += context.rank
agent_cfg["params"]["config"]["device"] = env_cfg.sim.device
agent_cfg["params"]["config"]["device_name"] = env_cfg.sim.device
agent_cfg["params"]["config"]["multi_gpu"] = True
env_cfg.seed = agent_cfg["params"]["seed"]
```

Enable/validate distributed args and write the run manifest only from rank 0 when distributed.

- [ ] **Step 5: Aggregate timings and retain rank-0 learning**

Use observer timings for distributed iteration/collection runtime and environment-recorder timings for step metrics. TensorBoard remains the rank-0 learning source. Populate global counts and the same `extra` mapping as RSL-RL, set `checkpoint_path=None`, annotate internal `run()` as `BenchmarkResult | None`, and finalize only on rank 0. Leave regular TensorBoard FPS behavior unchanged.

- [ ] **Step 6: Run dependency-free RL-Games and aggregation tests**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    source/isaaclab/test/benchmark/test_distributed.py -k 'rl_games or distributed' -q
```

- [ ] **Step 7: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/benchmark/entrypoints/backends/rl_games/benchmark_train_rl_games.py \
    scripts/benchmarks/test/test_training_adapters.py
./isaaclab.sh -f
git commit -m "Benchmark distributed RL-Games training"
```

### Task 6: Integrate skrl Torch distributed benchmark timing

**Files:**
- Modify: `source/isaaclab/isaaclab/benchmark/entrypoints/backends/skrl/benchmark_train_skrl.py`
- Modify: `scripts/benchmarks/test/test_training_adapters.py`

**Interfaces:**
- Consumes: the existing Torch `BenchmarkTrainer` timing arrays and Task 3 aggregation.
- Preserves: JAX and IPPO/MAPPO rejection.
- Produces: skrl Torch distributed setup and rank-0 aggregate output.

- [ ] **Step 1: Write failing distributed parser/config tests**

Keep the existing JAX/IPPO rejection assertions, replace unconditional distributed rejection with the common truth table, and add:

```python
args = SimpleNamespace(distributed=True)
agent_cfg = {"seed": 10}
env_cfg = SimpleNamespace(seed=10)
train_skrl._apply_distributed_config(args, agent_cfg, env_cfg, rank=2)
assert agent_cfg["seed"] == 12
assert env_cfg.seed == 12
```

- [ ] **Step 2: Run selected tests and verify they fail**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    -k 'skrl and (parser or distributed)' -q
```

- [ ] **Step 3: Enable only skrl Torch distributed mode**

Use `include_distributed=True`, the private marker, and common validation. Keep `choices=["torch"]` and algorithms AMP/PPO. Call `_common.validate_distributed_device()`, offset agent/environment seed by rank, and retain Torch Runner/SequentialTrainer imports.

- [ ] **Step 4: Aggregate the existing trainer timings**

Populate `LocalTrainingTiming` from `BenchmarkTrainer.collection_times_s`, `iter_times_s`, and the environment timer. All ranks reduce; non-main ranks return `None`; rank 0 builds the global runtime and distributed `extra`. Annotate internal `run()` as `BenchmarkResult | None`. Learning, success diagnostics, and checkpoint metadata remain rank-0 scoped. Regular skrl behavior stays unchanged.

- [ ] **Step 5: Run dependency-free skrl parser and aggregation tests**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_training_adapters.py \
    source/isaaclab/test/benchmark/test_distributed.py \
    -k 'skrl_parser or skrl_distributed or distributed' -q
```

The existing live `SequentialTrainer` test is allowed to skip when skrl is not installed; parser and helper tests must pass.

- [ ] **Step 6: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/benchmark/entrypoints/backends/skrl/benchmark_train_skrl.py \
    scripts/benchmarks/test/test_training_adapters.py
./isaaclab.sh -f
git commit -m "Benchmark distributed skrl training"
```

### Task 7: Document the workflow and add release metadata

**Files:**
- Modify: `docs/source/testing/benchmarks.rst`
- Modify: `docs/source/features/multi_gpu.rst`
- Modify: `.github/workflows/test-multi-gpu.yaml`
- Create: `source/isaaclab/changelog.d/multigpu-training-benchmark.rst`

**Interfaces:**
- Consumes: completed command and metric semantics.
- Produces: single-node/multi-node documentation and the package changelog fragment.

- [ ] **Step 1: Add the documented single-node command**

```bash
uv run --extra isaacsim isaaclab benchmark training_multigpu \
    --rl_library rsl_rl --num_gpus 2 \
    --task Isaac-Lift-KukaAllegro-Camera \
    presets=isaacsim_physx
```

- [ ] **Step 2: Add a multi-node rendezvous example**

```bash
uv run --extra isaacsim isaaclab benchmark training_multigpu \
    --rl_library rsl_rl --nnodes 2 --node_rank 0 --nproc_per_node 8 \
    --rdzv_backend c10d --rdzv_endpoint host0:29400 --rdzv_id lift-benchmark \
    --task Isaac-Lift-KukaAllegro-Camera \
    presets=isaacsim_physx
```

State that each node uses its own `--node_rank`; environments are per rank; rank 0 emits the bundle; throughput combines global work with slowest-rank time; learning/checkpoint are rank 0; resources cover the rank-0 node; JAX, SB3, video, sensors, early stop, play, and runtime are excluded; and multi-node execution was not hardware-validated in this change. Link from `multi_gpu.rst` without changing regular JAX documentation.

- [ ] **Step 3: Add the exact changelog fragment**

```rst
Added
^^^^^

* Added an experimental multi-GPU training benchmark for RSL-RL, RL-Games,
  and skrl Torch with single-node and multi-node launch support.
```

- [ ] **Step 4: Add one real two-GPU benchmark smoke to existing multi-GPU CI**

Extend the pull-request path filter for the new launcher, aggregation helper, and three benchmark adapters. In the existing PhysX/no-renderer matrix job only, run:

```bash
./isaaclab.sh -p scripts/benchmarks/training_multigpu.py \
    --rl_library rsl_rl --num_gpus 2 \
    --task Isaac-Cartpole-Direct --max_iterations 3 --num_envs 16 \
    --benchmark_formatter schema --output_path benchmark_results/multigpu
```

Load the emitted JSON with `./isaaclab.sh -p -c` and assert `extra.distributed` is true, `extra.world_size == 2`, `run.num_envs == 32`, `runtime.steps_per_iteration > 0`, and `runtime.total_fps.mean > 0`. Keep the existing action major versions and normal distributed-training matrix unchanged.

- [ ] **Step 5: Validate docs and focused CLI tests**

```bash
rg -n "training_multigpu|num_envs.*per rank|rank 0|skrl JAX" \
    docs/source/testing/benchmarks.rst docs/source/features/multi_gpu.rst
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_training_multigpu_command_building.py \
    source/isaaclab/test/cli/test_benchmark_entrypoint.py -q
```

- [ ] **Step 6: Format twice around staging and commit**

```bash
./isaaclab.sh -f
git add docs/source/testing/benchmarks.rst docs/source/features/multi_gpu.rst \
    .github/workflows/test-multi-gpu.yaml \
    source/isaaclab/changelog.d/multigpu-training-benchmark.rst
./isaaclab.sh -f
git commit -m "Document multi-GPU training benchmarks"
```

### Task 8: Verify the complete feature

**Files:**
- Review: every file changed since `origin/develop`
- Modify: only formatter edits or fixes required by verification

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a clean, evidence-backed branch ready for review.

- [ ] **Step 1: Verify launcher and dispatch tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/cli/test_train_multigpu_command_building.py \
    source/isaaclab/test/cli/test_training_multigpu_command_building.py \
    source/isaaclab/test/cli/test_benchmark_entrypoint.py -q
```

- [ ] **Step 2: Verify aggregation and adapters**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_distributed.py \
    scripts/benchmarks/test/test_training_adapters.py -q
```

Where optional SB3/skrl packages remain absent, record those known import failures and require all remaining tests to pass:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_distributed.py \
    scripts/benchmarks/test/test_training_adapters.py \
    -k 'not sb3_iteration_time_includes_policy_update and not skrl_reward_uses_episode_return_tracking' -q
```

- [ ] **Step 3: Inspect single-node and multi-node dry runs**

```bash
./isaaclab.sh -p scripts/benchmarks/training_multigpu.py --dry_run \
    --rl_library rsl_rl --num_gpus 2 --task Isaac-Cartpole-Direct
./isaaclab.sh -p scripts/benchmarks/training_multigpu.py --dry_run \
    --rl_library rl_games --nnodes 2 --node_rank 0 --nproc_per_node 2 \
    --rdzv_backend c10d --rdzv_endpoint host0:29400 --rdzv_id benchmark \
    --task Isaac-Cartpole-Direct
```

Both must target `scripts/benchmarks/training.py` and include the two private child flags. A skrl JAX dry run must exit 2 before spawning workers.

- [ ] **Step 4: Run full pre-commit and inspect repository state**

```bash
./isaaclab.sh -f
git diff --check origin/develop...HEAD
git diff --stat origin/develop...HEAD
git status --short
```

Require pre-commit PASS, no whitespace errors, and no uncommitted changes. Confirm no compiled changelog, version file, typed benchmark request, or unrelated file changed.

- [ ] **Step 5: Commit verification fixes only when needed**

Stage explicit fixed paths, rerun `./isaaclab.sh -f`, and use:

```bash
git commit -m "Fix multi-GPU benchmark verification"
```

Do not create an empty commit when verification changes nothing.
