# Performance Smoke Test — Module Interface Reference

Full function signatures, CLI arguments, and JSON schemas for every module.
All commands run from the `IsaacLab/` repository root unless otherwise noted.

---

## `backend_identity.py`

Canonical backend identity helpers. `backend_key` is always derived as `physics_backend` for physics-only runs, or `{physics_backend}_{render_backend}` when a render backend is set.

```python
def make_backend_key(physics_backend: str, render_backend: str | None = None) -> str
def backend_identity_from_launch_config(config: dict) -> BackendIdentity | None
def backend_identity_from_benchmark_info(info: dict) -> BackendIdentity | None
```

This is used by task loading, launch config generation, Phase 2 config-drift checks, aggregate baseline lookup, and baseline sample metadata. Render backends are workload identity, so `physx_newton_renderer`, `newton_newton_renderer`, and `newton_ovrtx_renderer` are separate baseline buckets.

## `gpu_identity.py`

Canonical GPU identity helpers. `gpu_model` is the baseline bucket key; `gpu_model_raw` is preserved for display/provenance. Existing `tasks.json` hard-floor maps can keep legacy keys such as `L40S` because floor lookup tries canonical, raw, and known legacy aliases.

```python
def canonical_gpu_model(value: Any) -> str
def normalize_gpu_fields(value: Any) -> dict[str, str]
def gpu_model_config_keys(value: Any) -> list[str]
```

Examples: `NVIDIA L40S -> l40s`, `RTX6000 -> rtx_6000`, `NVIDIA GeForce RTX 5090 -> geforce_rtx_5090`.

## `runtime_contract.py`

Builds the runtime compatibility contract used for baseline matching. The matching code only sees `runtime_contract_hash`; package/version field selection stays in `gate_config.py`.

```python
def build_runtime_contract(*, provenance: dict | None, gpu_diag: dict | None, backend: BackendIdentity, policy: Mapping[str, Any]) -> tuple[dict, str]
def build_runtime_publish_info(*, provenance: dict | None, gpu_diag: dict | None, policy: Mapping[str, Any]) -> dict
```

Default compatibility fields smoke test on top-level active-path packages: IsaacSim, IsaacLab, Torch, Warp, the active physics package (`isaaclab_physx` or `isaaclab_newton`/`newton`), and `isaaclab_ov` for renderer backends. CUDA version, NVIDIA driver, GPU memory, and compute capability are published for humans but do not affect the compatibility hash by default.

## `github_gate_context.py`

Resolves GitHub event metadata into the aggregate arguments used for baseline matching and publication policy.

```python
def resolve_gate_context(env=None, event=None) -> GateContext
```

Outputs: `base_sha`, `target_branch`, `source_branch`, `allow_update`, `trusted_source`, and `event_kind`.
Pull requests and merge-group events are read-only and use the event payload to recover the real base/source branches. Protected branch pushes (`main`, `develop`, and `release/**`) may publish baseline updates.

## `oracle.py`

### `compare()`

```python
def compare(
    bench_result: dict,
    baseline: Baseline | None,
    fps_mean_thresholds: list[FpsMeanThreshold],
    excluded_frames: frozenset[int],
    artifact_dir: Path,
    *,
    min_block_regression_pct: float = MIN_BLOCK_REGRESSION_PCT,
) -> OracleResult
```

The central verdict function. Reads `perf_smoke_test_info.json` from `artifact_dir`,
applies `excluded_frames`, computes mean FPS, and returns an `OracleResult`.

The final verdict is the **most severe** of the rolling-window/baseline verdict and
the verdict from any crossed gating threshold (`WARN` < `BLOCK`). A crossed
reporting-only threshold (one with no `threshold_verdict`) is recorded in
`OracleResult.crossed_thresholds` without changing the verdict.

| Parameter | Type | Description |
|---|---|---|
| `bench_result` | `dict` | Loaded `perf_smoke_test_result.json` |
| `baseline` | `Baseline \| None` | Rolling baseline stats; `None` for seed run |
| `fps_mean_thresholds` | `list[FpsMeanThreshold]` | Configured FPS floors/reference points for this task/backend; may be empty |
| `excluded_frames` | `frozenset[int]` | 0-based frame indices to drop before computing mean |
| `artifact_dir` | `Path` | Directory containing `perf_smoke_test_info.json` |
| `min_block_regression_pct` | `float` | Minimum % regression below the MAD block band required to BLOCK |

### `class FpsMeanThreshold`

```python
@dataclass(frozen=True)
class FpsMeanThreshold:
    name: str                       # informative label, e.g. "IsaacLab-2.0" (required)
    value: float                    # mean-FPS floor; 0.0 is valid and effectively non-gating
    verdict: OracleVerdict | None   # WARN/BLOCK to gate when crossed; None = reporting-only
```

A threshold is *crossed* when the measured mean FPS is below `value`. A crossed
gating threshold contributes its `verdict`; a crossed reporting-only threshold
(`verdict is None`) is surfaced in outputs but never changes the verdict.

### `apply_excluded_frames()`

```python
def apply_excluded_frames(fps_series: list[float], excluded_frames: frozenset[int]) -> list[float]
```

Returns `fps_series` with indices listed in `excluded_frames` removed.

### `class Baseline`

```python
@dataclass
class Baseline:
    median_fps: float       # Median FPS of the baseline window
    mad_fps: float          # Median absolute deviation of FPS in the window
    k_warn: float = 2.5     # MAD multiplier for WARN threshold
    k_block: float = 4.0    # MAD multiplier for BLOCK threshold
    sample_count: int = 0   # Number of samples in the window
```

Thresholds: `warn_thresh = median - k_warn × MAD`, `block_thresh = median - k_block × MAD`.

### `class OracleResult`

```python
@dataclass
class OracleResult:
    verdict: OracleVerdict          # PASS / WARN / BLOCK / HARD_FAILURE
    bisect_verdict: str             # "GOOD" / "BAD" / "SKIP"
    failure_phase: str | None       # From bench_result; see failure phase table
    measured_fps: float | None      # Mean FPS post-filter; None on HARD_FAILURE
    baseline_fps: float | None      # baseline.median_fps; None if no baseline
    regression_pct: float | None    # ((measured - baseline) / baseline) × 100; None if no baseline
    fps_median: float | None        # Median of filtered series [informational]
    fps_p5: float | None            # 5th-percentile of filtered series [informational]
    fps_p95: float | None           # 95th-percentile of filtered series [informational]
    gpu_mem_used_mb: float | None   # From bench_result.gpu_diag [informational]
    startup_time_s: float | None    # From bench_result [informational]
    wall_time_s: float | None       # From bench_result [informational]
    was_retried: bool               # Whether Phase 1 succeeded only after a retry
    task_id: str
    backend: str
    crossed_thresholds: list[dict]  # Crossed thresholds (name/value/verdict/gating) for reporting
```

`fps_median`, `fps_p5`, `fps_p95` are informational — they do not affect the verdict.
`crossed_thresholds` lists every threshold whose value the measured mean FPS fell below,
including reporting-only ones, and is surfaced in the aggregate summary and GitHub outputs.
`measured_fps` (mean of filtered series) is the blocking metric.

### `class OracleVerdict`

```python
class OracleVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    HARD_FAILURE = "HARD_FAILURE"
```

---

## `task_config.py`

### `load_tasks()`

```python
def load_tasks(tasks_json_path: Path | str | None = None) -> list[TaskConfig]
```

Loads all benchmark tasks from `tasks.json`, expanding each task's `backends` array into
one `TaskConfig` per `(task_id, backend)` combination. Applies `defaults` block to each task.

Default path: `tools/perf_smoke_test/tasks.json` (sibling of the module).

### `get_task()`

```python
def get_task(task_id: str, backend_key: str, tasks_json_path: Path | str | None = None) -> TaskConfig
```

Returns the `TaskConfig` for a specific `(task_id, backend_key)` pair.
Raises `KeyError` if not found.

### `caches_for_backend()`

```python
def caches_for_backend(backend: str) -> list[str]
```

Returns cache identifiers needed before benchmarking with a given physics backend.
Currently: `"newton"` → `["mjwarp_jit"]`; all others → `[]`.

### `class TaskConfig`

```python
@dataclass
class TaskConfig:
    task_id: str
    physics_backend: str            # "physx" or "newton"
    render_backend: str | None      # "newton_renderer", "ovrtx_renderer", or None
    preset: str                     # Hydra preset base (usually "default")
    num_envs: int
    num_frames: int
    excluded_frames_raw: list[int | list[int]]  # Raw JSON; use .excluded_frames
    camera_resolution: tuple[int, int] | None
    timeout_minutes: int
    fps_mean_thresholds: dict       # {"L40S": {"physx": [FpsMeanThreshold, ...]}}
    caches: list[str]               # Cache identifiers from caches_for_backend()
    tags: list[str]                 # ["always"] or ["camera"]
    task_type: str                  # "benchmark"
    runs_on: str                    # "gpu-l40s"
    seed: int | None                # Random seed for benchmark (default 42)

    @property
    def backend_key(self) -> str:
        # "{physics}_{render}" if render_backend else "{physics}"

    @property
    def excluded_frames(self) -> frozenset[int]:
        # Expands excluded_frames_raw ranges to individual indices

    def thresholds_for(self, gpu_model: str) -> list[FpsMeanThreshold]:
        # Resolves this task/backend's thresholds across canonical + legacy GPU keys
```

`excluded_frames_raw` supports two entry types:
- `[start, end]` — inclusive range, expanded to `range(start, end+1)`
- `N` — single index

Default value `[[0, 100]]` expands to indices 0–100 (101 frames excluded from 300 total).

---

## `baseline_manager.py`

### Flat-file operations (local / testing)

```python
def load_baseline(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None
) -> Baseline | None
```
Loads `samples.ndjson` for a task/backend pair and computes thresholds from compatible samples. Matching can require exact `gpu_model`, `task_id`, `backend_key`, `launch_config_hash`, `benchmark_contract_hash`, `runtime_contract_hash`, and `baseline_epoch`; with `base_sha`, samples must also come from ancestor commits. Returns `None` if no structured compatible samples exist.

```python
def update_baseline(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint=None
) -> None
```
Appends one structured sample to `samples.ndjson`. Thresholds are computed at read time.
Only call for PASS/WARN results — aggregate.py enforces this policy.

```python
def delete_baseline_files(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None
) -> None
```
Removes `samples.ndjson` for a task/backend pair. Used in tests and resets.

```python
def seed_baseline_with_spread(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str,
    center_fps: float, noise_fps: float = 5.0, n_samples: int = 10,
    seed: int = 0, fingerprint=None
) -> None
```
Populates a baseline window with `n_samples` Gaussian samples around `center_fps` ± `noise_fps`.
Used in tests to create deterministic baselines without running real benchmarks.

### Git operations (production)

```python
def refresh_baseline_branch(branch: str, remote: str | None = "origin") -> str | None
def load_baseline_git(ref: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None) -> Baseline | None
def update_baselines_git(branch: str, updates: list[BaselineUpdateRecord], remote: str | None = "origin") -> BaselinePushResult
def update_baseline_git(branch: str, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint: str | None) -> BaselinePushResult
```

`refresh_baseline_branch()` fetches the remote baseline branch and returns the exact SHA
used for reads. When `base_sha` is supplied, baseline matching requires each selected sample
to have a `commit_sha` that is an ancestor of that base. `update_baselines_git()` is the
production writer: it refetches the branch, applies queued structured samples in a temporary
worktree, commits once, and retries on non-fast-forward push races. Sample IDs make retries
idempotent if a previous push outcome was ambiguous.

### File paths (flat-file)

```
{baselines_dir}/{gpu_model}/{task_id}/{backend}/samples.ndjson

With fingerprint:
{baselines_dir}/{gpu_model}/{task_id}/{backend}/{fingerprint}/samples.ndjson
```

---

## `subprocess_runner.py`

### `classify_failure_phase()`

```python
def classify_failure_phase(
    stdout: str, stderr: str, exit_code: int, wall_time_s: float, timeout_s: float
) -> str | None
```

Classifies the failure phase of a benchmark run by scanning combined output. Priority order:

| Priority | Phase | Trigger |
|---|---|---|
| 1 | `"oom"` | `exit_code == 137` OR `"oom-kill"` in `stderr` |
| 2 | `"hang"` | `wall_time_s >= timeout_s * 0.95` |
| 3 | `"import"` | `"Traceback"` in combined AND `"AppLauncher"` NOT in `stdout` |
| 4 | `"driver"` | `"CudaError"` OR `"CUDA_ERROR_"` in combined |
| 5 | `"init"` | `exit_code != 0` AND `"AppLauncher initialization complete"` in `stdout` AND `"Step Frametimes"` NOT in `stdout` |
| 6 | `"runtime"` | `exit_code != 0` AND `"Step Frametimes"` in `stdout` |
| 7 | `None` | `exit_code == 0` |

### `run_benchmark()`

```python
def run_benchmark(cmd: list, timeout_s: float) -> dict
```

Runs `cmd` with `capture_test_output_with_timeout()` and returns:

```python
{
    "exit_code": int,
    "stdout_tail": str,       # last 2000 chars of combined stdout
    "wall_time_s": float,
    "startup_time_s": float,  # stub: 0.0
    "failure_phase": str | None,
}
```

### `capture_test_output_with_timeout()`

```python
def capture_test_output_with_timeout(
    cmd, timeout, env, startup_deadline=0, report_file=""
) -> tuple[int, bytes, bytes, str, float, str]
```

Returns `(returncode, stdout_bytes, stderr_bytes, kill_reason, wall_time, pre_kill_diag)`.

`kill_reason` values: `""` (normal), `"timeout"`, `"startup_hang"`, `"shutdown_hang"`.

Uses `select()` + non-blocking I/O for real-time streaming. Kills the entire process group
(including Kit / Isaac Sim child processes) on timeout.

---

## `tasks_to_ci_matrix.py`

```
python3 tools/perf_smoke_test/tasks_to_ci_matrix.py
```

No arguments. Reads `tasks.json` via `load_tasks()` and prints a JSON array to stdout,
one object per `(task_id, backend)` combination. Used by the `build_matrix` step in
`perf-smoke-test.yaml` to populate the GitHub Actions job matrix.

Each object contains: `task_id`, `physics_backend`, `render_backend` (empty string if none),
`num_envs`, `num_frames`, `seed`, `hydra_args`, `bench_timeout_s`, and `job_timeout_minutes`.

---

## `build_bench_result.py` CLI

```
python3 tools/perf_smoke_test/build_bench_result.py \
    --task_id <str>           task identifier (required)
    --physics_backend <str>   "physx" or "newton" (required)
    --render_backend <str>    render backend name or "" for none (default: "")
    --artifact_dir <path>     directory for artifacts (required)
    --exit_code <int>         Phase 1 process exit code (required)
    --wall_time_s <float>     Phase 1 wall-clock time in seconds (required)
    --timeout_s <float>       Phase 1 timeout in seconds (required)
    --log_file <path>         combined stdout+stderr log from Phase 1 (default: none)
    --launch_config <path>    launch_config.json from Phase 1 (default: artifact_dir/launch_config.json)
    --gate_config <path>      gate_config.json for runtime compatibility policy
    --attempt <int>           attempt number: 1 = first try, 2 = after retry (default: 1)
    --was_retried             flag: set when this result comes from a retry
```

**Output:** `{artifact_dir}/perf_smoke_test_result.json` (always written).

**Side effect:** Renames `benchmark_non_rl_{task_id}_{timestamp}.json` to
`perf_smoke_test_info.json` if the canonical name does not already exist.

When `perf_smoke_test_info.json` is present, `build_bench_result.py` also
calls `_extract_info_provenance()` to populate the FPS distribution, startup time,
GPU diagnostics, and full software/hardware/git provenance directly into the result JSON.
It also compares observed benchmark identity to `launch_config.json`, records `observed_backend`,
computes `runtime_contract_hash`, and publishes non-matching runtime diagnostics such as CUDA and driver version.
`nvidia-smi` is queried once at post-processing time to capture the driver version.

---

## `aggregate.py` CLI

```
python3 tools/perf_smoke_test/aggregate.py \
    --artifacts_dir <path>         root directory containing per-task artifact subdirectories (required)
    --gpu_model <str>              GPU model label for baseline lookup (default: L40S)
    --gate_config <path>           path to gate_config.json (default: perf_smoke_test/gate_config.json)
    --baseline_branch <str>        git branch for baseline storage (default: perf-baselines)
    --baseline_remote <str>        git remote that owns the baseline branch (default: origin; empty = local only)
    --baseline_push_retries <int>  max retry attempts for transactional baseline pushes (default: config)
    --baselines_dir <path>         flat-file baseline directory; bypasses git (default: None = use git)
    --allow_baseline_update <str>  "true"/"false": extend baseline window for PASS/WARN (default: false)
    --summary_file <path>          append step-summary markdown to this path (default: none)
    --base_sha <sha>               PR/protected branch base SHA for ancestry-aware matching
    --target_branch <str>          protected target branch name
    --source_branch <str>          source branch name recorded in baseline metadata
    --trusted_source <str>         audit label for written samples
```

**Exit codes:**
- `0` — smoke test is non-blocking, or all tasks PASS/WARN
- `1` — any BLOCK verdict and `gate_config.blocking == true`
- `2` — any HARD_FAILURE verdict and `gate_config.blocking == true`

**Environment variable:** When `GITHUB_OUTPUT` is set, aggregate writes baseline trace
outputs such as `baseline_read_sha`, `baseline_pushed_sha`, `baseline_push_attempts`, and
`baselines_updated`. The push has already happened inside aggregate when these outputs are
written.

---

## `local_runner.py` CLI

```
python3 tools/perf_smoke_test/local_runner.py \
    --tags <tag ...>          task tags to run (default: always)
    --gpu_model <str>         GPU model label for baselines (default: auto-detect with nvidia-smi)
    --artifacts_dir <path>    root for per-task artifacts (default: perf_smoke_test/artifacts/)
    --baselines_dir <path>    flat-file baseline dir (default: perf_smoke_test/local_baselines/)
    --allow_baseline_update   extend baseline window for PASS/WARN results
    --dry_run                 print task matrix and exit without running anything
    --skip_existing           skip tasks whose perf_smoke_test_result.json already exists
    --gate_config <path>      path to gate_config.json
```

Orchestrates Phase 1+2+3 sequentially. For each task:
1. Runs `./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py` with the derived command
2. On non-zero exit, retries once (sets `was_retried=True`)
3. Calls `build_bench_result.py` (Phase 2)
4. After all tasks, calls `aggregate.py` (Phase 3)

Returns aggregate.py's exit code.

---

## `gate_config.py`

```python
def load_gate_config(path: Path | str) -> dict
```

Loads `gate_config.json` when present and otherwise returns conservative defaults.
The dict contract includes `blocking`, `min_baseline_samples`, `max_baseline_samples`,
`min_block_regression_pct`, `baseline_push_retries`, and `runtime_compatibility`.
`runtime_compatibility` owns the policy for fields included in `runtime_contract_hash` vs
publish-only diagnostic fields.

---

## `dev/stub_benchmark.py` CLI

Simulates `benchmark_non_rl.py` for testing. Does not require IsaacSim.

```
python3 tools/perf_smoke_test/dev/stub_benchmark.py \
    --task_id <str>
    --backend <str>
    --num_envs <int>       (default: 1)
    --num_frames <int>     (default: 200)
    --out_dir <path>       (required)
    --fps_mean <float>     (default: 200.0)
    --failure_phase <str>  "none" | "import" | "init" | "runtime" (default: none)
```

On success: writes `perf_smoke_test_info.json` with a Gaussian FPS series centered on
`--fps_mean`, prints `"Step Frametimes"` to stdout, exits 0.

On failure modes:
- `import`: prints traceback-like output, exits 1 (no perf file written)
- `init`: prints `"AppLauncher initialization complete"`, exits 2 (no perf file written)
- `runtime`: writes perf file, prints frametimes, then prints error and exits 3

---

## `dev/sim_regression.py` CLI

Injects degraded FPS artifacts for demo/testing without re-running benchmarks.

```
python3 tools/perf_smoke_test/dev/sim_regression.py \
    --fps_scale <float>    multiply baseline FPS by this factor (default: 0.53 = 47% regression)
    --tags <tag ...>       task tags to include (default: always)
    --gpu_model <str>      GPU model label (default: L40S)
    --baselines_dir <path> (default: perf_smoke_test/local_baselines)
    --out_dir <path>       output artifacts directory (default: /tmp/sim_artifacts)
```

For each task with an existing baseline, loads `samples.ndjson`, computes
`regressed_fps = baseline.median_fps × fps_scale`, and writes:
- `{out_dir}/{task_id}/{backend_key}/perf_smoke_test_info.json`
- `{out_dir}/{task_id}/{backend_key}/perf_smoke_test_result.json`

Skips tasks with no baseline (prints `SKIP (no baseline)`).

---

## `tasks.json` Schema

```json
{
  "defaults": {
    "type": "benchmark",
    "runs_on": "gpu-l40s",
    "preset": "default",
    "seed": 42,
    "num_envs": 512,
    "num_frames": 300,
    "excluded_frames": [[0, 100]],
    "camera_resolution": null,
    "timeout_minutes": 10,
    "tags": ["always"]
  },
  "tasks": [
    {
      "task_id": "<str>",
      "num_envs": <int>,               // overrides defaults
      "timeout_minutes": <int>,        // overrides defaults
      "tags": ["always" | "camera"],   // overrides defaults
      "backends": [
        {"physics": "physx"},
        {"physics": "newton"},
        {"physics": "physx", "render": "newton_renderer"},
        {"physics": "newton", "render": "ovrtx_renderer"}
      ],
      "fps_mean_thresholds": {
        "<gpu_model>": {
          "<backend_key>": [
            // threshold_name is required; threshold_verdict is WARN|BLOCK, or
            // omitted for a reporting-only entry; threshold 0.0 is non-gating.
            {"threshold_verdict": "BLOCK", "threshold_name": "hard-floor", "threshold": <float>},
            {"threshold_name": "IsaacLab-2.0", "threshold": <float>}
          ]
        }
      }
    }
  ]
}
```

Each entry in `backends` becomes one `TaskConfig`. `backend_key = physics` when no
`render` field; `backend_key = {physics}_{render}` when `render` is present.

---

## Artifact Schemas

### `perf_smoke_test_result.json` (Phase 2 output)

```json
{
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend": "physx",
  "backend_key": "physx",
  "physics_backend": "physx",
  "render_backend": null,
  "preset": "default",
  "attempt": 1,
  "was_retried": false,
  "exit_code": 0,
  "failure_phase": null,
  "stdout_tail": "<last 2000 chars of log>",
  "wall_time_s": 23.7,
  "startup_time_s": 12.5,
  "perf_smoke_test_info_present": true,
  "raw_fps_mean": 1655000.0,
  "raw_fps_std": 9800.0,
  "raw_fps_min": 1632000.0,
  "raw_fps_max": 1680000.0,
  "raw_fps_median": 1655000.0,
  "raw_fps_p5": 1638000.0,
  "raw_fps_p95": 1672000.0,
  "outlier_count": null,
  "gpu_diag": {
    "gpu_name": "NVIDIA L40S",
    "gpu_total_memory_gb": 45.62,
    "cuda_version": "12.1",
    "nvidia_driver_version": "550.54.15",
    "gpu_mem_used_mb": 18432.0
  },
  "provenance": {
    "hardware": {
      "cpu_name": "Intel Xeon Gold 6438Y+",
      "cpu_physical_cores": 32,
      "total_ram_gb": 251.5,
      "gpu_device_count": 1,
      "gpu_name": "NVIDIA L40S",
      "gpu_total_memory_gb": 45.62,
      "gpu_compute_capability": "8.9",
      "gpu_multi_processor_count": 142,
      "cuda_version": "12.1"
    },
    "software": {
      "isaaclab": "2.1.0",
      "warp": "1.6.0",
      "isaacsim": "4.5.0",
      "torch": "2.3.0+cu121",
      "numpy": "1.26.4",
      "newton": "1.0.0",
      "mujoco_warp": "0.3.0"
    },
    "git": {
      "commit_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
      "commit_hash_short": "a1b2c3d4",
      "branch": "develop",
      "commit_date": "2024-06-10 09:00:00 +0000",
      "dirty": false
    }
  },
  "launch_config": {
    "gpu_model": "l40s",
    "gpu_model_raw": "NVIDIA L40S",
    "launch_config_hash": "...",
    "benchmark_contract_hash": "..."
  },
  "observed_backend": {
    "physics_backend": "physx",
    "render_backend": null,
    "backend_key": "physx"
  },
  "runtime_contract_hash": "...",
  "runtime_contract": {
    "runtime_contract_version": 1,
    "fields": {"software.warp": "1.6.0"}
  },
  "runtime_info": {
    "software": {"warp": "1.6.0"},
    "publish_only": {"gpu_diag.cuda_version": "12.1"}
  },
  "task_config_snapshot": {
    "task_id": "Isaac-Velocity-Flat-G1-v0",
    "backend": "physx",
    "backend_key": "physx",
    "physics_backend": "physx",
    "render_backend": null,
    "preset": "default",
    "num_envs": 512,
    "num_frames": 300,
    "excluded_frames_raw": [[0, 100]],
    "timeout_minutes": 12,
    "tags": ["always"],
    "seed": 42
  }
}
```

**Fields populated from `perf_smoke_test_info.json`** (all `null` when
`perf_smoke_test_info_present` is false):

| Field | Source | Notes |
|---|---|---|
| `raw_fps_mean` … `raw_fps_p95` | `runtime` phase, `Step Frametimes` measurement | Full distribution before excluded-frame filtering |
| `startup_time_s` | `startup` phase, `Total Start Time (Launch to Train)` measurement | `null` if startup phase absent |
| `gpu_diag.gpu_mem_used_mb` | `runtime` phase, `GPU Memory Used` measurement | Converted from GB |
| `gpu_diag.gpu_name`, `.cuda_version`, `.gpu_total_memory_gb` | `hardware_info` phase | |
| `gpu_diag.nvidia_driver_version` | `nvidia-smi` subprocess at post-processing time | `null` if nvidia-smi unavailable |
| `provenance.hardware` | `hardware_info` phase | CPU, GPU, RAM identity |
| `provenance.software` | `version_info` phase | Package versions; `_version` suffix stripped |
| `provenance.git` | `version_info` phase, `dev` dict | commit, branch, date, dirty flag |

Key fields the oracle reads: `perf_smoke_test_info_present`, `failure_phase`,
`was_retried`, `gpu_diag.gpu_mem_used_mb`, `startup_time_s`, `wall_time_s`.

The `raw_fps_*` fields capture the full unfiltered distribution and are for audit/debug;
`oracle.compare()` recomputes mean FPS independently after applying `excluded_frames`.

### `perf_smoke_test_info.json` (Phase 1 output, renamed from `benchmark_non_rl_*.json`)

A list of `TestPhase` objects serialized by `JSONFileMetrics`. Measurement and metadata
names are prefixed with `"{task_id} {phase_name} "` by the serializer.

```json
[
  {
    "phase_name": "hardware_info",
    "measurements": [],
    "metadata": [
      {"name": "<task_id> hardware_info cpu_name",         "data": "Intel Xeon Gold 6438Y+", "type": "string"},
      {"name": "<task_id> hardware_info physical_cores",   "data": 32,    "type": "int"},
      {"name": "<task_id> hardware_info total_ram_gb",     "data": 251.5, "type": "float"},
      {"name": "<task_id> hardware_info gpu_device_count", "data": 1,     "type": "int"},
      {"name": "<task_id> hardware_info cuda_version",     "data": "12.1","type": "string"},
      {"name": "<task_id> hardware_info gpu_devices",      "data": {
        "0": {"name": "NVIDIA L40S", "total_memory_gb": 45.62,
              "compute_capability": "8.9", "multi_processor_count": 142}
      }, "type": "dict"}
    ]
  },
  {
    "phase_name": "version_info",
    "measurements": [],
    "metadata": [
      {"name": "<task_id> version_info isaaclab_version", "data": "2.1.0",  "type": "string"},
      {"name": "<task_id> version_info warp_version",     "data": "1.6.0",  "type": "string"},
      {"name": "<task_id> version_info dev", "data": {
        "commit_hash": "a1b2c3d4...", "commit_hash_short": "a1b2c3d4",
        "branch": "develop", "commit_date": "2024-06-10 09:00:00 +0000", "dirty": false
      }, "type": "dict"}
    ]
  },
  {
    "phase_name": "runtime",
    "measurements": [
      {
        "name": "<task_id> runtime Step Frametimes",
        "value": {"Environment step effective FPS": [1644720.5, 1638400.0, ...]},
        "type": "dict"
      },
      {
        "name": "<task_id> runtime GPU Memory Used",
        "value": 18.0,
        "unit": "GB",
        "type": "single"
      }
    ],
    "metadata": []
  },
  {
    "phase_name": "startup",
    "measurements": [
      {
        "name": "<task_id> startup Total Start Time (Launch to Train)",
        "value": 12.5,
        "unit": "s",
        "type": "single"
      }
    ],
    "metadata": []
  }
]
```

The oracle looks for `phase_name == "runtime"`, then the measurement whose `name` ends with
`"Step Frametimes"`, then extracts `value["Environment step effective FPS"]` as the raw series.
`build_bench_result.py` reads the remaining phases for provenance extraction.

### `samples.ndjson` (baseline history)

One JSON object per accepted baseline sample, append-only. The exact metadata can grow
without changing the file contract; the required fields for threshold calculation are:

```json
{"fps": 1667482.3, "gpu_model": "l40s", "task_id": "Isaac-Cartpole-Direct-v0", "backend_key": "physx", "launch_config_hash": "...", "benchmark_contract_hash": "...", "runtime_contract_hash": "...", "baseline_epoch": 1}
```
