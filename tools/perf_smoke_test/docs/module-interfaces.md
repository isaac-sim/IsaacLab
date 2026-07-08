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

Canonical GPU identity helpers. `gpu_model` is the baseline bucket key; `gpu_model_raw` is preserved for display/provenance. `tasks.json` hard-floor maps are keyed by the canonical slug (e.g. `l40s`); floor lookup normalizes any GPU input (display name, raw name, or slug) to that canonical key.

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
    *,
    min_block_regression_pct: float = MIN_BLOCK_REGRESSION_PCT,
) -> OracleResult
```

The central verdict function. Reads the steady-state mean FPS from
`bench_result["raw_fps_mean"]` (which `build_bench_result` derives from the runtime
bundle's `runtime.total_fps.mean`, warmup already excluded at the source by
`perf_runtime.py --warmup_frames`) and returns an `OracleResult`. It no longer
re-parses `perf_smoke_test_info.json` or applies a post-hoc frame filter. Missing/
non-numeric `raw_fps_mean` → `HARD_FAILURE` (`missing_fps_mean`); `raw_fps_mean <= 0`
→ `HARD_FAILURE` (`zero_fps`).

The final verdict is the **most severe** of the rolling-window/baseline verdict and
the verdict from any crossed gating threshold (`WARN` < `BLOCK`). A crossed
reporting-only threshold (one with no `threshold_verdict`) is recorded in
`OracleResult.crossed_thresholds` without changing the verdict.

| Parameter | Type | Description |
|---|---|---|
| `bench_result` | `dict` | Loaded `perf_smoke_test_result.json` (mean FPS read from `raw_fps_mean`) |
| `baseline` | `Baseline \| None` | Rolling baseline stats; `None` for seed run |
| `fps_mean_thresholds` | `list[FpsMeanThreshold]` | Configured FPS floors/reference points for this task/backend; may be empty |
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
    measured_fps: float | None      # raw_fps_mean (steady-state mean); None on HARD_FAILURE
    baseline_fps: float | None      # baseline.median_fps; None if no baseline
    regression_pct: float | None    # ((measured - baseline) / baseline) × 100; None if no baseline
    gpu_mem_used_mb: float | None   # From bench_result.gpu_diag [informational]
    startup_time_s: float | None    # From bench_result [informational]
    wall_time_s: float | None       # From bench_result [informational]
    was_retried: bool               # Whether Phase 1 succeeded only after a retry
    task_id: str
    backend: str
    crossed_thresholds: list[dict]  # Crossed thresholds (name/value/verdict/gating) for reporting
```

`crossed_thresholds` lists every
threshold whose value the measured mean FPS fell below, including reporting-only ones,
and is surfaced in the aggregate summary and GitHub outputs. `measured_fps`
(`raw_fps_mean`, the steady-state mean) is the blocking metric.

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
    warmup_frames: int              # Leading steps excluded at the source before aggregation
    camera_resolution: tuple[int, int] | None
    timeout_minutes: int
    fps_mean_thresholds: dict[str, dict[str, list[FpsMeanThreshold]]]  # {"L40S": {"physx": [...]}}
    caches: list[str]               # Cache identifiers from caches_for_backend()
    tags: list[str]                 # ["always"] or ["camera"]
    task_type: str                  # "benchmark"
    runs_on: str                    # "gpu-l40s"
    seed: int | None                # Random seed for benchmark (default 42)
    baseline_epoch: int             # Baseline compatibility epoch (default 1)

    @property
    def backend_key(self) -> str:
        # "{physics}_{render}" if render_backend else "{physics}"

    def thresholds_for(self, gpu_model: str) -> list[FpsMeanThreshold]:
        # Resolves this task/backend's thresholds via the canonical GPU key
```

`warmup_frames` is a plain int: the number of leading cold-start steps
`perf_runtime.py` discards before aggregating FPS (steady-state warmup exclusion).
The default of `100` (from `tasks.json`) leaves 200 steady-state frames out of the
default 300. This replaces the old `excluded_frames_raw` range-list / `excluded_frames`
property, which filtered frames post-hoc in the oracle.

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

### Git operations (production)

```python
def refresh_baseline_branch(branch: str, remote: str | None = "origin") -> str | None
def load_baseline_git(ref: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None) -> Baseline | None
def update_baselines_git(branch: str, updates: list[BaselineUpdateRecord], remote: str | None = "origin") -> BaselinePushResult
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

**Side effect:** Globs `benchmark_runtime_{task_id}_*.json` (the schema-v1
`RuntimeBundle` written by `perf_runtime.py`) and **copies** it to
`perf_smoke_test_info.json` if the canonical name does not already exist.

When `perf_smoke_test_info.json` is present, `build_bench_result.py` loads the
`RuntimeBundle` and calls `benchmark_result_adapter.to_gate_fields()` to project it
into the FPS aggregates (`raw_fps_{mean,std,min,max}`), startup time, GPU diagnostics,
and full software/hardware/git provenance written into the result JSON.
It also compares the run's self-reported identity (`benchmark_result_adapter.benchmark_info()`)
to `launch_config.json`, records `observed_backend` and any `config_mismatch`,
computes `runtime_contract_hash`, and publishes non-matching runtime diagnostics such as CUDA and driver version.
The adapter queries `nvidia-smi` once at post-processing time to capture the driver version.

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
1. Runs `./isaaclab.sh -p tools/perf_smoke_test/perf_runtime.py` with the derived command
   (`--benchmark_formatter schema --warmup_frames <N>`)
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

Fakes a `perf_runtime.py` run without a GPU/sim: writes a schema-v1 `RuntimeBundle`
(`benchmark_runtime_{task_id}_{timestamp}.json`) identical in shape to what the real
driver produces, so `build_bench_result.py` → `benchmark_result_adapter` → `oracle`
can be exercised end-to-end offline. It uses the real `isaaclab.test.benchmark`
builders/serialize (pure-Python, no GPU), so run it with the Isaac Lab Python env.

```
python3 tools/perf_smoke_test/dev/stub_benchmark.py \
    --task_id <str>        (required)
    --backend <str>        (required) backend_key, e.g. "physx" or "newton_newton_renderer"
    --num_envs <int>       (default: 1)
    --num_frames <int>     (default: 200)
    --warmup_frames <int>  (default: 0)
    --seed <int>           (default: 42)
    --out_dir <path>       (required)
    --fps_mean <float>     (default: 200.0) target per-env-step effective FPS
    --failure_phase <str>  "none" | "import" | "init" | "runtime" (default: none)
```

On success (`--failure_phase none`): synthesizes a Gaussian steady-state FPS series
centered on `--fps_mean`, builds and writes the `RuntimeBundle`, prints
`"Step Frametimes"` to stdout, exits 0.

On failure modes:
- `import`: prints traceback-like output, exits 1 (no bundle written)
- `init`: prints `"AppLauncher initialization complete"`, exits 2 (no bundle written)
- `runtime`: writes the bundle, prints `"Step Frametimes"`, then prints an error and exits 3

**Producing a local BLOCK without the full suite:** run `stub_benchmark.py` with a
`--fps_mean` below the task's configured hard-floor into a per-task artifacts dir, then
run `aggregate.py` over that dir. The configured `BLOCK` `fps_mean_thresholds` floor is
crossed, so the oracle returns `BLOCK` (exit 1 when `blocking == true`).

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
    "warmup_frames": 100,
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
    "warmup_frames": 100,
    "timeout_minutes": 12,
    "tags": ["always"],
    "seed": 42
  },
  "schema_version": "1.0"
}
```

This artifact is a typed `BenchResult` (`contracts.py`): `build_bench_result` constructs
it, `BenchResult.to_dict()` serializes it (wire-stable + additive `schema_version`), and
`oracle`/`aggregate`/`seed_baselines` reconstruct it via `BenchResult.from_dict()`. The
adapter likewise returns a typed `RuntimeSample`. The `baseline_manager` storage layer
still works with the serialized dict form (callers pass `BenchResult.to_dict()`).

**Fields populated from `perf_smoke_test_info.json`** (all `null` when
`perf_smoke_test_info_present` is false):

All fields below are projected from the `RuntimeBundle` by
`benchmark_result_adapter.to_gate_fields()`. They are omitted/`null` when
`perf_smoke_test_info_present` is false.

| Field | Source (bundle path) | Notes |
|---|---|---|
| `raw_fps_mean`, `raw_fps_std`, `raw_fps_max` | `runtime.total_fps.{mean,std,peak}` | Steady-state (warmup excluded at source) |
| `raw_fps_min` | `run.num_envs / runtime.iteration_time_s.peak` | Recovered from the slowest steady-state step |
| `raw_fps_median`, `raw_fps_p5`, `raw_fps_p95`, `p99_over_median`, `outlier_count` | — (removed) | Not emitted: the schema keeps only aggregates (these needed the raw series), and they were never gating. `raw_fps_std`/`raw_fps_min` cover the tail |
| `startup_time_s` | sum of `runtime.startup_time_s.*` phases | app_launch + env_creation + first_step + python_imports |
| `gpu_diag.gpu_mem_used_mb` | `resources.gpu_mem_gb.mean` | Converted from GB |
| `gpu_diag.gpu_name`, `.gpu_total_memory_gb` | `hardware.gpu_devices[0].{name,mem_gb}` | |
| `gpu_diag.cuda_version` | `versions.cuda_bindings` | CUDA bindings version used as a display proxy (schema-v1 has no CUDA-runtime field) |
| `gpu_diag.nvidia_driver_version` | `nvidia-smi` subprocess at post-processing time | `null` if nvidia-smi unavailable |
| `provenance.hardware` | `hardware` snapshot | CPU, GPU, RAM identity |
| `provenance.software` | `versions` map (verbatim, minus `git_*` keys) | Package versions |
| `provenance.git` | `versions.git_commit`/`git_branch`/`git_dirty` | commit, branch, dirty flag |

Key fields the oracle reads: `perf_smoke_test_info_present`, `raw_fps_mean`,
`failure_phase`, `config_mismatch`, `was_retried`, `gpu_diag.gpu_mem_used_mb`,
`startup_time_s`, `wall_time_s`.

`raw_fps_mean` is the steady-state mean the oracle gates on; `raw_fps_std`/`_min`/`_max`
are audit/debug context. The distribution/percentile fields are `null` because the
runtime bundle serializes only aggregates, not the raw per-frame series.

### `perf_smoke_test_info.json` (Phase 1 output, copied from `benchmark_runtime_*.json`)

A schema-v1 `RuntimeBundle` — a nested dict serialized by
`isaaclab.test.benchmark.serialize.write_bundle_file` (defined by `RuntimeBundle` in
`source/isaaclab/isaaclab/test/benchmark/schema.py`). Top-level keys are `run`,
`versions`, `hardware`, `runtime`, `resources`, `extra`, and `schema_version`. This
replaces the legacy phase-array output; there are no more `TestPhase`/measurement
objects or prefixed measurement names. Abbreviated (many `versions.*` fields omitted):

```json
{
  "run": {
    "run_id": "physx-Isaac-Cartpole-Direct-42-20240610T090000",
    "framework": null,
    "config": {"physics_backend": "physx", "rendering_backend": "none", "presets": []},
    "task": "Isaac-Cartpole-Direct",
    "seed": 42,
    "start_time_utc": "2024-06-10T09:00:00+00:00",
    "end_time_utc": "2024-06-10T09:00:48+00:00",
    "duration_s": 48.0,
    "status": "completed",
    "num_envs": 4096,
    "max_iterations": null
  },
  "versions": {
    "isaaclab": "2.1.0", "isaacsim": "4.5.0", "kit": null, "newton": "1.0.0",
    "warp": "1.6.0", "mjwarp": "0.3.0", "torch": "2.3.0+cu121",
    "git_commit": "a1b2c3d4e5f6...", "git_branch": "develop", "git_dirty": false,
    "isaaclab_physx": "2.1.0", "isaaclab_newton": "2.1.0", "isaaclab_ov": null,
    "cuda_bindings": "12.1"
  },
  "hardware": {
    "hostname": "runner-01",
    "gpu_devices": [{"name": "NVIDIA L40S", "mem_gb": 45.62, "compute_cap": "8.9"}],
    "cpu_name": "Intel Xeon Gold 6438Y+", "cpu_count": 32, "ram_gb": 251.5
  },
  "runtime": {
    "startup_time_s": {"app_launch": 8.0, "env_creation": 3.0, "first_step": 0.3,
                       "python_imports": 1.2, "task_config": null},
    "iterations_completed": 200,
    "total_wall_time_s": 0.5,
    "steps_per_iteration": 4096,
    "iteration_time_s": {"mean": 0.00247, "std": 0.00002, "peak": 0.00251},
    "collection_fps": {"mean": 1655000.0, "std": 9800.0, "peak": 1680000.0},
    "total_fps": {"mean": 1655000.0, "std": 9800.0, "peak": 1680000.0},
    "iterations_per_s": {"mean": 404.9, "std": 3.2, "peak": 407.0}
  },
  "resources": {
    "gpu_util_pct": {"mean": 96.0, "std": 2.0, "peak": null},
    "gpu_mem_gb": {"mean": 18.0, "std": 0.1, "peak": 18.4},
    "cpu_util_pct": {"mean": 30.0, "std": 3.0, "peak": null},
    "ram_gb": {"mean": 12.0, "std": 0.4, "peak": 12.6}
  },
  "extra": {"num_frames": 300, "warmup_frames": 100},
  "schema_version": "1.0"
}
```

`benchmark_result_adapter` reads this bundle: the gate metric `raw_fps_mean` comes from
`runtime.total_fps.mean` (steady-state — warmup was excluded at the source by
`perf_runtime.py --warmup_frames`), `raw_fps_min` from `run.num_envs /
runtime.iteration_time_s.peak`, and provenance/`gpu_diag` from `versions`, `hardware`,
and `resources`. The bundle stores only aggregates, so the raw per-frame FPS series (and
hence percentiles) is not available.

### `samples.ndjson` (baseline history)

One JSON object per accepted baseline sample, append-only. The exact metadata can grow
without changing the file contract; the required fields for threshold calculation are:

```json
{"fps": 1667482.3, "gpu_model": "l40s", "task_id": "Isaac-Cartpole-Direct-v0", "backend_key": "physx", "launch_config_hash": "...", "benchmark_contract_hash": "...", "runtime_contract_hash": "...", "baseline_epoch": 1}
```
