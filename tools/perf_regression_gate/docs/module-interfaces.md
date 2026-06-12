# Performance Regression Gate — Module Interface Reference

Full function signatures, CLI arguments, and JSON schemas for every module.
All commands run from the `IsaacLab/` repository root unless otherwise noted.

---

## `oracle.py`

### `compare()`

```python
def compare(
    bench_result: dict,
    baseline: Baseline | None,
    fps_mean_floor: float,
    excluded_frames: frozenset[int],
    artifact_dir: Path,
    overrides: dict | None = None,
) -> OracleResult
```

The central verdict function. Reads `perf_regression_gate_info.json` from `artifact_dir`,
applies `excluded_frames`, computes mean FPS, and returns an `OracleResult`. Uses the
floored median+MAD bands (`spread = max(1.4826*MAD, min_spread_pct%*center)`), trusts the
window only at `sample_count >= MIN_WINDOW` (5), and applies per-task `overrides`.

| Parameter | Type | Description |
|---|---|---|
| `bench_result` | `dict` | Loaded `perf_regression_gate_result.json` |
| `baseline` | `Baseline \| None` | Rolling baseline stats; `None` for seed run |
| `fps_mean_floor` | `float` | Relative hard floor (`fps_floor_pct%*ref_fps`); 0.0 = disabled |
| `excluded_frames` | `frozenset[int]` | 0-based frame indices to drop before computing mean |
| `artifact_dir` | `Path` | Directory containing `perf_regression_gate_info.json` |
| `overrides` | `dict \| None` | Merged per-task/gpu overrides (`k_warn`/`k_block`/`min_spread_pct`/`pin_center_fps`/`pin_spread_fps`/`skip`/`tail_p99_warn`) |

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
```

`fps_median`, `fps_p5`, `fps_p95` are informational — they do not affect the verdict.
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

Default path: `tools/perf_regression_gate/tasks.json` (sibling of the module).

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
    excluded_frames_raw: list[int | list[int]]  # Raw JSON; use .excluded_frames (per-backend)
    camera_resolution: tuple[int, int] | None
    timeout_minutes: int
    ref_fps: dict[str, float]       # per-GPU calibrated reference, e.g. {"NVIDIA L40S": 276401.7}
    fps_floor_pct: float            # catastrophic floor as % of ref_fps; use .fps_floor(gpu)
    caches: list[str]               # Cache identifiers from caches_for_backend()
    tags: list[str]                 # ["always"] or ["camera"]
    enable_cameras: bool            # pass --enable_cameras (camera tasks)
    task_type: str                  # "benchmark"
    runs_on: str                    # "gpu-l40s"
    seed: int | None                # Random seed for benchmark (default 42)

    @property
    def backend_key(self) -> str:
        # "{physics}_{render}" if render_backend else "{physics}"

    @property
    def excluded_frames(self) -> frozenset[int]:
        # Expands excluded_frames_raw ranges to individual indices
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
Loads `stats.json` for a task/backend pair. Returns `None` if file does not exist (seed run).

```python
def update_baseline(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint=None
) -> None
```
Appends `fps` to `window.ndjson`, recomputes `median` and `MAD`, writes `stats.json`.
Only call for PASS/WARN results — aggregate.py enforces this policy.

```python
def delete_baseline_files(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None
) -> None
```
Removes `stats.json` and `window.ndjson` for a task/backend pair. Used in tests and resets.

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
def load_baseline_git(branch: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None) -> Baseline | None
def update_baseline_git(branch: str, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint: str | None) -> None
def seed_baseline_with_spread_git(branch, gpu_model, task_id, backend, center_fps, noise_fps, n_samples, seed, fingerprint) -> None
def delete_baseline_files_git(branch, gpu_model, task_id, backend, fingerprint) -> None
```

Git variants use `baseline_worktree()` — a context manager that creates a temporary git
worktree on `branch`, yields the worktree path, then commits any changes and removes the
worktree. Raises `RuntimeError` if `branch` is not found locally.

### File paths (flat-file)

```
{baselines_dir}/{gpu_model}/{task_id}/{backend}/stats.json
{baselines_dir}/{gpu_model}/{task_id}/{backend}/window.ndjson

With fingerprint:
{baselines_dir}/{gpu_model}/{task_id}/{backend}/{fingerprint}/stats.json
{baselines_dir}/{gpu_model}/{task_id}/{backend}/{fingerprint}/window.ndjson
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
python3 tools/perf_regression_gate/tasks_to_ci_matrix.py
```

No arguments. Reads `tasks.json` via `load_tasks()` and prints a JSON array to stdout,
one object per `(task_id, backend)` combination. Used by the `build_matrix` step in
`perf-regression-gate.yaml` to populate the GitHub Actions job matrix.

Each object contains: `task_id`, `physics_backend`, `render_backend` (empty string if none),
`num_envs`, `num_frames`, `bench_timeout_s`, `job_timeout_minutes`.

---

## `build_bench_result.py` CLI

```
python3 tools/perf_regression_gate/build_bench_result.py \
    --task_id <str>           task identifier (required)
    --physics_backend <str>   "physx" or "newton" (required)
    --render_backend <str>    render backend name or "" for none (default: "")
    --artifact_dir <path>     directory for artifacts (required)
    --exit_code <int>         Phase 1 process exit code (required)
    --wall_time_s <float>     Phase 1 wall-clock time in seconds (required)
    --timeout_s <float>       Phase 1 timeout in seconds (required)
    --log_file <path>         combined stdout+stderr log from Phase 1 (default: none)
    --attempt <int>           attempt number: 1 = first try, 2 = after retry (default: 1)
    --was_retried             flag: set when this result comes from a retry
```

**Output:** `{artifact_dir}/perf_regression_gate_result.json` (always written).

**Side effect:** Renames `benchmark_non_rl_{task_id}_{timestamp}.json` to
`perf_regression_gate_info.json` if the canonical name does not already exist.

When `perf_regression_gate_info.json` is present, `build_bench_result.py` also
calls `_extract_info_provenance()` to populate the FPS distribution, startup time,
GPU diagnostics, and full software/hardware/git provenance directly into the result JSON.
`nvidia-smi` is queried once at post-processing time to capture the driver version.

---

## `aggregate.py` CLI

```
python3 tools/perf_regression_gate/aggregate.py \
    --artifacts_dir <path>         root directory containing per-task artifact subdirectories (required)
    --gpu_model <str>              GPU model label for baseline lookup (default: L40S)
    --gate_config <path>           path to gate_config.json (default: perf_regression_gate/gate_config.json)
    --baseline_branch <str>        git branch for baseline storage (default: angehu/perf-baselines)
    --baselines_dir <path>         flat-file baseline directory; bypasses git (default: None = use git)
    --allow_baseline_update <str>  "true"/"false": extend baseline window for PASS/WARN (default: false)
    --summary_file <path>          append step-summary markdown to this path (default: none)
```

**Exit codes:**
- `0` — gate is non-blocking, or all tasks PASS/WARN
- `1` — any BLOCK verdict and `gate_config.blocking == true`
- `2` — any HARD_FAILURE verdict and `gate_config.blocking == true`

**Environment variable:** When `GITHUB_OUTPUT` is set, writes `baselines_updated=true` to
it after a successful baseline update. Used by the CI workflow to decide whether to push.

---

## `local_runner.py` CLI

```
python3 tools/perf_regression_gate/local_runner.py \
    --tags <tag ...>          task tags to run (default: always)
    --gpu_model <str>         GPU model label for baselines (default: L40S)
    --artifacts_dir <path>    root for per-task artifacts (default: perf_regression_gate/artifacts/)
    --baselines_dir <path>    flat-file baseline dir (default: perf_regression_gate/local_baselines/)
    --allow_baseline_update   extend baseline window for PASS/WARN results
    --dry_run                 print task matrix and exit without running anything
    --skip_existing           skip tasks whose perf_regression_gate_result.json already exists
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

Loads `gate_config.json`. Currently a stub that returns `{"blocking": False}`.
When the CI workflow PR lands, this will read the JSON file from disk.

The dict contract: `{"blocking": bool}`. `aggregate.py` uses `gate_config.get("blocking", False)`.

---

## `dev/stub_benchmark.py` CLI

Simulates `benchmark_non_rl.py` for testing. Does not require IsaacSim.

```
python3 tools/perf_regression_gate/dev/stub_benchmark.py \
    --task_id <str>
    --backend <str>
    --num_envs <int>       (default: 1)
    --num_frames <int>     (default: 200)
    --out_dir <path>       (required)
    --fps_mean <float>     (default: 200.0)
    --failure_phase <str>  "none" | "import" | "init" | "runtime" (default: none)
```

On success: writes `perf_regression_gate_info.json` with a Gaussian FPS series centered on
`--fps_mean`, prints `"Step Frametimes"` to stdout, exits 0.

On failure modes:
- `import`: prints traceback-like output, exits 1 (no perf file written)
- `init`: prints `"AppLauncher initialization complete"`, exits 2 (no perf file written)
- `runtime`: writes perf file, prints frametimes, then prints error and exits 3

---

## `dev/sim_regression.py` CLI

Injects degraded FPS artifacts for demo/testing without re-running benchmarks.

```
python3 tools/perf_regression_gate/dev/sim_regression.py \
    --fps_scale <float>    multiply baseline FPS by this factor (default: 0.53 = 47% regression)
    --tags <tag ...>       task tags to include (default: always)
    --gpu_model <str>      GPU model label (default: L40S)
    --baselines_dir <path> (default: perf_regression_gate/local_baselines)
    --out_dir <path>       output artifacts directory (default: /tmp/sim_artifacts)
```

For each task with an existing baseline, reads `stats.json`, computes
`regressed_fps = baseline.median_fps × fps_scale`, and writes:
- `{out_dir}/{task_id}/{backend_key}/perf_regression_gate_info.json`
- `{out_dir}/{task_id}/{backend_key}/perf_regression_gate_result.json`

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
      "fps_mean_floor": {
        "<gpu_model>": {
          "<backend_key>": <float>      // 0.0 = disabled
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

### `perf_regression_gate_result.json` (Phase 2 output)

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
  "perf_regression_gate_info_present": true,
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

**Fields populated from `perf_regression_gate_info.json`** (all `null` when
`perf_regression_gate_info_present` is false):

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

Key fields the oracle reads: `perf_regression_gate_info_present`, `failure_phase`,
`was_retried`, `gpu_diag.gpu_mem_used_mb`, `startup_time_s`, `wall_time_s`.

The `raw_fps_*` fields capture the full unfiltered distribution and are for audit/debug;
`oracle.compare()` recomputes mean FPS independently after applying `excluded_frames`.

### `perf_regression_gate_info.json` (Phase 1 output, renamed from `benchmark_non_rl_*.json`)

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

### `stats.json` (baseline statistics)

```json
{
  "median_fps": 1667482.3,
  "mad_fps": 8337.4,
  "k_warn": 2.5,
  "k_block": 4.0,
  "sample_count": 8
}
```

### `window.ndjson` (baseline raw window)

One FPS float per line, append-only:

```
1644720.5
1668000.1
1655000.0
...
```
