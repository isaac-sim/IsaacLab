# IsaacLab CI Performance Regression Gate — System Design
**Status:** POC / MVP complete: pipeline runs end-to-end locally, CI integration pending
**Date:** 2026-06-10
**Owners:** Angelina Hu, Neil Mehta

---

## 0. Unified POC (what this branch is)

This branch merges two prototypes into one gate. It keeps the **modular 3-phase
architecture** (`tasks.json` -> `benchmark_non_rl.py` -> `build_bench_result.py` ->
`aggregate.py` -> `oracle.py`, with git orphan-branch `baseline_manager` storage) as
the spine, and hardens the threshold/storage layer plus the run-provenance checks.

### What came from where
| Area | Source | Notes |
|---|---|---|
| 3-phase pipeline, `tasks.json` single source, oracle/aggregate/build split, failure-phase taxonomy, bisect verdicts, git orphan-branch storage | architecture spine | adopted wholesale |
| Spread floor, MAD-sample-count gate, capped rolling window, fingerprint fallback chain, baseline-push concurrency safety, relative hard floor | threshold/storage hardening | the core of this work |
| Config-drift guard, tail-p99 + warm-up step-time KPIs, `baseline_overrides.json`, `--cache-dir` warm-cache sidecar | robustness ports | folded into `build_bench_result` / `aggregate` / `local_runner` |

### Hardening changelog (the six real fixes)
1. **Spread floor (anti-flap):** `spread = max(1.4826*MAD, min_spread_pct%*center)` so a
   `MAD->0` window can't collapse the band onto the median and BLOCK on noise.
2. **No MAD at n<5:** the window must hold `MIN_WINDOW=5` samples before its median+MAD
   is trusted; below that the run seed-PASSes (rubber-stamp, team decision).
3. **Actual rolling window:** `window.ndjson` is a capped FIFO (`WINDOW_MAX=20`); the
   oldest sample is evicted on append so stats track recent behavior.
4. **Relative hard floor:** the catastrophic floor is `fps_floor_pct%` of a per-GPU
   calibrated `ref_fps` (not a unit-mismatched absolute that could never fire).
5. **Baseline-push concurrency safety:** orphan-branch writes use a bounded
   fetch -> rebase -> push retry loop so concurrent runs can't lose a sample.
6. **Finished fingerprint bucketing + fallback chain:** baselines bucket by
   `{backend_version}/{runtime_hash}/{code_fingerprint}`; loads relax outward to looser
   buckets so a dependency/driver bump still gates against the nearest history.

### Environment (Warp 1.12, shim-free)
Pinned stack: `isaacsim==6.0.0.1`, `warp-lang==1.12.0`, `mujoco-warp==3.8.1`. On Warp 1.12
`omni.replicator.core` imports natively, so the old `warp_replicator_shim.py` is **dropped**.
Trade-off: no Warp-1.13-only `wp.tile_query_valid` (rough-terrain Newton) — not used by our
task set (Cartpole + G1-flat Newton).

### Explicitly deferred
- Memory-regression gating (`gpu_mem` ceiling/WARN) — data is captured; gating is a clean follow-up.

---

## 1. Purpose and Use Cases

The performance regression gate runs a fixed benchmark matrix on every PR and blocks merge
when throughput drops below a MAD-derived threshold relative to a rolling baseline.

**Use cases:**

| When | What happens |
|---|---|
| Feature PR touches physics or RL code | 5 "always" benchmarks run automatically |
| PR touches camera/rendering paths | 5 additional Shadow-Vision camera benchmarks added |
| Any task regresses > k_block × MAD below baseline | Aggregate exits 1; GitHub required check fails |
| Gate is in advisory mode (`blocking: false`) | Verdicts print but PR is not blocked |
| Baseline does not yet exist | Seed run: PASS unconditionally (no regression to measure) |
| Protected branch (main/develop) merges | Baseline window extended with the new FPS sample |

---

## 2. Design Principles

1. **One source of truth** Task and backend parameters live in `tasks.json`.
   Python, shell, and GitHub Actions YAML all read from it so there is no duplication of info.

2. **Modularizable** Logic should be back-end and task agnostic; backend is a data dimension in
   `tasks.json`, not a logic branch in `oracle.py`, `subprocess_runner.py`, or `task_config.py`.
   Each pipeline stage should have proper separation of concerns. Individual components should
   be agnostic to environment/call method as long as contract is maintained.

3. **Minimal invasiveness** Bench jobs directly call `benchmark_non_rl.py --benchmark_backend json`.
    They do not invoke `tools/conftest.py` at runtime, interfere with existing tests, or modify task code.

4. **Traceability** Every stage leaves informative artifacts. Every bench job writes `perf_regression_gate_result.json`
   regardless of success or failure so the aggregator always has a structured artifact to read.

5. **Lightweight** Bench jobs are only run when necessary and with minimally sufficient configs.
  Warmed caches are pulled when needed (Newton).

---

## 3. System Diagram

```
PR opened (to main / release / develop)
    │
    ▼
┌─────────────────────────────────────────┐
│  .github/workflows/perf-gate.yml        │
│                                         │
│  1. Expand task matrix from tasks.json  │
│  2. Activate tags from changed files    │
│  3. Fan out: one runner per task        │
└────────────────┬────────────────────────┘
                 │  (parallel per task)
      ┌──────────┴──────────┐
      ▼                     ▼
┌───────────┐         ┌───────────┐
│ Phase 1   │   ...   │ Phase 1   │  benchmark_non_rl.py
│ Cartpole  │         │ G1/newton │  --benchmark_backend json
│ /physx    │         │           │  writes benchmark_non_rl_*.json
└─────┬─────┘         └─────┬─────┘
      │                     │
      ▼                     ▼
┌───────────┐         ┌───────────┐
│ Phase 2   │   ...   │ Phase 2   │  build_bench_result.py
│           │         │           │  renames → perf_regression_gate_info.json
│           │         │           │  classifies failure_phase
│           │         │           │  writes perf_regression_gate_result.json
└─────┬─────┘         └─────┬─────┘
      │                     │
      └──────────┬──────────┘
                 │  (artifacts dir)
                 ▼
        ┌────────────────┐
        │   Phase 3      │  aggregate.py
        │   aggregate    │  oracle.compare() per task
        │   + oracle     │  prints verdict table
        │                │  updates baseline window (if allowed)
        └───────┬────────┘
                │
        ┌───────┴────────┐
        │  exit 0        │  all PASS/WARN, or gate non-blocking
        │  exit 1        │  any BLOCK + blocking=true
        │  exit 2        │  any HARD_FAILURE + blocking=true
        └────────────────┘
```

---

## 4. Repository File Layout

```
IsaacLab/
├── .github/
│   └── workflows/
│       ├── build.yaml                    MODIFIED: add image_tag output to build job
│       └── perf-regression-gate.yaml     CI gate workflow
│
└── tools/
    ├── conftest.py                       MODIFIED: one-line import change
    ├── subprocess_runner.py              run_benchmark(), classify_failure_phase()
    │                                     capture_test_output_with_timeout() borrowed from
    │                                     existing conftest CI infrastructure
    │
    └── perf_regression_gate/
        ├── __init__.py
        ├── tasks.json                    SINGLE SOURCE OF TRUTH — task/backend matrix
        ├── task_config.py                TaskConfig dataclass, load_tasks(), get_task()
        ├── tasks_to_ci_matrix.py         Converts tasks.json → GitHub Actions matrix JSON
        │                                 (called by perf-regression-gate.yaml build_matrix step)
        ├── oracle.py                     compare() → OracleResult; PASS/WARN/BLOCK/HARD_FAILURE
        ├── build_bench_result.py         Phase 2: reads log + benchmark JSON,
        │                                 extracts FPS stats + SW/HW/git provenance,
        │                                 writes perf_regression_gate_result.json
        ├── aggregate.py                  Phase 3: scans result JSONs, calls oracle,
        │                                 prints table, updates baselines, exits 0/1/2
        ├── baseline_manager.py           load/update baseline, flat-file + git variants
        ├── gate_config.py                load_gate_config() — reads {"blocking": bool}
        ├── local_runner.py               LOCAL END-TO-END RUNNER: orchestrates Phase 1+2+3
        |                                 without Github/Docker/cloud platform dependencies
        │
        ├── dev/
        │   ├── stub_benchmark.py         Simulates benchmark_non_rl.py for unit tests
        │   └── sim_regression.py         Injects regressed FPS artifacts for demos
        │
        ├── docs/
        │   ├── system-design.md          High-level overview
        │   ├── module-interfaces.md      Full function/CLI interface reference
        │
        └── tests/                        Unit tests (no GPU)
```

**Baseline storage:**

```
Local (testing):
tools/perf_regression_gate/local_baselines/
  {gpu_model}/{task_id}/{backend_key}/
    stats.json      {"median_fps", "mad_fps", "k_warn", "k_block", "sample_count"}
    window.ndjson   append-only rolling window, one FPS float per line

Production (planned):
perf-baselines branch (git orphan)
  {gpu_model}/{task_id}/{backend_key}/{backend_version}/{runtime_hash}/{code_fingerprint}/
    stats.json
    window.ndjson
    meta.json
```

---

## 5. Three-Phase Pipeline

```
Phase 1 — bench       tasks.json → matrix → benchmark_non_rl.py (one process per task/backend)
Phase 2 — post-bench  build_bench_result.py (reads log + benchmark JSON → writes result JSON)
Phase 3 — aggregate   aggregate.py + oracle → verdict table → baseline update
```

### Phase 1: `benchmark_non_rl.py`

Called via `./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py`:

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096 \
    --num_frames 300 \
    --benchmark_backend json \
    --output_path <artifact_dir> \
    [presets=newton_mjwarp]
```

Output: `benchmark_non_rl_{task_id}_{timestamp}.json` in `artifact_dir`.

- Only step that depends on IsaacLab run-time.
- Need the `--benchmark_backend json` because the JSON backend preserves `DictMeasurement`
objects including the raw per-step FPS list but the OmniPerf backend drops these.

### Phase 2: `build_bench_result.py`

Runs once per task after Phase 1 completes:

- Renames `benchmark_non_rl_*.json` → `perf_regression_gate_info.json`
- Classifies failure phase by scanning the benchmark log
- Parses the info artifact to extract FPS distribution statistics, startup time, GPU diagnostics, and full SW/HW/git provenance
- Writes `perf_regression_gate_result.json` (always written, even on failure)

### Phase 3: `aggregate.py`

Plain Python. Scans `--artifacts_dir` recursively for `perf_regression_gate_result.json`,
calls `oracle.compare()` for each, prints the verdict table, optionally updates baselines.

Exit codes: 0 = all clear or non-blocking; 1 = BLOCK + blocking mode; 2 = HARD_FAILURE + blocking mode.

---

## 6. Run Modes and Tag System

Each task entry in `tasks.json` has a `"tags"` array. CI activates tags from the PR's changed
file list; the benchmark matrix is filtered to tasks whose tags intersect the activated set.

| Tag | Meaning | Tasks |
|---|---|---|
| `"always"` | Run on every PR | Cartpole ×2, Factory ×1, G1 ×2 (5 tasks) |
| `"camera"` | Run when camera/rendering paths change | Shadow-Vision ×5 |

Shadow-Vision has `"camera"` rather than `"always"` because its FPS is dominated by rendering
cost, not physics and Factory already covers manipulation and high-contact behavior, so it is
only a signal when camera code changes to save test time cost.

**Tag activation rules WIP (production):**
- Any changed file → `"always"` always activated
- Files matching `source/isaaclab/sensors/**` or `source/isaaclab/envs/**/*vision*` → `"camera"` also activated

---

## 7. Full Task Matrix

The migrated unified matrix: 6 (task, backend) combinations. "Effective FPS" =
per-env FPS × num_envs. `ref_fps` is the per-GPU calibrated reference (L40S; 300f,
post-warm-up); the catastrophic floor is `fps_floor_pct` (40%) of it. Warm-up
(`excluded_frames`) is **per-backend**: PhysX `[[0,1]]`, Newton `[[0,4]]`, camera `[[0,59]]`.

| task_id | backend_key | num_envs | frames | timeout | tags | ref_fps (L40S) |
|---|---|---|---|---|---|---|
| Isaac-Cartpole | physx | 4096 | 300 | 10 min | always | 276401.7 |
| Isaac-Cartpole | newton | 4096 | 300 | 10 min | always | 358461.3 |
| Isaac-Factory-GearMesh-Direct-v0 | physx | 512 | 300 | 15 min | always | 880.5 |
| Isaac-Velocity-Flat-G1-v0 | physx | 2048 | 300 | 12 min | always | 19213.7 |
| Isaac-Velocity-Flat-G1-v0 | newton | 2048 | 300 | 12 min | always | 69660.2 |
| Isaac-Repose-Cube-Shadow-Vision-Direct-v0 | physx_isaacsim_rtx_renderer | 128 | 300 | 20 min | camera | 1024.6 |

`backend_key` = `{physics}` or `{physics}_{render}`. The launch passes an explicit
`physics=` token (`physx` / `newton_mjwarp`) plus `presets=` for the renderer, so the
run's reported backend is verifiable by the config-drift guard.

The hard floor is `fps_floor_pct% × ref_fps`; with no `ref_fps` for the GPU it is 0
(disabled) and only the baseline median+MAD bands apply.

---

## 8. Oracle Logic

```
compare(bench_result, baseline, fps_mean_floor, excluded_frames, artifact_dir, overrides=None)
  → OracleResult
```

**Verdict decision tree (hardened):**

```
perf_regression_gate_info_present == False?
    → HARD_FAILURE (file-based check skipped)

failure_phase == "config_mismatch"?
    → HARD_FAILURE (run used a different config than the gate launched)

Load perf_regression_gate_info.json, extract fps_series from runtime phase
Apply excluded_frames filter
filtered empty?
    → HARD_FAILURE

mean_fps = statistics.mean(filtered)
spread   = max(1.4826 * baseline.mad, min_spread_pct% * center)   # spread floor

overrides.skip?
    → PASS (quarantine)
mean_fps < fps_mean_floor?
    → BLOCK (relative catastrophic floor: fps_floor_pct% of ref_fps)

baseline is None or baseline.sample_count < MIN_WINDOW (5)? (and no pin)
    → PASS (seed run)

mean_fps < center - k_block (6.0) * spread?
    → BLOCK
mean_fps < center - k_warn (3.0) * spread?
    → WARN
else
    → PASS

verdict == PASS and tail_p99_warn set and p99_over_median exceeds it?
    → WARN (advisory tail check)
verdict == PASS and was_retried?
    → downgrade to WARN
```

**Bisect verdicts:**

| Oracle verdict | Condition | Bisect |
|---|---|---|
| PASS | clean first attempt | GOOD |
| PASS | was_retried | SKIP |
| WARN | any | SKIP |
| BLOCK | any | BAD |
| HARD_FAILURE | failure_phase in {init, runtime} | BAD |
| HARD_FAILURE | failure_phase in {import, driver, oom, hang, None} | SKIP |

---

## 9. Failure Phase Classification

`classify_failure_phase()` in `subprocess_runner.py` scans combined stdout+stderr in priority
order. Classification is entirely string-pattern-based — no backend branching.

| `failure_phase` | Trigger | Bisect |
|---|---|---|
| `"oom"` | exit 137 or `"oom-kill"` in stderr | SKIP |
| `"hang"` | wall_time ≥ timeout × 0.95 | SKIP |
| `"import"` | `"Traceback"` in output, no `"AppLauncher"` seen | SKIP |
| `"driver"` | `"CudaError"` or `"CUDA_ERROR_"` in output | SKIP |
| `"init"` | exit ≠ 0, AppLauncher present, no `"Step Frametimes"` | BAD |
| `"runtime"` | exit ≠ 0, `"Step Frametimes"` present (partial run) | BAD |
| `null` | exit 0, no error markers | (clean) |

`import` → SKIP because import failures indicate environment mismatch, not code regression.
`init` and `runtime` → BAD because Isaac Sim started normally: the failure is the code's fault.

---

## 10. Baseline Storage

**Flat-file (local testing):**

```
local_baselines/{gpu_model}/{task_id}/{backend_key}/
  stats.json      {"median_fps", "mad_fps", "k_warn", "k_block", "sample_count"}
  window.ndjson   one FPS float per line, append-only
```

`baseline_manager.update_baseline()` appends to `window.ndjson`, recomputes median and MAD,
and rewrites `stats.json`. BLOCK results are never written.

**Git branch (production):** `perf-baselines` orphan branch. `baseline_manager.update_baseline_git()`
uses a temporary git worktree to write and commit atomically.

**Write policy:** Writes occur only on protected branches (main/develop/release/*) or with
`--allow_baseline_update` (local testing flag). Feature branch runs are read-only.

---

## 11. Artifact Schema (Key Fields)

`perf_regression_gate_result.json` (Phase 2 output):

```json
{
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend": "physx",
  "failure_phase": null,
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
  "gpu_diag": {
    "gpu_name": "NVIDIA L40S",
    "gpu_total_memory_gb": 45.62,
    "cuda_version": "12.1",
    "nvidia_driver_version": "550.54.15",
    "gpu_mem_used_mb": 18432.0
  },
  "provenance": {
    "hardware": { "cpu_name": "...", "gpu_name": "NVIDIA L40S", "cuda_version": "12.1", "..." : "..." },
    "software": { "isaaclab": "2.1.0", "warp": "1.6.0", "torch": "2.3.0+cu121", "...": "..." },
    "git":      { "commit_hash": "a1b2c3d4...", "branch": "develop", "dirty": false }
  },
  "task_config_snapshot": { "..." : "..." }
}
```

`raw_fps_*` fields capture the unfiltered per-step distribution for audit/debug;
`oracle.compare()` recomputes mean FPS independently after applying `excluded_frames`.
`provenance` enables cross-run comparison: when two baselines diverge, diff their
`provenance` blocks to identify driver, CUDA, or package version changes.

See `module-interfaces.md` for the full schema with all fields.

`perf_regression_gate_info.json` (Phase 1 output, renamed from `benchmark_non_rl_*.json`):

A list of `TestPhase` objects. The oracle reads `"Environment step effective FPS"` from the
`"runtime"` phase's `"Step Frametimes"` measurement. `build_bench_result.py` also reads
the `"hardware_info"`, `"version_info"`, and `"startup"` phases for provenance extraction.

---

## 12. Environment Requirements (Local Testing)

Validated configuration as of 2026-06-10:

| Component | Version |
|---|---|
| IsaacSim | 6.0.0.1 (NOT 6.0.0.0 — `omni.physics.tensors.api` moved in 6.0.0.1) |
| isaacsim-extscache-physics | 6.0.0.1 |
| warp-lang | 1.12.0 (NOT 1.13.0 — `warp.context` removed, breaks omni.replicator.core) |
| mujoco-warp | 3.8.1 |
| Python | 3.12 |
| GPU | RTX 5090 (local) / L40S (production CI) |

**Installation after fresh `./isaaclab.sh -i --extra rl`:**

```bash
pip install isaacsim==6.0.0.1 isaacsim-extscache-physics==6.0.0.1
pip install warp-lang==1.12.0
```

---

## 13. Alignment with OVPLC Testing Principles

### 13.1 PR-gated runs vs. nightly-authoritative runs

**Principle:** Authoritative runs SHOULD execute on a nightly schedule.
**Deviation:** We run per-PR and write baselines from protected branches.
**Defense:** We use wide tolerance bands (k_warn=2.5, k_block=4.0 MAD) to
absorb inter-run variance. and the purpose of our tests is to catch issues live rather
than nightly (existing OmniPerf test suite).

### 13.2 Single FPS sample per run vs. N=10 iterations

**Principle:** Minimum N=10 iterations per benchmark.
**Deviation:** Each CI run produces one FPS value (mean of 200 post-warmup frames)
**Defense:** Simulation steps ARE independent samples and this gate is meant to be light-weight.
The baseline window accumulates samples across runs, building MAD statistics over
the true run-to-run variance distribution.

### 13.3 Memory tracking scope

**Current state:** `gpu_diag.gpu_mem_used_mb` is captured and surfaced in `OracleResult`
as an informational field. No memory regression threshold yet.

---

## 14. Bisect Engine (Future / TBD)

`oracle.py` already populates `bisect_verdict` (`GOOD`/`BAD`/`SKIP`) on every run.
A bisect engine would consume it directly and execute a similar workflow to local_runner:

```
tools/bisect/
  env_resolver.py    resolve_env(commit) → ResolvedEnv { isaaclab_root, python_launcher, … }
  bisect_runner.py   run_at_commit(env, task_id, backend) → (Path|None, SubprocessResult)
  bisect_engine.py   git bisect loop using OracleResult.bisect_verdict
```
