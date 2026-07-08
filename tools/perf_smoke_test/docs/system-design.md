# IsaacLab CI Performance Smoke Test — System Design
**Status:** POC / MVP running locally, pending productionization, deployment, deployment features
**Date:** 2026-06-15
**Owners:** Angelina Hu, Neil Mehta

---

## 1. Purpose and Use Cases

The performance smoke test runs a fixed benchmark matrix on every PR and blocks merge
when throughput drops below an explicit hard floor or a MAD-derived threshold relative to compatible rolling baseline samples.

**Use cases:**

| When | What happens |
|---|---|
| Feature PR touches physics or RL code | 5 "always" benchmarks run automatically |
| PR touches camera/rendering paths | 4 additional Shadow-Vision camera benchmarks added |
| Any task regresses > k_block × MAD below baseline | Aggregate exits 1; GitHub required check fails |
| smoke test is in advisory mode (`blocking: false`) | Verdicts print but PR is not blocked |
| Baseline does not yet exist | Seed run: WARN with `no_baseline` (transparent, non-blocking in advisory mode) |
| Protected branch (main/develop/release) merges | Baseline history extended with structured PASS/WARN samples |

---

## 2. Design Principles

1. **One source of truth** Task and backend parameters live in `tasks.json`.
   Python, shell, and GitHub Actions YAML all read from it so there is no duplication of info.

2. **Modularizable** Logic should be back-end and task agnostic; backend is a data dimension in
   `tasks.json`, not a logic branch in `oracle.py`, `subprocess_runner.py`, or `task_config.py`.
   Each pipeline stage should have proper separation of concerns. Individual components should
   be agnostic to environment/call method as long as contract is maintained.

3. **Minimal invasiveness** Bench jobs directly call `perf_runtime.py --benchmark_formatter schema`.
    They do not invoke `tools/conftest.py` at runtime, interfere with existing tests, or modify task code.

4. **Traceability** Every stage leaves informative artifacts. Every bench job writes `perf_smoke_test_result.json`
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
│  .github/workflows/perf-smoke-test.yaml │
│                                         │
│  1. Expand task matrix from tasks.json  │
│  2. Activate tags from changed files    │
│  3. Fan out: one runner per task        │
└────────────────┬────────────────────────┘
                 │  (parallel per task)
      ┌──────────┴──────────┐
      ▼                     ▼
┌───────────┐         ┌───────────┐
│ Phase 1   │   ...   │ Phase 1   │  perf_runtime.py
│ Cartpole  │         │ G1/newton │  --benchmark_formatter schema
│ /physx    │         │           │  writes benchmark_runtime_*.json
└─────┬─────┘         └─────┬─────┘
      │                     │
      ▼                     ▼
┌───────────┐         ┌───────────┐
│ Phase 2   │   ...   │ Phase 2   │  build_bench_result.py
│           │         │           │  copies → perf_smoke_test_info.json
│           │         │           │  classifies failure_phase
│           │         │           │  writes perf_smoke_test_result.json
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
        │  exit 0        │  all PASS/WARN, or smoke test non-blocking
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
│       └── perf-smoke-test.yaml     CI smoke test workflow
│
└── tools/
    ├── conftest.py                       MODIFIED: one-line import change
    ├── subprocess_runner.py              run_benchmark(), classify_failure_phase()
    │                                     capture_test_output_with_timeout() borrowed from
    │                                     existing conftest CI infrastructure
    │
    └── perf_smoke_test/
        ├── __init__.py
        ├── tasks.json                    SINGLE SOURCE OF TRUTH — task/backend matrix
        ├── task_config.py                TaskConfig dataclass, load_tasks(), get_task()
        ├── backend_identity.py           canonical physics/render backend identity
        ├── gpu_identity.py               canonical GPU buckets for baselines + floor lookup
        ├── runtime_contract.py           runtime compatibility contract/hash builder
        ├── launch_config.py              artifact-carried launch intent + hashes
        ├── write_launch_config.py        CI/local helper to write launch_config.json
        ├── github_gate_context.py        PR/merge/push context + baseline write policy
        ├── gate_types.py                 verdict/failure/threshold enums
        ├── tasks_to_ci_matrix.py         Converts tasks.json → GitHub Actions matrix JSON
        │                                 (called by perf-smoke-test.yaml build_matrix step)
        ├── oracle.py                     compare() → OracleResult; PASS/WARN/BLOCK/HARD_FAILURE
        ├── perf_runtime.py               Phase 1 driver: thin wrapper over the merged
        │                                 Isaac Lab benchmark core; writes a schema-v1
        │                                 RuntimeBundle (benchmark_runtime_*.json)
        ├── benchmark_result_adapter.py   RuntimeBundle → typed RuntimeSample projection
        │                                 (FPS aggregates, provenance, gpu_diag)
        ├── contracts.py                  typed gate artifacts (schema v1): RuntimeSample
        │                                 + BenchResult (perf_smoke_test_result.json)
        ├── build_bench_result.py         Phase 2: copies the runtime bundle, projects it
        │                                 via benchmark_result_adapter (FPS stats + SW/HW/git
        │                                 provenance), writes perf_smoke_test_result.json
        ├── aggregate.py                  Phase 3: scans result JSONs, calls oracle,
        │                                 prints table, updates baselines, exits 0/1/2
        ├── baseline_manager.py           load/update baseline, flat-file + git variants
        ├── gate_config.py                policy constants + runtime compatibility defaults
        ├── local_runner.py               LOCAL END-TO-END RUNNER: orchestrates Phase 1+2+3
        |                                 without Github/Docker/cloud platform dependencies
        │
        ├── dev/
        │   └── stub_benchmark.py         Writes a schema-v1 RuntimeBundle stub (no
        │                                 GPU/sim) for offline end-to-end gate testing
        │
        ├── docs/
        │   ├── system-design.md          High-level overview
        │   ├── module-interfaces.md      Full function/CLI interface reference
        │
        └── tests/                        Unit tests (no GPU)
```

`perf_runtime.py` and `stub_benchmark.py` build on the merged Isaac Lab benchmark
core under `source/isaaclab/isaaclab/test/benchmark/` (`schema.py`, `builders.py`,
`capture.py`, `serialize.py`, `stepping.py`, `formatters.py`, …), brought in from
upstream Isaac Lab's "Part 1" benchmark refactor. The gate depends only on the
stable, merged building blocks (the typed `RuntimeBundle` schema plus the
builders/capture/serialize helpers), not on the still-unmerged standalone
`scripts/benchmarks/runtime.py`.

**Baseline storage:**

```
Local (testing):
tools/perf_smoke_test/local_baselines/
  {gpu_model}/{task_id}/{backend_key}/
    samples.ndjson  append-only structured baseline samples

Production:
perf-baselines branch (git orphan)
  {gpu_model}/{task_id}/{backend_key}/
    samples.ndjson  append-only structured samples; compatibility fields live in each sample
```

---

## 5. Three-Phase Pipeline

```
Phase 1 — bench       tasks.json → matrix → perf_runtime.py (one process per task/backend)
Phase 2 — post-bench  build_bench_result.py (reads log + runtime bundle → writes result JSON)
Phase 3 — aggregate   aggregate.py + oracle → verdict table → baseline update
```

### Phase 1: `perf_runtime.py`

Called via `./isaaclab.sh -p tools/perf_smoke_test/perf_runtime.py`:

```bash
./isaaclab.sh -p tools/perf_smoke_test/perf_runtime.py \
    --task Isaac-Cartpole-Direct \
    --num_envs 4096 \
    --num_frames 300 \
    --warmup_frames 100 \
    --benchmark_formatter schema \
    --output_path <artifact_dir> \
    [presets=newton_mjwarp]
```

Output: `benchmark_runtime_{task_id}_{timestamp}.json` in `artifact_dir` — a
schema-v1 `RuntimeBundle` (see §11).

- Only step that depends on IsaacLab run-time.
- `perf_runtime.py` is a thin driver over the merged Isaac Lab benchmark core; the
  `--benchmark_formatter schema` output is a typed `RuntimeBundle` (nested dict of
  aggregates), which replaces the legacy JSON phase-array output.
- Warmup is excluded **at the source**: `--warmup_frames N` slices the leading `N`
  cold steps before aggregation, so the reported `runtime.total_fps.mean` is
  steady-state. This replaces the gate's previous post-hoc `excluded_frames` filter
  (the bundle deliberately does not serialize the raw per-frame series).
- Prints `Step Frametimes` to stdout as the progress marker
  `subprocess_runner.classify_failure_phase()` keys on to separate init- from
  runtime-phase failures.

### Phase 2: `build_bench_result.py`

Runs once per task after Phase 1 completes:

- Globs `benchmark_runtime_{task_id}_*.json` and **copies** it to the canonical
  `perf_smoke_test_info.json`
- Classifies failure phase by scanning the benchmark log
- Projects the `RuntimeBundle` into the gate's flat fields via
  `benchmark_result_adapter` (FPS aggregates, startup time, GPU diagnostics, and full
  SW/HW/git provenance)
- Computes `runtime_contract_hash` and publish-only runtime info
- Writes `perf_smoke_test_result.json` (always written, even on failure)

### Phase 3: `aggregate.py`

Plain Python. Scans `--artifacts_dir` recursively for `perf_smoke_test_result.json`,
calls `oracle.compare()` for each, prints the verdict table, optionally updates baselines.

Exit codes: 0 = all clear or non-blocking; 1 = BLOCK + blocking mode; 2 = HARD_FAILURE + blocking mode.

---

## 6. Run Modes and Tag System

Each task entry in `tasks.json` has a `"tags"` array. CI activates tags from the PR's changed
file list; the benchmark matrix is filtered to tasks whose tags intersect the activated set.

| Tag | Meaning | Tasks |
|---|---|---|
| `"always"` | Run on every PR | Cartpole ×2, Factory ×1, G1 ×2 (5 tasks) |
| `"camera"` | Run when camera/rendering paths change | Shadow-Vision ×4 |

Shadow-Vision has `"camera"` rather than `"always"` because its FPS is dominated by rendering
cost, not physics and Factory already covers manipulation and high-contact behavior, so it is
only a signal when camera code changes to save test time cost.

**Tag activation rules WIP (production):**
- Any changed file → `"always"` always activated
- Files matching `source/isaaclab/sensors/**` or `source/isaaclab/envs/**/*vision*` → `"camera"` also activated

---

## 7. Full Task Matrix

9 (task, backend) combinations. "Effective FPS" = per-env FPS × num_envs. `warmup` leading
steps are discarded at the source (`perf_runtime.py --warmup_frames`) before aggregation, so
the reported mean is steady-state; the gate averages the remaining `frames − warmup` steps.

| task_id | backend_key | num_envs | frames | warmup | timeout | tags | floor (L40S) |
|---|---|---|---|---|---|---|---|
| Isaac-Cartpole-Direct | physx | 4096 | 300 | 100 | 10 min | always | 100 |
| Isaac-Cartpole-Direct | newton | 4096 | 300 | 100 | 10 min | always | 0 |
| Isaac-Factory-GearMesh-Direct-v0 | physx | 512 | 300 | 100 | 15 min | always | 30 |
| Isaac-Velocity-Flat-G1-v0 | physx | 512 | 300 | 100 | 12 min | always | 40 |
| Isaac-Velocity-Flat-G1-v0 | newton | 512 | 300 | 100 | 12 min | always | 0 |
| Isaac-Repose-Cube-Shadow-Vision-Direct-v0 | physx | 512 | 300 | 100 | 20 min | camera | 20 |
| Isaac-Repose-Cube-Shadow-Vision-Direct-v0 | physx_newton_renderer | 512 | 300 | 100 | 20 min | camera | 0 |
| Isaac-Repose-Cube-Shadow-Vision-Direct-v0 | newton | 512 | 300 | 100 | 20 min | camera | 0 |
| Isaac-Repose-Cube-Shadow-Vision-Direct-v0 | newton_newton_renderer | 512 | 300 | 100 | 20 min | camera | 0 |

`backend_key` = `{physics}` or `{physics}_{render}`. Preset tokens are derived automatically
by `local_runner.py` and the CI workflow.

The "floor" column shows each backend's `BLOCK` `fps_mean_thresholds` entry. A floor of
0 is a valid, effectively non-gating threshold (measured FPS never drops below it), so
only the baseline MAD thresholds apply in practice. Each backend may configure multiple
thresholds (e.g. a `WARN` reference point alongside a `BLOCK` floor) and reporting-only
entries that are surfaced without gating.

---

## 8. Oracle Logic

```
compare(bench_result, baseline, fps_mean_thresholds, *, min_block_regression_pct=…)
  → OracleResult
```

**Verdict decision tree:**

```
config_mismatch present?
    → HARD_FAILURE (config_mismatch)

perf_smoke_test_info_present == False?
    → HARD_FAILURE (file-based check skipped)

mean_fps = bench_result["raw_fps_mean"]        # runtime.total_fps.mean, warmup already
                                               # excluded at source by perf_runtime.py
mean_fps missing / non-numeric?
    → HARD_FAILURE (missing_fps_mean)
mean_fps <= 0?
    → HARD_FAILURE (zero_fps)

# threshold_verdict: worst gating threshold crossed (mean_fps < value); PASS if none.
#   crossed reporting-only thresholds are recorded but do not affect threshold_verdict.

# baseline_verdict:
baseline is None?
    → WARN (no_baseline)
baseline.sample_count < MIN_BASELINE_SAMPLES?
    → WARN (insufficient_baseline)
mean_fps < baseline.median - 4.0 * baseline.mad AND regression_pct <= -MIN_BLOCK_REGRESSION_PCT?
    → BLOCK
mean_fps < baseline.median - 2.5 * baseline.mad?
    → WARN
else
    → PASS

verdict = worst(threshold_verdict, baseline_verdict)   # most severe wins (e.g. WARN floor + baseline BLOCK → BLOCK)

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
  samples.ndjson  append-only structured baseline samples
```

`baseline_manager.update_baseline()` appends one structured sample per accepted result.
The rolling median/MAD thresholds are computed from the newest compatible samples at read time.
BLOCK results are never written.

**Git branch (production):** `perf-baselines` orphan branch. `aggregate.py` reads from a
freshly fetched baseline branch SHA. Accepted PASS/WARN samples are pushed through
`baseline_manager.update_baselines_git()`, which refetches before writing, commits
append-only `samples.ndjson` updates in a temporary worktree, and retries the push
if another runner updates the branch first.

**Trigger/write policy:** Non-draft PRs and merge-queue candidates run the smoke test but do
not publish baselines. Protected-branch push events (main/develop/release/*, plus the
POC branch while enabled) publish accepted PASS/WARN samples through the transactional
git writer. Feature branch runs are read-only unless `--allow_baseline_update` is set
for local testing.

---

## 11. Artifact Schema (Key Fields)

`perf_smoke_test_result.json` (Phase 2 output):

```json
{
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend": "physx",
  "failure_phase": null,
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
    "hardware": { "cpu_name": "...", "gpu_name": "NVIDIA L40S", "cuda_version": "12.1", "..." : "..." },
    "software": { "isaaclab": "2.1.0", "warp": "1.6.0", "torch": "2.3.0+cu121", "...": "..." },
    "git":      { "commit_hash": "a1b2c3d4...", "branch": "develop", "dirty": false }
  },
  "task_config_snapshot": { "..." : "..." },
  "schema_version": "1.0"
}
```

The result is a typed `BenchResult` (`contracts.py`) serialized to this JSON via
`BenchResult.to_dict()`; consumers reconstruct it with `BenchResult.from_dict()`. The
schema is wire-stable — same keys as before, plus the additive `schema_version`.

`raw_fps_mean`/`_std`/`_max` come straight from the bundle's `runtime.total_fps`
aggregates, and `raw_fps_min` is recovered as `num_envs / iteration_time_s.peak`
(slowest steady-state step). Percentile/distribution fields (median, p5/p95, p99/median,
outlier counts) are **not emitted**: the schema stores only aggregates, and they were never
gating — `raw_fps_std` (spread) and `raw_fps_min` (worst steady-state frame) cover the tail.
The oracle reads `raw_fps_mean` directly (warmup already excluded at source), so no post-hoc
filtering happens here. `provenance` enables cross-run comparison: when two baselines
diverge, diff their `provenance` blocks to identify driver, CUDA, or package version changes.

See `module-interfaces.md` for the full schema with all fields.

`perf_smoke_test_info.json` (Phase 1 output, copied from `benchmark_runtime_*.json`):

A schema-v1 `RuntimeBundle` — a nested dict with top-level keys `run` / `versions` /
`hardware` / `runtime` / `resources` / `extra` / `schema_version` (defined by
`RuntimeBundle` in `source/isaaclab/isaaclab/test/benchmark/schema.py`).
`benchmark_result_adapter` reads it and projects `runtime.total_fps`,
`runtime.startup_time_s`, `hardware`, `resources`, and `versions` into the gate's flat
result fields.

---

## 12. Environment Requirements (Local Testing)

Validated local smoke configuration as of 2026-06-15:

| Component | Version |
|---|---|
| IsaacSim | 6.0.0.1 (NOT 6.0.0.0 — `omni.physics.tensors.api` moved in 6.0.0.1) |
| isaacsim-extscache-physics | 6.0.0.1 |
| warp-lang | 1.13.0 (required exactly by Newton `v1.2.0rc2`; Isaac Sim ≥ 6.0.0.1 replicator is compatible with the 1.13 API) |
| mujoco-warp | 3.8.1 |
| Python | 3.12 |
| GPU | RTX 5090 (local) / RTX PRO 6000 target runners / L40S historical reference |

**Installation after fresh `./isaaclab.sh -i --extra rl`:**

```bash
pip install isaacsim==6.0.0.1 isaacsim-extscache-physics==6.0.0.1
pip install warp-lang==1.13.0
```

---

## 13. Alignment with OVPLC Testing Principles

### 13.1 PR-gated runs vs. nightly-authoritative runs

**Principle:** Authoritative runs SHOULD execute on a nightly schedule.
**Deviation:** We run on non-draft PRs and merge-queue candidates, and publish baselines
from protected-branch pushes.
**Defense:** The smoke test is intended to catch merge-time regressions before they land. Baseline
publication remains restricted to trusted protected-branch states, and matching prefers
compatible nearest-ancestor samples when a base SHA is available.

### 13.2 Single FPS sample per run vs. N=10 iterations

**Principle:** Minimum N=10 iterations per benchmark.
**Deviation:** Each CI run produces one FPS value (mean of 200 post-warmup frames)
**Defense:** Simulation steps ARE independent samples and this smoke test is meant to be light-weight.
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
