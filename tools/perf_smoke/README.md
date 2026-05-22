# Perf Smoke Gate (Phase 1)

Tooling for the Isaac Lab PR-time performance regression smoke gate.

## What this directory contains

| File | Purpose |
|---|---|
| `check_perf_regression.py` | Comparator. Reads a benchmark JSON + baseline, decides PASS / REGRESSION / HARD_FAILURE. |
| `baseline.json` | Per-task, per-GPU expected FPS and tolerance. |
| `test_check_perf_regression.py` | Stdlib `unittest` tests for the comparator. |

The CI workflow that drives this lives in `.github/workflows/build.yaml` (added once the platform decision lands).

## Exit-code contract

| Exit code | Meaning | What the CI workflow does |
|---|---|---|
| `0` | PASS — measured FPS within or above the baseline tolerance | Step succeeds. |
| `1` | REGRESSION — measured FPS dropped beyond `max_regression_pct` | Step exits non-zero; job is `continue-on-error: true` for Phase 1, so the PR check renders neutral. |
| `2` | HARD_FAILURE — structural problem (missing/malformed result, missing baseline entry, unknown GPU, NaN/zero FPS, etc.) | Step exits non-zero with a distinct annotation; this is the path that catches things like the `tomllib` incident. |

The split between `1` and `2` is intentional. A code-driven slowdown is not the same kind of signal as a broken environment, and the workflow renders them differently.

## Running the comparator

```bash
./isaaclab.sh -p tools/perf_smoke/check_perf_regression.py \
    --task Isaac-Cartpole-Direct-v0 \
    --results-dir ./perf-output \
    --baseline tools/perf_smoke/baseline.json
```

Useful flags:

- `--results-glob 'benchmark_non_rl_*.json'` — override the default glob (`benchmark_non_rl_<task>*.json`).
- `--allow-multiple` — pick the most recent file when multiple match.
- `--gpu-override 'NVIDIA L40'` — bypass the GPU auto-detection from the result JSON's `hardware_info` phase. Useful for local testing on different hardware.

The comparator prints a single structured line to stdout, e.g.:

```text
RESULT=PASS task=Isaac-Cartpole-Direct-v0 gpu=NVIDIA L40 baseline_fps=626772 measured_fps=635000 delta_pct=+1.31 threshold_pct=10.00
```

When `$GITHUB_STEP_SUMMARY` is set, the same line is appended to the job summary in a markdown code block.

## Running the tests

```bash
./isaaclab.sh -p tools/perf_smoke/test_check_perf_regression.py
# or
python3 tools/perf_smoke/test_check_perf_regression.py
```

Tests use stdlib `unittest` (not `pytest`) because `tools/conftest.py` blocks pytest collection beneath `tools/`. They have no Isaac Lab runtime dependencies and should pass on any Python ≥ 3.10.

## Baseline schema

`baseline.json` is keyed by task, then by GPU name. Substring matching is allowed: a baseline key `"L40"` matches a device named `"NVIDIA L40"`.

```jsonc
{
  "Isaac-Cartpole-Direct-v0": {
    "preset": "default",
    "num_envs": 4096,
    "num_frames": 100,
    "per_gpu": {
      "NVIDIA L40": {
        "baseline_fps": 626772,
        "max_regression_pct": 10.0,
        "n": 509,
        "window_days": 77,
        "source": "OmniPerf historical L40 Ada (EPYC_9124P_1XL40_ADA), P25 of WARM Mean Environment step effective FPS"
      }
    }
  }
}
```

`baseline_fps` and `max_regression_pct` are required. Other fields are informational and travel with the entry so reviewers can see how it was derived.

### Choosing `max_regression_pct`

For Phase 1 we use `max(3 × cross-runner CV%, 5%)` rounded up to the nearest whole percent, capped at 15%. The Phase 1 baseline above uses **10%** because:

1. The OmniPerf historical pooled CV on L40 Ada is 2.62% (3× ≈ 8%).
2. We do not yet have same-runner-vs-cross-runner CV decomposed; pooled may understate cross-runner.
3. Sporadic ~12-17% drops appear roughly 1-in-30 historical runs and recover on retry — likely contention/thermal, not code. A 10% threshold absorbs typical contention while still catching the 47% PR #5265-style real regression.

We will tighten once we have ≥ 20 of our own PR runs to characterize.

### Choosing `baseline_fps`

We use the **P25** of the WARM `Mean Environment step effective FPS` distribution from a stable post-`num_envs`-drift window, not the median. P25 absorbs more of the historical contention noise and gives natural headroom on a `max(3×CV, 5%)` threshold.

## Adding a new task

1. Pull the OmniPerf historical distribution for the task at the chosen `(num_envs, num_frames, preset)`.
2. Confirm pooled CV ≤ 4% on the L40 Ada subpool.
3. Pick `baseline_fps` = P25 of the WARM distribution from a stable window.
4. Pick `max_regression_pct` per the formula above.
5. Add the entry to `baseline.json`, including `source` and `notes`.
6. Add a CI matrix entry (workflow file, once it exists).
7. Validate against ≥ 20 historical PRs before adding the task to a blocking gate.

## Phase 1 invariants

These constraints exist so the gate can be reasoned about independently of the runner platform decision:

- The comparator is **runtime-free**: no torch, no Kit, no Isaac Sim imports. It runs from a stock Python ≥ 3.10.
- The result JSON shape is the OmniPerf backend output — see `source/isaaclab/isaaclab/test/benchmark/backends.py::OmniPerfKPIFile`. Other backends (`json`, `osmo`) write a different shape and are not supported.
- The metric path is fixed: `runtime["Mean Environment step effective FPS"]`. Other phases / metrics are out of scope for Phase 1.
- The GPU is read from `hardware_info.gpu_devices[hardware_info.gpu_current_device].name`, falling back to the first device in the map. Override via `--gpu-override` for local runs.

## Out of scope for Phase 1

- N-of-M sampling and same-job retries (Phase 1.5).
- Camera tasks, training tasks (`benchmark_rsl_rl.py`).
- Per-component (Isaac Sim, Newton, OVRTX, OVPhysX) bisection (Phase 2).
- PR comments, Slack notifications, dashboards.
- Writing baselines back to OmniPerf DB.
