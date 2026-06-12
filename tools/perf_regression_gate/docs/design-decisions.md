# Additions & Design Decisions — Unified Gate vs. the Original POC

**Status:** unified-POC branch; non-GPU pipeline validated end-to-end, RTX PRO 6000 run pending
**Scope:** what we changed on top of the original perf-gate POC, and *why*

---

## 0. Starting point

The original POC (the `tools/perf_regression_gate/` spine) is a clean, modular gate:

```
tasks.json  ──►  benchmark_non_rl.py  ──►  build_bench_result.py  ──►  aggregate.py  ──►  oracle.py
(single source)   (run a task)            (classify the run)          (decide gate)      (verdict math)
                                                                            │
                                                                  baseline_manager.py
                                                                  (git orphan-branch store)
```

We kept this architecture **wholesale** — the three-phase split, `tasks.json` as the
single source of truth, the failure-phase taxonomy, the bisect verdict mapping, and the
git orphan-branch baseline storage. It is the better skeleton and we did not rewrite it.

What follows is the layer we added: a hardened threshold/storage model, run-provenance
checks, and the operational wiring needed to actually run on the NVIDIA fleet. Every
behavioral change below is locked by a unit or property test under `tests/`.

---

## 1. Threshold model — six hardening fixes

The original gate compared a measured FPS mean against a baseline median ± a fixed number
of MADs (median absolute deviations). That core idea is sound; the edges were not safe.

| # | Original behavior | Problem (real, not hypothetical) | What we changed |
|---|---|---|---|
| 1 | `spread = MAD` | A freakishly stable task (or a small window) drives `MAD→0`. The band collapses onto the median, so trivial run-to-run noise trips a **false BLOCK**. | **Anti-flap spread floor:** `spread = max(1.4826·MAD, min_spread_pct%·center)`. The band can never be narrower than a small % of the center. |
| 2 | MAD trusted from sample #1 | With 1–2 samples the MAD is meaningless, yet it set the band — a single early sample could gate every later PR. | **`MIN_WINDOW = 5`:** below five samples the run *seed-PASSes* (rubber-stamp). No calibrated reference ⇒ no gating. |
| 3 | Cumulative history | The window grew without bound, so a genuine, intended speedup took longer and longer to become the new normal (old slow samples kept dragging the median). | **Capped FIFO rolling window** (`WINDOW_MAX = 20`): oldest sample evicted on append, so stats track *recent* behavior. |
| 4 | Absolute hard floor | The catastrophic floor was an absolute FPS number that, given unit/scale drift, could be set so low it **never fired** (a dead guardrail). | **Relative hard floor:** `fps_floor_pct%` of a per-GPU calibrated `ref_fps` in `tasks.json`. Scales with the task; `0` disables it. |
| 5 | Single push to baseline branch | Two PRs finishing at once both push the orphan branch; the second clobbers or rejects the first — a **lost sample**, silently. | **Concurrency-safe push:** bounded `fetch → rebase → push` retry loop in `baseline_manager`, so concurrent runs interleave instead of racing. |
| 6 | Single-bucket baselines | One baseline per (gpu, task, backend). A Warp/driver bump changes the performance regime but reuses the same bucket, comparing apples to oranges. | **Fingerprint bucketing + fallback chain:** baselines bucket by `{backend_version}/{runtime_hash}/{code_fingerprint}`; loads relax outward (`a/b/c → a/b → a → flat`) so a bump still gates against the *nearest* history instead of disabling the gate. |

Fixes 1–4 are pure `oracle.py` / `task_config.py` math; 5–6 live in `baseline_manager.py`.
Tests: `test_oracle.py`, `test_baseline_manager.py`, and the property checks in
`test_robustness.py` (monotonicity, spread-floor bounds, window-cap recency invariant).

---

## 2. Run-provenance checks (ported robustness)

A fast benchmark number is worthless if the run that produced it wasn't the run we asked
for. The original gate trusted the FPS at face value. We added two guards in
`build_bench_result.py`:

- **Config-drift guard.** Before trusting a number, we compare what the benchmark
  *actually ran* (task id, `num_envs`, seed, frame count, physics backend) against what the
  gate *launched*. A mismatch is reported as `failure_phase="config_mismatch"` and the
  oracle turns it into a **HARD_FAILURE**, not a regression. This catches the silent case
  where a config edit changes the workload and the FPS shift is misread as a perf change.
- **Step-time debug KPIs.** We extract `p99_over_median` (tail spike ratio), outlier
  count/magnitude, and a `warmup_flag` from the post-warm-up step times. These are
  diagnostics; one is optionally promotable to a verdict (next item).

> **Warm-up reconciliation.** The original used a single blanket `excluded_frames:[[0,100]]`.
> We moved to **per-backend** exclusion (`tasks.json`): PhysX `[[0,1]]`, Newton `[[0,4]]`,
> camera `[[0,59]]` — because Newton JIT and the RTX renderer have very different warm-up
> tails. A blanket window either keeps warm-up pollution (PhysX) or throws away real steady
> frames (camera). Per-backend is tighter and is validated in `test_task_config.py`.

---

## 3. Manual override file

We added `baseline_overrides.json` (committed with the PR, merged per task then per GPU):

- `k_warn` / `k_block` / `min_spread_pct` — per-task band tuning.
- `pin_center_fps` / `pin_spread_fps` — declare a new center for an **intended** perf change
  so the gate accepts it immediately instead of waiting for the window to re-converge.
- `skip` — quarantine a flaky task (forces PASS). Deliberately **cannot** rescue a crashed
  task: a missing-result/`config_mismatch` run stays HARD_FAILURE, because a crash is a real
  structural failure, not perf flakiness. Locked by
  `test_robustness.py::test_skip_does_not_rescue_a_crashed_task`.
- `tail_p99_warn` — opt-in advisory: WARN (never BLOCK) when the tail-spike ratio exceeds a
  per-task ceiling. Off by default; it surfaces tail regressions the FPS *mean* hides.

### A real bug this surfaced (pin-center band collapse)

Writing the robustness suite caught a genuine flaw in the merged oracle. When a *trusted*
window already existed, its spread was an absolute-FPS value (e.g. ±15 FPS around a
1,000-FPS center). Setting `pin_center_fps = 300000` moved the center but left that ±15
spread, so the band became ~0.005% of the new center and a run *right at* the intended new
center wrongly **BLOCKed** — the exact opposite of the override's purpose.

**Fix:** when `pin_center_fps` is set (and `pin_spread_fps` is not), re-floor the spread
against the *new* center: `spread = max(1.4826·MAD, min_spread_pct%·new_center)`. Locked by
`test_robustness.py::test_pinned_center_scales_its_own_spread`.

---

## 4. Environment: Warp 1.12, shim-free

The original explored Warp ≥1.13 plus a ~6-line `warp_replicator_shim.py` so the camera/RTX
task's `omni.replicator.core` import would coexist with the newer Warp. We **pinned the
validated stack** instead — `isaacsim==6.0.0.1`, `warp-lang==1.12.0`, `mujoco-warp==3.8.1` —
because on Warp 1.12 `omni.replicator.core` imports natively and the shim disappears
entirely. One fewer moving part in the container.

Trade-off: we give up the Warp-1.13-only `wp.tile_query_valid` path (rough-terrain Newton),
which our task set does not use. If a future task needs it, the escape hatch is Isaac Lab
3.0's kit-less OVRTX renderer (no replicator import), not the shim.

---

## 5. Operational wiring (so it actually runs on the fleet)

These are not POC-design changes, but they were required to turn the skeleton into a job
that runs on NVIDIA's self-hosted runners:

- **Container-based CI** (`perf-regression-gate.yaml`): runs the benchmark inside the NGC
  Isaac Sim image (Warp pinned at run time), instead of a ~20-min from-source
  `./isaaclab.sh --install` per job. Reproducible and avoids host Python (PEP 668) issues.
- **Warm JIT cache** (`actions/cache` + `--cache-dir` → `WARP_CACHE_PATH`/`CUDA_CACHE_PATH`):
  the cold cost is dominated by Newton JIT kernels, so carrying them across runs is the main
  speedup (RTX/shader caching is marginal — measured separately).
- **RTX PRO 6000 runner pinned.** The fork already has access via the `omniverse` project
  defaults, which include `nv-gpu-amd64-rtxpro6000-1gpu` — **no access PR was needed**. The
  workflow pins `linux-amd64-gpu-rtxpro6000-latest-1` rather than the generic
  `[self-hosted, gpu]`, because the fork is granted several GPU classes (H100, A40, …) and a
  generic label could land a baseline on the wrong hardware. Override via
  `vars.PERF_GATE_RUNS_ON`.
- **Per-task commit statuses + enriched step summary** (Center / Spread / Source / Note
  columns) so a reviewer sees *why* a verdict landed, not just the verdict.

---

## 6. Defaults & deferrals

- **Observe-only by default.** `gate_config` returns `blocking: False`, so the gate reports
  verdicts (and red/green per-task statuses) without failing PRs. The BLOCK→non-zero-exit
  path is wired and tested; flipping `blocking` to make it enforcing is a deliberate,
  separate PR once the baselines are trusted.
- **Deferred: memory-regression thresholding.** We record `gpu_mem_used_mb` but do not gate
  on it yet — left as future work to avoid scope creep.

---

## 7. Validation performed on this branch

- **58 unit/property tests** green (`pytest` under `tools/perf_regression_gate/`), isolated
  from the repo's parent `conftest.py` via a local `pytest.ini`.
- **Non-GPU end-to-end:** seed 12 samples → inject a 47% regression → gate **BLOCKs** every
  task with no baseline write → a healthy run **PASSes** and grows the window (12 → 13).
- **pre-commit** (ruff + format + hooks) clean on all touched files.
- **Live 4-task Warp-1.12 GPU sweep** is the remaining step, deferred to the RTX PRO 6000 run.
