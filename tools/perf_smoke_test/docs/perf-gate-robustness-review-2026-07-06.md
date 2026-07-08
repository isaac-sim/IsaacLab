# Perf Gate Robustness Review (2026-07-06): Context For A New Chat

This note captures the state of an in-progress review: **"Is the perf regression
gate robust enough to merge into real CI infra?"** Paste this file into a new
chat to resume without re-deriving everything below.

Provenance: current work lives on branch `neilm/bisection-on-perf-smoke`
(HEAD `619f9412e` at time of writing), repo `/home/horde/dev/IsaacLab-fork`.
The tool lives at `tools/perf_smoke_test/`.

---

## 0. Critical: "Unified POC" is not the branch you think it is

There is a local worktree/branch literally named `neilm/unified-poc-noise-flaky`
(`.worktrees/unified-poc-noise-flaky`). **Do not use it as the merge candidate.**

- It forks from an old commit (`1003c728c`) that predates the
  `perf_regression_gate -> perf_smoke_test` rename.
- It is missing the entire test suite, the evidence-pack tooling, and most of
  the bisection subsystem additions that now exist on current `HEAD`.
- It was never pushed to the shared `origin/unified-POC` remote branch (which
  itself sits at `46a6049c3`, already merged into `HEAD` back on 2026-06-24).
- Its one unique commit, `91ed8abf1` ("Calibrate perf gate spread and flaky
  handling"), is **not lost** — current `HEAD` independently re-implemented the
  same idea more thoroughly as a per-task/per-GPU/per-backend `noise_floor_pct`
  (see `oracle.py`, `aggregate.py`, `task_config.py`), backed by a passing test
  (`test_local_robustness.py::test_noise_floor_widens_rolling_threshold_for_jittery_cells`).

**Conclusion:** the actual merge candidate is current `HEAD`
(`neilm/bisection-on-perf-smoke`), not `neilm/unified-poc-noise-flaky`. That
branch should be treated as dead/abandoned.

**Bigger picture:** `origin/develop` (real NVIDIA-Omniverse/IsaacLab upstream)
has **zero trace** of `perf_smoke_test`/`perf_regression_gate` today. This has
never touched real CI infra — "merging it in" is a from-scratch integration,
not a small follow-up.

---

## 1. Unit tests: solid logic, but effectively invisible to CI

All 132 tests in `tools/perf_smoke_test/test/` pass. Coverage includes oracle
PASS/WARN/BLOCK boundaries, confirm-on-block re-run/demotion, noise floor
widening, runtime-contract/ancestry baseline filtering, env resolution, and the
bisection runner. Good foundation.

Two real gaps found:

### 1a. The documented test invocation is silently broken for this suite

`tools/conftest.py` (repo-wide) defines `pytest_sessionstart`, which
**unconditionally hijacks the pytest session** regardless of what path you
pass on the CLI. It walks only `source/` and `scripts/` for `test_*.py` files
(never `tools/`), then calls `pytest.exit(...)` to short-circuit normal
collection. Because pytest auto-discovers ancestor `conftest.py` files, simply
being under `tools/` triggers this hijack.

**Effect:** running the AGENTS.md-documented command,
`./isaaclab.sh -p -m pytest tools/perf_smoke_test/test/<file>`, does **not**
run your target tests. It silently redirects into running the *entire* main
IsaacLab test suite (hundreds of GPU/Isaac-Sim-dependent files) instead, with
no error telling you your requested tests never executed.

**Workaround (confirmed working):**
```bash
env_isaaclab/bin/python -m pytest tools/perf_smoke_test/test/ \
  --confcutdir=tools/perf_smoke_test -p no:cacheprovider
```
This bypasses `tools/conftest.py` entirely; all 132 tests run in ~1.3s (no
Isaac Sim dependency — pure logic tests).

### 1b. No CI workflow runs this test suite at all

`grep -rn "pytest" .github/workflows/perf-*.yaml` returns nothing.
`perf-smoke-test.yaml` only invokes the tool's scripts directly
(`validate_tasks.py`, `write_launch_config.py`, `build_bench_result.py`,
`aggregate.py`, `github_gate_context.py`) — never `pytest`. A regression in
`oracle.py`'s verdict math today would ship with zero automated protection.

**Action item:** fix the `conftest.py` hijack (or route around it) and add a
real CI job that runs the 132-test suite on every PR touching
`tools/perf_smoke_test/`.

---

## 2. Silent-fallback risk in the production hot path

`build_bench_result.py` (the actual script the CI workflow runs) calls
`fallback_launch_config()` from `launch_config.py` whenever `launch_config.json`
is missing on disk (e.g. an earlier `write_launch_config.py` step failed to
hand off the artifact). That fallback **hardcodes GPU key `"L40S"`**
(`launch_config.py` line ~157) for the hard-floor lookup, with **no warning
printed**. If this path fires on the real fleet (RTX PRO 6000 Blackwell, not
L40S), the gate would silently apply L40S-calibrated floors to the wrong
hardware instead of erroring loudly or resolving the actual runtime GPU model.

**Action item:** make this fail loudly, or resolve the real GPU model instead
of a hardcoded default.

---

## 3. Good news: it currently can't block anything

There is no `gate_config.json` anywhere in the repo (only `gate_config.py`,
the Python module with defaults). `load_gate_config()` defaults
`"blocking": False` when the file is absent — so **every gate run today is
advisory-only**. This matches Piotr's ask to keep it non-blocking for now, and
it's the main reason gaps #1/#2 haven't bitten anyone yet: nothing has failed
a PR because of them.

**Implication:** "merge into real CI infra" implicitly includes someone
deciding to author a `gate_config.json` with `"blocking": true`, and that flip
has never been exercised end-to-end. Treat it as a distinct, testable rollout
step, not a footnote.

---

## 4. Config coverage looks right for the actual fleet

`tasks.json` has calibration entries keyed by `rtx_pro_6000_blackwell` (not
just `l40s`), confirmed via `gpu_identity.py`'s GPU-model normalization. So the
primary tasks aren't running against an empty/default floor on the real
hardware — that part is in reasonable shape.

---

## 5. Self-documented known gaps

From `tools/perf_smoke_test/docs/poc-roadmap-next-steps.md` (already in-repo,
written by a prior session, not new info but worth restating):

1. **No automatic era->image resolution.** The gate always pulls
   `nvcr.io/nvidian/isaac-lab:latest-perf`; older-era testing (bisection,
   historical replay) requires manually setting `PERF_SMOKE_CI_IMAGE`.
2. **Bisection agent explicitly not ready to merge.** It was built against an
   older layout; the doc says outright it "should not be merged while it still
   depends on stale paths/imports" onto the current `perf_smoke_test` tree.

---

## 6. Statistical/hardware robustness gap (from prior investigation, same day)

Separate from code correctness — surfaced while investigating a 3.0-fork vs
2.3.2 performance regression using the evidence-collection workflow
(`.github/workflows/perf-smoke-regression-evidence.yaml`):

- **Runner fleet CPU heterogeneity confirmed directly from evidence artifacts.**
  Two evidence runs landed on the *same* GPU model (RTX PRO 6000 Blackwell
  Server Edition, different GPU UUID = different physical box) but
  *different* CPU SKUs: `INTEL(R) XEON(R) GOLD 5512U` on 4 of 5 runs vs
  `Intel(R) Xeon(R) 6731P` on one run (`perf-output/regression-evidence-28618991594`).
  Both are virtualized (hypervisor flag set), same vCPU count/cache topology —
  i.e. the GH Actions runner label (`linux-amd64-gpu-rtxpro6000-latest-1`) maps
  to a heterogeneous pool, not a fixed machine.
- **Within-run sample-order drift (JIT/warm-up bias) is large and
  task-dependent.** Three back-to-back repeat samples of the same task/host/
  commit showed monotonic drift that scales inversely with how overhead-bound
  the task is: Cartpole (overhead-bound) +28% sample1->sample3, G1 (mixed)
  +4.4%, Factory (physics-bound) +1.1%. Root cause: `run.log` shows
  `Warp UserWarning: Kernel cache artifacts from a previous Warp version were
  found in '/tmp/jit-cache/warp'. These will be ignored.` — the mounted JIT
  cache is stale/invalidated at run start, so repeat 1 pays a cold-compile tax
  and later repeats benefit from the now-warm cache.
- Ruled out as confounds (checked directly): GPU driver/CUDA version (identical
  `595.71.05`/CUDA 13.2), GPU power limit/clocks config, container-level
  multi-tenancy (`docker_stats.jsonl` showed exactly one container per host
  during sampled windows), `num_envs` config drift.
- **Gap:** CPU model isn't captured anywhere in the structured pipeline
  (`build_bench_result.py`/oracle output) — only exists as a raw `lscpu.txt`
  dumped by the evidence workflow. No automated way today to flag "this sample
  ran on a different CPU SKU" without manually diffing `lscpu.txt` files.
- **Question this raises for the noise/flaky calibration work:** does the
  current `noise_floor_pct` (flat calibrated percentage per task/GPU/backend)
  actually cover this magnitude of drift (up to +28% on Cartpole from warm-up
  alone, before even factoring in CPU-SKU swaps)? Not yet verified — worth
  stress-testing before flipping `blocking: true` on the real gate.

---

## 7. Recommended next steps (priority order)

1. **Fix `tools/conftest.py`** so it doesn't swallow test invocations rooted
   under `tools/`, and wire the 132-test suite into an actual CI job. Highest
   leverage, lowest risk — currently zero automated protection against a logic
   regression in the gate itself.
2. **Make the `"L40S"` fallback fail loudly** (or resolve the real GPU model)
   instead of silently mis-calibrating hard floors.
3. **Design + dry-run the `gate_config.json` / `blocking: true` flip** as an
   explicit, tested rollout step rather than an implicit side effect of
   "merging." Consider whether the sample-order/warm-up drift (item 6) needs a
   larger noise floor or a mandatory warm-up rep before this flip is safe.
4. **Do not merge the bisection agent yet** — its own roadmap doc says it's
   not ready (stale paths/imports).
5. **Treat `neilm/unified-poc-noise-flaky` as dead.** Propose `HEAD`'s lineage
   as the actual merge candidate; don't waste time reconciling/rebasing the
   stale branch.
6. Consider adding CPU model (`lscpu` "Model name") as structured provenance
   in `build_bench_result.py`'s output so cross-run comparisons can auto-flag
   a CPU-SKU mismatch instead of requiring manual detection.

---

## 8. Key file references

- Gate logic: `tools/perf_smoke_test/oracle.py`, `aggregate.py`,
  `gate_config.py`, `task_config.py`, `launch_config.py`
- Test suite: `tools/perf_smoke_test/test/` (run with
  `--confcutdir=tools/perf_smoke_test`, see §1a)
- CI workflows: `.github/workflows/perf-smoke-test.yaml`,
  `perf-smoke-seed-baselines.yaml`, `perf-smoke-publish-image.yaml`,
  `perf-smoke-regression-evidence.yaml`
- Roadmap/known-gaps doc: `tools/perf_smoke_test/docs/poc-roadmap-next-steps.md`
- System design: `tools/perf_smoke_test/docs/system-design.md`,
  `module-interfaces.md`
- Prior regression-evidence work (2.3.2 vs 3.0 fork, memory/nsys/CPU): see
  `tools/perf_smoke_test/docs/team_update_3_0_vs_2_3_2_evidence.md` and
  `3_0_vs_2_3_2_regression_evidence.md`
- Stale branch to ignore: `.worktrees/unified-poc-noise-flaky`
  (`neilm/unified-poc-noise-flaky`)
