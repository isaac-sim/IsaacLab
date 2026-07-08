# Perf Gate — Working Context Dump (2026-06-22)

Purpose: a self-contained context handoff so a fresh chat can pick up the
performance-regression-gate work without re-deriving everything. This captures
the live demo in flight, the state of baselines/branches/CI, decisions already
made, and the next strategic goal (GHCR -> ECR consolidation).

> This is a scratch/working doc, not user-facing API docs. Update or delete as
> the work lands. It is intentionally verbose.

---

## 0. TL;DR — where we are right now

- **Goal in flight:** demonstrate the perf gate catching a *real* regression on
  a *real* PR, end-to-end, single-branch (`unified-POC`).
- **Chosen regression:** the WrenchComposer dual-buffer regression from upstream
  `isaac-sim/IsaacLab#5265` (~47% throughput hit on locomotion/velocity tasks
  when an `apply_external_force_torque` event has all-zero ranges).
- **Demo matrix:** `Isaac-Velocity-Flat-G1-v0` (expected **RED**) +
  `Isaac-Cartpole-Direct` (expected **GREEN** control), physx + newton each = 4 cells.
- **Live gate run:** `27980523072` (event=push, branch=`pull-request/5`),
  build-from-source mode, in progress at time of writing.
- **CRITICAL UNDONE CLEANUP:** the repo variable `PERF_GATE_CI_IMAGE` is currently
  **DELETED** (unset) to force build-from-source. It MUST be restored after the
  gate run finishes:
  - Restore value: `ghcr.io/nvidia-omniverse/isaaclab-perf-gate:sha-6f97bf4`
  - Command: `gh variable set PERF_GATE_CI_IMAGE --body 'ghcr.io/nvidia-omniverse/isaaclab-perf-gate:sha-6f97bf4'`
- **Next strategic goal (post-demo):** stop maintaining the GHCR prebuilt-image
  path as a parallel system; converge perf-gate image handling onto the same
  ECR-backed `ecr-build-push-pull` action the rest of IsaacLab CI uses.

---

## 1. Repo / remote facts (don't get burned by these)

- Working dir: `/home/horde/dev/IsaacLab-fork`
- `origin` = `https://github.com/NVIDIA-Omniverse/IsaacLab.git`
  - This is the **working fork** used for all perf-gate work (NOT the public
    `isaac-sim/IsaacLab` upstream). Pushing perf-gate branches here is correct.
  - All perf-gate branches live here: `unified-POC`, `perf-baselines`,
    `pull-request/9000x`, `neil/*`, etc.
- Git identity is NOT configured in the repo. Commits need explicit env:
  `GIT_AUTHOR_NAME="Neil4561" GIT_AUTHOR_EMAIL="neilmehta456@gmail.com"`
  (and matching `GIT_COMMITTER_*`). Match the existing author on the branch.
- Default branch on the fork is `develop` — **`gh workflow run` defaults to it**.
  Always pass `--ref unified-POC` for perf-gate workflows, or it checks out
  `develop` (which lacks `tools/perf_regression_gate/`) and fails instantly.
- Network/CI: use `gh` with `required_permissions: ["full_network"]`.
  Mutating shared CI state (e.g. `gh variable delete/set`) can trip Auto-review;
  it needs the native approval card (request_smart_mode_approval on retry).

---

## 2. The gate in a nutshell (fundamentals)

The gate answers: *"for this task/backend/GPU/runtime, is this PR meaningfully
slower than comparable historical samples from the PR's own ancestry?"*

Pipeline:
1. **Trigger** (`.github/workflows/perf-regression-gate.yaml`): on push to
   `[main, develop, release/**, angehu/perf-gate-poc, pull-request/[0-9]+]` or
   `workflow_dispatch`. The real-PR path is copy-pr-bot mirroring a PR onto
   `pull-request/<N>`; that push triggers the gate.
2. **config + validate jobs**: load image config, build the benchmark matrix
   from `tools/perf_regression_gate/tasks.json`, statically validate task ids.
3. **bench jobs** (one per task/backend matrix cell, `continue-on-error: true`):
   run the benchmark inside Docker, emit a structured `bench_result`, upload
   artifacts, and post a per-task commit status (did the benchmark *run*).
4. **aggregate job**: resolve gate context, select baselines by ancestry, run
   the oracle (regression verdict), optionally update baselines, write a verdict
   table to the step summary and a **sticky PR comment**.

Key modules under `tools/perf_regression_gate/`:
- `github_gate_context.py` — resolves `base_sha`, `target_branch`,
  `source_branch`, `allow_update`, `trusted_source` from the GH event. For a
  `pull-request/<N>` push it calls the GitHub API for PR N and uses
  `pr.base.sha` / `pr.base.ref`.
- `baseline_manager.py` — sample matching + selection (`_sample_matches`,
  `_select_records`), baseline branch read/update, `make_sample_metadata`.
- `gate_config.py` — thresholds and the runtime-compatibility contract.
- `oracle.py` — the regression decision.
- `aggregate.py` — orchestrates the verdict + baseline update + summary.
- `seed_baselines.py` — the seeding orchestrator (walks commits, benchmarks
  each in the CI image, appends samples to `perf-baselines`).
- `tasks.json` — canonical task/backend matrix definition.

### Baseline selection = merge-base ancestry (NOT branch label)
`_sample_matches` requires exact match on
`gpu_model, task_id, backend_key, launch_config_hash, baseline_epoch,
benchmark_contract_hash, runtime_contract_hash`, then:

```python
base_sha = context.get("base_sha")
commit_sha = record.get("commit_sha")
if base_sha:
    if not commit_sha:
        return False
    if not _git_is_ancestor(str(commit_sha), str(base_sha), repo_dir=repo_dir):
        return False
```

So a baseline only counts if its commit is an ancestor of the PR base. The
`target_branch` field is metadata only. This is why "seed different target
branches" proves nothing new about isolation — isolation is the commit graph.

### Gate thresholds (`gate_config.py` defaults)
- `blocking = False` (advisory: comments, doesn't fail the check)
- `MIN_BASELINE_SAMPLES = 5` (below this -> "insufficient baseline")
- `MAX_BASELINE_SAMPLES = 20`
- `DEFAULT_K_WARN = 2.5`, `DEFAULT_K_BLOCK = 4.0` (robust z-score)
- `MIN_BLOCK_REGRESSION_PCT = 3.0` (floor so tiny but "significant" drops don't block)
- With baseline CV ~3-7%, a *blocking* regression needs a large drop (tens of %).
  The #5265 regression (~47%) is far beyond threshold = unambiguous RED.

---

## 3. The Docker image story (why this is the crux)

- `docker/Dockerfile.base` **bakes the IsaacLab source into the image**
  (`COPY ../source/ ...` and a final `COPY ../ ...`). So the benchmarked code is
  whatever was baked at build time.
- The **live gate does NOT source-mount** the PR checkout into the container; it
  benchmarks the baked code. (The *seeder* does source-mount historical commits
  into one prebuilt image — that's a seeding-only shortcut.)
- Therefore, to measure a PR's code you must build the image FROM that PR's
  source. Two image paths in `perf-regression-gate.yaml`:
  - **(A) GHCR override (current fast path):** if repo var `PERF_GATE_CI_IMAGE`
    is set, every bench cell just `docker pull`s that prebuilt image and retags
    it. Fast (~7.5 min cold pull vs ~26 min build) but benchmarks BAKED code,
    not the PR. Good for seeding/era-fixed runs; useless for measuring PR code.
  - **(B) build-from-source (default when var unset):** uses
    `./.github/actions/ecr-build-push-pull` to build the image from the checked
    out PR source. Production-faithful. This is what the live demo uses.

### "Era" = environment, and same-era caching
- An image "era" is defined by `{Isaac Sim base tag, Dockerfile.base, pinned
  deps/manifests}`. Code-only changes are NOT an era change.
- Same-era builds are fast because Docker reuses the expensive lower layers
  (base image, apt, `isaaclab.sh --install`) from cache; only the cheap
  source-copy tail re-runs. A true cold ~26-min build happens only on era bumps.
- Dockerfile wart worth noting for production: `COPY ../source/` happens *before*
  `isaaclab.sh --install`, so a source-only edit invalidates the install layer
  and re-runs it (more than necessary). Reorder (copy dep manifests -> install ->
  copy source last) to make code-only PRs cheaper.

---

## 4. GHCR vs ECR — the next strategic goal

**User's stated goal:** "switch from GHCR to ECR to match the pattern the rest of
IsaacLab CI does," to avoid maintaining two parallel image workflows.

### What exists today (the parallel GHCR system we want to retire)
- `.github/workflows/perf-gate-publish-image.yaml` — builds `Dockerfile.base`
  once and pushes to GHCR (`ghcr.io/<owner>/isaaclab-perf-gate:sha-<short>`).
- repo var `PERF_GATE_CI_IMAGE` points the gate/seeder at that image.
- `perf-regression-gate.yaml` has a bespoke "Pull prebuilt CI image (override)"
  step that logs into GHCR/NVCR and pulls.
- This whole path is documented in-repo as a **temporary fast path** because the
  NVIDIA-managed RTX PRO 6000 fleet has no provisioned ECR/registry cache.

### What the rest of IsaacLab CI does (the target pattern)
- `.github/workflows/build.yaml` ("Docker + Tests") builds via
  `./.github/actions/ecr-build-push-pull` with `cache-tag: cache-base`, then test
  jobs pull the image.
- `.github/actions/ecr-build-push-pull/action.yml` is the shared building block.
  It already implements the scalable, env-keyed caching we want:
  1. **Exact image check**: per-commit `ECR_URL:<image-tag>` — if it exists in
     ECR, skip build entirely (pull/alias).
  2. **Deps cache (`deps-<hash>`):** hashes `{Dockerfile, isaaclab.sh,
     environment.yml, source/.../cli, all repo manifest files (pyproject.toml,
     extension.toml, requirements*, uv.lock), base-image digest}` -> `deps-<hash>`.
     If that image exists, it registry-side aliases it as the commit image with
     **no rebuild**. **This is exactly the "env-era keying" I'd described as
     future work — it already exists here.**
  3. **Full build** with ECR registry layer cache (`--cache-from/--cache-to
     type=registry,ref=...:cache-base`) only on a deps-cache miss.
  4. **`gha-cache-scope`** opt-in: a registry-free GitHub Actions layer cache
     fallback for fleets where ECR can't be resolved (this is what perf-gate's
     build-from-source path currently leans on, scope `perf-gate-base`).
- ECR URL resolution order: `ecr-url` input -> `ECR_CACHE_URL` env -> SSM param
  `/github-runner/<instance-id>/ecr-cache-url` -> else skip ECR (local build).

### Why this consolidation is the right move
- **One image path, not two.** Deletes `perf-gate-publish-image.yaml`, the
  `PERF_GATE_CI_IMAGE` override step, and the manual variable juggling we just
  had to do for the demo (unset/restore).
- **Gets env-era caching for free.** The deps-cache solves both the "rebuild per
  PR" worry (same-era PRs hit deps cache) AND the multi-era seeding problem
  (each era resolves to its own `deps-<hash>` image, built once, cached in ECR).
- **Matches the org standard**, so perf-gate inherits future improvements to the
  shared action instead of diverging.

### The real blocker / open question
- The perf-gate runs on the NVIDIA-managed RTX PRO 6000 fleet, which (per the
  in-repo comments) has **no provisioned ECR** — `ecr-url` won't resolve there,
  so the action falls back to local build + gha cache. That's the original
  reason the GHCR fast path was created.
- So the consolidation hinges on one of:
  1. Getting ECR (or an equivalent registry cache) provisioned/reachable on the
     perf-gate fleet (SSM param or `ECR_CACHE_URL`), OR
  2. Accepting the `gha-cache-scope` fallback as the steady-state cache for
     perf-gate (no registry, but free same-era layer reuse), and dropping the
     GHCR prebuilt image entirely, OR
  3. Pointing `ecr-build-push-pull` at GHCR-as-registry-cache (the action's
     registry-cache logic is ECR-URL-shaped; would need generalization).
- Decision needed from the team: which of the above. (1) is cleanest and most
  "match the rest of CI"; (2) is the least-infra path and already partially in
  place; (3) is the most code change.

---

## 5. Baselines state (`perf-baselines` branch)

Path layout: `rtx_pro_6000_blackwell/<task_id>/<backend_key>/samples.ndjson`.

As of 2026-06-22, ancestry-selectable @ `6c76e4a0` (the unified-POC tip / PR #5
base) — these clear the 5-sample floor:

| task | backend | selectable | mean FPS | std (CV) |
|---|---|---|---|---|
| Isaac-Cartpole-Direct | physx | 8 | 287,391 | 21,137 (~7.4%) |
| Isaac-Cartpole-Direct | newton | 8 | 976,480 | 67,333 (~6.9%) |
| Isaac-Velocity-Flat-G1-v0 | physx | 6 | 10,952 | 395 (~3.6%) |
| Isaac-Velocity-Flat-G1-v0 | newton | 6 | — | — |

Notes:
- Older samples on these files have `commit_sha = null` (pre-fix) and are
  silently ignored under ancestry selection — harmless leftovers.
- Factory-GearMesh and Repose-Cube-Vision still have only 2 selectable samples
  (mostly null commit_sha); not seeded deeply (out of demo scope).

### The commit_sha fix (already on unified-POC, commit `6c76e4a06`)
- **Bug:** seeded samples relied on the in-container benchmark to self-capture
  git provenance, but on a detached-HEAD, runner-owned bind mount that capture
  returns nothing -> every seeded sample had `commit_sha = null` -> invisible to
  ancestry selection once a `base_sha` is in play (i.e. on every real PR).
- **Fix:** `make_sample_metadata(..., commit_sha=...)` now prefers an explicit
  SHA; `seed_baselines.py` threads the known checked-out commit
  (`_record_from_result(..., known_commit=commit)`). Verified: new samples carry
  the real SHA.

---

## 6. The demo artifacts (in flight)

- **Demo branch:** `neil/perf-demo-wrench-regression` (pushed to origin).
  - `9eb201450` — `[DEMO] Reintroduce WrenchComposer zero-range regression from
    #5265`: deletes the all-zero early-return guard in
    `source/isaaclab/isaaclab/envs/mdp/events.py` (`apply_external_force_torque`,
    ~line 1725). Honest commit message: states it's a demo, not for merge,
    references the real PR + 47% mechanism.
  - `38197210a` — `[DEMO] Scope perf-gate matrix to Cartpole + Velocity-Flat-G1`:
    trims `tasks.json` to those two tasks (drops Factory + Repose) for a fast,
    clean multi-row table.
- **PR #5** (`NVIDIA-Omniverse/IsaacLab`): base `unified-POC` @
  `6c76e4a068ca8456618de70d5fbfc5ee3ed2364e`, head `38197210a`. Body explains
  it's a demonstration PR mirroring upstream #5265.
- **Mirror branch:** `pull-request/5` pushed (= demo head) to trigger the gate.
- **Gate run `27980523072`:** event=push, branch=`pull-request/5`, matrix = the
  4 expected cells. config+validate passed; bench cells building from source.

### Expected verdict
- `Isaac-Velocity-Flat-G1-v0` -> **RED** (exercises the zero-range event; base
  cfg `velocity_env_cfg.py` `base_external_force_torque` has
  `force_range=(0.0,0.0)`, `torque_range=(-0.0,0.0)`).
- `Isaac-Cartpole-Direct` -> **GREEN** (no such event; unaffected control).

---

## 7. Immediate next steps (resume here)

1. **Watch `27980523072` to completion.** Most likely failure point is the
   from-source Docker build (cold ~26 min, or faster via gha cache scope
   `perf-gate-base`). Then benchmarks, then aggregate.
2. **Verify the verdict:** G1 RED + Cartpole GREEN in the sticky comment on
   PR #5 / step summary. Confirm aggregate selected the seeded baselines
   (ancestry) and didn't report "insufficient baseline".
3. **RESTORE `PERF_GATE_CI_IMAGE`** (see §0) once all bench cells are past their
   image step (safest: after the whole run completes). Leaving it unset breaks
   the seeder (it requires the var) and makes other gate runs build from source.
4. **(Optional) Healthy GREEN PR** for contrast (a no-op or trivially-correct
   change) to pair RED vs GREEN.
5. **Demo doc** (separate from this context dump): overview of the POC + how
   multi-branch/era would work in production + why scoped out for the demo.
6. **GHCR -> ECR consolidation** (see §4): get a team decision on the cache
   backend for the perf-gate fleet, then converge onto `ecr-build-push-pull`
   and retire `perf-gate-publish-image.yaml` + the override step.

---

## 8. Decisions already made (so we don't relitigate)

- **Demo is single-branch (`unified-POC`).** Multi-branch target seeding was
  dropped: it needs per-era images (heavy, serialized ~26-min builds on one
  runner, plus seeder work) and proves nothing extra because isolation is
  ancestry-based. Multi-branch already "just works" in real deployment
  (build-from-source per PR).
- **Real regression, not synthetic:** reproduce #5265 by removing the guard.
- **Matrix scoped to Cartpole + G1** for a fast, clean, multi-row RED/GREEN
  table (full 9-cell matrix would be multi-hour + insufficient-baseline noise).
- **Build-from-source for the demo** (unset `PERF_GATE_CI_IMAGE`) so the gate
  measures the PR's real code (production-faithful).
- **PR is honestly labeled** as a demonstration mirroring upstream #5265.

---

## 9. Command cheat-sheet

```bash
# Always target unified-POC for perf-gate workflows
gh workflow run perf-gate-seed-baselines.yaml --ref unified-POC \
  -f branches=unified-POC -f commit_count=1 -f samples_per_commit=6 \
  -f tasks=Isaac-Velocity-Flat-G1-v0 -f target_branch=unified-POC -f dry_run=false

# Watch a run
gh run view <run_id> --json status,jobs --jq '{status, jobs:[.jobs[]|{name,status,conclusion}]}'

# Inspect baseline sample counts / commit_sha
git fetch origin perf-baselines --depth=1
git show origin/perf-baselines:rtx_pro_6000_blackwell/<task>/<backend>/samples.ndjson \
  | python3 -c "import sys,json;rows=[json.loads(l) for l in sys.stdin if l.strip()];print(len(rows))"

# Restore the GHCR image var after the demo (IMPORTANT)
gh variable set PERF_GATE_CI_IMAGE --body 'ghcr.io/nvidia-omniverse/isaaclab-perf-gate:sha-6f97bf4'

# Gate context resolver / thresholds
#   tools/perf_regression_gate/github_gate_context.py
#   tools/perf_regression_gate/gate_config.py
```

---

## 10. Gotchas / lessons learned

- `gh workflow run` without `--ref unified-POC` runs on `develop` and fails
  (`seed_baselines.py: No such file`).
- Seeded samples need `commit_sha` or they're invisible on real PRs (fixed).
- Git LFS smudge can break historical checkouts; seeder sets
  `GIT_LFS_SKIP_SMUDGE=1` for clone/fetch/checkout.
- Root-owned `__pycache__` from prior container runs broke `git clean`; seeder
  chmods + best-effort cleans and sets `PYTHONDONTWRITEBYTECODE=1`.
- Era mismatch: seeding `main`/`develop` source into the unified-POC image
  fails (missing extensions / unregistered task ids). Only seed era-matching
  branches with a given image (skip-and-warn guard handles incompatibles).
- `benchmark_non_rl.py` `--benchmark_backend`: the unified-POC script accepts
  both `json` and `JSONFileMetrics`; the seeder uses `JSONFileMetrics`.
- Mutating shared CI vars trips Auto-review; expect an approval card.
