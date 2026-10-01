# Performance smoke comparisons

The **Performance smoke** section of a GitHub Actions run Summary contains the PR's
base-versus-tested-revision comparison. From the PR, open **Checks**, select
**Performance smoke comparison**, then click **Open rendered Performance smoke
Summary**. This takes you to the comparison for that exact run and attempt:

**🟢 Improved · 🔴 Regressed · ⚪ Not comparable**

The main table shows status, workload, baseline FPS, PR FPS and percentage change.
FPS is the median of the recorded samples; positive change means higher FPS.
These labels describe observed differences, not statistical significance. Exact
zero changes are shown as unchanged.

The comparison report check is informational. The separate `performance-smoke`
job check continues to report the existing rolling-history gate; its status and
exit behavior are unchanged.

## What runs

The existing CI label starts the initial PR run. Subsequent pushes trigger the
workflow through the PR's `synchronize` event.

The pair is resolved from GitHub's exact tested merge revision. Its first parent
is the comparison base, and its second parent must match the requested PR head.
Later changes to the target branch cannot change that pair. The originally
reported PR-event base is retained separately in the provenance.

For a PR without a reusable baseline, the performance job checks out that resolved
base commit and benchmarks it first. It then checks out and
benchmarks GitHub's tested merge revision on the same runner. The PR head SHA and
the tested merge SHA are both recorded, so the source actually tested is explicit.
Each revision resolves its dependency image from its own checked-out build inputs
and overlays its own source through the verified benchmark launcher.

When the tested merge's base parent is unchanged, the job can reuse that PR's
previously verified baseline artifact. It still benchmarks the current tested revision.
When the base changes, or the previous baseline is expired, incomplete or fails
verification, the job measures a fresh baseline. Artifact names include the
producing attempt, so reruns preserve earlier evidence.

Workloads are matched dynamically using their recorded task, physics backend,
renderer, environment count and presets. All shared compatible workloads are
compared. Missing, partial, unverified or incompatible results appear as not
comparable, with explanations in the details. A later PR can add a workload that
has no measurement in its saved base artifact; that absence stays visible.

## Reading the evidence

Collapsed details below the main table contain full workload identities, sample
values and counts, environment differences, and links to the exact base/PR source,
producing workflow runs and result artifacts. A reused baseline retains its
original run and artifact link; it is not presented as a fresh measurement on the
current runner.

Each sample's `source-revision.json` records loaded-source hashes and binds the
runtime result bytes to the verified source. `source-manifest.json` describes the
immutable checkout. The paired report also records a fingerprint of the selected
FPS producer implementation; different or unrecognized producer definitions do
not receive a forced percentage comparison.

The **Existing rolling-history CI gate** is a separate collapsed section. Its
comparison, recording behavior and exit status remain independent of the paired
build report. Baseline-only failures are reported as unavailable comparison
evidence; failures of the current benchmark retain the existing job behavior.
If source setup prevents a paired benchmark from running, the main Summary shows
the recorded cause once, with source diagnostics in collapsed details. Per-workload
rolling-history reports remain in the result artifact rather than appearing as
repeated missing-result tables on the PR's run Summary.
