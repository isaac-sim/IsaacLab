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

GitHub can group the informational comparison under another workflow, such as
**Pull Request Labeler**; expand that group to find the check. The native
`performance-smoke` job also includes an **Open rendered Performance smoke Summary**
annotation with the exact Summary URL. GitHub controls check-suite grouping.

## What runs

The existing label trigger starts the initial PR run. Subsequent pushes trigger
GPU CI through `synchronize` after a prior human or app-token label event. Removed
labels still count, so the existing `run-ci` command keeps working. Labels added
by `GITHUB_TOKEN` do not count: GitHub suppresses workflows triggered by that token.
Without activation, the required build check stops on a CPU runner and explains
how to start CI; it does not allocate a GPU or mark unexecuted required tests as
sufficient to merge. Updates cancel superseded running revisions.

The paired job retains its existing 450-minute limit. Retrieval requests have a
60-second timeout; baseline lookup and comparison generation each have a five-minute
step limit. A failed reuse lookup falls back to a fresh base measurement when the
base checkout and image are available. Comparison-generation failure leaves the
existing rolling-history gate and its Summary available.

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

When the fingerprint algorithm changes, a reused baseline's compatibility metadata
can be re-derived from its verified source checkout. The derived identity is bound
to the original artifact ID, checksum and commit; its measurements and original
recorded metadata are preserved.

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
