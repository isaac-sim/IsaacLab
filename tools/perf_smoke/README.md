# Performance smoke comparisons

The existing runtime benchmark runs in the CI GPU container. ASV runs on the host
and compares the resulting measurements; it does not install Isaac Sim or manage
the benchmark container. The runner uses `uv venv`, `uv pip`, and `uv run` for its
isolated host environment; the container uses `uv run --no-sync isaaclab` with its
already installed dependencies. Install the host dependencies locally with
`uv pip install -r tools/perf_smoke/requirements.txt`.

The gate retains the runtime compatibility contract, the last 20 comparable
develop measurements in Azure, per-task warning/failure thresholds, advisory
metrics, absolute FPS floors, and non-blocking SKIP/ERROR outcomes from the
original implementation. Only develop pushes write baselines, using the separate
write credential. Existing baseline rows remain readable.

## What ASV replaces

ASV calculates sample statistics and decides whether a change exceeds the
configured factor and is statistically significant. This replaces the custom
scaled median absolute deviation and two-sigma test. ASV uses its Mann–Whitney
test when there are enough samples and otherwise compares confidence intervals.
See the [ASV comparison implementation](https://github.com/airspeed-velocity/asv/blob/v0.6.6/asv/commands/compare.py)
and [statistical tests](https://github.com/airspeed-velocity/asv/blob/v0.6.6/asv/_stats.py).

CI collects three independent benchmark runs per combination, with one retry per
run. This approximately triples benchmark execution time; timeout budgets include
all six possible attempts. Each run keeps the original task, environment count,
seed, warmup, step count, backend matrix, and JIT-cache configuration. Per-frame
measurements are not presented as independent runs. Baseline writes use a stable
sample suffix on the CI run ID so retries do not duplicate samples.

At least three baseline runs and two candidate runs are required to compare. ASV
would otherwise bypass statistical testing for a singleton; the gate reports
SKIP instead. A configured absolute floor still fails if any candidate run
breaches it, including for an advisory-only task or an empty baseline.

ASV treats smaller values as better, so FPS samples are converted to seconds per
frame. An FPS-loss threshold `p` becomes the ASV factor `1 / (1 - p / 100)`;
time and memory thresholds use `1 + p / 100`. ASV uses strict factor comparisons,
so equality with a threshold is not a regression. Thresholds must be between 0
and 100 percent. Zero throughput cannot be inverted: it trips an applicable floor
or yields SKIP. The summary shows medians in the original units; the expanded ASV
tables show inverse FPS. For an even number of samples, the median of inverse FPS
is not exactly the inverse of the median FPS; ASV's transformed samples decide
the verdict.

## Local comparison and artifacts

```bash
uv run --no-project --with-requirements tools/perf_smoke/requirements.txt \
  python -m tools.perf_smoke.cli compare \
  --benchmark_result run-1.json run-2.json run-3.json \
  --output_json benchmark-output/comparison.json
```

Set `ISAACLAB_BLOB_URL` to the read-only baseline credential. Missing credentials
produce SKIP; store or dependency failures produce ERROR. Both exit successfully.
A measured regression exits 1. Candidate runs with different contracts are
rejected. `write` and `aggregate` retain their existing CLI interfaces.

The uploaded artifact includes raw bundles, `comparison.json`, and an `asv/`
directory containing native ASV benchmark metadata, machine metadata, and result
files with raw samples and statistics. `baseline` and `candidate` identify the
rolling reference and current run, **not Git commits**. These are comparison
snapshots, not a fabricated commit history suitable for `asv publish`.

With one combination's `asv/` artifact under `benchmark-output/asv`, the standard
ASV CLI can display it without the Azure credential or the simulator:

```bash
uv run --no-project --with-requirements tools/perf_smoke/requirements.txt \
  asv compare --config tools/perf_smoke/asv.conf.json baseline candidate --factor 1.1111111111111112
```

That factor represents a 10% FPS loss. This command displays all metrics using
one factor; the Isaac Lab gate applies the per-metric direction, warning/failure
policy, advisory flags, and floors. ASV's `compare` CLI does not itself return a
failure exit code for regressions.

The adapter uses the pinned ASV Python APIs instead of parsing console output or
depending on private statistical helpers. To validate changes or dependency bumps:

```bash
uv run --no-project --with-requirements tools/perf_smoke/requirements.txt \
  python -m unittest tools.perf_smoke.test_compare
```
