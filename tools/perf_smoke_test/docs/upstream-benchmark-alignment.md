# Upstream Benchmark Refactor — Simple Explanation

This README explains, in simple terms, the concepts in the upstream IsaacLab
benchmark refactor (a 4-part change by AntoineRichard) and how each one relates
to our unified POC performance gate.

The four PRs:

* Part 1/4 (#6197): backend-agnostic benchmark core
* Part 2/4 (#6198): unified runtime + startup scripts
* Part 3/4 (#6199): unified training dispatcher + RL adapters
* Part 4/4 (#6201): play (inference) benchmark

## First, How Our Gate Works Today

Our performance gate runs a benchmark inside a container like this:

```text
scripts/benchmarks/benchmark_non_rl.py --benchmark_backend json ...
```

It then reads the JSON the benchmark writes and decides pass/fail by comparing
the measured FPS to a stored baseline.

The important thing to know: **our gate depends on that script and that output
format.** The upstream refactor changes both. That is why these PRs matter to
us, not just as ideas but as things that can break or help our gate.

## The Concepts (Explained Simply)

### Benchmark "entry script"

An entry script is the file you actually run to start a benchmark.

* Old: `benchmark_non_rl.py` (sim stepping), `benchmark_startup.py` (startup).
* New: `runtime.py` (sim stepping), `startup.py` (startup).

Part 2 **deletes the old scripts**. Our gate calls `benchmark_non_rl.py`, so
when this lands, our gate breaks unless we switch to `runtime.py`.

### Backend selection with `presets=`

"Backend" here means which physics/render engine runs (PhysX, Newton, etc.).

The new scripts pick the backend using Hydra preset tokens, like:

```text
presets=newton_mjwarp
```

We already pass preset-style tokens, so this is a small alignment, not a big
change. We just need to match the exact token names they use.

### `--benchmark_backend` (output format, not engine)

Confusingly, "benchmark_backend" does NOT mean the physics engine. It means the
**output format** the benchmark writes.

* Old default for us: `json`.
* New default: `schema`.
* New ability: a comma-separated list, e.g. `schema,omniperf`, to write several
  formats in one run.

Our gate parses the old `json` shape, so if we keep calling it we must either
ask for `json` explicitly or, better, move to `schema` (see below).

### Schema bundles (the big one)

A "bundle" is a single structured file describing one benchmark run.

There are typed bundles:

* `RuntimeBundle` — sim stepping performance (this is the one our gate cares
  about).
* `StartupBundle` — startup timing.
* `TrainingBundle` — RL training performance.
* `PlayBundle` — inference (play) performance.

"Schema-v1" just means "version 1 of an agreed-upon shape for these files," so
different tools can read them reliably.

Why we care: a `RuntimeBundle` already contains the FPS numbers we compare AND
the environment details we need (see capture below). Instead of parsing raw JSON
ourselves, we could read the bundle directly.

### `capture` (versions / hardware / resources / run-id)

The `capture` module records, automatically, things like:

* software versions (IsaacLab, IsaacSim, etc.)
* hardware (GPU model, etc.)
* resource usage
* a unique run id

This is almost exactly what we built by hand for the gate:

* environment fingerprinting (to bucket results by environment)
* per-sample provenance (to know which commit/versions produced a number)
* the compatibility contract hash in `runtime_contract.py`

So `capture` overlaps heavily with our work. If we read their captured data, we
can delete some of ours and, importantly, bucket our baselines the **same way**
upstream/OmniPerf does, so numbers line up.

### `metrics` (MeanStd, peak, EMA, convergence)

The `metrics` module computes statistics from a run:

* `MeanStd` — mean and standard deviation.
* peak — maximum value.
* EMA — exponential moving average (a smoothed trend).
* convergence detection — did training settle?

Our oracle already compares an FPS mean to a floor/baseline. Their `MeanStd` is
the same idea. Using their definition keeps "what counts as a regression"
identical between our gate and upstream. EMA/convergence matter more only if we
extend the gate to training.

### Multi-backend output (`schema,omniperf`)

This means: run the benchmark once, write multiple output files.

For us this is valuable because:

* `schema` feeds our gate.
* `omniperf` feeds the historical OmniPerf dataset.

So one expensive run can serve both our gate and the baseline/OmniPerf data
collection we discussed, instead of running twice.

### Simulator-free core + test gating

The new core modules (capture, builders, metrics, stepping, profiling) import
nothing from Isaac Sim. That means they can be unit-tested on a normal machine
with no simulator.

Their tests also "gate" on Isaac Sim: if it is not installed, sim-dependent
tests skip instead of failing.

This matches what we already do (`validate_tasks.py` runs without Isaac Sim, and
our tests skip when Isaac Sim is missing). It is a good sign our approach is the
right pattern, and we could reuse their pure modules instead of writing our own.

### `stepping` (random-action loop)

A small helper that steps an environment with random actions (no trained policy)
and records per-frame times. This is essentially what the non-RL/runtime
benchmark does internally. Relevant because it is the heart of the throughput
number our gate checks.

### `profiling` (cProfile parsing)

`cProfile` is Python's built-in profiler (it measures where time is spent). The
`profiling` module parses its output. This powers the startup benchmark's
per-phase timings. Relevant only if we add startup timing to the gate.

### `startup.py` / `StartupBundle`

A benchmark that measures how long startup takes, split into 5 phases. This is a
**new kind of regression** we could watch (startup getting slower), separate
from FPS. Future scope for our gate.

### `training.py` + RL adapters

A single `training.py` that dispatches to one of several RL libraries
(`rsl_rl`, `rl_games`, `skrl`, `sb3`) via `--rl_library`. Each adapter runs real
training and emits a `TrainingBundle`.

For us this is future scope: it would let the gate watch **training throughput**,
not just raw sim stepping.

### `play.py` / `PlayBundle`

A benchmark that loads a trained checkpoint and measures **inference**
performance (how fast the policy runs), emitting a `PlayBundle` with inference
FPS plus reward/episode-length/success.

Also future scope: gating inference speed.

## Pitfalls Worth Knowing (From the PR Reviews)

If we start consuming their bundles/modules, the automated reviewer flagged some
bugs (several still open). Knowing them helps us avoid trusting bad numbers:

* `startup.py` captures start and end time back-to-back, so its `duration_s` is
  always near zero. Trust `runtime`'s duration, not `startup`'s, for now.
* Some scripts do not call `env.close()` in a `try/finally`, so a crash can leak
  the environment.
* `SuccessRateTracker` reports an iteration boundary at step 0 before any step is
  recorded.
* `resolve_play_checkpoint` can return a raw remote URI that adapters cannot
  load when no checkpoint is passed.
* `import_module_from_path` can cache a half-built module after an error.

These are upstream's to fix; we just shouldn't build on the broken parts yet.

## How This Maps To Priorities

Must-do (time-sensitive, avoids breakage):

1. Make our gate resilient to `benchmark_non_rl.py` being removed — either pin a
   known-good IsaacLab ref or detect `runtime.py` vs the old script. This is the
   only urgent item because the upstream merge will otherwise break us silently.

Should-do (clear wins):

2. Switch our result parsing to consume the `schema` `RuntimeBundle`, which also
   gives us versions/hardware/resources/run-id for free and aligns our
   compatibility bucket with upstream.
3. Use `--benchmark_backend schema,omniperf` for seed runs so one run feeds both
   our gate and OmniPerf.

Later (bigger scope):

4. Add startup-time gating (`StartupBundle`).
5. Add training/inference gating (`TrainingBundle` / `PlayBundle`).

## One-Sentence Summary

Upstream is turning ad-hoc benchmark scripts into a clean, typed, multi-format
system; the same system already captures the environment/provenance data our
gate hand-rolled, so we should ride on top of it — but first we must update our
gate before the old `benchmark_non_rl.py` entry point is deleted.
