# Odin — an OSMO dispatcher for Isaac Lab benchmarks

Odin submits Isaac Lab benchmark workflows to [OSMO](https://github.com/NVIDIA/OSMO).
It pins a commit into a container image, expands a task list into one OSMO task
per `(task, library, physics, seed)`, polls to completion, and pulls the emitted
benchmark bundles back.

Every task runs `uv run isaaclab benchmark training` directly. Odin owns no
runner, no schema, and no metrics code — those are Isaac Lab's
(`isaaclab.benchmark`).

An older Odin also managed a fleet of SSH-reachable machines. That half is not
part of this tool; it lives on the `antoiner/feat/odin` branch.

## Requirements

- The `osmo` CLI on `PATH` with a configured profile (`osmo login`).
- Docker, with `docker login nvcr.io` already done, for `build-image --push`.
- Network access for image build, push, and submission. Everything else,
  including `--dry-run`, works offline.

## Build an image

The image clones a git bundle of one commit and runs `uv sync --frozen` against
that commit's committed `uv.lock`, so a commit that changes a dependency is
tested with the changed dependency.

```bash
uv run python -m tools.odin.cli build-image \
    --config tools/odin/config/odin.yaml \
    --ref <sha|branch|tag> \
    --profile full \
    --push
```

`--dry-run` writes the build context and prints the tag without invoking git
bundling or Docker — useful for inspecting the generated Dockerfile.

One profile, `full`, holding every backend:

| Profile | Extras | Covers |
|---|---|---|
| `full` | `isaacsim, ovphysx, ovrtx, rsl-rl, skrl, rl-games, sb3, rerun` | Kit PhysX, OvPhysX, Newton, OVRTX, all four RL libraries |

A single virtualenv became possible once the root `pyproject.toml` widened the
`packaging` cap to `<27`: `ovphysx` capped it at `<24` while `isaacsim-core`
pinned `==26.0`, which had made the two extras unresolvable together. `teleop`,
`viser`, `mimic`, `test` and the `all` aggregate still conflict with `isaacsim`
and are excluded.

This matters beyond convenience. A preset token does **not** determine the
physics backend — `physics=physx` resolves to OvPhysX on 39 tasks and to Kit
PhysX on 5 — so with split profiles a row could only be routed correctly by
loading its env cfg first. One profile removes that dependency entirely.

The build warms `UV_CACHE_DIR`, so a task's `uv run --frozen --extra ...`
re-syncs from cache without touching the network.

Images are tagged `<registry>/<repository>:<short_sha>-<profiles>`. Pass the
resolved **digest** to `dispatch`, not the tag, so a retag cannot change what a
comparison compared.

## Run a dispatch

```bash
uv run python -m tools.odin.cli dispatch \
    --config tools/odin/config/odin.yaml \
    --tasks-yaml tools/odin/config/tasks.yaml \
    --image nvcr.io/nvidian/antoiner-isaac-lab@sha256:... \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant*'] \
    [--metadata-yaml tools/odin/config/task_metadata.yaml] \
    [--chunk-size 25] [--pool POOL] [--priority HIGH|NORMAL|LOW] \
    [--dry-run]
```

To compare two commits, build an image per commit and pass both:

```bash
    --image  <image-for-commit-A> \
    --image-b <image-for-commit-B>
```

Both sides run the identical row set. Side B's row keys and OSMO task names take
a `_b` suffix so the two coexist in one dispatch, and `dispatch.json` records the
image reference per side.

## Inspect and re-pull

```bash
uv run python -m tools.odin.cli status <dispatch_id>
uv run python -m tools.odin.cli fetch  <dispatch_id> --config tools/odin/config/odin.yaml
```

## Build per-task metadata from a run

The seed list in `config/tasks.yaml` carries only
`(task_id, rl_library, physics)`. `--num_envs` and `--max_iterations` default to
`None` upstream and fall back to each task's shipped agent config, so a first
dispatch runs everything at its shipped size. `harvest` then reads the emitted
bundles and writes what actually happened:

```bash
uv run python -m tools.odin.cli harvest <dispatch_id> \
    --output tools/odin/config/task_metadata.yaml
```

That file records resolved `num_envs` / `max_iterations`, a `timeout_s` derived
from the slowest observed run (`--timeout-headroom`, default 2.0), and a reward
baseline. Feed it back with `--metadata-yaml` on the next dispatch. Seed-list
values win over harvested ones, so a hand-set override survives a re-harvest.

The comparable upstream table is
`source/isaaclab_tasks/test/benchmarking/configs.yaml`, which is where these
values should eventually live.

## On-disk layout

```
odin_runs/
└── 20260729-120000/
    ├── dispatch.json                  # schema 2.0; per-side image digests
    ├── workflow.a0.yaml               # one per (side, chunk)
    └── rsl_rl_physx_Isaac-Ant_seed42/
        └── benchmark_training_Isaac-Ant_2026-07-29_12-00-00.json
```

## Preflight

`dispatch` refuses to submit until two server-side checks pass (override with
`--skip-preflight`):

1. `osmo data check <results_uri> -a WRITE` — a read-only bucket would swallow
   every result, and that is exactly how OSMO retired datasets.
2. `osmo workflow validate` on every rendered chunk — OSMO rejects unknown task
   fields, bad pool/platform pairs, unknown credential names, and unreachable
   images. Validating all chunks before submitting any avoids a half-submitted
   dispatch.

## Failure kinds

`dispatch.json` records one of four kinds per failed row:

| Kind | Meaning |
|---|---|
| `benchmark_crash` | The benchmark process exited non-zero. Not retried; real bugs repeat. |
| `timeout` | The task exceeded its OSMO `exec_timeout`. |
| `infrastructure` | Preemption, eviction, image-pull failure, queue timeout, backend or server error. OSMO reschedules these per `retry.reschedule_codes` before Odin ever sees them. |
| `malformed_bundle` | OSMO reported `COMPLETED` but no parseable schema bundle came back. |

Unrecognised `FAILED_*` states fall back to `infrastructure`, so OSMO version
drift cannot wedge the poll loop or be misread as a benchmark crash.

## Known gaps

- **DSS is not used.** OSMO datasets were retired (`outputs: - dataset:` now
  fails server-side with *"Bucket isaac mode is read-only"*). Publishing is
  declarative instead: each task carries an `outputs: - url: swift://...` block
  and OSMO performs the upload. Registering results as nv-datasets, if wanted,
  would be a controller-side step after fetch — the DSS client API could not be
  reached to design against.
- **The seed task list is interim.** It is replaced once the upstream
  task-discovery API lands, at which point `plan.load_task_rows` swaps its
  source and nothing downstream changes.
- **Play videos are not yet possible.** `PlayBundle.video_path` and
  `builders.build_play_bundle(video_path=...)` exist upstream, but none of the
  four play benchmark adapters has a `--video` flag or calls
  `wrap_record_video`; the video CLI arguments live in `add_common_train_args`
  and are training-only. That gap has to close upstream first.

## Running the tests

The suite is pure Python and runs offline — no simulator, no Docker daemon, no
live OSMO calls.

```bash
uv run --frozen --extra test python -m pytest \
    --rootdir=$PWD --confcutdir=$PWD/tools/odin/tests tools/odin/tests/ -q
```

`--extra test` is needed because pytest lives in the `test` extra.
`--confcutdir` bypasses the repository-level `tools/conftest.py`, which is a CI
orchestrator that hijacks pytest startup. `--frozen` stops an older local `uv`
from rewriting the committed `uv.lock`.
