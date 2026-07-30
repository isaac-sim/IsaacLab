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
  including `--dry_run`, works offline.

## Build an image

The image clones a git bundle of one commit and runs `uv sync --frozen` against
that commit's committed `uv.lock`, so a commit that changes a dependency is
tested with the changed dependency.

```bash
uv run python -m tools.odin.cli build-image \
    --config tools/odin/config/odin.yaml \
    --ref <sha|branch|tag> \
    --push
```

`--dry_run` writes the build context and prints the tag without invoking git
bundling or Docker — useful for inspecting the generated Dockerfile.

One virtualenv holds every backend. `plan.UV_EXTRAS` is the single list, used
both to build the image and to run each task:
`isaacsim, ovphysx, ovrtx, rsl-rl, skrl, rl-games, sb3, rerun` — Kit PhysX,
OvPhysX, Newton, OVRTX, and all four RL libraries. `teleop`, `viser`, `mimic`,
`test` and the `all` aggregate conflict with `isaacsim` and are excluded.

This matters beyond convenience. A preset token does **not** determine the
physics backend — `physics=physx` resolves to OvPhysX on most tasks and to Kit
PhysX on a few — so with split environments a row could only be routed correctly
by loading its env cfg first. One environment removes that dependency entirely.

The build warms `UV_CACHE_DIR`, so a task's `uv run --frozen --extra ...`
re-syncs from cache without touching the network.

Images are tagged `<registry>/<repository>:<short_sha>`. Pass the resolved
**digest** to `dispatch`, not the tag, so a retag cannot change what a
comparison compared.

## Generate the task list

The task list is generated, not shipped. Produce one before the first
dispatch:

```bash
uv run python -m tools.odin.cli discover
```

It writes `tools/odin/config/tasks.yaml`, which is gitignored: a committed copy
goes stale against the registry, and regenerating costs seconds.

Expansion is **total** — every task, every RL library it declares, and every
*legal mode* it supports — and filters are the only way to narrow.

A **mode** is a legal `(physics, renderer)` combination. Discovery returns those
directly rather than raw axes, so nothing downstream reconstructs the cross
product or reasons about which pairings are rejected. Tasks declaring renderers
are always expanded across them: benchmarking a camera task headless measures
everything except the thing under test.

The cross product is not all legal. OVRTX is kitless and cannot share a process
with Kit physics, so `isaacsim_physx + ovrtx` is rejected. Discovery checks each
pairing against the runtime validator, which costs a few seconds for the whole
registry and saves finding out on a GPU.

Domain presets — `depth`, `rgb`, `albedo`, `semantic_segmentation`, shading and
scene variants — are a third axis, selected with `presets=`. Backend names that
also appear under DOMAIN are filtered out, since those are chosen with
`physics=`.

**Known limitation:** discovery emits domain presets **one at a time** and never
combines them. Presets targeting the same field conflict outright
(`presets=depth,rgb` fails with *"Conflicting global presets"*), and there is no
rule available for which pairs compose. Orthogonal combinations that would be
valid — a data type plus a shading variant — are therefore not generated.
The **executor has no such limit**: a hand-written list may give
`presets: [depth, simple_shading_full_mdl]`, or the comma-separated string form,
and it is passed through as one `presets=a,b` token.

Current shape: **87 tasks → 1663 rows**, up from 429 before the preset axis.
That is far more than you would run, so whitelisting is the normal path:

```bash
# depth on cartpole, every legal physics/renderer pairing
uv run python -m tools.odin.cli discover \
    --include 'Isaac-Cartpole-Camera*' --presets depth --library rsl_rl
```

| Filter | Effect |
|---|---|
| `--library rsl_rl` | restrict the RL library axis; repeatable |
| `--physics newton_mjwarp` | restrict the physics axis; repeatable |
| `--renderer none` | restrict the renderer axis; repeatable (`none` keeps headless rows) |
| `--presets depth` | restrict the domain-preset axis; repeatable (`default` keeps rows selecting none) |
| `--scope {core,contrib,all}` | core vs contrib, from the explicit `scope` field |
| `--include` / `--exclude` | `task_id` globs |
| `--max_rows N` | deterministic head of the sorted order, as a cost valve |

```bash
# Newton vs OvPhysX on the same tasks, one library
uv run python -m tools.odin.cli discover \
    --physics newton_mjwarp --physics ovphysx --library rsl_rl

# Core only
uv run python -m tools.odin.cli discover --scope core --library rsl_rl
```

Two expansion rules worth knowing. `physx` is dropped from any task that also
declares `ovphysx`, because headless they resolve to the same backend and
running both is an exact duplicate. Tasks declaring no physics preset get one
row with the field omitted, since they reject any `physics=` token.

Beware that `physics=physx` does **not** mean Kit PhysX: it resolves to kitless
OvPhysX on most tasks and to Kit PhysX on only a few. `isaacsim_physx` is the
reliably Kit one. To sweep the whole PhysX family:
`--physics physx --physics ovphysx --physics isaacsim_physx`.

A hand-written list remains possible — `dispatch --tasks_yaml <file>` uses it
verbatim.

## Run a dispatch

```bash
uv run python -m tools.odin.cli dispatch \
    --config tools/odin/config/odin.yaml \
    --tasks_yaml tools/odin/config/tasks.yaml \
    --image nvcr.io/nvidian/antoiner-isaac-lab@sha256:... \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant*'] \
    [--metadata_yaml tools/odin/config/task_metadata.yaml] \
    [--chunk_size 25] [--pool POOL] [--priority HIGH|NORMAL|LOW] \
    [--dry_run]
```

To compare two commits, build an image per commit and pass both:

```bash
    --image  <image-for-commit-A> \
    --image_b <image-for-commit-B>
```

Both sides run the identical row set. Side B's row keys and OSMO task names take
a `_b` suffix so the two coexist in one dispatch, and `dispatch.json` records the
image reference per side.

## Videos and checkpoints

`--play` chains a play rollout after training **inside the same OSMO task**, so
it reads the checkpoint training just wrote with no cross-task handoff:

```bash
uv run python -m tools.odin.cli dispatch --config tools/odin/config/odin.yaml \
    --image <digest> --seeds 42 --play [--video_length 200]
```

The chained step is skipped when training failed or wrote no checkpoint, and the
task exits with the **training** exit code — a play failure must not turn a good
training run into a red row. Both steps receive the same physics, renderer and
preset tokens, so the rollout matches what was trained.

The checkpoint is read from the training bundle's `checkpoint_path`, which the
train adapters populate from what actually landed on disk.

### What ran, and what did not

Every row writes `odin-steps.json` alongside its bundles — a record of what
happened, as opposed to what a run measured:

```json
{
  "row_key": "rsl_rl_physx_Isaac-Ant_seed42",
  "training": {"ran": true, "exit_code": 0},
  "checkpoint": {"found": true, "path": "/workspace/logs/.../model_500.pt"},
  "play": {"status": "ran", "exit_code": 0},
  "artifacts": ["benchmark_training_...json", "videos/play/rl-video-step-0.mp4"]
}
```

`play.status` is one of `not_requested`, `skipped_training_failed`,
`skipped_no_checkpoint`, or `ran`, so a row that comes back with less than
expected says why. Writing the record can never fail the task.

## Recover a dispatch

A controller that dies mid-poll leaves rows stuck in `running` while OSMO keeps
going. Re-attach rather than re-submitting — the workflow ids are already
recorded, and submitting again would duplicate every task:

```bash
uv run python -m tools.odin.cli dispatch --config tools/odin/config/odin.yaml \
    --resume LATEST
```

Failed rows are never retried automatically: a benchmark crash usually repeats,
and silently re-running it wastes GPU time. Retry explicitly, which starts a new
dispatch containing only those rows and links it back via `parent_dispatch_id`:

```bash
uv run python -m tools.odin.cli dispatch --config tools/odin/config/odin.yaml \
    --image <digest> --retry_failed LATEST
```

Retries reuse the parent's seeds, so `--seeds` is not needed. Each identity is
paired with the seeds it actually failed on, so a seed that passed is not re-run.
An A/B dispatch cannot be retried: a retry runs one image, which would put side
B's rows on side A's image. Re-dispatch each side on its own instead.

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
from the slowest observed run (`--timeout_headroom`, default 2.0), and a reward
baseline. Feed it back with `--metadata_yaml` on the next dispatch. Seed-list
values win over harvested ones, so a hand-set override survives a re-harvest.

The comparable upstream table is
`source/isaaclab_tasks/test/benchmarking/configs.yaml`.

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
`--skip_preflight`):

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
- **Only `--output_path` is uploaded.** OSMO collects `{{output}}`; anything an
  RL library writes under `logs/` (tfevents, training videos) is discarded when
  the container exits. `--keep_checkpoints` copies the checkpoint across; nothing
  else is retained today.
- **The seed task list is interim.** It is replaced once the upstream
  task-discovery API lands, at which point `plan.load_task_rows` swaps its
  source and nothing downstream changes.

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

The rendered workflow YAML and Dockerfile are compared whole against the golden
files in `tools/odin/tests/golden/`. After an intended change to a template,
re-render them and review the diff:

```bash
ODIN_UPDATE_GOLDEN=1 uv run --frozen --extra test python -m pytest \
    --rootdir=$PWD --confcutdir=$PWD/tools/odin/tests tools/odin/tests/ -q
```
