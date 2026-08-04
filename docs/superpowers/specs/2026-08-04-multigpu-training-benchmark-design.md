# Multi-GPU Training Benchmark Design

## Objective

Add an experimental, training-only multi-GPU benchmark workflow that launches one Isaac Lab process per GPU and
emits one aggregate training benchmark bundle. Keep the existing single-process `benchmark training` workflow
unchanged and reuse the established `train_multigpu` launcher behavior wherever practical.

The primary command is:

```bash
uv run --extra isaacsim isaaclab benchmark training_multigpu \
    --rl_library rsl_rl \
    --num_gpus 2 \
    --task Isaac-Lift-KukaAllegro-Camera \
    presets=isaacsim_physx
```

## Scope

The initial workflow supports the Torch distributed modes already exercised by the non-benchmark multi-GPU
trainer:

- RSL-RL
- RL-Games
- skrl with `--ml_framework torch`
- single-node and multi-node launches
- the existing benchmark output formatters

The following are explicitly out of scope:

- skrl JAX
- Stable-Baselines3, which has no distributed training implementation in Isaac Lab
- multi-GPU play or runtime benchmarks
- video recording
- environment sensor capture
- multi-agent skrl algorithms
- success-based early stopping

`--num_envs` continues to mean environments per worker process, matching normal distributed training. The aggregate
bundle reports the total number of environments across all ranks.

## Command and Workflow Separation

`training_multigpu` is a distinct benchmark CLI workflow rather than a `--distributed` mode added to regular
`benchmark training`. This keeps the stable single-process benchmark contract and typed API unchanged while the
distributed result semantics remain experimental.

The repository gains a dedicated benchmark launcher script and entrypoint. It accepts the same Torch launcher
arguments as `train_multigpu`, including:

- `--num_gpus` / `--nproc_per_node`
- `--nnodes`
- `--node_rank`
- `--master_addr` and `--master_port`
- elastic rendezvous and worker-log options already supported by `torch.distributed.run`
- `--dry_run`

All remaining arguments are forwarded verbatim to the normal training benchmark dispatcher. The launcher adds the
distributed training flag and a private benchmark-mode marker. Backend adapters accept distributed execution only
when that marker is present, so invoking `benchmark training ... --distributed` directly continues to fail with the
existing guidance.

## Launcher Reuse

The command construction, validation, signal forwarding, and subprocess execution currently implemented by
`scripts/reinforcement_learning/train_multigpu.py` move behind a private shared helper. The regular and benchmark
scripts configure that helper with their target training script and supported backend set.

The refactor must preserve the existing `train_multigpu` command, accepted arguments, JAX behavior, dry-run output,
and error messages. The benchmark configuration permits only the three Torch-backed libraries in scope and rejects
skrl JAX before launching workers.

This boundary prevents the benchmark launcher from copying and subsequently drifting from the normal multi-GPU
launcher while keeping benchmark aggregation out of the launcher itself.

## Worker Execution

Each Torch worker launches one independent Isaac Lab application and simulation environment on its local-rank GPU.
Backend adapters reuse their corresponding non-benchmark distributed setup for:

- process-group initialization
- global and local rank discovery
- device selection and validation
- rank-offset seeds
- distributed backend configuration
- rank-safe training log directories and checkpoints

The existing benchmark instrumentation remains responsible for collecting local startup, training, environment-step,
learning, and resource measurements. A small shared distributed benchmark layer normalizes those measurements into a
backend-independent rank payload. It communicates only numeric measurements and scalar metadata; benchmark bundles
and arbitrary Python objects are not exchanged.

All ranks participate in final aggregation. Only global rank 0 builds and formats the aggregate bundle. Other ranks
return after aggregation without calling benchmark formatters, preventing output-file collisions.

## Aggregate Metric Semantics

Distributed workers progress in lockstep, so elapsed time is governed by the slowest worker while completed work is
the sum across workers. For each aligned sample:

- startup phase duration is the maximum duration across ranks;
- iteration duration is the maximum duration across ranks;
- collection duration is the maximum duration across ranks;
- environment-step and synchronized simulation-step durations are the maximum across ranks;
- global steps per iteration are the sum of per-rank steps;
- collection and total throughput are recomputed as global steps divided by the corresponding maximum duration.

Ranks must report compatible iteration and timing-series lengths. A mismatch is treated as an invalid benchmark and
fails with an error identifying the metric and per-rank lengths rather than silently truncating data.

The learning curve, success rate, and checkpoint path come from rank 0. Distributed training synchronizes policy
updates, but rank-local episode samples can differ, so the result explicitly records `learning_scope=rank0` rather
than presenting the curve as a global episode aggregate.

`RunIdentity.num_envs`, `Runtime.steps_per_iteration`, and `Runtime.frames_per_environment_step` contain global
totals. Experimental distributed metadata is stored in `TrainingBundle.extra`:

- `distributed=true`
- `world_size`
- `local_world_size`
- `num_nodes`
- `num_envs_per_rank`
- `learning_scope=rank0`
- `resource_scope=rank0_node`

These keys remain outside the stable schema contract until distributed benchmark semantics have seen broader use.

## Hardware and Resource Scope

Hardware and resource capture currently describe one host. The initial multi-node implementation therefore retains
rank 0's hardware and resource snapshot and marks it with `resource_scope=rank0_node`. It does not imply that CPU,
RAM, or GPU utilization from other nodes was collected.

Single-node rank 0 capture should include all GPUs visible on that host. Cross-node resource aggregation and a
schema-level cluster hardware model are deferred rather than encoded ambiguously in the existing `Hardware` and
`Resources` fields.

## Output and Logging

Rank 0 writes one bundle using the requested formatter set and a multi-GPU-specific output prefix. The output
directory therefore receives one logical benchmark result, not one result per GPU.

Backend training logs remain rank-safe and may exist on each node according to the backend's established distributed
layout. The bundle exposes only rank 0's checkpoint. The workflow must not require a filesystem shared across nodes;
all metric aggregation uses Torch distributed collectives.

Video, sensor capture, and success-based early stopping are rejected during argument validation. These features can
create rank-owned files or cause workers to leave the collective training loop at different times, so silently
enabling them would risk corrupt output or a distributed deadlock.

## Error Handling

Validation happens before worker launch when possible:

- reject unsupported libraries, skrl JAX, and incompatible benchmark features;
- reject invalid or unavailable local GPU counts using the same checks as `train_multigpu`;
- preserve Torch launcher's validation of node and rendezvous arguments.

After launch, failure of any worker fails the overall command through `torch.distributed.run`. Aggregation validates
world metadata and series shapes before rank 0 writes output. No partial aggregate bundle is emitted after a worker
or aggregation failure.

Multi-node commands are supported through the existing Torch launcher mechanics. They are documented as not
hardware-validated as part of this change, since the development environment cannot exercise multiple nodes.

## Testing

Fast tests cover:

- single-node and multi-node benchmark command construction;
- forwarding of benchmark, backend, Hydra preset, and Kit arguments;
- dry-run output;
- rejection of SB3, skrl JAX, video, sensor capture, and success early stopping;
- preservation of every existing `train_multigpu` command-building test after launcher refactoring;
- regular `benchmark training` continuing to reject distributed execution;
- backend parsers accepting the private multi-GPU mode only with the distributed flag;
- pure aggregation of synthetic rank measurements, including global work totals and slowest-rank timing;
- clear failure for inconsistent world metadata or series lengths;
- rank-0-only formatter execution.

A lightweight two-process CPU/Gloo test exercises the collective aggregation layer without launching Isaac Sim or an
RL backend. GPU smoke coverage is added only where existing CI provides multiple GPUs; it is marked appropriately and
does not make the ordinary test suite require multi-GPU hardware. Multi-node launch construction is tested, while
end-to-end multi-node execution remains unverified locally.

All focused tests and the full pre-commit suite run through `./isaaclab.sh` as required by the repository guidelines.

## Documentation and Changelog

The benchmark documentation gains the new command, per-rank environment-count semantics, supported libraries,
multi-node examples, aggregation rules, and current feature exclusions. The multi-GPU documentation links the
benchmark workflow separately from normal distributed training.

One changelog fragment is added for each touched package. The entry describes the new experimental multi-GPU
training benchmark and its supported Torch backends; it does not advertise skrl JAX.

## Non-Goals

- Changing the stable benchmark schema or typed `BenchmarkTrainingRequest` API.
- Making distributed training a mode of the regular training benchmark.
- Comparing multi-GPU training throughput with single-process runtime benchmarks as if they had identical semantics.
- Aggregating hardware or resource samples across multiple hosts.
- Adding a filesystem dependency between nodes.
- Changing the algorithms or distributed semantics of the underlying RL libraries.
