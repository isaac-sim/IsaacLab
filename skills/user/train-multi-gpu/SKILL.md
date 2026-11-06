---
name: isaaclab-training-multi-gpu
description: Launches and debugs Isaac Lab multi-GPU and multi-node RL training, including NCCL hangs and collective failures. Use when running train_multigpu, choosing num_gpus, setting up multi-node jobs, or diagnosing distributed training that hangs, deadlocks, or aborts in ProcessGroupNCCL.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Training on Multiple GPUs

## When To Use

Use this skill when a user wants to scale an Isaac Lab RL job across several GPUs or nodes, or when a
distributed run hangs, deadlocks, or fails inside NCCL.

Do not use this skill for single-GPU training setup, framework selection, or agent config wiring. Use
`isaaclab-training-rl-agents` for those. Use `isaaclab-debugging-rl-training` when the job runs to
completion but the reward curves or metrics look wrong.

## Workflow

1. Confirm the job trains correctly on a single GPU first. Do not debug a distributed failure before
   the same task, backend, and preset combination is known to work with one rank.
2. Launch with `uv run isaaclab train_multigpu`, which wraps `torchrun` and injects `--distributed`
   for you. Pass all training arguments exactly as for `uv run isaaclab train`.
3. Select the rank count with `--num_gpus`. The default `gpu` uses every visible device. Use
   `--dry_run` to print the resolved launcher command without running it.
4. For skrl with JAX, `--num_gpus` must be an integer and the torchrun-only options are rejected. Use
   `--coordinator_address` instead.
5. Expect console output from local rank 0 on each node only. The launcher filters the other ranks by
   default because each one repeats the same startup and warning output. Use `--log_all_ranks` when
   ranks may disagree with each other, or `--tee 3 --log_dir <dir>` to keep per-rank logs on disk.
6. Separate a launcher problem from a communication problem before changing training code. A hang
   with all participating GPUs at 100% utilization and no error is a NCCL stall, not a task bug.
7. When a run hangs, reproduce it at a different world size and with the standalone NCCL probe in
   [Reference](reference.md) before touching Isaac Lab code. World size 2 is the most common failure
   point on systems without NVLink.
8. Apply NCCL environment workarounds one at a time, and confirm each one against the probe before
   applying it to training. Record which variable was needed and why.
9. Treat NCCL workarounds as machine-specific. Do not bake them into committed configs or scripts
   without noting the affected hardware.
10. For multi-node jobs, establish rendezvous with `--nnodes`, `--node_rank`, and either
    `--master_addr`/`--master_port` or the `--rdzv_*` options, and verify a two-node job before
    scaling further.

## Validation

Use this checklist:

1. Confirm the same command trains on a single GPU.
2. Confirm every rank reaches parameter synchronization in the logs, not just rank 0.
3. Confirm the run reports a training time and exits with status 0.
4. Confirm scaling changes throughput rather than silently running one rank.
5. When a NCCL workaround is applied, confirm the run also fails without it, so the workaround is
   known to be the cause of the fix.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/features/multi_gpu.rst`,
`docs/source/refs/troubleshooting.rst`, and the launcher at
`scripts/reinforcement_learning/train_multigpu.py`. The NCCL workaround list is maintained in the
multi-GPU feature doc. Add new workarounds there first and keep this skill as a router.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Training RL agents skill](../train-rl-agents/SKILL.md)
- [Debug RL training skill](../debug-rl-training/SKILL.md)
- [Multi-GPU and multi-node training](../../../docs/source/features/multi_gpu.rst)
- [Troubleshooting](../../../docs/source/refs/troubleshooting.rst)
- [Multi-GPU launcher](../../../scripts/reinforcement_learning/train_multigpu.py)
