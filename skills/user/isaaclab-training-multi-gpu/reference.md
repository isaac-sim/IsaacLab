# Multi-GPU Training Reference

## Contents

- [Launching](#launching)
- [Console output from multiple ranks](#console-output-from-multiple-ranks)
- [Isolating a distributed hang](#isolating-a-distributed-hang)
- [NCCL workarounds](#nccl-workarounds)

## Launching

```bash
# All visible GPUs (default).
uv run isaaclab train_multigpu --rl_library rsl_rl --task Isaac-Cartpole

# Fixed rank count.
uv run isaaclab train_multigpu --rl_library rsl_rl --num_gpus 2 --task Isaac-Cartpole

# Print the resolved torchrun command without running it.
uv run isaaclab train_multigpu --rl_library rsl_rl --num_gpus 2 --task Isaac-Cartpole --dry_run
```

The launcher forwards unrecognized arguments to `scripts/reinforcement_learning/train.py` and adds
`--distributed`, so training flags are written exactly as for `uv run isaaclab train`. Supported
libraries are `rsl_rl`, `rl_games`, and `skrl`. skrl with `--ml_framework jax` uses skrl's own
launcher, requires an integer `--num_gpus`, and takes `--coordinator_address` instead of the
torchrun rendezvous options.

## Console output from multiple ranks

Every rank runs the full training script, so every rank repeats the same startup banners, simulation
warnings, and model summaries. The launcher passes torchrun's `--local_ranks_filter` by default so
only local rank 0 reaches the console. Per-iteration metrics already come from global rank 0 alone, so
nothing is lost. The filter is per node, so a multi-node launch shows the startup output once per node.

Crashes still surface: `train.py` wraps its entry point in `record`, so torchrun names the failing
rank and prints its traceback as the root cause even when that rank's output is hidden.

```bash
# Raw output from every rank, for example when ranks disagree about what they built.
uv run isaaclab train_multigpu --rl_library rsl_rl --task Isaac-Cartpole --log_all_ranks

# Clean console, full per-rank logs on disk under --log_dir.
uv run isaaclab train_multigpu --rl_library rsl_rl --task Isaac-Cartpole --tee 3 --log_dir /tmp/ranklogs
```

Rank filtering is a torchrun feature. skrl with `--ml_framework jax` uses skrl's own launcher and
always writes every rank to the console.

## Isolating a distributed hang

A distributed run that stops producing output while every participating GPU sits at 100%
utilization is a stalled NCCL collective. NCCL busy-waits, so there is no traceback and no timeout
by default. Bisect from the outside in, and stop at the first layer that reproduces:

1. **Change the world size.** Run at 2, 3, and 4 ranks. A failure that appears only at one world
   size points at the NCCL transport rather than at the task.
2. **Drop the launcher.** Run `torchrun` directly. If this still hangs, `train_multigpu` is not
   involved:

   ```bash
   uv run python -m torch.distributed.run --nproc_per_node 2 \
     scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Cartpole --distributed
   ```

3. **Drop Isaac Lab.** Run the standalone probe below. If the probe hangs, the problem is in NCCL or
   the system topology, and no change to Isaac Lab or the RL library will fix it.

   ```python
   # nccl_probe.py
   import os, torch, torch.distributed as dist

   local_rank = int(os.environ["LOCAL_RANK"])
   torch.cuda.set_device(local_rank)
   dist.init_process_group("nccl")
   tensor = torch.ones(1024, device=f"cuda:{local_rank}")
   dist.all_reduce(tensor)
   torch.cuda.synchronize()
   print(f"rank {dist.get_rank()} ok", flush=True)
   dist.destroy_process_group()
   ```

   ```bash
   uv run python -m torch.distributed.run --nproc_per_node 2 nccl_probe.py
   ```

4. **Check the topology.** `nvidia-smi topo -m` shows the interconnect. Systems with only `PHB`,
   `NODE`, or `SYS` links have no NVLink, and inter-GPU peer-to-peer traffic crosses PCIe, which is
   where the known hangs occur.
5. **Confirm the fix both ways.** Re-run the hanging case with the candidate environment variable
   set, then re-run it unset. A workaround is only established when the run fails without it.

Useful signals while a job is stalled:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

## NCCL workarounds

`docs/source/features/multi_gpu.rst` is the maintained list. Add new entries there, not here.

| Symptom | Variable |
| --- | --- |
| Hang at world size 2 on a system without NVLink | `NCCL_P2P_DISABLE=1` |
| `CUDA error: an illegal memory access was encountered` in `ProcessGroupNCCL` | `NCCL_SHM_DISABLE=1` |
| Collective timeout with rendering enabled across NUMA nodes | `NCCL_CUMEM_HOST_ENABLE=0`, then `NCCL_CUMEM_ENABLE=0` |
| Persistent init or transport failures | `NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring` |

Set `NCCL_DEBUG=INFO` to see which transport NCCL selected.

These variables change how ranks communicate and can reduce bandwidth. `NCCL_P2P_DISABLE=1` in
particular routes traffic through host memory instead of a direct GPU-to-GPU link. Scope them to the
affected machine rather than committing them into shared configs.
