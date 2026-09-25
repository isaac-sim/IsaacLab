# Multi-GPU Training Evaluations

## Scenario 1: Launching a Multi-GPU Run

Query: "How do I train Isaac-Cartpole across all my GPUs?"

Expected behavior:

- Uses `uv run isaaclab train_multigpu` with `--rl_library` and the suffixless task name.
- Explains that the default `--num_gpus gpu` uses every visible device, and that training flags are
  passed exactly as for single-GPU training.
- Suggests confirming the single-GPU run first, and mentions `--dry_run` to inspect the resolved
  command.

Known failure modes:

- Hand-writes a `torchrun` invocation instead of using the supported launcher.
- Adds `--distributed` manually, which the launcher already injects.
- Recommends a rank count without checking how many GPUs are visible.

## Scenario 2: Silent Hang During Distributed Training

Query: "My 2-GPU training run just hangs after 'Synchronizing parameters' and never prints anything else."

Expected behavior:

- Identifies the signature: no traceback, no timeout, participating GPUs pinned at 100% utilization,
  which indicates a stalled NCCL collective rather than a task bug.
- Bisects outward-in: retries at a different world size, then bypasses `train_multigpu` with plain
  `torchrun`, then runs the standalone NCCL probe from `reference.md`.
- Calls out world size 2 on systems without NVLink as the most common cause, and checks
  `nvidia-smi topo -m`.
- Proposes `NCCL_P2P_DISABLE=1` and verifies the run fails without it and passes with it.

Known failure modes:

- Blames Isaac Lab, the task, or the RL library without reproducing outside them.
- Starts editing environment or training code before establishing where the stall occurs.
- Applies several NCCL variables at once, leaving the actual cause unknown.
- Treats "works on 4 GPUs, hangs on 2" as intermittent flakiness instead of a world-size-dependent
  transport problem.

## Scenario 3: Choosing a Rank Count for skrl JAX

Query: "Can I run multi-GPU training with skrl and JAX?"

Expected behavior:

- Notes that skrl JAX uses skrl's own distributed launcher rather than torchrun.
- Requires an integer `--num_gpus` and routes coordination through `--coordinator_address`.
- Explains that the torchrun-only rendezvous options are rejected for this combination.

Known failure modes:

- Passes `--num_gpus gpu` for skrl JAX, which the launcher cannot resolve.
- Mixes `--master_addr`/`--rdzv_*` options into a skrl JAX launch.

## Scenario 4: Persisting a NCCL Workaround

Query: "NCCL_P2P_DISABLE=1 fixed my hang. Should I add it to the repo so nobody else hits this?"

Expected behavior:

- Explains that the variable is machine-specific and can reduce inter-GPU bandwidth, since traffic
  is routed through host memory instead of a direct link.
- Recommends scoping it to the affected machine rather than committing it into shared configs or
  launch scripts.
- Points to `docs/source/features/multi_gpu.rst` as the maintained place to document a newly
  confirmed workaround, including the hardware it applies to.

Known failure modes:

- Adds the variable to a committed script or config that runs on unaffected hardware.
- Documents the workaround without recording the topology or world size that triggered the hang.
