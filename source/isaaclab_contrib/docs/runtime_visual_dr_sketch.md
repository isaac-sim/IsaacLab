# Runtime visual DR: first implementation sketch

Companion to [the working requirements](runtime_visual_dr.md). This is a small
integration boundary to review together, not an operational Cosmos demo yet.
The code lives in `source/isaaclab_contrib/isaaclab_contrib/visual_dr/`.

## What this slice does

- `runtime.py`: explicit observation identity, consumed-frame selection, per-camera
  caching, foreground compositing and application-owned activate/offload/close.
- `observations.py`: camera RGB/depth/semantic IDs to GPU tensors, then DR before
  observation normalization and history. Unknown semantic IDs remain protected.
- `cosmos.py`: configuration and a factory boundary for a tensor-native backend.
  Without an adapter, construction raises `NotImplementedError`; it cannot silently
  call a file-based service or return unchanged images as successful generation.
- `nixl.py`: lazy optional agent creation and the communication sequence to fill in.
  There is no implemented remote backend, buffer pool or server in this version.
- `demo.py`: an opt-in clone of the existing Franka stacking Cosmos camera config.
  It enables uncolored segmentation on both cameras, keeps RGB uint8, and removes
  auxiliary images from policy observations. Instantiate `FrankaStackRuntimeDRCfg`
  directly; no new Gym registration or core environment changes are needed yet.

The demo keeps the robot, cubes, table and unknown objects intact; only explicit
`ground` and `BACKGROUND` labels are replaceable. Broaden that choice after visual
inspection. Rendering still runs on skipped action-chunk frames; inference does not.
The first version is synchronous, single-threaded, one environment, one view per
request. It clones GPU buffers to isolate sensor storage and cached observations.
Depth conversion, model resize/crop rules and seed derivation belong in the adapter.

## Application wiring

After starting Isaac Sim with the usual AppLauncher, construct a
`ManagerBasedRLEnv(cfg=FrankaStackRuntimeDRCfg())`. Attach the runtime after
construction so observation-dimension probes never load or invoke the model.
This is illustrative wiring; `tensor_backend_factory` is still to build:

```python
backend = create_cosmos_backend(CosmosDRCfg(checkpoint, prompt), tensor_backend_factory)
env.visual_dr_runtime = runtime = VisualDRRuntime(backend)
runtime.activate()
runtime.begin(DRObservation(sequence=0, episode=0, seed=42))
observations, info = env.reset()
# Before EACH subsequent step or explicit reset, begin a new increasing sequence.
# Set consumed=False only when no policy, history or value-bootstrap reader needs it.
# For K actions, mark actual decision boundaries, not every K physics ticks.
# Retain the exact processed policy inputs in rollout storage for learning.
runtime.offload()  # backend must finish inference AND transfers before returning
# ... learner uses the released memory ...
runtime.activate()
runtime.begin(DRObservation(sequence=1, episode=0, seed=43))
# ... next environment observation ...
runtime.close()  # application calls before env.close(), including on exceptions
```

No automatic reset/terminal hooks exist yet. A new scope must precede each reset;
multiple renders within a scope are unsupported. Do not use this prototype for
autoreset RL: distinguish pre-reset terminal/bootstrap observations from reset
observations in the core environment before adding vectorized/partial resets.
Reusing a sequence with different context raises instead of reusing stale pixels.
The base stacking task has no reward configuration and ships imitation-learning
configuration. A reward, compatible policy and clean baseline are still needed for RL.

## Cosmos installation in the shared environment

The optional `isaaclab_contrib[cosmos-runtime]` extra carries inference support
dependencies from Cosmos metadata and the existing shared installer, including
dependencies previously supplied indirectly by training packages. It does not
depend on a trainer, Ray, ZMQ, or Cosmos's training/serving extras.
The extra alone does **not** install Cosmos: use the patched source checkout.

For a Linux Python 3.12 environment with IsaacLab already installed, this is the
incremental recipe **after resolving the version mismatch below**:

```bash
# From the IsaacLab checkout, targeting the same active environment:
uv pip install -e 'source/isaaclab_contrib[cosmos-runtime]'
uv pip install --no-deps -e /path/to/patched/cosmos-framework
# Optional, only on a separately communicating GPU worker/client:
uv pip install -e 'source/isaaclab_contrib[nixl]'
```

The source pin inspected was Cosmos `5e02e643c458ce06c7232244271f567dce1dec7a`.
Its metadata requests diffusers >=0.39.0; the shared setup deliberately uses
0.35.1 and installs Cosmos with `--no-deps`. The extra records that exception;
this is not a clean resolution of upstream Cosmos metadata. Audit `uv pip check`
and imports in the final environment; do not paper over additional conflicts.
Root `uv sync` needs corresponding extras/lockfile work once the union is tested.

The shared setup finishes on torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0,
CUDA 12.8 and a CPython 3.12 FlashAttention 2.8.3 wheel built for torch 2.10.
Its early comment saying torch 2.9 is stale. Transformers is pinned to 4.57.3.
IsaacLab `a8b4da3` now declares torch 2.11 and transformers 4.57.6. **That union is
unvalidated.** Select and test a compatible torch/FlashAttention build before
claiming a shared install works on this revision. This sketch does not downgrade
IsaacLab or transplant the old binary wheel. Windows contract tests cannot settle it.

FP8 remains optional adapter work: the existing recipe uses torchao 0.14.0 with
`--no-deps`; 0.16 removes an import used by diffusers 0.35.1. Revalidate it with the
chosen torch version. Neither torchao nor its precision behavior is enabled here.
The current depth-guided Nano path omits natten and transformer-engine; do not
assume those omissions hold for another model or attention configuration.

## NIXL communication direction

Start in-process with CUDA tensors. A separate worker can implement the same
`DRBackend` contract with NIXL; its owner handles residency and replicas.
Register long-lived CUDA input/output buffers at both peers, exchange only
descriptors and request metadata, transfer inputs, run inference, then transfer
results. Agent creation follows the [NVIDIA Python example](https://github.com/ai-dynamo/nixl/blob/main/examples/python/expanded_two_peers.py).
The `nixl` extra is independent of Cosmos and imported only when requested.

The next slice needs bounded leases, CUDA synchronization, deadlines, restart
handling and teardown. Keep buffers registered until inference, transfers and all
consumers finish. Do not treat a timeout as permission to reuse memory.
NIXL targets [Linux](https://github.com/ai-dynamo/nixl#supported-platforms).
Selecting UCX is not proof of a GPU-only route: [UCX supports host staging too](https://openucx.readthedocs.io/en/master/faq.html#working-with-gpu).
Qualify both directions and all batch sizes on the deployed topology; reject
unsupported paths. No image serialization or CPU transport fallback is proposed.
NVIDIA Dynamo could later own worker discovery/control; it is unnecessary for
this first boundary and is not an installation dependency.

## Overrides to account for before real inference

These are local compatibility findings, not requirements on every application:

- **GPU-native entry point:** the current Cosmos integration wraps file-oriented
  loading/output and converts images to CPU tensors. NIXL alone cannot fix that.
  Extract CUDA preprocessing, sampling and decoding into the new backend first.
- **Mask guidance spans files:** Cosmos overrides add transfer arguments, masked
  latent state projection in `transfer.py`, sampler hooks in fixed-step/UniPC and
  forwarding in `omni_mot_model.py`. Keep these together when porting; final
  compositing alone does not reproduce masked denoising behavior.
- **Batching spans files:** `inference.py`, `transfer.py`, model attention and network
  overrides support multi-sample control packing. Validate those together before
  enabling multi-environment/view batches. This slice accepts N=1.
- **Attention guard changes:** overrides raise the Flash2 version ceiling and
  disable its variable-length rejection. Those edits do not establish ABI or
  numerical compatibility with torch 2.11.
- **Checkpoint aliases:** public model configuration and export helpers contain
  namespace/snapshot-path fixes. Include a representative existing checkpoint in
  the first import/load/generate test; a clean upstream checkout is insufficient.
- **IsaacLab interfaces:** camera outputs now use Warp-backed `ProxyArray`; this
  term uses `.torch`. Existing timing and Mimic overrides do not provide the
  observation lifecycle hooks needed for terminal images or partial resets.

## Where the next features belong

Offload belongs to the model owner behind `DRBackend`; frame selection belongs to
the environment application's consumption context. FP8 and `torch.compile` belong
in the Cosmos factory, with unsupported options rejected. Replicas and multi-node
placement belong to worker orchestration, with NIXL handling GPU payload movement.
None should require importing a trainer or reading its configuration files.

Kitless tests cover scopes, skips, cache isolation, foreground masks, lifecycle
admission, CPU rejection and the missing-adapter error. They use CPU tensors with
the CUDA guard isolated only in the control-flow test. They do not validate Isaac
Sim rendering, model output, CUDA stream ordering, memory paging or NIXL transfers.
Those are the next Linux GPU checks after resolving the installation blockers.
