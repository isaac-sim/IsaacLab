# Runtime visual DR: environment setup

Cosmos runs inside Isaac Lab's own environment. It does **not** need a separate
Python version, a separate torch, or a patched Cosmos checkout.

## Verified configuration

| | |
|---|---|
| Python | 3.12 |
| torch | 2.12 (cu130), Isaac Lab's own pin |
| transformers | 5.10.4, Isaac Lab's own pin |
| diffusers | 0.39.0 |
| flash-attn | not installed, and not required |
| checkpoint | `nvidia/Cosmos3-Nano` (public, ungated) |

## Install

```bash
# --extra teleop: the stacking task imports XrCfg, which now lives in
#   isaaclab_teleop, so the task will not import without it.
# --extra video:  only for AppLauncher's own --video; the visual DR scripts
#   write MP4s through imageio-ffmpeg, which the cosmos-runtime extra supplies.
uv sync --extra isaacsim --extra teleop --extra video

uv pip install -e 'source/isaaclab_contrib[cosmos-runtime]'
uv pip install --no-deps -e /path/to/cosmos-framework
```

### Re-running uv sync will remove all of this

``uv sync`` is exact: it uninstalls whatever the lockfile does not name. Because
Cosmos is installed over the top of the locked environment, a later plain ``uv
sync`` removes ``cosmos-framework`` and every dependency the extra brought in --
around fifty packages. Use:

```bash
uv sync --extra isaacsim --extra teleop --extra video --inexact
```

``--inexact`` leaves extraneous packages alone, which preserves the overlay.
Otherwise re-run the two ``uv pip install`` lines above after every sync.

Cosmos cannot currently be locked alongside Isaac Lab, but it is closer than it
looks: its base dependencies do **not** pin torch at all -- torch appears only in
the optional CUDA groups, which is what ``--no-deps`` avoids. The single blocker is
its base pin of ``transformers>=4.57.1,<5`` against Isaac Lab's ``5.10.4``.
Relaxing that, via a uv dependency override and a git source for Cosmos, would let
it live in the lockfile and survive a sync. That is a packaging decision for the
repository rather than something this feature should impose.

`--no-deps` is the important part. Cosmos publishes dependency *groups* that pin
torch 2.10 or 2.13 to match the CUDA wheel variants it ships, and installing those
would fight Isaac Lab's pin. They are not hard requirements: Cosmos itself declares
`requires-python >= 3.10`, and the depth and segmentation transfer paths were
verified end to end on the table above. The `cosmos-runtime` extra supplies the
pure-Python dependencies Cosmos actually imports.

FlashAttention is the one component whose wheels are built against a specific torch
ABI. It is not needed here: the transfer path runs without it, falling back to
cuDNN attention, or to torch's own Flash kernels when cuDNN cannot compile a new
shape -- which is what happens once Omniverse Kit has started.

## Running generation on other GPUs

Generation is slow enough (seconds per frame) that sharing a GPU with the simulator
hurts. `RemoteCosmosBackendCfg` runs it in worker processes on their own GPUs,
passing frames by CUDA IPC so image data never goes through host memory. Device
indices are indices into what the process can see, so `CUDA_VISIBLE_DEVICES` must
cover the union of the workers' GPUs and the simulator's.

## A different Cosmos checkout, or a different checkpoint

Both are supported without touching the code.

```bash
# any local path, https URL, or ssh URL, optionally pinned with '.git@<ref>'
scripts/visual_dr/install_cosmos.sh \
    ssh://git@<host>:<port>/<group>/cosmos-framework.git@<ref>
```

The script installs the `cosmos-runtime` extra, then Cosmos itself with
`--no-deps`, and reports which checkout it ended up with and whether that checkout
supports guided generation. Splitting on `.git@` keeps the `@` in an ssh URL's
`git@host` from being mistaken for a ref.

`CosmosBackendCfg.checkpoint` accepts a registered name, an `s3://` URI, or a local
directory, so a downloaded checkpoint needs no special handling:

```bash
hf download nvidia/<repo> --revision <rev> \
    --include '<checkpoint-name>/**' \
    --local-dir <local-path>
```

```python
checkpoint = "<local-path>/<checkpoint-name>"
```

### A custom checkout may need FlashAttention

The base `Cosmos3-Nano` path needs no FlashAttention: it falls back to cuDNN, or to
torch's own Flash kernels. A checkout whose models use variable-length (packed)
attention is different -- cuDNN has no varlen path, so FlashAttention becomes
mandatory and generation otherwise fails with *"Could not find a compatible
Attention backend"*.

The Cosmos dependency index publishes FlashAttention only for CPython 3.13 against
torch 2.9 or 2.10, which does not fit Isaac Lab. Dao-AILab's own releases do:

```bash
uv pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

That wheel is built against torch 2.10 and imports and runs correctly on torch
2.12 -- the ABI held in testing, including the varlen kernel. It is a pragmatic pin
rather than a guaranteed one: check it still imports after any torch upgrade, since
nothing enforces the match.

### Distilled checkpoints want different sampling

The defaults here -- 16 sampler steps at `guidance=3.0` -- are tuned for the base
`Cosmos3-Nano`. A DMD2-distilled checkpoint is trained to run in about four steps
*without* classifier-free guidance, so it wants roughly `num_steps=4` and
`guidance=1.0`. Leaving the base defaults in place is both slower and wrong: high
guidance on a model distilled without it tends to produce hard contrast rather than
detail.
