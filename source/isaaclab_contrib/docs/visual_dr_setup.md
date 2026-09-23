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
uv sync --extra isaacsim
uv pip install -e 'source/isaaclab_contrib[cosmos-runtime]'
uv pip install --no-deps -e /path/to/cosmos-framework
```

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
