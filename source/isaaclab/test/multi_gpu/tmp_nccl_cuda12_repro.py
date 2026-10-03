# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEMP (revert before review): does OVRTX's bundled CUDA 12 stack break NCCL in a CUDA 13 process?

Run under ``torchrun --nproc_per_node 2`` with one variant; each adds one step to the previous:

- ``none``: NCCL all-reduce only (control).
- ``load``: also load OVRTX's ``libcudart.so.12`` and ``libnvrtc.so.12``.
- ``cudart``: also initialize the CUDA 12 runtime on this rank's device.
- ``jit``: also compile a kernel with NVRTC 12 and load it through the driver.

Imports nothing from Isaac Lab and never imports ``ovrtx`` (its import could load the libraries).
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

variant = sys.argv[1]
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.ones(1, device=f"cuda:{rank}")  # create the primary context first, as torch does in training

plugins = Path(importlib.util.find_spec("ovrtx").origin).parent / "bin" / "plugins"
steps = ["none", "load", "cudart", "jit"]
if steps.index(variant) >= 1:
    cudart = ctypes.CDLL(str(plugins / "gpu.foundation" / "libcudart.so.12"))
    nvrtc = ctypes.CDLL(str(plugins / "rtx" / "libnvrtc.so.12"))
if steps.index(variant) >= 2:
    assert cudart.cudaSetDevice(rank) == 0
    assert cudart.cudaFree(None) == 0
if steps.index(variant) >= 3:
    major, minor = torch.cuda.get_device_capability(rank)
    prog = ctypes.c_void_p()
    src = b'extern "C" __global__ void k(float* x) { x[threadIdx.x] += 1.0f; }'
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None) == 0
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=sm_{major}{minor}".encode())
    assert nvrtc.nvrtcCompileProgram(prog, 1, opts) == 0
    size = ctypes.c_size_t()
    assert nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)) == 0
    cubin = ctypes.create_string_buffer(size.value)
    assert nvrtc.nvrtcGetCUBIN(prog, cubin) == 0
    cuda = ctypes.CDLL("libcuda.so.1")
    module, fn = ctypes.c_void_p(), ctypes.c_void_p()
    assert cuda.cuModuleLoadData(ctypes.byref(module), cubin) == 0
    assert cuda.cuModuleGetFunction(ctypes.byref(fn), module, b"k") == 0

dist.init_process_group(backend="nccl")
value = torch.ones(1, device=f"cuda:{rank}")
dist.all_reduce(value)
torch.cuda.synchronize()
print(f"[REPRO] variant={variant} rank={rank} all_reduce={value.item()} OK", flush=True)
dist.destroy_process_group()
