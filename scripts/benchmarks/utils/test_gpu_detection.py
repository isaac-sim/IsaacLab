#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test GPU detection for standalone benchmark."""

from contextlib import suppress

print("Testing GPU detection methods...\n")

# Test GPUtil
print("=" * 60)
print("Testing GPUtil:")
print("=" * 60)
try:
    import GPUtil

    print("✓ GPUtil is installed")
    gpus = GPUtil.getGPUs()
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            print(f"    Memory Used: {gpu.memoryUsed:.2f} MB")
            print(f"    Memory Total: {gpu.memoryTotal:.2f} MB")
            print(f"    Load: {gpu.load * 100:.1f}%")
    else:
        print("✗ No GPUs found via GPUtil")
except ImportError:
    print("✗ GPUtil not installed. Install with: pip install GPUtil")
except Exception as e:
    print(f"✗ Error using GPUtil: {e}")

print()

# Test pynvml
print("=" * 60)
print("Testing pynvml (nvidia-ml-py3):")
print("=" * 60)
try:
    import pynvml

    print("✓ pynvml is installed")
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"✓ Found {device_count} GPU(s)")

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)

        print(f"  GPU {i}: {name}")
        print(f"    Memory Used: {mem_info.used / (1024**2):.2f} MB")
        print(f"    Memory Total: {mem_info.total / (1024**2):.2f} MB")
        print(f"    GPU Utilization: {util_rates.gpu}%")
        print(f"    Memory Utilization: {util_rates.memory}%")

    pynvml.nvmlShutdown()
except ImportError:
    print("✗ pynvml not installed. Install with: pip install nvidia-ml-py3")
except Exception as e:
    print(f"✗ Error using pynvml: {e}")
    with suppress(Exception):
        pynvml.nvmlShutdown()

print()

# Test nvidia-smi directly
print("=" * 60)
print("Testing nvidia-smi (direct system call):")
print("=" * 60)
try:
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if result.returncode == 0:
        print("✓ nvidia-smi is available")
        lines = result.stdout.strip().split("\n")
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                print(f"  GPU {i}: {parts[0]}")
                print(f"    Memory Used: {float(parts[1]):.2f} MB")
                print(f"    Memory Total: {float(parts[2]):.2f} MB")
                print(f"    GPU Utilization: {float(parts[3]):.1f}%")
    else:
        print(f"✗ nvidia-smi returned error code: {result.returncode}")
        print(f"  stderr: {result.stderr}")
except FileNotFoundError:
    print("✗ nvidia-smi not found. NVIDIA drivers may not be installed.")
except subprocess.TimeoutExpired:
    print("✗ nvidia-smi timed out")
except Exception as e:
    print(f"✗ Error running nvidia-smi: {e}")

print()

# Test torch CUDA
print("=" * 60)
print("Testing PyTorch CUDA:")
print("=" * 60)
try:
    import torch

    print("✓ PyTorch is installed")
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"    Memory Allocated: {mem_allocated:.2f} GB")
            print(f"    Memory Reserved: {mem_reserved:.2f} GB")
    else:
        print("✗ CUDA not available in PyTorch")
except ImportError:
    print("✗ PyTorch not installed")
except Exception as e:
    print(f"✗ Error using PyTorch: {e}")

print()
print("=" * 60)
print("Summary:")
print("=" * 60)
print("GPU Detection Methods (in priority order):")
print("1. GPUtil - Python library, fast but affected by driver mismatch")
print("2. pynvml - Official NVIDIA library, affected by driver mismatch")
print("3. PyTorch CUDA - Shows PyTorch-allocated memory (may be 0)")
print("4. nvidia-smi - Direct system call, MOST RELIABLE, bypasses driver issues")
print()
print("If no GPUs were detected, possible reasons:")
print("1. No NVIDIA GPU in the system")
print("2. NVIDIA drivers not installed (check: nvidia-smi)")
print("3. Driver/library version mismatch (ERROR: 'Driver/library version mismatch')")
print("   → FIX: sudo reboot  (reloads driver and libraries)")
print("4. GPU libraries not installed")
print("5. Permission issues (try: sudo usermod -a -G video $USER)")
print()
print("Best Solution for 'Driver/library version mismatch':")
print("  → Use nvidia-smi method (Method 4) - works despite mismatch")
print("  → Or reboot system to sync driver/library versions")
print()
print("Recommended install command:")
print("  pip install psutil GPUtil nvidia-ml-py3")
print()
print("Note: PyTorch is usually already installed in Isaac Lab environments")
print("      and nvidia-smi is the most reliable fallback.")
