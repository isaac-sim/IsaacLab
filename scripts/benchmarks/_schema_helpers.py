# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the v1.0 benchmark bundle writers.

Used by ``benchmark_startup.py``, ``benchmark_rsl_rl.py``, and
``benchmark_skrl.py`` to build schema-v1 ``Versions`` and ``Hardware``
dataclasses from the benchmark's manual recorders, and to synthesise a
fallback run_id when the caller did not provide one.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone

from isaaclab.test.benchmark import BaseIsaacLabBenchmark
from isaaclab.benchmark.schema import GpuDeviceInfo, Hardware, Versions


def capture_versions(bm: BaseIsaacLabBenchmark) -> Versions:
    """Build a :class:`Versions` from the benchmark's ``VersionInfoRecorder``.

    Must be called before :meth:`BaseIsaacLabBenchmark._finalize_impl`, which
    clears ``_manual_recorders``.
    """
    meta = {m.name: m.data for m in bm._manual_recorders["VersionInfo"].get_data().metadata}
    dev = meta.get("dev", {}) or {}
    return Versions(
        isaaclab=meta.get("isaaclab_version", "unknown"),
        isaacsim=meta.get("isaacsim_version"),
        kit=meta.get("kit_version"),
        newton=meta.get("newton_version"),
        warp=meta.get("warp_version"),
        mjwarp=meta.get("mujoco_warp_version"),
        torch=meta.get("torch_version", "unknown"),
        rsl_rl=meta.get("rsl_rl_version"),
        skrl=meta.get("skrl_version"),
        git_commit=dev.get("commit_hash"),
        git_branch=dev.get("branch"),
        git_dirty=bool(dev.get("dirty", False)),
    )


def capture_hardware(bm: BaseIsaacLabBenchmark) -> Hardware:
    """Build a :class:`Hardware` from GPU/CPU/Memory recorders.

    Must be called before :meth:`BaseIsaacLabBenchmark._finalize_impl`, which
    clears ``_manual_recorders``.
    """
    gpu_meta = {m.name: m.data for m in bm._manual_recorders["GPUInfo"].get_data().metadata}
    cpu_meta = {m.name: m.data for m in bm._manual_recorders["CPUInfo"].get_data().metadata}
    mem_meta = {m.name: m.data for m in bm._manual_recorders["MemoryInfo"].get_data().metadata}
    devices_raw = gpu_meta.get("gpu_devices", {}) or {}
    devices = [
        GpuDeviceInfo(
            name=str(d.get("name", "unknown")),
            mem_gb=float(d.get("total_memory_gb", 0.0) or 0.0),
            compute_cap=str(d.get("compute_capability", "unknown")),
        )
        for d in devices_raw.values()
    ]
    return Hardware(
        hostname=socket.gethostname(),
        gpu_devices=devices,
        cpu_name=str(cpu_meta.get("cpu_name", "unknown")),
        cpu_count=int(cpu_meta.get("physical_cores", 0) or 0),
        ram_gb=float(mem_meta.get("total_ram_gb", 0.0) or 0.0),
    )


def synth_run_id(framework: str, backend: str, task: str, seed: int) -> str:
    """Fallback run_id when the caller did not supply ``--run_id``.

    Format: ``<framework>_<backend>_<task>_<YYYYMMDD-HHMMSS>_seed<seed>``,
    with underscores in ``framework`` replaced by hyphens (so ``rsl_rl``
    becomes ``rsl-rl``).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fw = framework.replace("_", "-")
    return f"{fw}_{backend}_{task}_{stamp}_seed{seed}"
