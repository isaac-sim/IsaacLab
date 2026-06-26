# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture helpers for benchmark data extraction.

Reads recorder data off a ``BaseIsaacLabBenchmark``-like object and maps the
raw measurements and metadata to the typed schema dataclasses
:class:`~isaaclab.test.benchmark.schema.Versions`,
:class:`~isaaclab.test.benchmark.schema.Hardware`, and
:class:`~isaaclab.test.benchmark.schema.Resources`.

Pure stdlib — no torch, isaacsim, or RL-library imports.  The benchmark
object is accepted at call time; its recorder classes are never imported here.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from isaaclab.test.benchmark.schema import GpuDeviceInfo, Hardware, MeanStd, Resources, RunConfig, Versions

###
# Private helpers
###


def _metadata(metadata_list: Any) -> dict[str, Any]:
    """Build a name-keyed dict from a list of metadata objects.

    Args:
        metadata_list: Sequence of metadata objects, each with ``.name`` and
            ``.data`` attributes.

    Returns:
        Mapping from ``m.name`` to ``m.data`` for every entry in the list.
    """
    if not metadata_list:
        return {}
    return {m.name: m.data for m in metadata_list}


def _find_value(measurements: Any, name: str, default: float = 0.0) -> float:
    """Scan a measurement list for a :class:`~.measurements.SingleMeasurement` by name.

    Args:
        measurements: Sequence of measurement objects, each with ``.name`` and
            ``.value`` attributes.
        name: Exact measurement name to look up.
        default: Value to return when the name is not found. Defaults to ``0.0``.

    Returns:
        The ``float`` value of the first matching measurement, or *default*.
    """
    if not measurements:
        return default
    for m in measurements:
        if m.name == name:
            return float(m.value)
    return default


def _get_recorder_data(bm: Any, key: str) -> Any | None:
    """Safely retrieve :meth:`get_data` output from a named recorder.

    Args:
        bm: Benchmark-like object with a ``_manual_recorders`` attribute.
        key: Recorder key (e.g. ``"VersionInfo"``).

    Returns:
        The :class:`~.interfaces.MeasurementData` returned by the recorder, or
        ``None`` when the recorders dict is absent or the key is missing.
    """
    recorders = getattr(bm, "_manual_recorders", None)
    if recorders is None:
        return None
    rec = recorders.get(key)
    if rec is None:
        return None
    return rec.get_data()


###
# Public API — run-identity helpers
###


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 formatted current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def synth_run_id(
    framework: str | None,
    physics_backend: str,
    task: str,
    seed: int,
    stamp: str,
) -> str:
    """Synthesise a stable run identifier from run parameters.

    Args:
        framework: RL framework name, or ``None`` for non-learning runs
            (substituted with ``"runtime"``).
        physics_backend: Physics backend preset string.
        task: Gym task id.
        seed: Environment/agent seed.
        stamp: Timestamp string (e.g. ``"20260612-150000"``).

    Returns:
        Underscore-joined run identifier string.
    """
    fw = framework or "runtime"
    return f"{fw}_{physics_backend}_{task}_{stamp}_seed{seed}"


###
# Public API — schema capture functions
###


def capture_versions(bm: Any) -> Versions:
    """Read software version metadata from a benchmark object.

    Reads ``bm._manual_recorders["VersionInfo"].get_data().metadata`` and maps
    the ``*_version`` :class:`~.measurements.StringMetadata` entries plus the
    ``dev`` :class:`~.measurements.DictMetadata` to a :class:`~.schema.Versions`
    instance.  All fields default gracefully when the recorder is absent.

    Args:
        bm: Benchmark-like object exposing ``._manual_recorders``.

    Returns:
        Populated :class:`~.schema.Versions` dataclass; never raises.
    """
    data = _get_recorder_data(bm, "VersionInfo")
    if data is None:
        return Versions(
            isaaclab="unknown",
            isaacsim=None,
            kit=None,
            newton=None,
            warp=None,
            mjwarp=None,
            torch="unknown",
            rsl_rl=None,
            rl_games=None,
            skrl=None,
            sb3=None,
            git_commit=None,
            git_branch=None,
            git_dirty=False,
        )

    md = _metadata(data.metadata)
    dev: dict[str, Any] = md.get("dev") or {}

    return Versions(
        isaaclab=md.get("isaaclab_version", "unknown"),
        isaacsim=md.get("isaacsim_version", None),
        kit=md.get("kit_version", None),
        newton=md.get("newton_version", None),
        warp=md.get("warp_version", None),
        mjwarp=md.get("mujoco_warp_version", None),
        torch=md.get("torch_version", "unknown"),
        rsl_rl=md.get("rsl_rl_version", None),
        rl_games=md.get("rl_games_version", None),
        skrl=md.get("skrl_version", None),
        sb3=md.get("stable_baselines3_version", None),
        git_commit=dev.get("commit_hash"),
        git_branch=dev.get("branch"),
        git_dirty=dev.get("dirty", False),
    )


def capture_hardware(bm: Any) -> Hardware:
    """Read hardware metadata from a benchmark object.

    Reads GPU, CPU, and memory recorder metadata to populate a
    :class:`~.schema.Hardware` instance.  All fields default gracefully when
    recorders are absent.

    Args:
        bm: Benchmark-like object exposing ``._manual_recorders``.

    Returns:
        Populated :class:`~.schema.Hardware` dataclass; never raises.
    """
    gpu_data = _get_recorder_data(bm, "GPUInfo")
    cpu_data = _get_recorder_data(bm, "CPUInfo")
    mem_data = _get_recorder_data(bm, "MemoryInfo")

    # GPU devices
    gpu_devices: list[GpuDeviceInfo] = []
    if gpu_data is not None:
        gpu_md = _metadata(gpu_data.metadata)
        raw_devices: dict[str, Any] = gpu_md.get("gpu_devices") or {}
        for idx_str in sorted(raw_devices.keys(), key=lambda k: int(k)):
            d = raw_devices[idx_str]
            gpu_devices.append(
                GpuDeviceInfo(
                    name=d["name"],
                    mem_gb=float(d["total_memory_gb"]),
                    compute_cap=str(d["compute_capability"]),
                )
            )

    # CPU
    cpu_name = "unknown"
    cpu_count = 0
    if cpu_data is not None:
        cpu_md = _metadata(cpu_data.metadata)
        cpu_name = cpu_md.get("cpu_name", "unknown")
        cpu_count = int(cpu_md.get("physical_cores", 0))

    # RAM
    ram_gb = 0.0
    if mem_data is not None:
        mem_md = _metadata(mem_data.metadata)
        ram_gb = float(mem_md.get("total_ram_gb", 0.0))

    return Hardware(
        hostname=socket.gethostname(),
        gpu_devices=gpu_devices,
        cpu_name=cpu_name,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
    )


_PHYSICS_PRESETS = {"physx", "newton_mjwarp", "newton_kamino", "ovphysx"}
_PHYSICS_ALIASES = {"newton": "newton_mjwarp", "kamino": "newton_kamino"}
_RENDERING_PRESETS = {
    "isaacsim_rtx_renderer": "isaacsim_rtx",
    "ovrtx_renderer": "ovrtx",
    "newton_renderer": "newton",
}


def run_config_from_presets(tokens: Sequence[str]) -> RunConfig:
    """Best-effort :class:`~isaaclab.test.benchmark.schema.RunConfig` from active Hydra preset tokens.

    Picks the physics/rendering backend from recognised tokens (physics defaults
    to ``"physx"``, rendering to ``"none"``) and stores ALL tokens verbatim in
    ``presets``.

    Args:
        tokens: Active preset tokens (e.g. ``["newton_mjwarp", "rgb"]``).

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.RunConfig`.
    """
    physics = "physx"
    rendering = "none"
    for t in tokens:
        if t in _PHYSICS_PRESETS:
            physics = t
        elif t in _PHYSICS_ALIASES:
            physics = _PHYSICS_ALIASES[t]
        elif t in _RENDERING_PRESETS:
            rendering = _RENDERING_PRESETS[t]
    return RunConfig(physics_backend=physics, rendering_backend=rendering, presets=list(tokens))


def capture_resources(bm: Any) -> Resources:
    """Read resource-utilisation measurements from a benchmark object.

    Reads GPU utilisation/memory, CPU utilisation, and system RAM from the
    corresponding recorders and maps them to :class:`~.schema.Resources`.

    Utilisation fields leave ``peak`` as ``None``; memory fields populate it.

    Args:
        bm: Benchmark-like object exposing ``._manual_recorders``.

    Returns:
        Populated :class:`~.schema.Resources` dataclass; never raises.
    """
    gpu_data = _get_recorder_data(bm, "GPUInfo")
    cpu_data = _get_recorder_data(bm, "CPUInfo")
    mem_data = _get_recorder_data(bm, "MemoryInfo")

    # --- GPU ---
    gpu_meas = gpu_data.measurements if gpu_data is not None else []

    gpu_util_mean = _find_value(gpu_meas, "GPU Utilization")
    gpu_util_std = _find_value(gpu_meas, "GPU Utilization std")

    gpu_mem_mean = _find_value(gpu_meas, "GPU Memory Used")
    gpu_mem_std = _find_value(gpu_meas, "GPU Memory Used std")
    _gpu_mem_peak_raw = _find_value(gpu_meas, "GPU Memory Used peak", default=0.0)
    gpu_mem_peak = max(gpu_mem_mean, _gpu_mem_peak_raw)

    # --- CPU ---
    cpu_meas = cpu_data.measurements if cpu_data is not None else []

    cpu_util_mean = _find_value(cpu_meas, "CPU Utilization")
    cpu_util_std = _find_value(cpu_meas, "CPU Utilization std")

    # --- Memory ---
    mem_meas = mem_data.measurements if mem_data is not None else []

    ram_mean = _find_value(mem_meas, "System Memory RSS")
    ram_std = _find_value(mem_meas, "System Memory RSS std")
    _ram_peak_raw = _find_value(mem_meas, "System Memory RSS peak", default=0.0)
    ram_peak = max(ram_mean, _ram_peak_raw)

    return Resources(
        gpu_util_pct=MeanStd(mean=gpu_util_mean, std=gpu_util_std, peak=None),
        gpu_mem_gb=MeanStd(mean=gpu_mem_mean, std=gpu_mem_std, peak=gpu_mem_peak),
        cpu_util_pct=MeanStd(mean=cpu_util_mean, std=cpu_util_std, peak=None),
        ram_gb=MeanStd(mean=ram_mean, std=ram_std, peak=ram_peak),
    )
