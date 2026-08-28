# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture helpers for benchmark data extraction.

Reads recorder data off a ``BaseIsaacLabBenchmark``-like object and maps the
raw measurements and metadata to the typed schema dataclasses
:class:`~isaaclab.benchmark.schema.Versions`,
:class:`~isaaclab.benchmark.schema.Hardware`, and
:class:`~isaaclab.benchmark.schema.Resources`.

Import-time dependencies stay light: no torch, isaacsim, or RL-library
imports. The benchmark object is accepted at call time; its recorder classes
are never imported here.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from typing import Any

from isaaclab.benchmark.schema import (
    GpuDeviceInfo,
    GpuResources,
    Hardware,
    MeanStd,
    Resources,
    RunConfig,
    Versions,
)


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
        physics_backend: Physics backend name.
        task: Gym task id.
        seed: Environment/agent seed.
        stamp: Timestamp string (e.g. ``"20260612-150000"``).

    Returns:
        Underscore-joined run identifier string.
    """
    fw = framework or "runtime"
    return f"{fw}_{physics_backend}_{task}_{stamp}_seed{seed}"


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

    md = {m.name: m.data for m in data.metadata or []}
    dev: dict[str, Any] = md.get("dev") or {}

    return Versions(
        isaaclab=md.get("isaaclab_version", "unknown"),
        isaacsim=md.get("isaacsim_version"),
        kit=md.get("kit_version"),
        newton=md.get("newton_version"),
        warp=md.get("warp_version"),
        mjwarp=md.get("mujoco_warp_version"),
        torch=md.get("torch_version", "unknown"),
        rsl_rl=md.get("rsl_rl_version"),
        rl_games=md.get("rl_games_version"),
        skrl=md.get("skrl_version"),
        sb3=md.get("stable_baselines3_version"),
        git_commit=dev.get("commit_hash"),
        git_branch=dev.get("branch"),
        git_dirty=dev.get("dirty", False),
        numpy=md.get("numpy_version"),
        isaaclab_newton=md.get("isaaclab_newton_version"),
        isaaclab_physx=md.get("isaaclab_physx_version"),
        isaaclab_ov=md.get("isaaclab_ov_version"),
        isaaclab_tasks=md.get("isaaclab_tasks_version"),
        isaaclab_rl=md.get("isaaclab_rl_version"),
        ovrtx=md.get("ovrtx_version"),
        ovphysx=md.get("ovphysx_version"),
        mujoco=md.get("mujoco_version"),
        cuda_bindings=md.get("cuda_bindings_version"),
        usd_core=md.get("usd_core_version"),
        usd_exchange=md.get("usd_exchange_version"),
        isaaclab_release=md.get("isaaclab_release_version"),
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
        gpu_md = {m.name: m.data for m in gpu_data.metadata or []}
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
        cpu_md = {m.name: m.data for m in cpu_data.metadata or []}
        cpu_name = cpu_md.get("cpu_name", "unknown")
        cpu_count = int(cpu_md.get("physical_cores", 0))

    # RAM
    ram_gb = 0.0
    if mem_data is not None:
        mem_md = {m.name: m.data for m in mem_data.metadata or []}
        ram_gb = float(mem_md.get("total_ram_gb", 0.0))

    return Hardware(
        hostname=socket.gethostname(),
        gpu_devices=gpu_devices,
        cpu_name=cpu_name,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
    )


def _backends_from_env_cfg(env_cfg: object) -> tuple[str | None, str | None]:
    """Return active backend names from a concrete environment configuration."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    physics_type = type(physics_cfg)
    physics_descriptor = (
        "physx"
        if physics_cfg is None
        else f"{physics_type.__module__}.{physics_type.__name__} {getattr(physics_cfg, 'class_type', '')}".lower()
    )
    physics = next(
        (
            name
            for marker, name in (
                ("ovphysx", "ovphysx"),
                ("kamino", "newton_kamino"),
                ("mjwarp", "newton_mjwarp"),
                ("physx", "physx"),
            )
            if marker in physics_descriptor
        ),
        None,
    )

    renderer_names = {"isaac_rtx": "isaacsim_rtx", "ovrtx": "ovrtx", "newton_warp": "newton"}
    rendering = None
    stack = [env_cfg]
    visited: set[int] = set()
    while stack and rendering is None:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        rendering = renderer_names.get(getattr(node, "renderer_type", None))
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            try:
                children = vars(node).values()
            except TypeError:
                continue
        stack.extend(
            child
            for child in children
            if child is not None and not isinstance(child, (str, bytes, int, float, bool, type))
        )
    return physics, rendering


def _is_camera_cfg(node: object) -> bool:
    """Return whether *node* derives from a supported camera configuration class."""
    camera_cfg_names = {"CameraCfg", "RayCasterCameraCfg"}
    return any(base.__name__ in camera_cfg_names for base in type(node).__mro__)


def _camera_resolution(node: object) -> tuple[int, int] | None:
    """Return the resolved ``(width, height)`` for one camera configuration."""
    width = getattr(node, "width", None)
    height = getattr(node, "height", None)
    if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
        pattern_cfg = getattr(node, "pattern_cfg", None)
        width = getattr(pattern_cfg, "width", None)
        height = getattr(pattern_cfg, "height", None)
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        return None
    return width, height


def camera_resolutions_from_env_cfg(env_cfg: object) -> dict[str, dict[str, int]]:
    """Collect resolved image dimensions from camera configurations in an environment config.

    The returned keys are config paths rooted at ``env`` so benchmark results identify the
    exact camera field that was resolved by Hydra. RTX camera configurations expose dimensions
    directly, while ray-cast cameras expose them through their pattern configuration.

    Args:
        env_cfg: Concrete task environment configuration after command-line overrides are applied.

    Returns:
        Camera config paths mapped to their resolved image width and height in pixels.
    """
    resolutions: dict[str, dict[str, int]] = {}
    stack: list[tuple[str, object]] = [("env", env_cfg)]
    visited: set[int] = set()

    while stack:
        path, node = stack.pop()
        if node is None or isinstance(node, (str, bytes, int, float, bool, type)) or id(node) in visited:
            continue
        visited.add(id(node))

        if _is_camera_cfg(node):
            resolution = _camera_resolution(node)
            if resolution is not None:
                width, height = resolution
                resolutions[path] = {"width": width, "height": height}
            continue

        if isinstance(node, dict):
            children = [(f"{path}.{key}", value) for key, value in node.items()]
        elif isinstance(node, (list, tuple)):
            children = [(f"{path}[{index}]", value) for index, value in enumerate(node)]
        else:
            try:
                children = [(f"{path}.{name}", value) for name, value in vars(node).items() if not name.startswith("_")]
            except TypeError:
                continue
        stack.extend(reversed(children))

    return dict(sorted(resolutions.items()))


def camera_resolution_metadata_from_env_cfg(env_cfg: object) -> list[dict[str, object]]:
    """Build workflow metadata containing resolved camera resolutions, when present.

    Args:
        env_cfg: Concrete task environment configuration after command-line overrides are applied.

    Returns:
        A workflow metadata entry for ``camera_resolutions``, or an empty list when the task has no
        configured cameras with concrete image dimensions.
    """
    resolutions = camera_resolutions_from_env_cfg(env_cfg)
    return [{"name": "camera_resolutions", "data": resolutions}] if resolutions else []


def run_config_from_env_cfg(env_cfg: object) -> RunConfig:
    """Build a :class:`~isaaclab.benchmark.RunConfig` from a concrete task config.

    Args:
        env_cfg: Concrete task environment configuration.

    Returns:
        Populated :class:`~isaaclab.benchmark.RunConfig`.

    Raises:
        ValueError: If the config does not contain a supported concrete physics backend.
    """
    physics, rendering = _backends_from_env_cfg(env_cfg)
    if physics is None:
        physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
        raise ValueError(f"Unsupported concrete physics config: {type(physics_cfg).__name__}.")

    return RunConfig(
        physics_backend=physics,
        rendering_backend=rendering or "none",
    )


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
    gpu_metadata = {m.name: m.data for m in gpu_data.metadata or []} if gpu_data is not None else {}
    # Without the recorder there is nothing to attribute to a device, so the per-device mapping
    # stays empty rather than claiming an idle device 0.
    device_count = int(gpu_metadata.get("gpu_device_count", 1)) if gpu_data is not None else 0
    current_device = int(gpu_metadata.get("gpu_current_device", 0))

    # The recorder drops the device index from the measurement names on a single-GPU host.
    devices = {
        str(index): _gpu_device_resources(gpu_meas, f"GPU {index} " if device_count > 1 else "GPU ")
        for index in range(device_count)
    }
    current = devices.get(str(current_device), _gpu_device_resources(gpu_meas, "GPU "))

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
        gpu_util_pct=current.util_pct,
        gpu_mem_gb=current.mem_gb,
        cpu_util_pct=MeanStd(mean=cpu_util_mean, std=cpu_util_std, peak=None),
        ram_gb=MeanStd(mean=ram_mean, std=ram_std, peak=ram_peak),
        devices=devices,
    )


def _gpu_device_resources(measurements: Any, prefix: str) -> GpuResources:
    """Read one GPU device's utilisation and memory from prefixed measurement names."""
    mem_mean = _find_value(measurements, f"{prefix}Memory Used")
    return GpuResources(
        util_pct=MeanStd(
            mean=_find_value(measurements, f"{prefix}Utilization"),
            std=_find_value(measurements, f"{prefix}Utilization std"),
            peak=None,
        ),
        mem_gb=MeanStd(
            mean=mem_mean,
            std=_find_value(measurements, f"{prefix}Memory Used std"),
            peak=max(mem_mean, _find_value(measurements, f"{prefix}Memory Used peak", default=0.0)),
        ),
    )
