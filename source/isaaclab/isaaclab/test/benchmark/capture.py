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

Import-time dependencies stay light: no torch, isaacsim, or RL-library
imports. Preset metadata is imported lazily when run_config_from_presets()
needs it. The benchmark object is accepted at call time; its recorder classes
are never imported here.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, get_args

from isaaclab.test.benchmark.schema import (
    GpuDeviceInfo,
    Hardware,
    MeanStd,
    PhysicsBackend,
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
        physics_backend: Physics backend preset string.
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


def _preset_target_metadata() -> tuple[str, str, str, dict[str, str]]:
    """Return preset selector labels and physics aliases from the preset CLI layer."""
    from isaaclab_tasks.utils.preset_target import PresetTarget

    return (
        PresetTarget.PHYSICS.value,
        PresetTarget.RENDERER.value,
        PresetTarget.DOMAIN.value,
        dict(PresetTarget.PHYSICS.legacy_aliases),
    )


def _physics_backend_names() -> set[str]:
    """Return physics backend names accepted by the benchmark schema."""
    return set(get_args(PhysicsBackend))


def _rendering_backend_by_preset() -> dict[str, str]:
    """Return renderer preset names mapped to benchmark rendering backend names."""
    try:
        from isaaclab_tasks.utils.hydra import _preset_fields
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg
    except ImportError:
        return {}

    return {
        name: name.removesuffix("_renderer") for name in _preset_fields(MultiBackendRendererCfg()) if name != "default"
    }


def _backend_defaults_from_env_cfg(env_cfg: object) -> tuple[str | None, str | None]:
    """Return active backend names from a resolved environment configuration."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    physics_descriptor = (
        f"{type(physics_cfg).__module__}.{type(physics_cfg).__name__} {getattr(physics_cfg, 'class_type', '')}"
    ).lower()
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


def _expand_preset_tokens(tokens: Sequence[str]) -> list[tuple[str | None, str]]:
    """Expand Hydra-style preset tokens into ``(selector, value)`` pairs."""
    physics_label, renderer_label, domain_label, _ = _preset_target_metadata()
    expanded: list[tuple[str | None, str]] = []
    for token in tokens:
        selector, has_value, raw_value = token.partition("=")
        if not has_value:
            value = token.strip()
            if value:
                expanded.append((None, value))
            continue

        selector = selector.strip()
        if selector == domain_label:
            expanded.extend((selector, value) for value in (v.strip() for v in raw_value.split(",")) if value)
        elif selector in (physics_label, renderer_label):
            value = raw_value.strip()
            if value:
                expanded.append((selector, value))
        else:
            value = token.strip()
            if value:
                expanded.append((None, value))
    return expanded


def run_config_from_presets(tokens: Sequence[str], *, env_cfg: object | None = None) -> RunConfig:
    """Build a :class:`~isaaclab.test.benchmark.RunConfig` from presets and resolved task config.

    Picks backend defaults from ``env_cfg`` when provided, then applies recognised
    preset tokens. Without a resolved config, physics defaults to ``"physx"`` and
    rendering to ``"none"``. Accepts bare preset names as well as Hydra-style
    ``physics=...``, ``renderer=...``, and ``presets=...`` tokens.

    Args:
        tokens: Active preset tokens (e.g. ``["newton_mjwarp", "rgb"]``).
        env_cfg: Optional resolved task environment configuration. Its active physics
            and renderer configurations take precedence over token inference.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.RunConfig`.
    """
    if not tokens and env_cfg is None:
        return RunConfig(physics_backend="physx", rendering_backend="none", presets=[])

    physics = "physx"
    rendering = "none"
    expanded_tokens = _expand_preset_tokens(tokens)
    physics_label, renderer_label, _, physics_aliases = _preset_target_metadata()
    physics_backends = _physics_backend_names()
    rendering_backends: dict[str, str] | None = None

    def rendering_backend_for(preset_name: str) -> str | None:
        nonlocal rendering_backends
        if rendering_backends is None:
            rendering_backends = _rendering_backend_by_preset()
        if preset_name in rendering_backends:
            return rendering_backends[preset_name]
        if preset_name.endswith("_renderer"):
            return preset_name.removesuffix("_renderer")
        return None

    for selector, token in expanded_tokens:
        if selector == physics_label:
            physics = physics_aliases.get(token, token)
        elif selector == renderer_label:
            rendering = rendering_backend_for(token) or token
        else:
            canonical = physics_aliases.get(token, token)
            if canonical in physics_backends:
                physics = canonical
            elif token.endswith("_renderer"):
                renderer = rendering_backend_for(token)
                if renderer is not None:
                    rendering = renderer

    if env_cfg is not None:
        active_physics, active_rendering = _backend_defaults_from_env_cfg(env_cfg)
        physics = active_physics or physics
        rendering = active_rendering or rendering

    return RunConfig(
        physics_backend=physics,
        rendering_backend=rendering,
        presets=[token for _, token in expanded_tokens],
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
    gpu_prefix = (
        f"GPU {gpu_metadata.get('gpu_current_device', 0)} " if gpu_metadata.get("gpu_device_count", 1) > 1 else "GPU "
    )

    gpu_util_mean = _find_value(gpu_meas, f"{gpu_prefix}Utilization")
    gpu_util_std = _find_value(gpu_meas, f"{gpu_prefix}Utilization std")

    gpu_mem_mean = _find_value(gpu_meas, f"{gpu_prefix}Memory Used")
    gpu_mem_std = _find_value(gpu_meas, f"{gpu_prefix}Memory Used std")
    _gpu_mem_peak_raw = _find_value(gpu_meas, f"{gpu_prefix}Memory Used peak", default=0.0)
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
