# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark capture helpers, driven by recorder-shaped fakes."""

from types import SimpleNamespace

import pytest

from isaaclab.benchmark.capture import (
    capture_hardware,
    capture_resources,
    capture_versions,
    run_config_from_env_cfg,
    synth_run_id,
)
from isaaclab.benchmark.interfaces import MeasurementData
from isaaclab.benchmark.measurements import DictMetadata, FloatMetadata, IntMetadata, SingleMeasurement, StringMetadata

pytestmark = pytest.mark.benchmark


def _recorder(values: dict[str, float] | None = None, metadata: list | None = None) -> SimpleNamespace:
    measurements = [SingleMeasurement(name=name, value=value, unit="") for name, value in (values or {}).items()]
    data = MeasurementData(measurements=measurements, metadata=metadata or [])
    return SimpleNamespace(get_data=lambda: data)


def _benchmark(**recorders) -> SimpleNamespace:
    return SimpleNamespace(_manual_recorders=recorders or None)


def test_capture_versions_maps_recorder_metadata():
    metadata = [
        StringMetadata(name="isaaclab_version", data="4.6.8"),
        StringMetadata(name="torch_version", data="2.5.1"),
        StringMetadata(name="mujoco_warp_version", data="0.0.4"),
        StringMetadata(name="stable_baselines3_version", data="2.3.0"),
        StringMetadata(name="isaaclab_newton_version", data="1.0.2"),
        StringMetadata(name="ovrtx_version", data=None),
        StringMetadata(name="usd_core_version", data="25.11"),
        StringMetadata(name="isaaclab_release_version", data="3.0.0"),
        DictMetadata(name="dev", data={"commit_hash": "abc123", "branch": "develop", "dirty": True}),
    ]

    versions = capture_versions(_benchmark(VersionInfo=_recorder(metadata=metadata)))

    assert (versions.isaaclab, versions.torch, versions.mjwarp, versions.sb3) == ("4.6.8", "2.5.1", "0.0.4", "2.3.0")
    assert (versions.isaaclab_newton, versions.usd_core, versions.isaaclab_release) == ("1.0.2", "25.11", "3.0.0")
    assert (versions.git_commit, versions.git_branch, versions.git_dirty) == ("abc123", "develop", True)
    assert versions.isaacsim is None and versions.ovrtx is None


def test_capture_hardware_orders_devices_numerically():
    devices = {
        str(index): {"name": f"H100-{index}", "total_memory_gb": 80.0, "compute_capability": "9.0"}
        for index in (10, 2, 0)
    }
    hardware = capture_hardware(
        _benchmark(
            GPUInfo=_recorder(metadata=[DictMetadata(name="gpu_devices", data=devices)]),
            CPUInfo=_recorder(
                metadata=[StringMetadata(name="cpu_name", data="EPYC"), IntMetadata(name="physical_cores", data=64)]
            ),
            MemoryInfo=_recorder(metadata=[FloatMetadata(name="total_ram_gb", data=512.0)]),
        )
    )

    assert [device.name for device in hardware.gpu_devices] == ["H100-0", "H100-2", "H100-10"]
    assert (hardware.gpu_devices[0].mem_gb, hardware.gpu_devices[0].compute_cap) == (80.0, "9.0")
    assert (hardware.cpu_name, hardware.cpu_count, hardware.ram_gb) == ("EPYC", 64, 512.0)
    assert hardware.hostname


def test_capture_resources_reads_peaks_and_omits_utilization_peaks():
    resources = capture_resources(
        _benchmark(
            GPUInfo=_recorder(
                {
                    "GPU Utilization": 80.0,
                    "GPU Utilization std": 5.0,
                    "GPU Memory Used": 10.0,
                    "GPU Memory Used std": 0.5,
                    "GPU Memory Used peak": 12.0,
                }
            ),
            CPUInfo=_recorder({"CPU Utilization": 30.0, "CPU Utilization std": 4.0}),
            MemoryInfo=_recorder(
                {"System Memory RSS": 20.0, "System Memory RSS std": 1.0, "System Memory RSS peak": 24.0}
            ),
        )
    )

    assert (resources.gpu_util_pct.mean, resources.gpu_util_pct.peak) == (80.0, None)
    assert (resources.gpu_mem_gb.mean, resources.gpu_mem_gb.peak) == (10.0, 12.0)
    assert (resources.cpu_util_pct.std, resources.cpu_util_pct.peak) == (4.0, None)
    assert (resources.ram_gb.mean, resources.ram_gb.peak) == (20.0, 24.0)
    assert list(resources.devices) == ["0"]


def test_capture_resources_clamps_missing_peaks_to_mean():
    """Missing peak rows must not produce a MeanStd whose peak is below its mean."""
    resources = capture_resources(
        _benchmark(
            GPUInfo=_recorder({"GPU Memory Used": 10.0, "GPU Memory Used std": 0.5}),
            MemoryInfo=_recorder({"System Memory RSS": 10.0, "System Memory RSS std": 0.2}),
        )
    )

    assert resources.gpu_mem_gb.peak == pytest.approx(10.0)
    assert resources.ram_gb.peak == pytest.approx(10.0)


def test_capture_resources_uses_current_gpu_for_multiple_devices():
    resources = capture_resources(
        _benchmark(
            GPUInfo=_recorder(
                {"GPU 1 Utilization": 80.0, "GPU 1 Memory Used": 10.0, "GPU 1 Memory Used peak": 12.0},
                metadata=[IntMetadata(name="gpu_device_count", data=2), IntMetadata(name="gpu_current_device", data=1)],
            )
        )
    )

    assert resources.gpu_util_pct.mean == pytest.approx(80.0)
    assert resources.gpu_mem_gb.peak == pytest.approx(12.0)
    assert set(resources.devices) == {"0", "1"}
    assert resources.devices["0"].mem_gb.mean == 0.0


def test_capture_without_recorders_returns_defaults():
    benchmark = _benchmark()

    assert capture_versions(benchmark).isaaclab == "unknown"
    assert capture_hardware(benchmark).gpu_devices == []
    assert capture_resources(benchmark).devices == {}


def test_synth_run_id():
    assert synth_run_id("rsl_rl", "physx", "Isaac-Ant-Direct-v0", 42, "20260612-150000") == (
        "rsl_rl_physx_Isaac-Ant-Direct-v0_20260612-150000_seed42"
    )
    assert synth_run_id(None, "physx", "task", 0, "stamp").startswith("runtime_")


def test_run_config_uses_concrete_backend_configuration():
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=SimpleNamespace(class_type="isaaclab_newton.physics:NewtonMJWarpManager")),
        camera=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx")),
    )
    cfg = run_config_from_env_cfg(env_cfg)
    assert (cfg.physics_backend, cfg.rendering_backend, cfg.presets) == ("newton_mjwarp", "isaacsim_rtx", [])

    physx_env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(class_type="PhysXManager")))
    assert run_config_from_env_cfg(physx_env_cfg).physics_backend == "physx"
    assert run_config_from_env_cfg(SimpleNamespace(sim=SimpleNamespace(physics=None))).physics_backend == "physx"

    with pytest.raises(ValueError, match="Unsupported concrete physics config"):
        run_config_from_env_cfg(SimpleNamespace(sim=SimpleNamespace(physics=object())))
