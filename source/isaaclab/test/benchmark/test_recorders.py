# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark recorders."""

import builtins
import statistics
import sys
import types

import psutil
import pytest
import torch

from isaaclab.benchmark.interfaces import MeasurementData
from isaaclab.benchmark.recorders import CPUInfoRecorder, GPUInfoRecorder, MemoryInfoRecorder, VersionInfoRecorder
from isaaclab.benchmark.recorders._stats import RunningStats, bytes_to_gb

pytestmark = pytest.mark.benchmark

GIB = 1024**3


def _values(data: MeasurementData) -> dict[str, object]:
    return {m.name: m.value for m in data.measurements}


def test_running_stats_match_reference():
    stats = RunningStats()
    assert (stats.mean, stats.std, stats.peak, stats.n) == (0.0, 0.0, 0.0, 0)

    samples = [3.0, 1.0, 4.0, 1.0, 5.0]
    for index, value in enumerate(samples, start=1):
        stats.update(value)
        assert stats.n == index
        assert stats.mean == pytest.approx(statistics.mean(samples[:index]))
        assert stats.std == pytest.approx(statistics.stdev(samples[:index]) if index > 1 else 0.0)
        assert stats.peak == max(samples[:index])
    assert bytes_to_gb(1.5 * GIB) == 1.5


def test_cpu_recorder_reports_process_utilization():
    recorder = CPUInfoRecorder()
    metadata = recorder.get_initial_data()["cpu_metadata"]
    assert metadata["physical_cores"] > 0 and isinstance(metadata["name"], str)
    assert recorder.get_runtime_data() == {"cpu_utilization": {}}

    for _ in range(3):
        recorder.update()

    utilization = recorder.get_runtime_data()["cpu_utilization"]
    assert utilization["n"] == 3 and utilization["mean"] >= 0.0 and utilization["std"] >= 0.0
    data = recorder.get_data()
    assert set(_values(data)) == {"CPU Utilization", "CPU Utilization std", "CPU Utilization n"}
    assert {m.name for m in data.metadata} == {"cpu_name", "physical_cores"}


def test_memory_recorder_reports_process_memory():
    recorder = MemoryInfoRecorder()
    assert recorder.get_initial_data()["memory_metadata"]["total_ram_gb"] > 0
    assert _values(recorder.get_data())["System Memory RSS peak"] == 0.0

    for _ in range(3):
        recorder.update()

    memory = recorder.get_runtime_data()["memory_utilization"]
    assert memory["rss_n"] == memory["vms_n"] == 3
    assert memory["rss_mean"] > 0 and memory["rss_std"] >= 0 and memory["rss_peak"] >= memory["rss_mean"]
    data = recorder.get_data()
    assert {"System Memory RSS", "System Memory RSS std", "System Memory RSS peak", "System Memory VMS n"} <= set(
        _values(data)
    )
    assert [m.name for m in data.metadata] == ["total_ram_gb"]


def test_memory_recorder_tracks_peak(monkeypatch):
    scripted = iter([300 * GIB, 50 * GIB, 200 * GIB])
    monkeypatch.setattr(
        psutil.Process, "memory_info", lambda self: types.SimpleNamespace(rss=(value := next(scripted)), vms=value)
    )

    recorder = MemoryInfoRecorder()
    for _ in range(3):
        recorder.update()

    values = _values(recorder.get_data())
    assert values["System Memory RSS peak"] == values["System Memory VMS peak"] == 300.0
    assert values["System Memory RSS"] == pytest.approx(bytes_to_gb(statistics.mean([300, 50, 200]) * GIB))


@pytest.fixture
def fake_cuda(monkeypatch):
    """Present one fake CUDA device so GPU recorder tests run without hardware or NVML."""
    props = types.SimpleNamespace(name="FakeGPU", total_memory=80 * GIB, major=9, minor=0, multi_processor_count=132)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr("isaaclab.benchmark.recorders.record_gpu_info.subprocess.run", lambda *a, **k: 1 / 0)


def test_gpu_recorder_tracks_memory_with_torch_fallback(fake_cuda, monkeypatch):
    recorder = GPUInfoRecorder()
    metadata = recorder.get_initial_data()["gpu_metadata"]
    assert metadata["available"] and metadata["device_count"] == 1
    assert metadata["devices"][0]["name"] == "FakeGPU"
    assert recorder.get_runtime_data() == {"gpu_utilization": {}}
    assert _values(recorder.get_data())["GPU Memory Used peak"] == 0.0

    scripted = iter([10 * GIB, 50 * GIB, 30 * GIB])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: next(scripted))
    for _ in range(3):
        recorder.update()

    device = recorder.get_runtime_data()["gpu_utilization"]["devices"][0]
    assert device["memory_n"] == 3 and device["memory_used_peak_bytes"] == 50 * GIB
    assert "utilization_mean_percent" not in device
    data = recorder.get_data()
    values = _values(data)
    assert values["GPU Memory Used peak"] == 50.0 and values["GPU Memory Used"] == 30.0
    assert "GPU Utilization" not in values and "GPU Utilization peak" not in values
    assert {m.name: m.data for m in data.metadata}["gpu_devices"] == {
        "0": {"name": "FakeGPU", "total_memory_gb": 80.0, "compute_capability": "9.0", "multi_processor_count": 132}
    }


def test_gpu_recorder_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    recorder = GPUInfoRecorder()
    recorder.update()

    assert recorder.get_initial_data() == {"gpu_metadata": {"available": False}}
    assert recorder.get_runtime_data() == {"gpu_utilization": {}}
    assert recorder.get_data() == MeasurementData()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gpu_recorder_samples_every_device():
    recorder = GPUInfoRecorder()
    recorder.update()
    recorder.update()

    device_count = recorder.get_initial_data()["gpu_metadata"]["device_count"]
    devices = recorder.get_runtime_data()["gpu_utilization"]["devices"]
    assert len(devices) == device_count
    assert all(device["memory_n"] == 2 and device["memory_used_mean_bytes"] >= 0 for device in devices)
    names = set(_values(recorder.get_data()))
    prefix = "GPU " if device_count == 1 else "GPU 0 "
    assert {f"{prefix}Memory Used", f"{prefix}Memory Used peak"} <= names


def _install_kit(monkeypatch, app) -> None:
    omni, kit, kit_app = types.ModuleType("omni"), types.ModuleType("omni.kit"), types.ModuleType("omni.kit.app")
    kit_app.get_app = lambda: app
    omni.kit, kit.app = kit, kit_app
    for name, module in (
        ("omni", omni),
        ("omni.kit", kit),
        ("omni.kit.app", kit_app),
        ("carb", types.ModuleType("carb")),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_version_recorder_captures_environment():
    recorder = VersionInfoRecorder()
    versions = recorder.get_initial_data()["version_metadata"]
    assert all(isinstance(versions[name], str) and versions[name] for name in ("torch", "numpy", "isaaclab"))
    for name, version in versions.items():
        assert version is None or isinstance(version, str)
        assert version is None and name in {"kit", "isaacsim", "ovrtx", "ovphysx"} or version
    assert recorder.get_runtime_data() == {}

    data = recorder.get_data()
    metadata = {m.name: m.data for m in data.metadata}
    assert data.measurements == []
    assert {"torch_version", "numpy_version", "isaaclab_version"} <= set(metadata)
    dev = metadata.get("dev", {})
    if "commit_hash" in dev:
        assert len(dev["commit_hash"]) == 40 and dev["commit_hash_short"] == dev["commit_hash"][:8]
        assert isinstance(dev["dirty"], bool)


def test_version_recorder_reads_kit_versions_from_install(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("6.0.0-test")
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    _install_kit(monkeypatch, types.SimpleNamespace(get_build_version=lambda: "110.1.1-test"))

    versions = VersionInfoRecorder().get_initial_data()["version_metadata"]

    assert versions["kit"] == "110.1.1-test"
    assert versions["isaacsim"] == "6.0.0-test"


def test_version_recorder_reads_isaacsim_version_from_runtime(monkeypatch):
    monkeypatch.delenv("ISAAC_PATH", raising=False)
    _install_kit(monkeypatch, types.SimpleNamespace(get_kit_version=lambda: "110.1.1-test"))
    version = types.ModuleType("isaacsim.core.version")
    version.get_version = lambda: ("6.0.0", "rc.59", "6", "0", "0", "rc", "59", "main.0.test")
    monkeypatch.setitem(sys.modules, "isaacsim", types.ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", types.ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.version", version)

    assert VersionInfoRecorder().get_initial_data()["version_metadata"]["isaacsim"] == "6.0.0-rc.59+main.0.test"


def test_version_recorder_records_null_kit_versions_without_kit(monkeypatch):
    monkeypatch.delenv("ISAAC_PATH", raising=False)
    for module_name in ("omni.kit.app", "omni.kit", "omni"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def import_without_kit(name, *args, **kwargs):
        assert name != "omni.kit.app", "Kit must not be imported for Kitless runs"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_kit)

    recorder = VersionInfoRecorder()
    versions = recorder.get_initial_data()["version_metadata"]
    metadata = {m.name: m.data for m in recorder.get_data().metadata}

    assert versions["kit"] is None and versions["isaacsim"] is None
    assert metadata["kit_version"] is None and metadata["isaacsim_version"] is None


def test_version_recorder_records_optional_runtime_versions(monkeypatch):
    available = {"ovrtx": "0.3.1"}
    monkeypatch.setattr(VersionInfoRecorder, "_get_pkg_version", lambda self, name: available.get(name))

    recorder = VersionInfoRecorder()
    versions = recorder.get_initial_data()["version_metadata"]
    metadata = {m.name: m.data for m in recorder.get_data().metadata}

    assert versions["ovrtx"] == metadata["ovrtx_version"] == "0.3.1"
    assert versions["ovphysx"] is None and metadata["ovphysx_version"] is None
    assert "newton" not in versions
