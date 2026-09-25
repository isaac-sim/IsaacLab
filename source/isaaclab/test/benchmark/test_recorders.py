# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for benchmark recorder classes."""

import builtins
import sys
import types

import pytest

from isaaclab.benchmark.interfaces import MeasurementData
from isaaclab.benchmark.recorders.record_cpu_info import CPUInfoRecorder
from isaaclab.benchmark.recorders.record_gpu_info import GPUInfoRecorder
from isaaclab.benchmark.recorders.record_memory_info import MemoryInfoRecorder
from isaaclab.benchmark.recorders.record_version_info import VersionInfoRecorder

pytestmark = pytest.mark.benchmark

# ==============================================================================
# CPUInfoRecorder Tests
# ==============================================================================


class TestCPUInfoRecorder:
    """Tests for CPUInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a CPUInfoRecorder fixture."""
        return CPUInfoRecorder()

    def test_get_runtime_data_after_updates(self, recorder):
        """Test that get_runtime_data returns stats after updates."""
        for _ in range(5):
            recorder.update()

        data = recorder.get_runtime_data()
        assert "cpu_utilization" in data
        assert "mean" in data["cpu_utilization"]
        assert "std" in data["cpu_utilization"]
        assert "n" in data["cpu_utilization"]
        assert data["cpu_utilization"]["n"] == 5
        assert isinstance(data["cpu_utilization"]["mean"], float)
        assert isinstance(data["cpu_utilization"]["std"], float)
        assert isinstance(data["cpu_utilization"]["n"], int)

    def test_get_data_measurement_names(self, recorder):
        """Test get_data before the first update and the measurement names after updates."""
        assert len(recorder.get_data().measurements) == 3
        for _ in range(3):
            recorder.update()

        data = recorder.get_data()
        names = [m.name for m in data.measurements]
        assert "CPU Utilization" in names
        assert "CPU Utilization std" in names
        assert "CPU Utilization n" in names
        assert len(data.measurements) == 3
        metadata = {m.name: m.data for m in data.metadata}
        assert len(data.metadata) == 2
        assert isinstance(metadata["physical_cores"], int) and metadata["physical_cores"] > 0
        assert isinstance(metadata["cpu_name"], str)


# ==============================================================================
# GPUInfoRecorder Tests
# ==============================================================================


class TestGPUInfoRecorder:
    """Tests for GPUInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a GPUInfoRecorder fixture."""
        return GPUInfoRecorder()

    def test_get_runtime_data_after_updates(self, recorder):
        """Test that get_runtime_data returns stats after updates."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        for _ in range(5):
            recorder.update()

        runtime_data = recorder.get_runtime_data()
        assert "gpu_utilization" in runtime_data
        assert "devices" in runtime_data["gpu_utilization"]
        # Check first device
        device_runtime = runtime_data["gpu_utilization"]["devices"][0]
        assert "memory_used_mean_bytes" in device_runtime
        assert "memory_used_std_bytes" in device_runtime
        assert "memory_n" in device_runtime
        assert device_runtime["memory_n"] == 5
        assert isinstance(device_runtime["memory_used_mean_bytes"], float)
        assert isinstance(device_runtime["memory_used_std_bytes"], float)
        assert isinstance(device_runtime["memory_n"], int)
        for device_runtime in runtime_data["gpu_utilization"]["devices"]:
            assert device_runtime["memory_used_mean_bytes"] >= 0
            assert device_runtime["memory_used_std_bytes"] >= 0

    def test_get_data_returns_measurement_data(self, recorder):
        """Test that get_data returns a MeasurementData object."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        for _ in range(3):
            recorder.update()

        measurement_data = recorder.get_data()
        assert isinstance(measurement_data, MeasurementData)
        # Assert memory peak is present and utilization peak is absent for each GPU.
        num_gpus = data["gpu_metadata"]["device_count"]
        names = {m.name for m in measurement_data.measurements}
        for i in range(num_gpus):
            prefix = f"GPU {i} " if num_gpus > 1 else "GPU "
            assert f"{prefix}Memory Used peak" in names
            assert f"{prefix}Utilization peak" not in names
        # 4 metadata entries: device_count, current_device, cuda_version, gpu_devices dict
        assert len(measurement_data.metadata) == 4
        names = [m.name for m in measurement_data.metadata]
        # Global metadata
        assert "gpu_device_count" in names
        assert "gpu_current_device" in names
        assert "cuda_version" in names
        # Per-device data in dict
        assert "gpu_devices" in names

    def test_get_data_devices_dict_structure(self, recorder):
        """Test that gpu_devices dict contains per-device data."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        for _ in range(3):
            recorder.update()

        measurement_data = recorder.get_data()
        # Find the gpu_devices metadata
        gpu_devices = None
        for m in measurement_data.metadata:
            if m.name == "gpu_devices":
                gpu_devices = m.data
                break

        assert gpu_devices is not None
        device_count = data["gpu_metadata"]["device_count"]
        assert len(gpu_devices) == device_count
        assert sorted(gpu_devices, key=int) == [str(i) for i in range(device_count)]

        # Check first device has expected hardware fields
        device_0 = gpu_devices["0"]
        assert "name" in device_0
        assert "total_memory_gb" in device_0
        assert "compute_capability" in device_0
        assert "multi_processor_count" in device_0

    def test_mem_peak_tracks_running_max(self, monkeypatch):
        """Feed the recorder a scripted memory sequence; peak must match the max."""
        import torch

        from isaaclab.benchmark.recorders.record_gpu_info import GPUInfoRecorder

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

        class _FakeProps:
            name = "FakeGPU"
            total_memory = 80 * 1024**3
            major = 9
            minor = 0
            multi_processor_count = 132

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _FakeProps())

        rec = GPUInfoRecorder()
        mem_peak_rows = [m for m in rec.get_data().measurements if "Memory" in m.name and "peak" in m.name.lower()]
        assert mem_peak_rows, "expected a GPU memory peak row before any update"
        assert mem_peak_rows[0].value == 0.0

        # Bypass nvml / nvidia-smi entirely and drive memory_allocated.
        scripted_mem = iter([10 * 1024**3, 50 * 1024**3, 30 * 1024**3])  # 10 GB, 50 GB, 30 GB
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: next(scripted_mem))
        rec._nvml_available = False
        rec._nvidia_smi_available = False

        for _ in range(3):
            rec.update()

        data = rec.get_data()
        mem_peak_rows = [m for m in data.measurements if "Memory" in m.name and "peak" in m.name.lower()]
        assert mem_peak_rows, "expected a GPU memory peak row"
        # 50 GB is the max.
        assert mem_peak_rows[0].value == 50.0, f"expected 50.0 GB peak, got {mem_peak_rows[0].value}"


# ==============================================================================
# MemoryInfoRecorder Tests
# ==============================================================================


class TestMemoryInfoRecorder:
    """Tests for MemoryInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a MemoryInfoRecorder fixture."""
        return MemoryInfoRecorder()

    def test_get_runtime_data_after_updates(self, recorder):
        """Test that get_runtime_data returns stats after updates."""
        for _ in range(5):
            recorder.update()

        data = recorder.get_runtime_data()
        assert "memory_utilization" in data
        # RSS stats
        assert "rss_mean" in data["memory_utilization"]
        assert "rss_std" in data["memory_utilization"]
        assert "rss_n" in data["memory_utilization"]
        # VMS stats
        assert "vms_mean" in data["memory_utilization"]
        assert "vms_std" in data["memory_utilization"]
        assert "vms_n" in data["memory_utilization"]
        # Check counts
        assert data["memory_utilization"]["rss_n"] == 5
        assert data["memory_utilization"]["vms_n"] == 5
        assert isinstance(data["memory_utilization"]["rss_mean"], float)
        assert isinstance(data["memory_utilization"]["rss_std"], float)
        assert isinstance(data["memory_utilization"]["rss_n"], int)
        assert isinstance(data["memory_utilization"]["vms_mean"], float)
        assert isinstance(data["memory_utilization"]["vms_std"], float)
        assert isinstance(data["memory_utilization"]["vms_n"], int)
        assert data["memory_utilization"]["rss_mean"] > 0
        assert data["memory_utilization"]["vms_mean"] > 0
        assert data["memory_utilization"]["rss_std"] >= 0
        assert data["memory_utilization"]["vms_std"] >= 0

    def test_get_data_measurement_names(self, recorder):
        """Test that get_data returns measurements with correct names."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_data()
        names = [m.name for m in data.measurements]
        # RSS measurements should always be present
        assert "System Memory RSS" in names
        assert "System Memory RSS std" in names
        assert "System Memory RSS n" in names
        # VMS measurements should always be present
        assert "System Memory VMS" in names
        assert "System Memory VMS std" in names
        assert "System Memory VMS n" in names
        # USS measurements may be present depending on platform (mean, std, peak, n)
        assert 8 <= len(data.measurements) <= 12
        metadata = {m.name: m.data for m in data.metadata}
        assert len(data.metadata) == 1
        assert isinstance(metadata["total_ram_gb"], float) and metadata["total_ram_gb"] > 0

    def test_rss_peak_tracks_running_max(self, monkeypatch):
        """Test that RSS peak tracks the running maximum and Welford stats match the samples."""
        import psutil

        from isaaclab.benchmark.recorders.record_memory_info import MemoryInfoRecorder

        # Scripted RSS sequence; peak must equal the max seen so far.
        scripted_values = [100 * 1024**3, 200 * 1024**3, 150 * 1024**3]  # bytes
        scripted_iter = iter(scripted_values)

        class _FakeMemInfo:
            def __init__(self, rss):
                self.rss = rss
                self.vms = rss  # mirror so VMS also moves
                # USS is read via memory_full_info, not memory_info; leave alone.

        def _fake_memory_info(self):  # noqa: ARG001 — bound method, self is the process
            return _FakeMemInfo(next(scripted_iter))

        monkeypatch.setattr(psutil.Process, "memory_info", _fake_memory_info)

        rec = MemoryInfoRecorder()
        rss_peak = next(m for m in rec.get_data().measurements if m.name == "System Memory RSS peak")
        assert rss_peak.value == 0.0

        rec.update()
        runtime = rec.get_runtime_data()["memory_utilization"]
        # A single sample has no spread.
        assert runtime["rss_std"] == 0
        assert runtime["vms_std"] == 0
        for _ in scripted_values[1:]:
            rec.update()

        # Mean of 100/200/150 GiB is 150 GiB; the sample standard deviation is 50 GiB.
        runtime = rec.get_runtime_data()["memory_utilization"]
        assert runtime["rss_mean"] == pytest.approx(150 * 1024**3)
        assert runtime["rss_std"] == pytest.approx(50 * 1024**3)

        data = rec.get_data()
        rss_peak = next(m for m in data.measurements if m.name == "System Memory RSS peak")
        # The recorder emits GB; input was in bytes. 200 GiB -> 200.0 after rounding.
        assert rss_peak.value == 200.0, f"expected peak=200.0 GB, got {rss_peak.value}"

        vms_peak = next(m for m in data.measurements if m.name == "System Memory VMS peak")
        assert vms_peak.value == 200.0


# ==============================================================================
# VersionInfoRecorder Tests
# ==============================================================================


class TestVersionInfoRecorder:
    """Tests for VersionInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a VersionInfoRecorder fixture."""
        return VersionInfoRecorder()

    @pytest.mark.parametrize("versions_by_distribution", [{"ovrtx": "0.3.1", "ovphysx": "0.5.9"}, {}])
    def test_captures_renderer_runtime_versions(self, monkeypatch, versions_by_distribution):
        """Renderer runtime versions are recorded when installed and null otherwise."""
        monkeypatch.setattr(
            VersionInfoRecorder,
            "_get_pkg_version",
            lambda _self, distribution: versions_by_distribution.get(distribution),
        )

        recorder = VersionInfoRecorder()
        versions = recorder.get_initial_data()["version_metadata"]
        metadata = {entry.name: entry.data for entry in recorder.get_data().metadata}
        for distribution in ("ovrtx", "ovphysx"):
            assert versions[distribution] == versions_by_distribution.get(distribution)
            assert metadata[f"{distribution}_version"] == versions_by_distribution.get(distribution)

    def test_captures_active_kit_versions(self, monkeypatch, tmp_path):
        """Test that versions are captured from an active Kit runtime."""
        isaac_path = tmp_path / "isaacsim"
        isaac_path.mkdir()
        (isaac_path / "VERSION").write_text("6.0.0-test")
        monkeypatch.setenv("ISAAC_PATH", str(isaac_path))

        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        app = types.ModuleType("omni.kit.app")
        app.get_app = lambda: types.SimpleNamespace(get_build_version=lambda: "110.1.1-test")
        omni.kit = kit
        kit.app = app
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "omni.kit", kit)
        monkeypatch.setitem(sys.modules, "omni.kit.app", app)

        versions = VersionInfoRecorder().get_initial_data()["version_metadata"]

        assert versions["kit"] == "110.1.1-test"
        assert versions["isaacsim"] == "6.0.0-test"

    def test_captures_isaacsim_version_without_isaac_path(self, monkeypatch):
        """Test that an active Kit runtime provides the Isaac Sim application version."""
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        app = types.ModuleType("omni.kit.app")
        app.get_app = lambda: types.SimpleNamespace(get_kit_version=lambda: "110.1.1-test")
        omni.kit = kit
        kit.app = app
        isaacsim = types.ModuleType("isaacsim")
        core = types.ModuleType("isaacsim.core")
        version = types.ModuleType("isaacsim.core.version")
        version.get_version = lambda: ("6.0.0", "rc.59", "6", "0", "0", "rc", "59", "main.0.test")
        isaacsim.core = core
        core.version = version
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "omni.kit", kit)
        monkeypatch.setitem(sys.modules, "omni.kit.app", app)
        monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
        monkeypatch.setitem(sys.modules, "isaacsim.core", core)
        monkeypatch.setitem(sys.modules, "isaacsim.core.version", version)
        monkeypatch.setitem(sys.modules, "carb", types.ModuleType("carb"))

        versions = VersionInfoRecorder().get_initial_data()["version_metadata"]

        assert versions["isaacsim"] == "6.0.0-rc.59+main.0.test"

    def test_records_null_kit_versions_without_active_kit(self, monkeypatch):
        """Test that Kit versions are null when no Kit runtime is active."""
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        app = types.ModuleType("omni.kit.app")
        app.get_app = lambda: None
        omni.kit = kit
        kit.app = app
        isaacsim = types.ModuleType("isaacsim")
        isaacsim.__version__ = "should-not-be-recorded"
        carb = types.ModuleType("carb")
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
        monkeypatch.setitem(sys.modules, "carb", carb)
        monkeypatch.setitem(sys.modules, "omni.kit", kit)
        monkeypatch.setitem(sys.modules, "omni.kit.app", app)

        versions = VersionInfoRecorder().get_initial_data()["version_metadata"]

        assert versions["kit"] is None
        assert versions["isaacsim"] is None

    def test_records_null_kit_versions_without_importing_kit(self, monkeypatch):
        """Test that Kitless runs do not import the Kit application module."""
        monkeypatch.delenv("ISAAC_PATH", raising=False)
        for module_name in ("omni.kit.app", "omni.kit", "omni"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

        kit_imports = []
        original_import = builtins.__import__

        def import_without_kit(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "omni.kit.app":
                kit_imports.append(name)
                raise AssertionError("Kit must not be imported for Kitless runs")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_without_kit)

        versions = VersionInfoRecorder().get_initial_data()["version_metadata"]

        assert kit_imports == []
        assert versions["kit"] is None
        assert versions["isaacsim"] is None

    def test_version_values_are_strings_or_null(self, recorder):
        """Test that version values are strings or null for runtime packages."""
        data = recorder.get_initial_data()
        nullable_versions = {"kit", "isaacsim", "ovrtx", "ovphysx"}
        for name, version in data["version_metadata"].items():
            if name in nullable_versions:
                assert version is None or isinstance(version, str)
            else:
                assert isinstance(version, str)
                assert len(version) > 0

    def test_commit_hash_format(self, recorder):
        """Test that git info carries the fields capture reads, with a well-formed commit hash."""
        data = recorder.get_initial_data()
        dev = data["dev"]
        if dev:
            assert {"commit_hash", "branch", "dirty"} <= dev.keys()
        if "commit_hash" in dev:
            # Full hash should be 40 hex characters
            assert len(dev["commit_hash"]) == 40
            assert all(c in "0123456789abcdef" for c in dev["commit_hash"])
        if "commit_hash_short" in dev:
            # Short hash should be 8 characters
            assert len(dev["commit_hash_short"]) == 8

    def test_get_data_metadata_names(self, recorder):
        """Test that get_data returns metadata with version names."""
        data = recorder.get_data()
        assert len(data.measurements) == 0
        names = [m.name for m in data.metadata]
        # Check that version suffixes are present
        assert "torch_version" in names
        assert "numpy_version" in names
        assert "isaaclab_version" in names
        # Dev info is now in a DictMetadata entry named "dev" if git info is available
        # We check if it's present (it may not be in all environments)
        if any(name == "dev" for name in names):
            # If dev metadata is present, verify it's a dict
            dev_meta = next(m for m in data.metadata if m.name == "dev")
            assert hasattr(dev_meta, "data")
            assert isinstance(dev_meta.data, dict)
