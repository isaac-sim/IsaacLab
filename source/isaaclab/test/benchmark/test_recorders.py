# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for benchmark recorder classes."""

import pytest

from isaaclab.test.benchmark.interfaces import MeasurementData
from isaaclab.test.benchmark.recorders.record_cpu_info import CPUInfoRecorder
from isaaclab.test.benchmark.recorders.record_gpu_info import GPUInfoRecorder
from isaaclab.test.benchmark.recorders.record_memory_info import MemoryInfoRecorder
from isaaclab.test.benchmark.recorders.record_version_info import VersionInfoRecorder

# ==============================================================================
# CPUInfoRecorder Tests
# ==============================================================================


class TestCPUInfoRecorder:
    """Tests for CPUInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a CPUInfoRecorder fixture."""
        return CPUInfoRecorder()

    def test_initialization(self, recorder):
        """Test that CPUInfoRecorder initializes correctly."""
        assert recorder._cpu_hardware_info is not None
        assert recorder._cpu_runtime_info is not None
        assert recorder._mean == 0
        assert recorder._std == 0
        assert recorder._n == 0
        assert recorder._m2 == 0

    def test_get_initial_data_structure(self, recorder):
        """Test that get_initial_data returns correct structure."""
        data = recorder.get_initial_data()
        assert "cpu_metadata" in data
        assert "physical_cores" in data["cpu_metadata"]
        assert "name" in data["cpu_metadata"]

    def test_get_initial_data_values(self, recorder):
        """Test that get_initial_data returns valid values."""
        data = recorder.get_initial_data()
        assert isinstance(data["cpu_metadata"]["physical_cores"], int)
        assert data["cpu_metadata"]["physical_cores"] > 0
        assert isinstance(data["cpu_metadata"]["name"], str)

    def test_update_increments_count(self, recorder):
        """Test that update increments the sample count."""
        assert recorder._n == 0
        recorder.update()
        assert recorder._n == 1
        recorder.update()
        assert recorder._n == 2

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

    def test_runtime_data_types(self, recorder):
        """Test that runtime data has correct types."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_runtime_data()
        assert isinstance(data["cpu_utilization"]["mean"], float)
        assert isinstance(data["cpu_utilization"]["std"], float)
        assert isinstance(data["cpu_utilization"]["n"], int)

    def test_get_data_returns_measurement_data(self, recorder):
        """Test that get_data returns a MeasurementData object."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_data()
        assert isinstance(data, MeasurementData)
        assert len(data.measurements) == 3
        assert len(data.metadata) == 2

    def test_get_data_measurement_names(self, recorder):
        """Test that get_data returns measurements with correct names."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_data()
        names = [m.name for m in data.measurements]
        assert "CPU Utilization" in names
        assert "CPU Utilization std" in names
        assert "CPU Utilization n" in names

    def test_get_data_metadata_names(self, recorder):
        """Test that get_data returns metadata with correct names."""
        recorder.update()
        data = recorder.get_data()
        names = [m.name for m in data.metadata]
        assert "cpu_name" in names
        assert "physical_cores" in names


# ==============================================================================
# GPUInfoRecorder Tests
# ==============================================================================


class TestGPUInfoRecorder:
    """Tests for GPUInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a GPUInfoRecorder fixture."""
        return GPUInfoRecorder()

    def test_initialization(self, recorder):
        """Test that GPUInfoRecorder initializes correctly."""
        assert recorder._gpu_hardware_info is not None
        assert recorder._gpu_runtime_info is not None
        # These are now lists (one entry per GPU)
        assert isinstance(recorder._mem_mean, list)
        assert isinstance(recorder._mem_n, list)
        assert isinstance(recorder._util_mean, list)
        assert isinstance(recorder._util_n, list)

    def test_get_initial_data_structure(self, recorder):
        """Test that get_initial_data returns correct structure."""
        data = recorder.get_initial_data()
        assert "gpu_metadata" in data
        assert "available" in data["gpu_metadata"]

    def test_get_initial_data_with_gpu(self, recorder):
        """Test hardware info when GPU is available."""
        data = recorder.get_initial_data()
        if data["gpu_metadata"]["available"]:
            assert "devices" in data["gpu_metadata"]
            assert "device_count" in data["gpu_metadata"]
            assert "current_device" in data["gpu_metadata"]
            # Check first device has expected fields
            assert len(data["gpu_metadata"]["devices"]) > 0
            device = data["gpu_metadata"]["devices"][0]
            assert "name" in device
            assert "total_memory_gb" in device
            assert "compute_capability" in device
            assert "multi_processor_count" in device

    def test_multiple_gpu_info(self, recorder):
        """Test that all GPUs are recorded."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        device_count = data["gpu_metadata"]["device_count"]
        assert len(data["gpu_metadata"]["devices"]) == device_count
        # Each device should have an index
        for i, device in enumerate(data["gpu_metadata"]["devices"]):
            assert device["index"] == i

    def test_update_increments_count(self, recorder):
        """Test that update increments the sample count."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        assert all(n == 0 for n in recorder._mem_n)
        recorder.update()
        assert all(n == 1 for n in recorder._mem_n)
        recorder.update()
        assert all(n == 2 for n in recorder._mem_n)

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

    def test_runtime_data_types(self, recorder):
        """Test that runtime data has correct types."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        for _ in range(3):
            recorder.update()

        runtime_data = recorder.get_runtime_data()
        device_runtime = runtime_data["gpu_utilization"]["devices"][0]
        assert isinstance(device_runtime["memory_used_mean_bytes"], float)
        assert isinstance(device_runtime["memory_used_std_bytes"], float)
        assert isinstance(device_runtime["memory_n"], int)

    def test_memory_values_non_negative(self, recorder):
        """Test that memory values are non-negative."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        for _ in range(5):
            recorder.update()

        runtime_data = recorder.get_runtime_data()
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
        # GPU data includes measurements (memory and utilization stats)
        # 6 measurements per GPU: memory (mean, std, n) + utilization (mean, std, n)
        num_gpus = data["gpu_metadata"]["device_count"]
        assert len(measurement_data.measurements) == 6 * num_gpus
        # 4 metadata entries: device_count, current_device, cuda_version, gpu_devices dict
        assert len(measurement_data.metadata) == 4

    def test_get_data_metadata_names(self, recorder):
        """Test that get_data returns metadata with correct names."""
        data = recorder.get_initial_data()
        if not data["gpu_metadata"]["available"]:
            pytest.skip("GPU not available")

        recorder.update()
        measurement_data = recorder.get_data()
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

        # Check first device has expected hardware fields
        device_0 = gpu_devices["0"]
        assert "name" in device_0
        assert "total_memory_gb" in device_0
        assert "compute_capability" in device_0
        assert "multi_processor_count" in device_0


# ==============================================================================
# MemoryInfoRecorder Tests
# ==============================================================================


class TestMemoryInfoRecorder:
    """Tests for MemoryInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a MemoryInfoRecorder fixture."""
        return MemoryInfoRecorder()

    def test_initialization(self, recorder):
        """Test that MemoryInfoRecorder initializes correctly."""
        assert recorder._memory_hardware_info is not None
        assert recorder._memory_runtime_info is not None
        assert recorder._rss_mean == 0
        assert recorder._rss_n == 0
        assert recorder._vms_mean == 0
        assert recorder._vms_n == 0
        assert recorder._uss_mean == 0
        assert recorder._uss_n == 0

    def test_get_initial_data_structure(self, recorder):
        """Test that get_initial_data returns correct structure."""
        data = recorder.get_initial_data()
        assert "memory_metadata" in data
        assert "total_ram_gb" in data["memory_metadata"]

    def test_get_initial_data_values(self, recorder):
        """Test that get_initial_data returns valid values."""
        data = recorder.get_initial_data()
        assert isinstance(data["memory_metadata"]["total_ram_gb"], float)
        assert data["memory_metadata"]["total_ram_gb"] > 0

    def test_update_increments_count(self, recorder):
        """Test that update increments the sample count."""
        assert recorder._rss_n == 0
        assert recorder._vms_n == 0
        recorder.update()
        assert recorder._rss_n == 1
        assert recorder._vms_n == 1
        recorder.update()
        assert recorder._rss_n == 2
        assert recorder._vms_n == 2

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

    def test_runtime_data_types(self, recorder):
        """Test that runtime data has correct types."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_runtime_data()
        assert isinstance(data["memory_utilization"]["rss_mean"], float)
        assert isinstance(data["memory_utilization"]["rss_std"], float)
        assert isinstance(data["memory_utilization"]["rss_n"], int)
        assert isinstance(data["memory_utilization"]["vms_mean"], float)
        assert isinstance(data["memory_utilization"]["vms_std"], float)
        assert isinstance(data["memory_utilization"]["vms_n"], int)

    def test_memory_values_positive(self, recorder):
        """Test that memory values are positive."""
        for _ in range(5):
            recorder.update()

        data = recorder.get_runtime_data()
        assert data["memory_utilization"]["rss_mean"] > 0
        assert data["memory_utilization"]["vms_mean"] > 0

    def test_std_non_negative(self, recorder):
        """Test that standard deviation values are non-negative."""
        for _ in range(5):
            recorder.update()

        data = recorder.get_runtime_data()
        assert data["memory_utilization"]["rss_std"] >= 0
        assert data["memory_utilization"]["vms_std"] >= 0

    def test_get_data_returns_measurement_data(self, recorder):
        """Test that get_data returns a MeasurementData object."""
        for _ in range(3):
            recorder.update()

        data = recorder.get_data()
        assert isinstance(data, MeasurementData)
        # 6 measurements for RSS and VMS (mean, std, n for each)
        # Plus potentially 3 more for USS if available (mean, std, n)
        assert len(data.measurements) >= 6
        assert len(data.measurements) <= 9
        assert len(data.metadata) == 1

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
        # USS measurements may be present depending on platform
        # We don't assert their presence since they're platform-dependent

    def test_get_data_metadata_names(self, recorder):
        """Test that get_data returns metadata with correct names."""
        recorder.update()
        data = recorder.get_data()
        names = [m.name for m in data.metadata]
        assert "total_ram_gb" in names


# ==============================================================================
# VersionInfoRecorder Tests
# ==============================================================================


class TestVersionInfoRecorder:
    """Tests for VersionInfoRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create a VersionInfoRecorder fixture."""
        return VersionInfoRecorder()

    def test_initialization(self, recorder):
        """Test that VersionInfoRecorder initializes correctly."""
        assert recorder._version_info is not None
        assert recorder._dev_info is not None

    def test_get_initial_data_structure(self, recorder):
        """Test that get_initial_data returns correct structure."""
        data = recorder.get_initial_data()
        assert "version_metadata" in data
        assert "dev" in data

    def test_captures_core_versions(self, recorder):
        """Test that core package versions are captured."""
        data = recorder.get_initial_data()
        versions = data["version_metadata"]
        # These should always be available in the test environment
        assert "torch" in versions
        assert "numpy" in versions
        assert "isaaclab" in versions

    def test_version_values_are_strings(self, recorder):
        """Test that version values are strings."""
        data = recorder.get_initial_data()
        for version in data["version_metadata"].values():
            assert isinstance(version, str)
            assert len(version) > 0

    def test_git_info_structure(self, recorder):
        """Test that git info has expected fields when available."""
        data = recorder.get_initial_data()
        dev = data["dev"]
        # If git info is available, check structure
        if dev:
            # At least one of these should be present if git is available
            possible_keys = ["commit_hash", "commit_hash_short", "branch", "commit_date", "dirty"]
            assert any(key in dev for key in possible_keys)

    def test_commit_hash_format(self, recorder):
        """Test that commit hash has correct format when available."""
        data = recorder.get_initial_data()
        dev = data["dev"]
        if "commit_hash" in dev:
            # Full hash should be 40 hex characters
            assert len(dev["commit_hash"]) == 40
            assert all(c in "0123456789abcdef" for c in dev["commit_hash"])
        if "commit_hash_short" in dev:
            # Short hash should be 8 characters
            assert len(dev["commit_hash_short"]) == 8

    def test_update_is_noop(self, recorder):
        """Test that update doesn't change anything."""
        data_before = recorder.get_initial_data()
        recorder.update()
        data_after = recorder.get_initial_data()
        assert data_before == data_after

    def test_get_runtime_data_is_empty(self, recorder):
        """Test that runtime data is empty (versions don't change)."""
        data = recorder.get_runtime_data()
        assert data == {}

    def test_get_data_returns_measurement_data(self, recorder):
        """Test that get_data returns a MeasurementData object."""
        data = recorder.get_data()
        assert isinstance(data, MeasurementData)
        # No measurements, only metadata
        assert len(data.measurements) == 0
        # Should have metadata for versions + dev info
        assert len(data.metadata) > 0

    def test_get_data_metadata_names(self, recorder):
        """Test that get_data returns metadata with version names."""
        data = recorder.get_data()
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


# ==============================================================================
# Welford's Algorithm Verification Tests
# ==============================================================================


class TestWelfordAlgorithm:
    """Tests to verify Welford's algorithm implementation in recorders."""

    def test_memory_recorder_welford_convergence(self):
        """Test that MemoryInfoRecorder's Welford implementation produces stable results."""
        recorder = MemoryInfoRecorder()

        # Run many updates
        for _ in range(100):
            recorder.update()

        data = recorder.get_runtime_data()

        # Mean should be positive (process is using memory)
        assert data["memory_utilization"]["rss_mean"] > 0

        # Std should be defined after multiple samples
        assert data["memory_utilization"]["rss_n"] == 100

    def test_single_update_std_is_zero(self):
        """Test that std is zero after a single update (no variance with one sample)."""
        recorder = MemoryInfoRecorder()
        recorder.update()

        data = recorder.get_runtime_data()
        # With n=1, std should be 0 (or undefined, but we initialize to 0)
        assert data["memory_utilization"]["rss_std"] == 0
        assert data["memory_utilization"]["vms_std"] == 0
