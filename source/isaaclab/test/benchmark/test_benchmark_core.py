# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BaseIsaacLabBenchmark class."""

import json
import os
import tempfile

import pytest

from isaaclab.test.benchmark import backends
from isaaclab.test.benchmark.benchmark_core import BaseIsaacLabBenchmark
from isaaclab.test.benchmark.measurements import SingleMeasurement, StringMetadata

# ==============================================================================
# BaseIsaacLabBenchmark Tests
# ==============================================================================


class TestBaseIsaacLabBenchmark:
    """Tests for BaseIsaacLabBenchmark."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture(autouse=True)
    def reset_backends(self):
        """Reset backend instances before each test."""
        backends.MetricsBackend.reset_instances()
        yield
        backends.MetricsBackend.reset_instances()

    def test_initialization(self, temp_output_dir):
        """Test benchmark initializes correctly."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.output_path == temp_output_dir
        assert "test_" in benchmark.output_prefix

    def test_initialization_creates_output_dir(self):
        """Test that initialization creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "output")
            _benchmark = BaseIsaacLabBenchmark(  # noqa: F841
                benchmark_name="test_benchmark",
                backend_type="omniperf",
                output_path=output_path,
                use_recorders=False,
            )
            assert os.path.exists(output_path)

    def test_initialization_with_recorders(self, temp_output_dir):
        """Test benchmark initializes with recorders enabled."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=True,
        )
        assert benchmark._use_recorders is True
        assert "CPUInfo" in benchmark._manual_recorders
        assert "GPUInfo" in benchmark._manual_recorders
        assert "MemoryInfo" in benchmark._manual_recorders
        assert "VersionInfo" in benchmark._manual_recorders

    def test_initialization_without_recorders(self, temp_output_dir):
        """Test benchmark initializes with recorders disabled."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
        )
        assert benchmark._use_recorders is False
        assert not hasattr(benchmark, "_manual_recorders") or benchmark._manual_recorders is None

    def test_add_measurement(self, temp_output_dir):
        """Test adding measurements to phases."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
        )
        measurement = SingleMeasurement(name="test_metric", value=42.0, unit="ms")
        benchmark.add_measurement("test_phase", measurement=measurement)
        assert "test_phase" in benchmark._phases
        assert len(benchmark._phases["test_phase"].measurements) == 1
        assert benchmark._phases["test_phase"].measurements[0].name == "test_metric"

    def test_add_multiple_measurements(self, temp_output_dir):
        """Test adding multiple measurements to a phase."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
        )
        measurements = [
            SingleMeasurement(name="metric1", value=10.0, unit="ms"),
            SingleMeasurement(name="metric2", value=20.0, unit="ms"),
        ]
        benchmark.add_measurement("test_phase", measurement=measurements)
        assert len(benchmark._phases["test_phase"].measurements) == 2

    def test_add_metadata(self, temp_output_dir):
        """Test adding metadata to phases."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
        )
        metadata = StringMetadata(name="test_key", data="test_value")
        benchmark.add_measurement("test_phase", metadata=metadata)
        assert "test_phase" in benchmark._phases
        # Phase metadata includes automatic "phase" and "workflow_name" entries plus our custom one
        assert len(benchmark._phases["test_phase"].metadata) == 3
        metadata_names = [m.name for m in benchmark._phases["test_phase"].metadata]
        assert "test_key" in metadata_names
        assert "phase" in metadata_names
        assert "workflow_name" in metadata_names

    def test_update_manual_recorders(self, temp_output_dir):
        """Test updating manual recorders."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=True,
        )
        # Should not raise
        benchmark.update_manual_recorders()
        # Check recorders were updated - CPUInfoRecorder has _n attribute
        assert benchmark._manual_recorders["CPUInfo"]._n >= 1
        assert benchmark._manual_recorders["MemoryInfo"]._proc_n >= 1

    def test_update_manual_recorders_disabled(self, temp_output_dir):
        """Test that update_manual_recorders is a no-op when recorders are disabled."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
        )
        # Should not raise
        benchmark.update_manual_recorders()

    def test_finalize_generates_output(self, temp_output_dir):
        """Test that finalize creates output file."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=True,
            output_prefix="test",
        )
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms")
        )
        benchmark.update_manual_recorders()
        benchmark._finalize_impl()

        # Check output file exists
        assert os.path.exists(benchmark.output_file_path)

    def test_finalize_output_contains_measurements(self, temp_output_dir):
        """Test that finalized output contains added measurements."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms")
        )
        benchmark._finalize_impl()

        # Read and verify output
        with open(benchmark.output_file_path) as f:
            data = json.load(f)

        # Check that runtime phase is present with our measurement
        assert "runtime" in data
        assert "execution_time" in data["runtime"]
        assert data["runtime"]["execution_time"] == 100.5

    def test_finalize_cleans_up_recorders(self, temp_output_dir):
        """Test that finalize cleans up recorders."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test_benchmark",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=True,
            output_prefix="test",
        )
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms")
        )
        benchmark.update_manual_recorders()
        benchmark._finalize_impl()

        # Recorders should be set to None
        assert benchmark._manual_recorders is None
        assert benchmark._automatic_recorders is None

    def test_workflow_metadata_in_output(self, temp_output_dir):
        """Test that workflow name and timestamp metadata are in output."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="my_workflow",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        benchmark._finalize_impl()

        with open(benchmark.output_file_path) as f:
            data = json.load(f)

        # Check benchmark phase has workflow metadata
        assert "benchmark" in data
        assert "workflow_name" in data["benchmark"]
        assert data["benchmark"]["workflow_name"] == "my_workflow"
        assert "timestamp" in data["benchmark"]


# ==============================================================================
# MetricsBackend Factory Tests
# ==============================================================================


class TestMetricsBackendFactory:
    """Tests for MetricsBackend factory class."""

    @pytest.fixture(autouse=True)
    def reset_backends(self):
        """Reset backend instances before each test."""
        backends.MetricsBackend.reset_instances()
        yield
        backends.MetricsBackend.reset_instances()

    def test_get_json_backend(self):
        """Test getting JSON backend instance."""
        backend = backends.MetricsBackend.get_instance("json")
        assert isinstance(backend, backends.JSONFileMetrics)

    def test_get_osmo_backend(self):
        """Test getting Osmo backend instance."""
        backend = backends.MetricsBackend.get_instance("osmo")
        assert isinstance(backend, backends.OsmoKPIFile)

    def test_get_omniperf_backend(self):
        """Test getting OmniPerf backend instance."""
        backend = backends.MetricsBackend.get_instance("omniperf")
        assert isinstance(backend, backends.OmniPerfKPIFile)

    def test_invalid_backend_type_raises_error(self):
        """Test that invalid backend type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            backends.MetricsBackend.get_instance("invalid_type")

    def test_backend_instance_is_cached(self):
        """Test that backend instances are cached and reused."""
        backend1 = backends.MetricsBackend.get_instance("omniperf")
        backend2 = backends.MetricsBackend.get_instance("omniperf")
        assert backend1 is backend2

    def test_reset_instances(self):
        """Test that reset_instances clears the cache."""
        backend1 = backends.MetricsBackend.get_instance("omniperf")
        backends.MetricsBackend.reset_instances()
        backend2 = backends.MetricsBackend.get_instance("omniperf")
        assert backend1 is not backend2
