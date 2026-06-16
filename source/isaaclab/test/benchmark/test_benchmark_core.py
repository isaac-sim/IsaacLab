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
            output_prefix="test",
        )
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.output_path == temp_output_dir
        assert "test_" in benchmark.output_prefix
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
        assert benchmark._manual_recorders["MemoryInfo"]._rss_n >= 1

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
        assert benchmark._frametime_recorders is None

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

        # Check benchmark_info phase has workflow metadata
        assert "benchmark_info" in data
        assert "workflow_name" in data["benchmark_info"]
        assert data["benchmark_info"]["workflow_name"] == "my_workflow"
        assert "timestamp" in data["benchmark_info"]


# ==============================================================================
# Multi-backend support Tests
# ==============================================================================


def _minimal_runtime_bundle():
    """Build a minimal but schema-valid RuntimeBundle for the schema backend."""
    from isaaclab.test.benchmark.schema import (
        GpuDeviceInfo,
        Hardware,
        MeanStd,
        Resources,
        RunConfig,
        RunIdentity,
        Runtime,
        RuntimeBundle,
        StartupTime,
        Versions,
    )

    return RuntimeBundle(
        run=RunIdentity(
            run_id="runtime_newton_mjwarp_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework=None,
            config=RunConfig(physics_backend="newton_mjwarp", rendering_backend="none"),
            task="Isaac-Ant-Direct-v0",
            seed=42,
            start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:15:10Z",
            duration_s=10.0,
            status="completed",
            num_envs=16,
        ),
        versions=Versions(
            isaaclab="4.6.8",
            isaacsim=None,
            kit=None,
            newton=None,
            warp=None,
            mjwarp=None,
            torch="2.5.1",
            rsl_rl=None,
            rl_games=None,
            skrl=None,
            sb3=None,
            git_commit=None,
            git_branch=None,
            git_dirty=False,
        ),
        hardware=Hardware(
            hostname="benchmark-host",
            gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
            cpu_name="AMD EPYC 7763",
            cpu_count=64,
            ram_gb=512.0,
        ),
        runtime=Runtime(
            startup_time_s=StartupTime(app_launch=1.0, env_creation=2.0, first_step=0.5),
            iterations_completed=1,
            total_wall_time_s=4.0,
            steps_per_iteration=24,
            iteration_time_s=MeanStd(mean=1.0, std=0.0),
            collection_fps=MeanStd(mean=100.0, std=0.0),
            total_fps=MeanStd(mean=100.0, std=0.0),
            iterations_per_s=MeanStd(mean=1.0, std=0.0),
        ),
        resources=Resources(
            gpu_util_pct=MeanStd(mean=80.0, std=5.0),
            gpu_mem_gb=MeanStd(mean=10.0, std=0.5, peak=12.0),
            cpu_util_pct=MeanStd(mean=30.0, std=4.0),
            ram_gb=MeanStd(mean=20.0, std=1.0, peak=24.0),
        ),
    )


class TestMultiBackend:
    """Tests for multi-backend support in BaseIsaacLabBenchmark."""

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

    def _backend_keys(self, benchmark) -> list[str]:
        """Return the ordered list of backend type keys stored on the benchmark."""
        return [key for key, _ in benchmark._metrics]

    def test_single_string_backend_yields_one_pair(self, temp_output_dir):
        """A single backend string normalizes to one (key, backend) pair."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        assert self._backend_keys(benchmark) == ["omniperf"]

    def test_comma_separated_backends_preserve_order(self, temp_output_dir):
        """A comma-separated string normalizes to multiple pairs in order."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="schema,omniperf",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        assert self._backend_keys(benchmark) == ["schema", "omniperf"]

    def test_duplicate_backends_are_deduped(self, temp_output_dir):
        """A repeated backend registers only once (no duplicate finalize / output file)."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="json,json",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        assert self._backend_keys(benchmark) == ["json"]

    def test_empty_backend_string_defaults_to_omniperf(self, temp_output_dir):
        """An empty/whitespace backend string falls back to the default omniperf backend."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="   ",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        assert self._backend_keys(benchmark) == ["omniperf"]

    def test_single_backend_filename_has_no_key_suffix(self, temp_output_dir):
        """A single-backend run writes <output_prefix>.json with no backend-key suffix."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="json",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms")
        )
        benchmark._finalize_impl()

        expected = os.path.join(temp_output_dir, f"{benchmark.output_prefix}.json")
        assert os.path.exists(expected)
        # No suffixed files should be produced for a single backend.
        assert not os.path.exists(os.path.join(temp_output_dir, f"{benchmark.output_prefix}_json.json"))

    def test_two_backends_write_distinct_suffixed_files(self, temp_output_dir):
        """A two-backend run writes one suffixed file per backend, both present and distinct."""
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="test",
            backend_type="schema,json",
            output_path=temp_output_dir,
            use_recorders=False,
            output_prefix="test",
        )
        benchmark.attach_bundle(_minimal_runtime_bundle())
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms")
        )
        benchmark._finalize_impl()

        schema_path = os.path.join(temp_output_dir, f"{benchmark.output_prefix}_schema.json")
        json_path = os.path.join(temp_output_dir, f"{benchmark.output_prefix}_json.json")
        assert os.path.exists(schema_path)
        assert os.path.exists(json_path)

        with open(schema_path) as f:
            schema_data = json.load(f)
        with open(json_path) as f:
            json_data = json.load(f)
        # The schema backend serializes the typed bundle; the json backend serializes flat phases.
        assert schema_data != json_data
        assert schema_data["run"]["task"] == "Isaac-Ant-Direct-v0"


# ==============================================================================
# MetricsBackend Factory Tests
# ==============================================================================


class TestMetricsBackendFactory:
    """Tests for MetricsBackend factory class."""

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

    def test_get_summary_backend(self):
        """Test getting Summary backend instance."""
        backend = backends.MetricsBackend.get_instance("summary")
        assert isinstance(backend, backends.SummaryMetrics)

    def test_summary_backend_finalize_writes_json(self, temp_output_dir):
        """Test that SummaryMetrics finalize writes JSON output (and does not raise)."""
        backend = backends.MetricsBackend.get_instance("summary")
        from isaaclab.test.benchmark.measurements import StringMetadata, TestPhase

        phase = TestPhase(phase_name="runtime")
        phase.measurements.append(SingleMeasurement(name="Test FPS", value=60.0, unit="FPS"))
        phase.metadata.append(StringMetadata(name="runtime workflow_name", data="summary_test"))
        phase.metadata.append(StringMetadata(name="runtime phase", data="runtime"))
        backend.add_metrics(phase)
        output_path = temp_output_dir
        output_filename = "summary_test"
        backend.finalize(output_path, output_filename)
        expected_path = os.path.join(output_path, f"{output_filename}.json")
        assert os.path.exists(expected_path)
        with open(expected_path) as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) >= 1
        assert any(p.get("phase_name") == "runtime" for p in data)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
