# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for metrics backends."""

import json
import logging
import os

import pytest

from isaaclab.test.benchmark import backends
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


def _minimal_runtime_bundle() -> RuntimeBundle:
    """Build a minimal but schema-valid RuntimeBundle for serialization tests."""
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


class TestSchemaBundleFile:
    """Tests for the schema bundle metrics backend."""

    @pytest.fixture(autouse=True)
    def reset_backends(self):
        """Reset backend instances before and after each test."""
        backends.MetricsBackend.reset_instances()
        yield
        backends.MetricsBackend.reset_instances()

    def test_schema_backend_is_registered(self):
        """The factory resolves "schema" to a SchemaBundleFile instance."""
        backend = backends.MetricsBackend.get_instance("schema")
        assert isinstance(backend, backends.SchemaBundleFile)

    def test_finalize_with_bundle_writes_json(self, tmp_path):
        """finalize with a bundle attached writes valid JSON with the bundle's fields."""
        backend = backends.SchemaBundleFile()
        bundle = _minimal_runtime_bundle()
        backend.finalize(str(tmp_path), "runtime", bundle=bundle)

        expected_path = os.path.join(str(tmp_path), "runtime.json")
        assert os.path.exists(expected_path)
        with open(expected_path) as f:
            data = json.load(f)
        assert data["run"]["task"] == "Isaac-Ant-Direct-v0"
        assert data["run"]["framework"] is None
        assert data["runtime"]["total_fps"]["mean"] == pytest.approx(100.0)
        assert data["resources"]["gpu_mem_gb"]["peak"] == pytest.approx(12.0)
        assert "schema_version" in data

    def test_finalize_without_bundle_writes_nothing_and_warns(self, tmp_path, caplog):
        """finalize without a bundle writes no file and logs a warning."""
        backend = backends.SchemaBundleFile()
        with caplog.at_level(logging.WARNING, logger="isaaclab.test.benchmark.backends"):
            backend.finalize(str(tmp_path), "runtime", bundle=None)

        assert not os.path.exists(os.path.join(str(tmp_path), "runtime.json"))
        assert any("no bundle" in record.message.lower() for record in caplog.records)

    def test_add_metrics_is_noop(self, tmp_path):
        """add_metrics ignores accumulated phases; only the attached bundle is serialized."""
        from isaaclab.test.benchmark.measurements import SingleMeasurement, TestPhase

        backend = backends.SchemaBundleFile()
        phase = TestPhase(phase_name="runtime")
        phase.measurements.append(SingleMeasurement(name="Test FPS", value=60.0, unit="FPS"))
        # add_metrics must not raise and must not influence the serialized output.
        backend.add_metrics(phase)

        backend.finalize(str(tmp_path), "runtime", bundle=_minimal_runtime_bundle())
        with open(os.path.join(str(tmp_path), "runtime.json")) as f:
            data = json.load(f)
        # The flat measurement phase is not present; only the typed bundle is serialized.
        assert "Test FPS" not in json.dumps(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
