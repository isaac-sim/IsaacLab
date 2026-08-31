# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark capture helpers (Isaac-Sim-free, fake recorders)."""

from types import SimpleNamespace

import pytest

from isaaclab.benchmark.capture import (
    camera_resolution_metadata_from_env_cfg,
    camera_resolutions_from_env_cfg,
    capture_hardware,
    capture_resources,
    capture_versions,
    run_config_from_env_cfg,
    synth_run_id,
)
from isaaclab.benchmark.interfaces import MeasurementData
from isaaclab.benchmark.measurements import (
    DictMetadata,
    FloatMetadata,
    IntMetadata,
    SingleMeasurement,
    StringMetadata,
)
from isaaclab.benchmark.schema import Hardware, Resources, Versions
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, patterns


class _Rec:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class _Bm:
    def __init__(self, recorders):
        self._manual_recorders = recorders


def _camera_cfg(width: int = 64, height: int = 48) -> CameraCfg:
    return CameraCfg(prim_path="/World/Camera", spawn=None, width=width, height=height)


def _ray_caster_camera_cfg(width: int = 32, height: int = 24) -> RayCasterCameraCfg:
    return RayCasterCameraCfg(
        prim_path="/World/RayCamera",
        mesh_prim_paths=["/World/Ground"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(width=width, height=height),
    )


def test_capture_versions_renames_and_defaults():
    md = [
        StringMetadata(name="isaaclab_version", data="4.6.8"),
        StringMetadata(name="torch_version", data="2.5.1"),
        StringMetadata(name="mujoco_warp_version", data="0.0.4"),
        StringMetadata(name="stable_baselines3_version", data="2.3.0"),
        DictMetadata(name="dev", data={"commit_hash": "abc123", "branch": "develop", "dirty": True}),
    ]
    bm = _Bm({"VersionInfo": _Rec(MeasurementData(measurements=[], metadata=md, artefacts=[]))})
    v = capture_versions(bm)
    assert isinstance(v, Versions)
    assert v.isaaclab == "4.6.8" and v.torch == "2.5.1"
    assert v.mjwarp == "0.0.4"
    assert v.sb3 == "2.3.0"
    assert v.git_commit == "abc123" and v.git_branch == "develop" and v.git_dirty is True
    assert v.isaacsim is None


def test_capture_versions_preserves_runtime_packages():
    md = [
        StringMetadata(name="numpy_version", data="2.4.4"),
        StringMetadata(name="isaaclab_newton_version", data="1.0.2"),
        StringMetadata(name="isaaclab_physx_version", data="2.0.1"),
        StringMetadata(name="isaaclab_ov_version", data="0.4.6"),
        StringMetadata(name="isaaclab_tasks_version", data="8.0.1"),
        StringMetadata(name="isaaclab_rl_version", data="0.6.1"),
        StringMetadata(name="ovrtx_version", data=None),
        StringMetadata(name="ovphysx_version", data="3.0.5"),
        StringMetadata(name="mujoco_version", data="3.8.1"),
        StringMetadata(name="cuda_bindings_version", data="12.9.4"),
        StringMetadata(name="usd_core_version", data="25.11"),
        StringMetadata(name="isaaclab_release_version", data="3.0.0"),
    ]
    bm = _Bm({"VersionInfo": _Rec(MeasurementData(measurements=[], metadata=md, artefacts=[]))})

    versions = capture_versions(bm)

    assert versions.numpy == "2.4.4"
    assert versions.isaaclab_newton == "1.0.2"
    assert versions.isaaclab_physx == "2.0.1"
    assert versions.isaaclab_ov == "0.4.6"
    assert versions.isaaclab_tasks == "8.0.1"
    assert versions.isaaclab_rl == "0.6.1"
    assert versions.ovrtx is None
    assert versions.ovphysx == "3.0.5"
    assert versions.mujoco == "3.8.1"
    assert versions.cuda_bindings == "12.9.4"
    assert versions.usd_core == "25.11"
    assert versions.isaaclab_release == "3.0.0"


def test_capture_resources_peaks():
    gpu = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="GPU Utilization", value=80.0, unit="%"),
                SingleMeasurement(name="GPU Utilization std", value=5.0, unit="%"),
                SingleMeasurement(name="GPU Memory Used", value=10.0, unit="GB"),
                SingleMeasurement(name="GPU Memory Used std", value=0.5, unit="GB"),
                SingleMeasurement(name="GPU Memory Used peak", value=12.0, unit="GB"),
            ],
            metadata=[],
            artefacts=[],
        )
    )
    cpu = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="CPU Utilization", value=30.0, unit="%"),
                SingleMeasurement(name="CPU Utilization std", value=4.0, unit="%"),
            ],
            metadata=[],
            artefacts=[],
        )
    )
    mem = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="System Memory RSS", value=20.0, unit="GB"),
                SingleMeasurement(name="System Memory RSS std", value=1.0, unit="GB"),
                SingleMeasurement(name="System Memory RSS peak", value=24.0, unit="GB"),
            ],
            metadata=[],
            artefacts=[],
        )
    )
    r = capture_resources(_Bm({"GPUInfo": gpu, "CPUInfo": cpu, "MemoryInfo": mem}))
    assert isinstance(r, Resources)
    assert r.gpu_util_pct.peak is None
    assert r.gpu_mem_gb.peak == pytest.approx(12.0)
    assert r.ram_gb.peak == pytest.approx(24.0)
    assert r.cpu_util_pct.peak is None


def test_capture_resources_uses_current_gpu_for_multiple_devices():
    gpu = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="GPU 1 Utilization", value=80.0, unit="%"),
                SingleMeasurement(name="GPU 1 Utilization std", value=5.0, unit="%"),
                SingleMeasurement(name="GPU 1 Memory Used", value=10.0, unit="GB"),
                SingleMeasurement(name="GPU 1 Memory Used std", value=0.5, unit="GB"),
                SingleMeasurement(name="GPU 1 Memory Used peak", value=12.0, unit="GB"),
            ],
            metadata=[
                IntMetadata(name="gpu_device_count", data=2),
                IntMetadata(name="gpu_current_device", data=1),
            ],
            artefacts=[],
        )
    )
    resources = capture_resources(_Bm({"GPUInfo": gpu}))

    assert resources.gpu_util_pct.mean == pytest.approx(80.0)
    assert resources.gpu_mem_gb.peak == pytest.approx(12.0)


def test_capture_hardware():
    gpu = _Rec(
        MeasurementData(
            measurements=[],
            metadata=[
                DictMetadata(
                    name="gpu_devices",
                    data={"0": {"name": "H100", "total_memory_gb": 80.0, "compute_capability": "9.0"}},
                ),
            ],
            artefacts=[],
        )
    )
    cpu = _Rec(
        MeasurementData(
            measurements=[],
            metadata=[
                StringMetadata(name="cpu_name", data="EPYC"),
                IntMetadata(name="physical_cores", data=64),
            ],
            artefacts=[],
        )
    )
    mem = _Rec(
        MeasurementData(
            measurements=[],
            metadata=[FloatMetadata(name="total_ram_gb", data=512.0)],
            artefacts=[],
        )
    )
    h = capture_hardware(_Bm({"GPUInfo": gpu, "CPUInfo": cpu, "MemoryInfo": mem}))
    assert isinstance(h, Hardware)
    assert h.gpu_devices[0].name == "H100" and h.gpu_devices[0].mem_gb == pytest.approx(80.0)
    assert h.gpu_devices[0].compute_cap == "9.0"
    assert h.cpu_name == "EPYC" and h.cpu_count == 64 and h.ram_gb == pytest.approx(512.0)
    assert isinstance(h.hostname, str) and h.hostname


def test_capture_handles_missing_recorders():
    bm = _Bm(None)
    assert isinstance(capture_versions(bm), Versions)
    assert isinstance(capture_hardware(bm), Hardware)
    assert isinstance(capture_resources(bm), Resources)


def test_synth_run_id():
    rid = synth_run_id("rsl_rl", "physx", "Isaac-Ant-Direct-v0", 42, "20260612-150000")
    assert "rsl_rl" in rid and "physx" in rid and "42" in rid


def test_run_config_uses_concrete_backend_configuration():
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=SimpleNamespace(class_type="isaaclab_newton.physics:NewtonMJWarpManager")),
        camera=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx")),
    )
    cfg = run_config_from_env_cfg(env_cfg)
    assert cfg.physics_backend == "newton_mjwarp"
    assert cfg.rendering_backend == "isaacsim_rtx"
    assert cfg.presets == []

    physx_env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(class_type="PhysXManager")))
    cfg = run_config_from_env_cfg(physx_env_cfg)
    assert cfg.physics_backend == "physx"

    default_env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=None))
    assert run_config_from_env_cfg(default_env_cfg).physics_backend == "physx"

    with pytest.raises(ValueError, match="Unsupported concrete physics config"):
        run_config_from_env_cfg(SimpleNamespace(sim=SimpleNamespace(physics=object())))


def test_camera_resolutions_omit_no_camera_metadata():
    env_cfg = SimpleNamespace(viewport=SimpleNamespace(width=1920, height=1080))
    assert camera_resolutions_from_env_cfg(env_cfg) == {}
    assert camera_resolution_metadata_from_env_cfg(env_cfg) == []


def test_camera_resolutions_report_distinct_cameras_sorted_by_path():
    env_cfg = SimpleNamespace(
        z_camera=_camera_cfg(),
        a_ray_camera=_ray_caster_camera_cfg(),
    )

    expected = {
        "env.a_ray_camera": {"width": 32, "height": 24},
        "env.z_camera": {"width": 64, "height": 48},
    }
    assert camera_resolutions_from_env_cfg(env_cfg) == expected
    assert camera_resolution_metadata_from_env_cfg(env_cfg) == [{"name": "camera_resolutions", "data": expected}]


def test_camera_resolutions_preserve_aliased_camera_paths():
    shared_camera = _camera_cfg()
    env_cfg = SimpleNamespace(scene=SimpleNamespace(tiled_camera=shared_camera, duplicate=shared_camera))

    assert camera_resolutions_from_env_cfg(env_cfg) == {
        "env.scene.duplicate": {"width": 64, "height": 48},
        "env.scene.tiled_camera": {"width": 64, "height": 48},
    }


def test_camera_resolutions_ignore_unrelated_same_named_class():
    class CameraCfg:
        width = 64
        height = 48

    assert camera_resolutions_from_env_cfg(SimpleNamespace(camera=CameraCfg())) == {}


def test_camera_resolutions_omit_unconfigured_dimensions():
    camera_cfg = _camera_cfg()
    camera_cfg.width = object()
    camera_cfg.height = object()

    env_cfg = SimpleNamespace(camera=camera_cfg)
    assert camera_resolutions_from_env_cfg(env_cfg) == {}
    assert camera_resolution_metadata_from_env_cfg(env_cfg) == []


def test_capture_resources_peak_clamped_to_mean_when_peak_row_absent():
    # Build a recorder that has mean/std rows but no peak rows.
    # capture_resources must clamp peak to mean rather than leaving it at 0.0
    # (which would violate MeanStd.__post_init__ since peak < mean).
    gpu = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="GPU Utilization", value=5.0, unit="%"),
                SingleMeasurement(name="GPU Utilization std", value=1.0, unit="%"),
                SingleMeasurement(name="GPU Memory Used", value=10.0, unit="GB"),
                SingleMeasurement(name="GPU Memory Used std", value=0.5, unit="GB"),
                # No "GPU Memory Used peak" row — peak defaults to 0.0 before clamping.
            ],
            metadata=[],
            artefacts=[],
        )
    )
    mem = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="System Memory RSS", value=10.0, unit="GB"),
                SingleMeasurement(name="System Memory RSS std", value=0.2, unit="GB"),
                # No "System Memory RSS peak" row.
            ],
            metadata=[],
            artefacts=[],
        )
    )
    cpu = _Rec(
        MeasurementData(
            measurements=[
                SingleMeasurement(name="CPU Utilization", value=20.0, unit="%"),
                SingleMeasurement(name="CPU Utilization std", value=2.0, unit="%"),
            ],
            metadata=[],
            artefacts=[],
        )
    )
    # Must not raise ValueError from MeanStd.__post_init__.
    r = capture_resources(_Bm({"GPUInfo": gpu, "CPUInfo": cpu, "MemoryInfo": mem}))
    assert r.gpu_mem_gb.peak == pytest.approx(10.0)
    assert r.ram_gb.peak == pytest.approx(10.0)


def test_capture_hardware_gpu_devices_sorted_by_numeric_index():
    # Keys "10", "2", "0" should be returned in numeric order 0, 2, 10 — not lexical "0","10","2".
    gpu = _Rec(
        MeasurementData(
            measurements=[],
            metadata=[
                DictMetadata(
                    name="gpu_devices",
                    data={
                        "10": {"name": "H100-10", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                        "2": {"name": "H100-2", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                        "0": {"name": "H100-0", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                    },
                ),
            ],
            artefacts=[],
        )
    )
    bm = _Bm({"GPUInfo": gpu})
    h = capture_hardware(bm)
    names = [d.name for d in h.gpu_devices]
    assert names == ["H100-0", "H100-2", "H100-10"]
