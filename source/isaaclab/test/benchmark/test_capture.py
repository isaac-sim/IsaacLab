# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark capture helpers (Isaac-Sim-free, fake recorders)."""

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonSolverCfg,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.benchmark.capture import (
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

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerEntryCfg, CouplerProxyCfg
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg


class _Rec:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class _Bm:
    def __init__(self, recorders):
        self._manual_recorders = recorders


def test_capture_versions_renames_and_defaults():
    md = [
        StringMetadata(name="isaaclab_version", data="4.6.8"),
        StringMetadata(name="torch_version", data="2.5.1"),
        StringMetadata(name="mujoco_warp_version", data="0.0.4"),
        StringMetadata(name="stable_baselines3_version", data="2.3.0"),
        DictMetadata(name="dev", data={"commit_hash": "abc123", "branch": "develop", "dirty": True}),
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
    v = capture_versions(bm)
    assert isinstance(v, Versions)
    assert v.isaaclab == "4.6.8" and v.torch == "2.5.1"
    assert v.mjwarp == "0.0.4"
    assert v.sb3 == "2.3.0"
    assert v.git_commit == "abc123" and v.git_branch == "develop" and v.git_dirty is True
    assert v.isaacsim is None
    assert v.numpy == "2.4.4"
    assert v.isaaclab_newton == "1.0.2"
    assert v.isaaclab_physx == "2.0.1"
    assert v.isaaclab_ov == "0.4.6"
    assert v.isaaclab_tasks == "8.0.1"
    assert v.isaaclab_rl == "0.6.1"
    assert v.ovrtx is None
    assert v.ovphysx == "3.0.5"
    assert v.mujoco == "3.8.1"
    assert v.cuda_bindings == "12.9.4"
    assert v.usd_core == "25.11"
    assert v.isaaclab_release == "3.0.0"


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
                    # Numeric keys must be ordered 0, 2, 10, not lexically.
                    data={
                        "10": {"name": "H100-10", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                        "2": {"name": "H100-2", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                        "0": {"name": "H100", "total_memory_gb": 80.0, "compute_capability": "9.0"},
                    },
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
    assert [d.name for d in h.gpu_devices] == ["H100", "H100-2", "H100-10"]
    assert h.gpu_devices[0].mem_gb == pytest.approx(80.0)
    assert h.gpu_devices[0].compute_cap == "9.0"
    assert h.cpu_name == "EPYC" and h.cpu_count == 64 and h.ram_gb == pytest.approx(512.0)
    assert isinstance(h.hostname, str) and h.hostname


def test_capture_handles_missing_recorders():
    bm = _Bm(None)
    versions = capture_versions(bm)
    assert versions.isaaclab == "unknown"
    assert versions.torch == "unknown"
    assert versions.git_commit is None and versions.git_dirty is False
    hardware = capture_hardware(bm)
    assert hardware.gpu_devices == []
    assert hardware.cpu_name == "unknown" and hardware.cpu_count == 0 and hardware.ram_gb == 0.0
    resources = capture_resources(bm)
    assert resources.devices == {}
    assert resources.ram_gb.mean == 0.0 and resources.gpu_mem_gb.mean == 0.0


def test_synth_run_id():
    rid = synth_run_id("rsl_rl", "physx", "Isaac-Ant-Direct-v0", 42, "20260612-150000")
    assert rid == "rsl_rl_physx_Isaac-Ant-Direct-v0_20260612-150000_seed42"
    assert synth_run_id(None, "physx", "Isaac-Ant-Direct-v0", 42, "20260612-150000").startswith("runtime_")


@pytest.mark.parametrize(
    "physics_cfg, expected_backend, expected_solvers, expected_coupling",
    [
        (None, "physx", ["physx"], None),
        (NewtonCfg(), "newton_mjwarp", ["newton_mjwarp"], None),
        (PhysxCfg(), "physx", ["physx"], None),
        (OvPhysxCfg(), "ovphysx", ["ovphysx"], None),
        (NewtonCfg(solver_cfg=FeatherstoneSolverCfg()), "newton_featherstone", ["newton_featherstone"], None),
        (NewtonCfg(solver_cfg=XPBDSolverCfg()), "newton_xpbd", ["newton_xpbd"], None),
        (NewtonCfg(solver_cfg=VBDSolverCfg()), "newton_vbd", ["newton_vbd"], None),
        (NewtonCfg(solver_cfg=MPMSolverCfg()), "newton_mpm", ["newton_mpm"], None),
        (
            NewtonCfg(
                solver_cfg=CouplerProxyCfg(
                    entries=[
                        CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg()),
                        CouplerEntryCfg(name="soft", solver_cfg=VBDSolverCfg()),
                    ]
                )
            ),
            "newton_mjwarp",
            ["newton_mjwarp", "newton_vbd"],
            "proxy",
        ),
        (
            NewtonCfg(
                solver_cfg=CouplerAdmmCfg(
                    entries=[
                        CouplerEntryCfg(name="second", solver_cfg=MJWarpSolverCfg()),
                        CouplerEntryCfg(name="first", solver_cfg=KaminoPADMMSolverCfg()),
                        CouplerEntryCfg(name="duplicate", solver_cfg=MJWarpSolverCfg()),
                    ]
                )
            ),
            "newton_kamino",
            ["newton_kamino", "newton_mjwarp"],
            "admm",
        ),
        (
            NewtonCfg(solver_cfg=CoupledMJWarpVBDSolverCfg(coupling_mode="one_way")),
            "newton_mjwarp",
            ["newton_mjwarp", "newton_vbd"],
            "custom_one_way",
        ),
    ],
)
def test_run_config_uses_concrete_backend_configuration(
    physics_cfg, expected_backend, expected_solvers, expected_coupling
):
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=physics_cfg),
        camera=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx")),
    )
    cfg = run_config_from_env_cfg(env_cfg)
    assert cfg.physics_backend == expected_backend
    assert cfg.physics_solvers == expected_solvers
    assert cfg.physics_coupling == expected_coupling
    assert cfg.rendering_backend == "isaacsim_rtx"
    assert cfg.presets == []

    if physics_cfg is not None:
        # Runtime-class resolution and incidental configuration nodes do not define solver identity.
        physics_cfg.class_type = "unrelated.module:RenamedManager"
        physics_cfg.inactive_alternative = PhysxCfg()
        assert run_config_from_env_cfg(env_cfg) == cfg


def test_run_config_rejects_unknown_physics():
    with pytest.raises(ValueError, match="Unsupported concrete physics config"):
        run_config_from_env_cfg(SimpleNamespace(sim=SimpleNamespace(physics=object())))
    incomplete = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(name="known", solver_cfg=MJWarpSolverCfg()),
                CouplerEntryCfg(name="unknown", solver_cfg=NewtonSolverCfg()),
            ]
        )
    )
    with pytest.raises(ValueError, match="Unsupported Newton solver config"):
        run_config_from_env_cfg(SimpleNamespace(sim=SimpleNamespace(physics=incomplete)))


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
