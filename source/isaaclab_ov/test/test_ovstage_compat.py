# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVStage 0.1 / 0.2 hierarchy computation model compatibility."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging

import pytest
from packaging.version import Version

_REQUIRED_MODULES = ("isaaclab_ov",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.ovstage_compat import (  # noqa: E402
        detect_ovstage_version,
        resolve_hierarchy_computation_model,
        supports_gpu_hierarchy_computation,
    )
else:
    detect_ovstage_version = None
    resolve_hierarchy_computation_model = None
    supports_gpu_hierarchy_computation = None


def test_detect_ovstage_version_reads_distribution_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.1.355824")
    assert detect_ovstage_version() == Version("0.1.1.355824")


def test_detect_ovstage_version_returns_none_when_uninstalled(monkeypatch: pytest.MonkeyPatch):
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert detect_ovstage_version() is None


def test_detect_ovstage_version_returns_none_for_unparseable_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "internal-build")
    assert detect_ovstage_version() is None


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, False),
        (Version("0.1.0.346039"), False),
        (Version("0.1.1.355824"), False),
        (Version("0.1.2"), False),
        (Version("0.2"), True),
        (Version("0.2.0.377349"), True),
        (Version("1.0"), True),
    ],
)
def test_supports_gpu_hierarchy_computation_switches_at_ovstage_02(version: Version | None, expected: bool):
    assert supports_gpu_hierarchy_computation(version) is expected


@pytest.mark.parametrize("version", [None, Version("0.1.1.355824")])
def test_ovstage_01_uses_the_host_hierarchy_model(version: Version | None):
    """OVStage 0.1 places objects out of position under the device model, so it must stay on the host."""
    assert resolve_hierarchy_computation_model(version) == "CPU_INCREMENTAL"


@pytest.mark.parametrize("version", [Version("0.2"), Version("0.2.0.377349"), Version("1.0")])
def test_ovstage_02_uses_the_device_hierarchy_model(version: Version):
    assert resolve_hierarchy_computation_model(version) == "GPU_INCREMENTAL"


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_published_model_names_exist_on_the_installed_enum():
    """Both names must resolve, since :func:`create_ovstage` looks the published one up by name."""
    import ovstage

    for name in ("CPU_INCREMENTAL", "GPU_INCREMENTAL"):
        assert isinstance(getattr(ovstage.HierarchyComputationModel, name), ovstage.HierarchyComputationModel)


@pytest.fixture()
def captured_stage_config(monkeypatch: pytest.MonkeyPatch):
    """Replace stage construction with fakes and yield the requested hierarchy model.

    :func:`~isaaclab_ov.stage.create_ovstage` reaches ovstage through module attributes, so
    patching them intercepts the config without building a real process-scoped stage.
    """
    import ovstage

    captured: dict[str, object] = {}

    def fake_stage_config(*, runtime_default_hierarchy_computation_model):
        captured["model"] = runtime_default_hierarchy_computation_model
        return object()

    monkeypatch.setattr(ovstage, "StageConfig", fake_stage_config)
    monkeypatch.setattr(ovstage, "Stage", lambda name, config: object())
    return captured


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_create_ovstage_requests_the_selected_model(monkeypatch: pytest.MonkeyPatch, captured_stage_config: dict):
    import ovstage
    from isaaclab_ov import stage as stage_module

    monkeypatch.setattr(stage_module, "HIERARCHY_COMPUTATION_MODEL", "GPU_INCREMENTAL")
    stage_module.create_ovstage("compat.selected")

    assert captured_stage_config["model"] is ovstage.HierarchyComputationModel.GPU_INCREMENTAL


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_create_ovstage_falls_back_to_the_host_model_for_an_unknown_name(
    monkeypatch: pytest.MonkeyPatch, captured_stage_config: dict, caplog: pytest.LogCaptureFixture
):
    """A model this ovstage does not carry must degrade to the host model, not fail the stage."""
    import ovstage
    from isaaclab_ov import stage as stage_module

    monkeypatch.setattr(stage_module, "HIERARCHY_COMPUTATION_MODEL", "MODEL_FROM_A_FUTURE_OVSTAGE")
    with caplog.at_level(logging.WARNING, logger=stage_module.__name__):
        stage_module.create_ovstage("compat.fallback")

    assert captured_stage_config["model"] is ovstage.HierarchyComputationModel.CPU_INCREMENTAL
    assert "MODEL_FROM_A_FUTURE_OVSTAGE" in caplog.text
