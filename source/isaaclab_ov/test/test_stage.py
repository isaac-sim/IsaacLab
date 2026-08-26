# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared ovstage construction helpers."""

import types

from isaaclab_ov import stage as stage_module


def test_create_ovstage_configures_supported_runtime(monkeypatch):
    """StageConfig-capable runtimes receive the requested hierarchy model."""
    configured_model = object()
    created = object()
    calls = {}

    class RecordingStageConfig:
        def __init__(self, *, runtime_default_hierarchy_computation_model):
            calls["model"] = runtime_default_hierarchy_computation_model

    def create_stage(name, *, config):
        calls["name"] = name
        calls["config"] = config
        return created

    runtime = types.SimpleNamespace(
        HierarchyComputationModel=types.SimpleNamespace(CPU_INCREMENTAL=configured_model),
        Stage=create_stage,
        StageConfig=RecordingStageConfig,
    )
    monkeypatch.setattr(stage_module, "ovstage", runtime)

    assert stage_module.create_ovstage("configured") is created
    assert calls["name"] == "configured"
    assert calls["model"] is configured_model
    assert isinstance(calls["config"], RecordingStageConfig)


def test_create_ovstage_supports_runtime_without_stage_config(monkeypatch):
    """Earlier runtimes that expose only ``Stage(name)`` remain supported."""
    created = object()
    calls = []

    def create_stage(name):
        calls.append(name)
        return created

    runtime = types.SimpleNamespace(Stage=create_stage)
    monkeypatch.setattr(stage_module, "ovstage", runtime)

    assert stage_module.create_ovstage("legacy") is created
    assert calls == ["legacy"]
