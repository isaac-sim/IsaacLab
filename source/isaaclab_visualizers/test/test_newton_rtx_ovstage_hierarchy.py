# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVStage hierarchy-model stage held by :class:`NewtonViewerRTX`.

Newton's ``ViewerRTX`` creates its ``ovstage.Stage`` with the default hierarchy model, which leaves
world transforms stale on OVStage 0.2. ``NewtonViewerRTX`` therefore holds a stage configured by
Isaac Lab for its lifetime. These tests pin that lifecycle without constructing a real renderer.
"""

from __future__ import annotations

import sys
import types

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pytest
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerRTX

from isaaclab.utils.backend_utils import FactoryBase

pytestmark = [pytest.mark.unit]


class _FakeStage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> list[_FakeStage]:
    """Replace the viewer base class and ``isaaclab_ov.stage`` with lightweight fakes."""
    created: list[_FakeStage] = []

    def create_ovstage(name: str) -> _FakeStage:
        created.append(_FakeStage(name))
        return created[-1]

    package = types.ModuleType("isaaclab_ov")
    package.__path__ = []
    stage_module = types.ModuleType("isaaclab_ov.stage")
    stage_module.create_ovstage = create_ovstage
    monkeypatch.setitem(sys.modules, "isaaclab_ov", package)
    monkeypatch.setitem(sys.modules, "isaaclab_ov.stage", stage_module)

    monkeypatch.setattr(newton_visualizer.ViewerRTX, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(newton_visualizer.ViewerRTX, "close", lambda self: None)
    monkeypatch.setattr(NewtonViewerRTX, "register_ui_callback", lambda self, *args, **kwargs: None, raising=False)
    monkeypatch.setattr(FactoryBase, "_get_backend", staticmethod(lambda: "physx"))
    return created


def test_stage_is_created_on_construction_and_released_once_on_close(stages):
    """Construction creates one named stage; close() destroys it exactly once, even when called twice."""
    viewer = NewtonViewerRTX()

    assert [stage.name for stage in stages] == ["isaaclab.newton_rtx_hierarchy_model"]
    assert stages[0].destroy_calls == 0

    viewer.close()
    assert stages[0].destroy_calls == 1

    viewer.close()
    assert stages[0].destroy_calls == 1


def test_stage_is_released_when_viewer_close_fails(stages, monkeypatch):
    """A failing teardown must not pin the process-wide hierarchy model."""

    def failing_close(self) -> None:
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(newton_visualizer.ViewerRTX, "close", failing_close)
    viewer = NewtonViewerRTX()

    with pytest.raises(RuntimeError, match="teardown failed"):
        viewer.close()
    assert stages[0].destroy_calls == 1


def test_no_stage_is_acquired_when_construction_fails(stages, monkeypatch):
    """A constructor failure must not leave a stage pinning the process-wide hierarchy model."""

    def failing_register(self, *args, **kwargs) -> None:
        raise RuntimeError("ui registration failed")

    monkeypatch.setattr(NewtonViewerRTX, "register_ui_callback", failing_register)

    with pytest.raises(RuntimeError, match="ui registration failed"):
        NewtonViewerRTX()
    assert stages == []


def test_no_stage_is_held_without_ovstage(stages, monkeypatch):
    """Without ovstage, ViewerRTX does not use it, so nothing is held and close() still works."""
    monkeypatch.setitem(sys.modules, "isaaclab_ov.stage", None)
    viewer = NewtonViewerRTX()

    assert viewer._ovstage_hierarchy_holder is None
    assert stages == []
    viewer.close()
