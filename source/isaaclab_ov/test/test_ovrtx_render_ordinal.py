# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ordinal propagation from the render strategies into ovrtx stepping."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer_strategies import _AsyncRenderStrategy, _SyncRenderStrategy


class _RecordingRenderer:
    """Captures the ordinal ovrtx would receive, mimicking ovrtx's attached/standalone contract."""

    def __init__(self, *, attached: bool) -> None:
        self._attached = attached
        self.ordinals: list[int | None] = []

    def _check(self, ordinal: int | None) -> None:
        if self._attached and ordinal is None:
            raise RuntimeError("ordinal is required while an ovstage is attached")
        if not self._attached and ordinal is not None:
            raise RuntimeError("ordinal is only valid while an ovstage is attached")
        self.ordinals.append(ordinal)

    def step(self, render_products: set[str], delta_time: float, *, ordinal: int | None = None) -> dict:
        self._check(ordinal)
        return {}

    def step_async(self, render_products: set[str], delta_time: float, *, ordinal: int | None = None) -> Any:
        self._check(ordinal)
        return _CompletedOp()


class _CompletedOp:
    """Stands in for an ovrtx step operation whose products are already available."""

    def wait(self) -> _CompletedOp:
        return self

    def fetch(self) -> dict:
        return {}


def _render(strategy, renderer, ordinal: int | None) -> None:
    strategy.render(renderer, {"/product"}, 1.0 / 60.0, None, lambda render_data, products: None, ordinal=ordinal)


def test_sync_strategy_forwards_ordinal_when_attached():
    renderer = _RecordingRenderer(attached=True)

    _render(_SyncRenderStrategy(), renderer, 7)

    assert renderer.ordinals == [7]


def test_sync_strategy_omits_ordinal_when_standalone():
    renderer = _RecordingRenderer(attached=False)

    _render(_SyncRenderStrategy(), renderer, None)

    assert renderer.ordinals == [None]


def test_async_strategy_forwards_ordinal_when_attached():
    renderer = _RecordingRenderer(attached=True)

    _render(_AsyncRenderStrategy(), renderer, 11)

    assert renderer.ordinals == [11]


def test_async_strategy_omits_ordinal_when_standalone():
    renderer = _RecordingRenderer(attached=False)

    _render(_AsyncRenderStrategy(), renderer, None)

    assert renderer.ordinals == [None]


def test_async_strategy_forwards_each_frame_ordinal():
    renderer = _RecordingRenderer(attached=True)
    strategy = _AsyncRenderStrategy()

    for ordinal in (3, 4, 5):
        _render(strategy, renderer, ordinal)

    assert renderer.ordinals == [3, 4, 5]
