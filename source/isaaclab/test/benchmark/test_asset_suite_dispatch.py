# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the asset micro-benchmark provider seam."""

from types import SimpleNamespace

import pytest

from isaaclab.benchmark.asset_suites import dispatch as asset_dispatch
from isaaclab.benchmark.asset_suites import get_asset_benchmark_adapter

pytestmark = pytest.mark.benchmark


def test_provider_lookup_is_lazy_and_component_specific(monkeypatch) -> None:
    """The future factory should import only the selected physics provider."""
    adapter = object()
    imports: list[str] = []

    def fake_import_module(module_name: str):
        imports.append(module_name)
        return SimpleNamespace(get_asset_benchmark_adapter=lambda component: (component, adapter))

    monkeypatch.setattr(asset_dispatch.importlib, "import_module", fake_import_module)

    assert get_asset_benchmark_adapter("newton", "articulation") == ("articulation", adapter)
    assert imports == ["isaaclab_newton.benchmark.assets"]


@pytest.mark.parametrize(
    ("physics", "component", "message"),
    (
        ("unknown", "articulation", "Unsupported physics backend"),
        ("physx", "unknown", "Unsupported asset component"),
    ),
)
def test_provider_lookup_rejects_unknown_selection(physics: str, component: str, message: str) -> None:
    """Invalid future factory selections should fail before backend imports."""
    with pytest.raises(ValueError, match=message):
        get_asset_benchmark_adapter(physics, component)
