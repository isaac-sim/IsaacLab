# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the asset micro-benchmark provider seam."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab.benchmark.asset_suites import dispatch as asset_dispatch
from isaaclab.benchmark.asset_suites import get_asset_benchmark_adapter

pytestmark = pytest.mark.benchmark


def test_provider_lookup_is_lazy_and_component_specific(monkeypatch) -> None:
    """Provider lookup should import only the selected physics package."""

    @dataclass(frozen=True)
    class FakeAdapter:
        component: str
        marker: object
        physics_variant: str = "newton_mjwarp"

    marker = object()
    imports: list[str] = []

    def fake_import_module(module_name: str):
        imports.append(module_name)
        return SimpleNamespace(get_asset_benchmark_adapter=lambda component: FakeAdapter(component, marker))

    monkeypatch.setattr(asset_dispatch.importlib, "import_module", fake_import_module)

    adapter = get_asset_benchmark_adapter("newton_kamino", "articulation")
    assert adapter.component == "articulation"
    assert adapter.marker is marker
    assert adapter.physics_variant == "newton_kamino"
    assert imports == ["isaaclab_newton.benchmark.assets"]


@pytest.mark.parametrize(
    ("physics", "component", "message"),
    (
        ("unknown", "articulation", "Unsupported asset physics selector"),
        ("physx", "unknown", "Unsupported asset component"),
    ),
)
def test_provider_lookup_rejects_unknown_selection(physics: str, component: str, message: str) -> None:
    """Invalid provider selections should fail before backend imports."""
    with pytest.raises(ValueError, match=message):
        get_asset_benchmark_adapter(physics, component)


@pytest.mark.parametrize(
    ("variant", "family"),
    (
        ("physx", "physx"),
        ("isaacsim_physx", "physx"),
        ("ovphysx", "ovphysx"),
        ("newton_mjwarp", "newton"),
        ("newton_kamino", "newton"),
    ),
)
def test_exact_physics_variant_is_preserved_separately_from_family(variant, family) -> None:
    """Exact selectors should survive provider resolution without changing package family dispatch."""
    adapter = get_asset_benchmark_adapter(variant, "articulation")

    assert adapter.physics == family
    assert adapter.physics_variant == variant


def test_unsupported_exact_physics_variant_is_explicit() -> None:
    """Unknown exact variants should fail rather than silently normalizing to a family."""
    with pytest.raises(ValueError, match="Unsupported asset physics selector"):
        get_asset_benchmark_adapter("newton_unknown", "articulation")
