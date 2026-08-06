# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that scene operations resolve to the selected backend without gaps."""

from __future__ import annotations

import importlib.util

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
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer, _SceneBackendOperation

# Declared as :class:`_SceneBackendOperation`; each needs an ``_ovstage`` and a ``_legacy`` method.
_BACKEND_OPERATIONS = (
    "init_fields",
    "initialize_from_spec",
    "setup_xform_bindings",
    "setup_deformable_bindings",
    "setup_particle_bindings",
    "update_transforms",
    "update_geometries",
    "update_camera",
    "render",
    "close",
)


@pytest.mark.parametrize("operation", _BACKEND_OPERATIONS)
@pytest.mark.parametrize("suffix", ["ovstage", "legacy"])
def test_both_backends_implement_every_operation(operation, suffix):
    assert callable(getattr(OVRTXRenderer, f"_{operation}_{suffix}", None)), (
        f"the {suffix} backend is missing _{operation}_{suffix}"
    )


def test_declared_operations_match_the_paired_implementations():
    """An ``_x_ovstage``/``_x_legacy`` pair that is not declared would still need manual dispatch."""
    paired = {
        name.removesuffix("_ovstage").removeprefix("_")
        for name in vars(OVRTXRenderer)
        if name.endswith("_ovstage") and f"{name.removesuffix('_ovstage')}_legacy" in vars(OVRTXRenderer)
    }
    declared = {
        name.removeprefix("_")
        for name, value in vars(OVRTXRenderer).items()
        if isinstance(value, _SceneBackendOperation)
    }

    assert declared == paired == set(_BACKEND_OPERATIONS)


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_operations_resolve_to_the_selected_backend(use_ovstage):
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._use_ovstage = use_ovstage
    suffix = "ovstage" if use_ovstage else "legacy"

    for operation in _BACKEND_OPERATIONS:
        assert getattr(renderer, f"_{operation}").__func__ is getattr(OVRTXRenderer, f"_{operation}_{suffix}")
