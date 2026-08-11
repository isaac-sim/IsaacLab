# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that scene operations dispatch to the selected backend without gaps."""

from __future__ import annotations

import importlib.util
import inspect

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
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

# Each operation needs an ``_<operation>_ovstage`` and an ``_<operation>_legacy`` implementation, plus a
# dispatch method selecting between them on ``_use_ovstage``.
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


def _dispatch_method(operation: str):
    """Return the dispatch method for ``operation``, whether it is named ``_x`` or ``x``."""
    return getattr(OVRTXRenderer, f"_{operation}", None) or getattr(OVRTXRenderer, operation)


@pytest.mark.parametrize("operation", _BACKEND_OPERATIONS)
@pytest.mark.parametrize("suffix", ["ovstage", "legacy"])
def test_both_backends_implement_every_operation(operation, suffix):
    assert callable(getattr(OVRTXRenderer, f"_{operation}_{suffix}", None)), (
        f"the {suffix} backend is missing _{operation}_{suffix}"
    )


def test_every_paired_implementation_is_dispatched():
    """A backend pair missing from :data:`_BACKEND_OPERATIONS` would never be covered here."""
    paired = {
        name.removesuffix("_ovstage").removeprefix("_")
        for name in vars(OVRTXRenderer)
        if name.endswith("_ovstage") and f"{name.removesuffix('_ovstage')}_legacy" in vars(OVRTXRenderer)
    }

    assert paired == set(_BACKEND_OPERATIONS)


@pytest.mark.parametrize("operation", _BACKEND_OPERATIONS)
@pytest.mark.parametrize("use_ovstage", [False, True])
def test_operation_calls_only_the_selected_backend(operation, use_ovstage, monkeypatch):
    """Calling the dispatch method must invoke the selected backend and only that one."""
    calls: list[str] = []
    for suffix in ("ovstage", "legacy"):
        name = f"_{operation}_{suffix}"
        monkeypatch.setattr(OVRTXRenderer, name, lambda self, *a, _n=name, **kw: calls.append(_n))

    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._use_ovstage = use_ovstage
    # close() drains the shared strategy before dispatching; that is not part of backend selection.
    renderer._strategy = type("_Drained", (), {"cleanup": lambda *a, **kw: None})()
    renderer._consume_products = None

    dispatch = getattr(renderer, f"_{operation}", None) or getattr(renderer, operation)
    parameters = list(inspect.signature(dispatch).parameters.values())
    required = [p for p in parameters if p.default is inspect.Parameter.empty]
    dispatch(*[None] * len(required))

    assert calls == [f"_{operation}_{'ovstage' if use_ovstage else 'legacy'}"]
