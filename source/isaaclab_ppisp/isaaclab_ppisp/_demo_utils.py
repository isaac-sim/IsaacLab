# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PPISP camera-selection helpers for demo scripts."""

from __future__ import annotations

from typing import Any

from pxr import Sdf

from .cfg import has_ppisp_camera_attrs

PpispCameraBinding = tuple[str, Any | None, Any]


def find_ppisp_camera_bindings(stage: Any) -> list[PpispCameraBinding]:
    """Return cameras with recognized PPISP attributes and optional render-product metadata."""
    bindings = []
    for prim in stage.Traverse():
        if not has_ppisp_camera_attrs(prim):
            continue
        camera_path = str(prim.GetPath())
        bindings.append((camera_path, find_render_product_for_camera(stage, camera_path), prim))
    return bindings


def find_render_product_for_camera(stage: Any, camera_prim_path: str) -> Any | None:
    """Return the first RenderProduct targeting ``camera_prim_path``, if any."""
    target_path = Sdf.Path(camera_prim_path)
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        camera_rel = prim.GetRelationship("camera")
        if camera_rel and target_path in camera_rel.GetTargets():
            return prim
    return None


def order_ppisp_bindings_by_camera(stage: Any, ppisp_bindings: list[PpispCameraBinding]) -> list[PpispCameraBinding]:
    """Return PPISP bindings ordered by source camera prim traversal."""
    binding_by_camera_path = {}
    for binding in ppisp_bindings:
        binding_by_camera_path.setdefault(binding[0], binding)

    ordered_bindings = []
    seen_paths = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera":
            continue
        camera_path = str(prim.GetPath())
        binding = binding_by_camera_path.get(camera_path)
        if binding is not None:
            ordered_bindings.append(binding)
            seen_paths.add(camera_path)

    for binding in ppisp_bindings:
        if binding[0] not in seen_paths:
            ordered_bindings.append(binding)
            seen_paths.add(binding[0])
    return ordered_bindings


def format_available_ppisp_cameras(ppisp_bindings: list[PpispCameraBinding]) -> str:
    """Format cameras with PPISP attributes for CLI error messages."""
    return "\n  ".join(dict.fromkeys(binding[0] for binding in ppisp_bindings))
