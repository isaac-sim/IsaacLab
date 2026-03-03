# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer factory helpers and base types."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from isaaclab.utils.module import lazy_export

from .visualizer import Visualizer

lazy_export(packages=["isaaclab_physx.visualizers", "isaaclab_newton.visualizers"])

_VISUALIZER_REGISTRY: dict[str, Any] = {}
_VISUALIZER_BACKEND_SPECS: dict[str, tuple[str, str]] = {
    "kit": ("isaaclab_physx.visualizers", "KitVisualizer"),
    "newton": ("isaaclab_newton.visualizers", "NewtonVisualizer"),
    "rerun": ("isaaclab_newton.visualizers", "RerunVisualizer"),
}


def get_visualizer_class(name: str) -> type[Visualizer] | None:
    """Get a visualizer class by type name."""
    if name in _VISUALIZER_REGISTRY:
        return _VISUALIZER_REGISTRY[name]

    try:
        module_name, class_name = _VISUALIZER_BACKEND_SPECS[name]
    except KeyError:
        return None

    try:
        visualizer_class = getattr(import_module(module_name), class_name)
        _VISUALIZER_REGISTRY[name] = visualizer_class
        return visualizer_class
    except ImportError as exc:
        import warnings

        warnings.warn(f"Failed to load visualizer '{name}': {exc}", ImportWarning)
        return None
