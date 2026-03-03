# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating visualizer instances."""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_visualizer import BaseVisualizer

_VISUALIZER_TYPE_TO_BACKEND = {"kit": "physx", "newton": "newton", "rerun": "newton"}


class Visualizer(FactoryBase, BaseVisualizer):
    """Factory for creating visualizer instances."""

    _backend_class_names = {"kit": "KitVisualizer", "newton": "NewtonVisualizer", "rerun": "RerunVisualizer"}

    @classmethod
    def _get_backend(cls, cfg, *args, **kwargs) -> str:
        visualizer_type = getattr(cfg, "visualizer_type", None)
        if visualizer_type not in _VISUALIZER_TYPE_TO_BACKEND:
            raise ValueError(
                f"Visualizer type '{visualizer_type}' is not registered. Valid types: "
                f"{', '.join(repr(k) for k in _VISUALIZER_TYPE_TO_BACKEND)}."
            )
        return visualizer_type

    @classmethod
    def _get_module_name(cls, backend: str) -> str:
        package_backend = _VISUALIZER_TYPE_TO_BACKEND[backend]
        return f"isaaclab_{package_backend}.{cls._module_subpath}"

    def __new__(cls, cfg, *args, **kwargs) -> BaseVisualizer:
        """Create a new visualizer instance based on the visualizer type."""
        result = super().__new__(cls, cfg, *args, **kwargs)
        if not isinstance(result, BaseVisualizer):
            name = type(result).__name__
            raise TypeError(f"Backend visualizer {name!r} must inherit from BaseVisualizer.")
        return result
