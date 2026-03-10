# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating visualizer instances."""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_visualizer import BaseVisualizer

# Visualizer types; each loads from isaaclab_visualizers.<type> for minimal deps.
_VISUALIZER_TYPES = ("kit", "newton", "rerun", "viser")


class Visualizer(FactoryBase, BaseVisualizer):
    """Factory for creating visualizer instances."""

    _backend_class_names = {
        "kit": "KitVisualizer",
        "newton": "NewtonVisualizer",
        "rerun": "RerunVisualizer",
        "viser": "ViserVisualizer",
    }

    @classmethod
    def _get_backend(cls, cfg, *args, **kwargs) -> str:
        """Resolve backend key from visualizer config.

        Args:
            cfg: Visualizer configuration instance.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Backend key used by the factory.

        Raises:
            ValueError: If visualizer type is not registered.
        """
        visualizer_type = getattr(cfg, "visualizer_type", None)
        if visualizer_type not in _VISUALIZER_TYPES:
            raise ValueError(
                f"Visualizer type '{visualizer_type}' is not registered. Valid types: "
                f"{', '.join(repr(k) for k in _VISUALIZER_TYPES)}."
            )
        return visualizer_type

    @classmethod
    def _get_module_name(cls, backend: str) -> str:
        """Return module path for a visualizer backend.

        Args:
            backend: Backend key.

        Returns:
            Module import path for the backend.
        """
        return f"isaaclab_visualizers.{backend}"

    def __new__(cls, cfg, *args, **kwargs) -> BaseVisualizer:
        """Create a new visualizer instance based on the visualizer type.

        Args:
            cfg: Visualizer configuration instance.
            *args: Additional constructor positional arguments.
            **kwargs: Additional constructor keyword arguments.

        Returns:
            Instantiated backend visualizer.

        Raises:
            TypeError: If backend class does not inherit from ``BaseVisualizer``.
        """
        result = super().__new__(cls, cfg, *args, **kwargs)
        if not isinstance(result, BaseVisualizer):
            name = type(result).__name__
            raise TypeError(f"Backend visualizer {name!r} must inherit from BaseVisualizer.")
        return result
