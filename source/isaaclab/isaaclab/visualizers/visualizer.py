# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating visualizer instances."""

from __future__ import annotations

import importlib
import logging

from isaaclab.utils.backend_utils import FactoryBase

from .base_visualizer import BaseVisualizer

# Visualizer types; each loads from isaaclab_visualizers.<type> for minimal deps.
_VISUALIZER_TYPES = ("kit", "newton", "rerun", "viser")

logger = logging.getLogger(__name__)


class Visualizer(FactoryBase, BaseVisualizer):
    """Factory for creating visualizer instances."""

    _backend_class_names = {
        "kit": "KitVisualizer",
        "newton": "NewtonVisualizer",
        "rerun": "RerunVisualizer",
        "viser": "ViserVisualizer",
    }
    _cfg_import_targets = {
        "kit": ("isaaclab_visualizers.kit.kit_visualizer_cfg", "KitVisualizerCfg"),
        "newton": ("isaaclab_visualizers.newton.newton_visualizer_cfg", "NewtonVisualizerCfg"),
        "rerun": ("isaaclab_visualizers.rerun.rerun_visualizer_cfg", "RerunVisualizerCfg"),
        "viser": ("isaaclab_visualizers.viser.viser_visualizer_cfg", "ViserVisualizerCfg"),
    }

    @classmethod
    def _get_backend(cls, cfg, *args, **kwargs) -> str:
        visualizer_type = getattr(cfg, "visualizer_type", None)
        if visualizer_type not in _VISUALIZER_TYPES:
            raise ValueError(
                f"Visualizer type '{visualizer_type}' is not registered. Valid types: "
                f"{', '.join(repr(k) for k in _VISUALIZER_TYPES)}."
            )
        return visualizer_type

    @classmethod
    def _get_module_name(cls, backend: str) -> str:
        return f"isaaclab_visualizers.{backend}"

    @classmethod
    def _resolve_impl_class_for_type(cls, visualizer_type: str) -> type[BaseVisualizer] | None:
        """Resolve backend visualizer class for a visualizer type."""
        if visualizer_type not in _VISUALIZER_TYPES:
            return None
        if visualizer_type in cls._registry:
            return cls._registry[visualizer_type]
        module_name = cls._get_module_name(visualizer_type)
        class_name = cls._backend_class_names.get(visualizer_type, cls.__name__)
        try:
            module = importlib.import_module(module_name)
            module_class = getattr(module, class_name)
            cls.register(visualizer_type, module_class)
            return module_class
        except Exception as exc:
            logger.debug(
                "[Visualizer] Failed to resolve implementation class for type '%s': %s",
                visualizer_type,
                exc,
            )
            return None

    @classmethod
    def _resolve_cfg_class_for_type(cls, visualizer_type: str) -> type | None:
        """Resolve visualizer cfg class for a visualizer type."""
        if visualizer_type not in _VISUALIZER_TYPES:
            return None
        target = cls._cfg_import_targets.get(visualizer_type)
        if target is None:
            return None
        module_name, class_name = target
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            logger.debug(
                "[Visualizer] Failed to resolve config class for type '%s': %s",
                visualizer_type,
                exc,
            )
            return None

    @classmethod
    def get_requirements_for_type(cls, visualizer_type: str) -> tuple[bool, bool]:
        """Return (requires_newton_model, requires_usd_stage) for a visualizer type."""
        if visualizer_type not in _VISUALIZER_TYPES:
            raise ValueError(
                f"Visualizer type '{visualizer_type}' is not registered. Valid types: "
                f"{', '.join(repr(k) for k in _VISUALIZER_TYPES)}."
            )
        cfg_class = cls._resolve_cfg_class_for_type(visualizer_type)
        if cfg_class is None:
            logger.debug(
                "[Visualizer] Using default requirements (False, False) for type '%s' because config class could not "
                "be imported.",
                visualizer_type,
            )
            return False, False
        try:
            cfg_obj = cfg_class()
        except Exception as exc:
            logger.debug(
                "[Visualizer] Using default requirements (False, False) for type '%s' because config could not be "
                "instantiated: %s",
                visualizer_type,
                exc,
            )
            return False, False
        return bool(getattr(cfg_obj, "requires_newton_model", False)), bool(getattr(cfg_obj, "requires_usd_stage", False))

    def __new__(cls, cfg, *args, **kwargs) -> BaseVisualizer:
        """Create a new visualizer instance based on the visualizer type."""
        result = super().__new__(cls, cfg, *args, **kwargs)
        if not isinstance(result, BaseVisualizer):
            name = type(result).__name__
            raise TypeError(f"Backend visualizer {name!r} must inherit from BaseVisualizer.")
        return result
