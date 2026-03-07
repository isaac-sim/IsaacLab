# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating renderer instances."""

from __future__ import annotations

import importlib
import logging

from isaaclab.utils.backend_utils import FactoryBase

from .base_renderer import BaseRenderer
from .renderer_cfg import RendererCfg

# This is mapping of where backends live in the isaaclab_<backend> package.
_RENDERER_TYPE_TO_BACKEND = {"isaac_rtx": "physx", "newton_warp": "newton", "ovrtx": "ov"}

logger = logging.getLogger(__name__)


class Renderer(FactoryBase, BaseRenderer):
    """Factory for creating renderer instances."""

    _backend_class_names = {
        "physx": "IsaacRtxRenderer",
        "newton": "NewtonWarpRenderer",
        "ov": "OVRTXRenderer",
    }
    _renderer_cfg_import_targets = {
        "isaac_rtx": ("isaaclab_physx.renderers.isaac_rtx_renderer_cfg", "IsaacRtxRendererCfg"),
        "newton_warp": ("isaaclab_newton.renderers.newton_warp_renderer_cfg", "NewtonWarpRendererCfg"),
        "ovrtx": ("isaaclab_ov.renderers.ovrtx_renderer_cfg", "OVRTXRendererCfg"),
    }

    @classmethod
    def _get_backend(cls, cfg: RendererCfg, *args, **kwargs) -> str:
        return _RENDERER_TYPE_TO_BACKEND.get(cfg.renderer_type, "physx")

    @classmethod
    def _resolve_impl_class_for_renderer_type(cls, renderer_type: str) -> type[BaseRenderer] | None:
        """Resolve backend renderer class for a renderer type."""
        backend = _RENDERER_TYPE_TO_BACKEND.get(renderer_type, "physx")
        if backend in cls._registry:
            return cls._registry[backend]
        module_name = cls._get_module_name(backend)
        class_name = cls._backend_class_names.get(backend, cls.__name__)
        try:
            module = importlib.import_module(module_name)
            module_class = getattr(module, class_name)
            cls.register(backend, module_class)
            return module_class
        except Exception as exc:
            logger.debug(
                "[Renderer] Failed to resolve implementation class for renderer '%s' (backend '%s'): %s",
                renderer_type,
                backend,
                exc,
            )
            return None

    @classmethod
    def _resolve_cfg_class_for_renderer_type(cls, renderer_type: str) -> type | None:
        """Resolve backend renderer cfg class for a renderer type."""
        target = cls._renderer_cfg_import_targets.get(renderer_type)
        if target is None:
            return None
        module_name, class_name = target
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            logger.debug(
                "[Renderer] Failed to resolve config class for renderer '%s': %s",
                renderer_type,
                exc,
            )
            return None

    @classmethod
    def get_requirements_for_type(cls, renderer_type: str) -> tuple[bool, bool]:
        """Return (requires_newton_model, requires_usd_stage) for a renderer type."""
        cfg_class = cls._resolve_cfg_class_for_renderer_type(renderer_type)
        if cfg_class is None:
            logger.debug(
                "[Renderer] Using default requirements (False, False) for renderer '%s' because backend config "
                "class could not be imported.",
                renderer_type,
            )
            return False, False
        try:
            cfg_obj = cfg_class()
        except Exception as exc:
            logger.debug(
                "[Renderer] Using default requirements (False, False) for renderer '%s' because backend config "
                "could not be instantiated: %s",
                renderer_type,
                exc,
            )
            return False, False
        return bool(getattr(cfg_obj, "requires_newton_model", False)), bool(
            getattr(cfg_obj, "requires_usd_stage", False)
        )

    def __new__(cls, cfg: RendererCfg, *args, **kwargs) -> BaseRenderer:
        """Create a new instance of a renderer based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific renderer class.
        result = super().__new__(cls, cfg, *args, **kwargs)
        if not isinstance(result, BaseRenderer):
            name = type(result).__name__
            bases = type(result).__bases__
            raise TypeError(
                f"Backend renderer {name!r} must inherit from BaseRenderer, but it inherits from {bases!r}."
            )
        return result
