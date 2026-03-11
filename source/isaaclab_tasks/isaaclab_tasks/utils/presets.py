# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class PresetCfg:
    """Base class for declarative preset definitions.

    Subclass this and define fields as preset options.
    The field named ``default`` holds the config instance used
    when no CLI override is given. All other fields are named
    alternative presets.

    Example::

        @configclass
        class PhysicsCfg(PresetCfg):
            default: PhysxCfg = PhysxCfg()
            newton: NewtonCfg = NewtonCfg()
    """

    pass


class UnavailablePreset:
    """Sentinel for a preset whose backend package is not installed.

    Carries an install hint so the error message can tell the user how to fix it.
    """

    def __init__(self, install_cmd: str) -> None:
        self.install_cmd = install_cmd


# Backend-specific renderer imports — each is optional depending on the installation.
# Use try/except to catch transitive import failures (e.g. isaaclab_ov installed
# with --no-deps so the ovrtx dependency is missing).
try:
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
except ImportError:
    IsaacRtxRendererCfg = None

try:
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
except ImportError:
    NewtonWarpRendererCfg = None

try:
    from isaaclab_ov.renderers import OVRTXRendererCfg
except ImportError:
    OVRTXRendererCfg = None


def _renderer_or_unavailable(cls, install_cmd: str) -> object:
    return cls() if cls is not None else UnavailablePreset(install_cmd)


# Pick the first available renderer as the default so that resolve_preset_defaults
# always produces a serializable config, even on partial installs (e.g. Newton-only).
_default_renderer_cls = next(
    (cls for cls in (IsaacRtxRendererCfg, NewtonWarpRendererCfg, OVRTXRendererCfg) if cls is not None),
    None,
)


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: object = _default_renderer_cls() if _default_renderer_cls is not None else None
    isaacsim_rtx_renderer: object = _renderer_or_unavailable(IsaacRtxRendererCfg, "./isaaclab.sh -i isaacsim")
    newton_renderer: object = _renderer_or_unavailable(NewtonWarpRendererCfg, "./isaaclab.sh -i newton")
    ovrtx_renderer: object = _renderer_or_unavailable(OVRTXRendererCfg, "./isaaclab.sh -i ovrtx")
