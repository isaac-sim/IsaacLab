# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Driver shader-cache redirection for the OVRTX renderer.

Kept apart from ``ovrtx_renderer`` so the redirect policy imports without the
ovrtx runtime and can be tested on any runner; the runtime is only touched
inside :func:`_acquire_settings_applier`, which imports the bindings lazily.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovrtx import RendererConfig

logger = logging.getLogger(__name__)

# Runtime environment variable selecting the directory the NVIDIA driver writes
# its compiled shader cache to. Unset leaves the driver at its default path.
SHADER_CACHE_PATH_ENV = "OVRTX_SHADER_CACHE_PATH"

_APPLY_SETTINGS_EXT = b"ovrtx.settings.apply_settings"

# Both are needed: the driver keeps its own cache and the per-application cache
# in separate trees, and leaving either at the default path still compiles cold.
SHADER_CACHE_SETTINGS = (
    "/rtx/shaderDb/driverShaderCachePath",
    "/rtx/shaderDb/driverAppShaderCachePath",
)


def _acquire_settings_applier(config: RendererConfig) -> Callable[[str], bool] | None:
    """Return a callable that applies one ``--/setting=value`` string, or ``None``.

    ``None`` means this runtime does not ship the settings extension;
    :func:`redirect_shader_cache` turns that into a ``RuntimeError`` rather than
    silently leaving the driver at its default cache path.

    Args:
        config: The configuration the renderer is about to be built with. Querying
            the extension loads the bindings, which runs ``ovrtx_initialize`` once
            per process, so the real config has to be supplied here or the log
            sink, log level and keep-alive are silently dropped.
    """
    # Deliberately local, not moved to the top with `ctypes`: this is the only place
    # the ovrtx runtime is touched, so an import here is what lets the rest of this
    # module - and the tests that exercise it - import without the ovrtx runtime
    # installed (see the module docstring).
    from ovrtx import Renderer
    from ovrtx._src import bindings as ovrtx_bindings

    class _ApplySettingsVTable(ctypes.Structure):
        _fields_ = [
            (
                "apply_settings",
                ctypes.CFUNCTYPE(ovrtx_bindings.ovrtx_result_t, ovrtx_bindings.ovx_string_t),
            ),
        ]

    loader = ovrtx_bindings._ovrtx_loader
    if loader._lib is None:
        loader.create_bindings(Renderer._to_c_config(config))
    lib = loader._lib

    lib.ovrtx_query_extension.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ovrtx_query_extension.restype = ovrtx_bindings.ovrtx_result_t

    vtable_ptr = ctypes.c_void_p()
    query = lib.ovrtx_query_extension(_APPLY_SETTINGS_EXT, ctypes.byref(vtable_ptr))
    if query.status != ovrtx_bindings.OVRTX_API_SUCCESS:
        return None

    vtable = ctypes.cast(vtable_ptr, ctypes.POINTER(_ApplySettingsVTable)).contents

    def apply_setting(setting: str) -> bool:
        result = vtable.apply_settings(ovrtx_bindings.ovx_string_t(setting))
        return result.status == ovrtx_bindings.OVRTX_API_SUCCESS

    return apply_setting


def apply_shader_cache_settings(apply_setting: Callable[[str], bool], cache_path: str) -> None:
    """Point every driver shader-cache setting at ``cache_path``.

    Args:
        apply_setting: Applies one setting string, returning whether it took effect.
        cache_path: Directory the driver should write its compiled shaders to.

    Raises:
        RuntimeError: A setting was rejected, so the cache would silently stay at
            its default path despite having been explicitly redirected.
    """
    for setting in SHADER_CACHE_SETTINGS:
        if not apply_setting(f"--{setting}={cache_path}"):
            raise RuntimeError(
                f"{SHADER_CACHE_PATH_ENV} requested {cache_path!r}, but the ovrtx settings"
                f" extension rejected {setting!r}."
            )
    # Only once every setting took effect - a partial redirect still compiles cold.
    logger.info("OVRTX driver shader cache redirected to %r.", cache_path)


def redirect_shader_cache(config: RendererConfig) -> None:
    """Redirect the driver shader cache to :data:`SHADER_CACHE_PATH_ENV` when set.

    Must be called **before** :class:`~ovrtx.Renderer` is constructed: the
    settings are read at renderer-creation time and cannot be changed on a live
    instance. No-ops when the variable is unset, so every construction path can
    call it unconditionally.

    Args:
        config: The configuration the renderer will be constructed with, forwarded
            to :func:`_acquire_settings_applier`.

    Raises:
        RuntimeError: The variable is set but the redirect could not be applied, either
            because a setting was rejected or because this runtime has no settings
            extension. A silent fallback to the default cache path would report the
            same cache-restore hit CI already logged before this call ran, even
            though this run recompiles from scratch at a path that is never mounted
            or published - so an unusable redirect is surfaced the same way a
            rejected setting is, not degraded to a warning.
    """
    cache_path = os.environ.get(SHADER_CACHE_PATH_ENV)
    if not cache_path:
        return

    apply_setting = _acquire_settings_applier(config)
    if apply_setting is None:
        raise RuntimeError(
            f"{SHADER_CACHE_PATH_ENV} requested {cache_path!r}, but this ovrtx runtime has no"
            " settings extension, so the driver shader cache cannot be redirected."
        )

    apply_shader_cache_settings(apply_setting, cache_path)
