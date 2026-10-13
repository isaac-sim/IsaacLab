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
from pathlib import Path

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


def _acquire_settings_applier() -> Callable[[str], bool] | None:
    """Return a callable that applies one ``--/setting=value`` string, or ``None``.

    ``None`` means this runtime does not ship the settings extension;
    :func:`redirect_shader_cache` turns that into a ``RuntimeError`` rather than
    silently leaving the driver at its default cache path.

    Only loads the ovrtx library. ``ovrtx_initialize`` and its configuration are left
    to the first renderer, and settings applied before it still take effect.
    """
    from ovrtx._src import bindings as ovrtx_bindings

    class _ApplySettingsVTable(ctypes.Structure):
        _fields_ = [
            (
                "apply_settings",
                ctypes.CFUNCTYPE(ovrtx_bindings.ovrtx_result_t, ovrtx_bindings.ovx_string_t),
            ),
        ]

    loader = ovrtx_bindings._ovrtx_loader
    lib = loader._lib
    if lib is None:
        hint = ovrtx_bindings.OVRTX_LIBRARY_PATH_HINT
        lib = loader._load_library([Path(hint)] if hint else None)

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


def redirect_shader_cache() -> None:
    """Redirect the driver shader cache to :data:`SHADER_CACHE_PATH_ENV` when set.

    Must be called **before** :class:`~ovrtx.Renderer` is constructed: the
    settings are read at renderer-creation time and cannot be changed on a live
    instance. No-ops when the variable is unset, so every construction path can
    call it unconditionally. Does not initialize ovrtx, so it needs no renderer
    configuration.

    Raises:
        RuntimeError: The variable is set but the runtime has no settings extension or
            rejected a setting; silently compiling at the default path would hide it.
    """
    cache_path = os.environ.get(SHADER_CACHE_PATH_ENV)
    if not cache_path:
        return

    apply_setting = _acquire_settings_applier()
    if apply_setting is None:
        raise RuntimeError(
            f"{SHADER_CACHE_PATH_ENV} requested {cache_path!r}, but this ovrtx runtime has no"
            " settings extension, so the driver shader cache cannot be redirected."
        )

    for setting in SHADER_CACHE_SETTINGS:
        if not apply_setting(f"--{setting}={cache_path}"):
            raise RuntimeError(
                f"{SHADER_CACHE_PATH_ENV} requested {cache_path!r}, but the ovrtx settings"
                f" extension rejected {setting!r}."
            )
    # Only once every setting took effect - a partial redirect still compiles cold.
    logger.info("OVRTX driver shader cache redirected to %r.", cache_path)
