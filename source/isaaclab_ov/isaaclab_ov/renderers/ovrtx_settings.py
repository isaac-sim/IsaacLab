# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forwarding of Isaac Lab RTX settings into the OVRTX renderer's Carbonite instance.

The OVRTX backend runs without Kit, so :class:`~isaaclab.app.settings_manager.SettingsManager` has no
``carb.settings`` to delegate to and records values in a plain dictionary instead. OVRTX brings up its
own Carbonite instance, and ``ovrtx`` 0.4 exposes no typed Python API for arbitrary RTX setting paths
(:class:`ovrtx.RendererConfig` covers only its own closed set of options). The one route that reaches
those paths is the ``ovrtx.settings.apply_settings`` extension, obtained through the public
``ovrtx_query_extension`` entry point and invoked with Kit-style ``--/path=value`` tokens.

That extension is internal and unsupported: ``ovrtx.h`` documents ``ovrtx_query_extension`` as public
while stating that all extension names and vtable contracts are internal. It is used here because the
only alternative is requiring users to export ``OVRTX_*`` environment variables before starting the
process. Every call is therefore best-effort — a missing or changed extension degrades to a warning,
never an exception.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLY_SETTINGS_EXTENSION = b"ovrtx.settings.apply_settings"
_OVRTX_API_SUCCESS = 0


def _format_value(value: Any) -> str:
    """Format a setting value as a Kit command-line token value.

    Carbonite auto-types the token, so booleans must be spelled ``true``/``false``: ``0``/``1`` parse
    as integers instead, which only happen to coerce to ``bool`` for bool-typed settings.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def query_apply_settings_fn() -> Any | None:
    """Look up the OVRTX settings extension.

    Returns:
        The extension's ``apply_settings`` function, or None if this OVRTX build does not expose it.
    """
    from ovrtx._src import bindings as ovrtx_bindings

    class _ApplySettingsVTable(ctypes.Structure):
        """Mirror of ``ovrtx_extension_apply_settings_t``: one function taking ``ovx_string_t``."""

        _fields_ = [("apply_settings", ctypes.CFUNCTYPE(ovrtx_bindings.ovrtx_result_t, ovrtx_bindings.ovx_string_t))]

    # Loading the loader library does not initialize OVRTX; the extension is documented to be callable
    # before the runtime is up, which is exactly when settings must be queued.
    lib = ovrtx_bindings._ovrtx_loader._load_library()
    lib.ovrtx_query_extension.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ovrtx_query_extension.restype = ovrtx_bindings.ovrtx_result_t

    vtable_ptr = ctypes.c_void_p()
    result = lib.ovrtx_query_extension(_APPLY_SETTINGS_EXTENSION, ctypes.byref(vtable_ptr))
    if result.status.value != _OVRTX_API_SUCCESS or not vtable_ptr.value:
        return None
    return ctypes.cast(vtable_ptr, ctypes.POINTER(_ApplySettingsVTable)).contents.apply_settings


def apply_carb_settings(settings: dict[str, Any]) -> bool:
    """Apply Carbonite settings to the OVRTX runtime.

    Must be called before the first :class:`ovrtx.Renderer` is constructed in the process: OVRTX
    caches the queued values and flushes them once, while preparing the Carbonite settings system
    during the first renderer's bring-up. Values queued afterwards do not take effect.

    Args:
        settings: Setting ``path -> value`` pairs, with paths in Carbonite form (e.g.
            ``"/rtx/rtpt/gaussian/skipTonemapping/enabled"``).

    Returns:
        True if the settings were queued, False if they could not be applied.
    """
    if not settings:
        return True

    tokens = " ".join(f"--{path}={_format_value(value)}" for path, value in settings.items())
    # The whole mechanism reaches through an internal extension and private ctypes bindings, so any
    # failure mode it can produce — a moved binding, a changed vtable, a rejected argument — must
    # degrade to a warning rather than break the renderer that is being constructed.
    try:
        from ovrtx._src import bindings as ovrtx_bindings

        apply_settings = query_apply_settings_fn()
        if apply_settings is None:
            logger.warning(
                "This ovrtx build does not expose the %s extension, so the RTX settings %s were not applied."
                " Export them as OVRTX_<path with underscores> environment variables instead.",
                _APPLY_SETTINGS_EXTENSION.decode(),
                sorted(settings),
            )
            return False

        apply_result = apply_settings(ovrtx_bindings.ovx_string_t(tokens))
        if apply_result.status.value != _OVRTX_API_SUCCESS:
            logger.warning("ovrtx rejected the RTX settings '%s' (status %s).", tokens, apply_result.status.value)
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not apply the RTX settings '%s' to ovrtx: %s", tokens, exc)
        return False

    logger.info("Applied RTX settings to ovrtx: %s", tokens)
    return True


def apply_pending_rtx_settings() -> None:
    """Forward every ``/rtx/`` setting recorded by Isaac Lab's settings manager to OVRTX.

    Sensors record backend-agnostic RTX settings through
    :class:`~isaaclab.app.settings_manager.SettingsManager` before the renderer exists — for example
    :class:`~isaaclab.sensors.camera.Camera` disables Gaussian tonemapping when an ISP or the HDR AOV
    needs scene-referred Gaussian radiance. Called from the renderer's constructor, before the OVRTX
    renderer is created, so those writes reach the RTX runtime.
    """
    from isaaclab.app.settings_manager import get_settings_manager

    apply_carb_settings(get_settings_manager().get_with_prefix("/rtx/"))
