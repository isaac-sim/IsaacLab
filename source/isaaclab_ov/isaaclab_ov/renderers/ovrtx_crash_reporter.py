# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse crash-report upload for the OVRTX renderer.

Kept apart from ``ovrtx_renderer`` for the same reason as ``ovrtx_shader_cache``:
the policy imports without the ovrtx runtime and can be tested on any runner, and
the runtime is only touched inside :func:`_acquire_settings_applier`.

The settings below are carb settings, so they only reach the process that applies
them. That is why this runs inside the renderer's own process rather than from a
CI setup step: a crash reporter configured in a short-lived helper dies with it,
leaving the process that actually renders - and actually crashes - unconfigured.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovrtx import RendererConfig

logger = logging.getLogger(__name__)

# Runtime environment variable opting a process into uploading its crash dumps.
# Unset leaves the crash reporter alone, which is the default everywhere except
# the develop-gated wheelhouse jobs in .github/workflows/build.yaml.
CRASH_UPLOAD_ENV = "ISAACLAB_OVRTX_CRASH_UPLOAD"

CRASH_REPORT_URL = "https://services.nvidia.com/submit"
"""Endpoint the Omniverse crash reporter submits minidumps to."""

CRASH_REPORT_PRODUCT = "Omniverse.ovrtx"
"""Product uploaded reports are filed under."""


def crash_report_settings(ovrtx_version: str) -> tuple[str, ...]:
    """Return the crash-reporter settings that enable upload.

    Args:
        ovrtx_version: Version recorded on each uploaded report, so a dump can be
            attributed to the runtime that produced it.

    Returns:
        One ``--/setting=value`` string per setting, in the order they are applied.
    """
    return (
        "--/crashreporter/enabled=true",
        f'--/crashreporter/url="{CRASH_REPORT_URL}"',
        f'--/crashreporter/product="{CRASH_REPORT_PRODUCT}"',
        f'--/crashreporter/version="{ovrtx_version}"',
        # Keep the dump on disk as well as uploading it, so a job artifact can still carry it.
        "--/crashreporter/preserveDump=true",
        # Nothing is attended here, so never block a crashing run on a user-story prompt.
        "--/crashreporter/gatherUserStory=false",
        # A CI runner is not an end-user machine and no one is present to consent,
        # so upload unconditionally rather than honouring local privacy state.
        "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=true",
        "--/crashreporter/alwaysUpload=true",
    )


def _acquire_settings_applier(config: RendererConfig) -> Callable[[str], None] | None:
    """Return a callable that applies one ``--/setting=value`` string, or ``None``.

    ``None`` means ``ovrtx_extensions`` is not installed; it ships on NVIDIA's
    internal index rather than PyPI, so that is the normal case everywhere the
    wheelhouse has not run. :func:`enable_crash_upload` turns it into a
    ``RuntimeError`` only when upload was explicitly asked for.

    Args:
        config: The configuration the renderer is about to be built with. Applying
            a setting can be what loads the bindings, which runs ``ovrtx_initialize``
            once per process, and ``ovrtx_extensions`` has no config of its own to
            hand it - so the real one is loaded here first, or the renderer silently
            loses its log sink, log level and keep-alive.
    """
    # Deliberately local, mirroring ovrtx_shader_cache: these imports are the only
    # place the ovrtx runtime is touched, which is what lets this module - and the
    # tests that exercise it - import without ovrtx installed.
    try:
        from ovrtx_extensions import apply_settings
    except ImportError:
        return None

    from ovrtx import Renderer
    from ovrtx._src import bindings as ovrtx_bindings

    loader = ovrtx_bindings._ovrtx_loader
    if loader._lib is None:
        loader.create_bindings(Renderer._to_c_config(config))

    return apply_settings


def apply_crash_report_settings(apply_setting: Callable[[str], None], ovrtx_version: str) -> None:
    """Apply every crash-reporter setting through ``apply_setting``.

    Args:
        apply_setting: Applies one setting string.
        ovrtx_version: Version recorded on each uploaded report.
    """
    for setting in crash_report_settings(ovrtx_version):
        apply_setting(setting)
    logger.info("OVRTX crash-report upload enabled for ovrtx %s as %r.", ovrtx_version, CRASH_REPORT_PRODUCT)


def enable_crash_upload(config: RendererConfig) -> None:
    """Enable crash-report upload when :data:`CRASH_UPLOAD_ENV` is set.

    Must be called **before** :class:`~ovrtx.Renderer` is constructed, so the
    reporter is configured before there is anything to crash. No-ops when the
    variable is unset, so every construction path can call it unconditionally.

    Args:
        config: The configuration the renderer will be constructed with, forwarded
            to :func:`_acquire_settings_applier`.

    Raises:
        RuntimeError: The variable is set but ``ovrtx_extensions`` is missing. The
            variable is only set once the wheelhouse reports having installed that
            package, so a missing import means a broken install rather than an
            unsupported runtime - and a silent skip would leave a run looking like
            it was uploading dumps it never collected.
    """
    if not os.environ.get(CRASH_UPLOAD_ENV):
        return

    apply_setting = _acquire_settings_applier(config)
    if apply_setting is None:
        raise RuntimeError(
            f"{CRASH_UPLOAD_ENV} requested crash-report upload, but ovrtx_extensions is not"
            " installed, so the crash reporter cannot be configured."
        )

    apply_crash_report_settings(apply_setting, version("ovrtx"))
