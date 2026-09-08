# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn on Omniverse crash-report upload for the OVRTX runtime.

Run inside the test container right after the wheelhouse install, which is the
only place ``ovrtx_extensions`` exists: it is published to NVIDIA's internal
index rather than PyPI, so it reaches nothing but the develop-gated wheelhouse
jobs in ``.github/workflows/build.yaml``. Every other job, and every external
checkout, never calls this.

The strings below are Omniverse crash-reporter carb settings.
``ovrtx_extensions.apply_settings`` hands them to the runtime the same way the
``ovrtx.settings.apply_settings`` extension does for the shader-cache redirect
in ``isaaclab_ov.renderers.ovrtx_shader_cache``.

Deliberately its own process. Settings apply per process, and applying one is
what first loads the ovrtx library -- ``ovrtx_initialize`` runs once, with the
config of whichever caller loads it first (see ``ovrtx._src.bindings``). Running
this ahead of the renderer inside pytest would therefore initialize the runtime
with a config that is not ``OVRTXRenderer``'s, silently dropping the log file
path, log level and keep-alive that ``tools/ovrtx_log.py`` reads back per test.

Lives under ``tools/`` like ``verify_ovrtx_shader_cache`` because
``.dockerignore`` excludes ``.github/``.
"""

from __future__ import annotations

import sys
from importlib.metadata import version

CRASH_REPORT_URL = "https://services.nvidia.com/submit"
"""Endpoint the Omniverse crash reporter submits minidumps to."""

CRASH_REPORT_PRODUCT = "Omniverse.ovrtx"
"""Product uploaded reports are filed under."""


def crash_report_settings(ovrtx_version: str) -> tuple[str, ...]:
    """Return the ``--/crashreporter/...`` settings that enable upload.

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
        # A CI runner is not an end-user machine and no one is present to give
        # consent, so upload unconditionally rather than honouring local privacy state.
        "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=true",
        "--/crashreporter/alwaysUpload=true",
    )


def main() -> int:
    # Only called once the wheelhouse claims to have installed the package, so a
    # failed import means a broken wheelhouse rather than an unsupported job.
    try:
        from ovrtx_extensions import apply_settings
    except ImportError as exc:
        print(f"[enable_ovrtx_crash_upload] ovrtx_extensions is not importable: {exc}", file=sys.stderr)
        return 1

    ovrtx_version = version("ovrtx")
    for setting in crash_report_settings(ovrtx_version):
        apply_settings(setting)

    print(f"[enable_ovrtx_crash_upload] crash upload enabled for ovrtx {ovrtx_version} as {CRASH_REPORT_PRODUCT!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
