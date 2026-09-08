# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Best-effort ovrtx runtime setup shared by every ovrtx consumer.

Two independent problems show up whenever something (the :class:`~isaaclab_ov.renderers.OVRTXRenderer`
camera backend, or Newton's ``ViewerRTX`` wrapped by ``NewtonRTXVisualizer``) is the *first* code in the
process to load the ``ovrtx`` native library:

- On Linux, the pip-installed ``libovrtx-dynamic.so``'s RPATH does not resolve its own dependencies
  under the pip layout, so a bare ``ctypes.CDLL``/``dlopen`` of it (or of anything that loads it) can
  fail unless ``LD_LIBRARY_PATH``/``OMNI_USD_PLUGINS_BASE_PATH`` point at its ``bin`` directory first.
- ``ovrtx`` ships its own USD schema plugins (e.g. the ``rtx_settings`` plugin defining
  ``OmniRtx*API``), but does not publish their paths to USD's plug registry unless
  ``ovrtx.register_schema_paths()`` is called before that registry's one-shot-per-process
  initialization. Doing this too late (e.g. after a physics backend has already touched ``pxr``)
  leaves the schemas unregistered, and applying them later logs benign but noisy
  ``FindAppliedAPIPrimDefinition(...) returned nothing`` errors.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_ovrtx_runtime() -> None:
    """Patch library search paths and register ovrtx's USD schema paths, if ``ovrtx`` is installed.

    Safe to call unconditionally, repeatedly, and as early as possible in process startup (ideally
    before anything else touches ``pxr``): a no-op when ``ovrtx`` is not installed, and otherwise
    best-effort -- failures (missing native loader library, USD version mismatch, etc.) are logged
    at WARNING and swallowed rather than raised, matching ``ovrtx``'s own opt-in auto-register hook
    (``OVRTX_PXR_SCHEMA_AUTO_REGISTER``). ``ovrtx.register_schema_paths()`` itself is documented as
    first-call-wins and idempotent, so calling this from multiple entry points in the same process
    (e.g. both the launcher and a visualizer constructed without going through it) is safe.
    """
    import importlib.util

    spec = importlib.util.find_spec("ovrtx")
    if spec is None or spec.origin is None:
        return

    if sys.platform.startswith("linux"):
        # Patch environment so ovrtx's CRenderApiLibLoader can find libovrtx.dylib.so.
        # libovrtx-dynamic.so's built-in RPATH uses paths from the original deploy layout
        # which don't match the pip install layout. LD_LIBRARY_PATH (read by glibc at each
        # dlopen call) and OMNI_USD_PLUGINS_BASE_PATH (read by CRenderApiLibLoader) redirect
        # the search to the correct location.
        bin_dir = Path(spec.origin).parent / "bin"
        extra_dirs = os.pathsep.join([str(bin_dir / "plugins" / "rtx"), str(bin_dir / "plugins"), str(bin_dir)])
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if str(bin_dir / "plugins" / "rtx") not in existing:
            os.environ["LD_LIBRARY_PATH"] = extra_dirs + (os.pathsep + existing if existing else "")
        os.environ.setdefault("OMNI_USD_PLUGINS_BASE_PATH", str(bin_dir))

    try:
        import ovrtx

        ovrtx.register_schema_paths()
    except Exception:
        logger.warning(
            "Failed to register ovrtx's USD schema paths. RTX render products may log benign"
            " 'FindAppliedAPIPrimDefinition(...) returned nothing' errors for OmniRtx*API schemas.",
            exc_info=True,
        )
