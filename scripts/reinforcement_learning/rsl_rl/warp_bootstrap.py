# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Warp/newton bootstrap for training with warp_renderer.

Isaac Sim requires that no omniverse/pxr modules are imported before
SimulationApp(...). Importing torch/warp/newton before AppLauncher can pull in
pxr (e.g. via torch or warp deps) and triggers "extension class wrapper has not
been created yet" errors. So we do not load warp/newton here.

When using warp_renderer, Newton needs standalone warp (with DeviceLike). That
typically requires the env's warp to be loaded before Isaac Sim's bundled warp.
Without an early load, the warp renderer will raise a clear ImportError when
used. See newton_warp_renderer and the task docs for workarounds (e.g. run
without Isaac Sim, or use an RTX renderer instead).
"""

from __future__ import annotations


def ensure_warp_newton_ready() -> None:
    """No-op: loading warp/newton before AppLauncher would break Isaac Sim (pxr load order)."""
    pass
