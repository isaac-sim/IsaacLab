# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detection of source-versus-image dependency skew in a failed benchmark.

The gate bind-mounts Isaac Lab source over a prebuilt CI image, but the image
supplies the installed third-party packages (Newton, Warp, Isaac Sim). Between a
dependency-pin change landing on ``develop`` and the next image publish, a PR's
source can reference a symbol the installed package does not have yet, which
crashes every affected task before any FPS is measured.

That crash says nothing about the PR's performance, so it must not read as a
performance failure. This module recognizes the crash signature so the gate can
report a stale image and stay advisory for the affected tasks instead.

Only packages the *image* installs are eligible. A missing symbol in Isaac Lab's
own source is a genuine defect in the change under test and still fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Packages installed into the CI image rather than bind-mounted from the PR.
IMAGE_PROVIDED_PACKAGES: frozenset[str] = frozenset(
    {
        "carb",
        "isaacsim",
        "mujoco",
        "mujoco_warp",
        "newton",
        "omni",
        "pxr",
        "warp",
    }
)

# Python spells "this name is not in the installed package" three ways.
_MISSING_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ImportError: cannot import name ['\"](?P<symbol>\w+)['\"] from ['\"](?P<module>[\w.]+)['\"]"),
    re.compile(r"ModuleNotFoundError: No module named ['\"](?P<module>[\w.]+)['\"]"),
    re.compile(r"AttributeError: module ['\"](?P<module>[\w.]+)['\"] has no attribute ['\"](?P<symbol>\w+)['\"]"),
)


@dataclass(frozen=True)
class DependencySkew:
    """One detected mismatch between the PR's source and the image's packages."""

    package: str
    module: str
    symbol: str | None

    def describe(self) -> str:
        """Return a one-line reviewer-facing description of the mismatch."""
        if self.symbol:
            return f"`{self.module}` in the CI image has no `{self.symbol}`"
        return f"`{self.module}` is not installed in the CI image"


def detect_dependency_skew(log_text: str | None) -> DependencySkew | None:
    """Return the dependency skew a benchmark log indicates, if any.

    Args:
        log_text: Captured benchmark output, typically ``BenchResult.stdout_tail``.

    Returns:
        The detected mismatch, or ``None`` when the log shows no missing symbol
        from an image-provided package.
    """
    if not log_text:
        return None
    for pattern in _MISSING_NAME_PATTERNS:
        match = pattern.search(log_text)
        if match is None:
            continue
        module = match.group("module")
        package = module.split(".", 1)[0]
        if package not in IMAGE_PROVIDED_PACKAGES:
            continue
        return DependencySkew(package=package, module=module, symbol=match.groupdict().get("symbol"))
    return None
