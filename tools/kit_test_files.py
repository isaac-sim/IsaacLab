# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List the test files in a directory that can share one Kit app.

The ``kit`` / ``kit_cameras`` / ``kit_solo`` markers already record which files can share a
Kit app; this turns that into the file list a runner needs, so the two never drift. Anything
that hardcodes such a list has to be updated by hand whenever a file is added, renamed, or
reclassified, and a stale list is silently wrong rather than loudly broken.

One profile at a time. ``kit`` and ``kit_cameras`` files cannot share a process in either
direction: cameras cannot be enabled after startup, and a camera-enabled app is not a drop-in
replacement for a plain one because some tests assert that offscreen rendering is off. Each
profile is a separate batch, so the caller asks for one.

Selection: files marked with the requested profile, minus those also marked ``kit_solo`` and
those in :data:`tools.test_settings.TESTS_TO_SKIP`.

Markers are read from the file's source rather than by importing it, because importing a
Kit-dependent test module boots Kit.

Usage::

    python3 tools/kit_test_files.py source/isaaclab/test/sim --profile kit --format paths
    python3 tools/kit_test_files.py source/isaaclab/test/sim --profile kit_cameras --format names
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# `kit` must not match `kit_cameras` or `kit_solo`, hence the boundary on the plain pattern.
_MARK_KIT = re.compile(r"pytest\.mark\.kit(?![\w])")
_MARK_CAMERAS = re.compile(r"pytest\.mark\.kit_cameras\b")
_MARK_SOLO = re.compile(r"pytest\.mark\.kit_solo\b")


def _tests_to_skip() -> frozenset[str]:
    """Names from ``tools/test_settings.py``, which the per-file runner also honours."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from test_settings import TESTS_TO_SKIP  # noqa: PLC0415
    except ImportError:
        return frozenset()
    return frozenset(TESTS_TO_SKIP)


def shareable_test_files(directory: Path, profile: str = "kit") -> list[Path]:
    """Return the files under ``directory`` that can share one Kit app of ``profile``.

    Args:
        directory: Directory to scan, non-recursively matching ``test_*.py``.
        profile: Which launch configuration to select, ``"kit"`` or ``"kit_cameras"``.

    Returns:
        The selected files, sorted by name.

    Raises:
        ValueError: If ``profile`` is not a known launch configuration.
    """
    if profile not in ("kit", "kit_cameras"):
        raise ValueError(f"unknown profile {profile!r}; expected 'kit' or 'kit_cameras'")

    skip = _tests_to_skip()
    selected = []
    for path in sorted(directory.glob("test_*.py")):
        if path.name in skip:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if _MARK_SOLO.search(source):
            continue
        # `kit_cameras` implies the file also matches the plain `kit` pattern's prefix, so
        # classify on the more specific marker first.
        if _MARK_CAMERAS.search(source):
            if profile == "kit_cameras":
                selected.append(path)
        elif _MARK_KIT.search(source) and profile == "kit":
            selected.append(path)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory to scan for test files")
    parser.add_argument(
        "--profile",
        choices=("kit", "kit_cameras"),
        default="kit",
        help="which launch configuration to select; the two never share a process",
    )
    parser.add_argument(
        "--format",
        choices=("paths", "names"),
        default="paths",
        help="'paths' for space-separated repo paths (pytest arguments); "
        "'names' for comma-separated file names (the include-files input)",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    files = shareable_test_files(args.directory, args.profile)
    if not files:
        parser.error(f"no {args.profile} test files found in {args.directory}")

    if args.format == "paths":
        print(" ".join(path.as_posix() for path in files))
    else:
        print(",".join(path.name for path in files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
