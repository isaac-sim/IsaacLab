# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List the test files in a directory that can share one Kit app, in a safe order.

The ``kit`` / ``kit_cameras`` / ``kit_solo`` markers already record which files can share a
Kit app; this turns that into the file list a runner needs, so the two never drift. Anything
that hardcodes such a list has to be updated by hand whenever a file is added, renamed, or
reclassified, and a stale list is silently wrong rather than loudly broken.

Selection: every file marked ``kit`` or ``kit_cameras``, minus those marked ``kit_solo`` and
those in :data:`tools.test_settings.TESTS_TO_SKIP`.

Order: ``kit_cameras`` files first. A camera-enabled app can serve tests that do not need
cameras, but cameras cannot be enabled after startup, so a plain ``kit`` file booting first
would make a later ``launch_kit(cameras=True)`` raise.

Markers are read from the file's source rather than by importing it, because importing a
Kit-dependent test module boots Kit.

Usage::

    python3 tools/kit_test_files.py source/isaaclab/test/sim --format paths
    python3 tools/kit_test_files.py source/isaaclab/test/sim --format names
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


def shareable_test_files(directory: Path) -> list[Path]:
    """Return the files under ``directory`` that can share a Kit app, cameras first.

    Args:
        directory: Directory to scan, non-recursively matching ``test_*.py``.

    Returns:
        The selected files: ``kit_cameras`` ones first, each group sorted by name.
    """
    skip = _tests_to_skip()
    cameras, plain = [], []
    for path in sorted(directory.glob("test_*.py")):
        if path.name in skip:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if _MARK_SOLO.search(source):
            continue
        if _MARK_CAMERAS.search(source):
            cameras.append(path)
        elif _MARK_KIT.search(source):
            plain.append(path)
    return cameras + plain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory to scan for test files")
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

    files = shareable_test_files(args.directory)
    if not files:
        parser.error(f"no Kit-marked test files found in {args.directory}")

    if args.format == "paths":
        print(" ".join(path.as_posix() for path in files))
    else:
        print(",".join(path.name for path in files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
