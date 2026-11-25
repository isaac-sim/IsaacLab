# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pin links and clone commands in generated 2.x documentation to their release tag."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_VERSION_PATTERN = re.compile(r"^v2\.\d+\.\d+(?:-[A-Za-z0-9.]+)?$")
_ISAACLAB_REPOSITORIES = (
    "git@github.com:isaac-sim/IsaacLab.git",
    "https://github.com/isaac-sim/IsaacLab.git",
)
_PYGMENTS_SPACE = '<span class="w"> </span>'


def _pin_html(html: str, version: str) -> str:
    """Return historical HTML with rolling-branch references pinned to ``version``."""
    replacements = {
        "https://isaac-sim.github.io/IsaacLab/main/": f"https://isaac-sim.github.io/IsaacLab/{version}/",
        "https://github.com/isaac-sim/IsaacLab/blob/main/": f"https://github.com/isaac-sim/IsaacLab/blob/{version}/",
        "https://github.com/isaac-sim/IsaacLab/tree/main/": f"https://github.com/isaac-sim/IsaacLab/tree/{version}/",
        "https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/": (
            f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{version}/"
        ),
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    for repository in _ISAACLAB_REPOSITORIES:
        html = html.replace(
            f"git clone {repository}",
            f"git clone --branch {version} {repository}",
        )
        html = html.replace(
            f"git{_PYGMENTS_SPACE}clone{_PYGMENTS_SPACE}{repository}",
            (
                f"git{_PYGMENTS_SPACE}clone{_PYGMENTS_SPACE}--branch{_PYGMENTS_SPACE}"
                f"{version}{_PYGMENTS_SPACE}{repository}"
            ),
        )
    return html


def pin_historical_versions(build_dir: Path) -> int:
    """Pin rolling references in every generated 2.x version and return the changed file count."""
    changed = 0
    for version_dir in sorted(path for path in build_dir.iterdir() if path.is_dir()):
        version = version_dir.name
        if not _VERSION_PATTERN.fullmatch(version):
            continue
        for path in version_dir.rglob("*.html"):
            html = path.read_text(encoding="utf-8")
            pinned = _pin_html(html, version)
            if pinned != html:
                path.write_text(pinned, encoding="utf-8")
                changed += 1
    return changed


def main() -> None:
    """Pin the generated historical documentation under a Sphinx build directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="Sphinx multi-version build directory")
    args = parser.parse_args()
    if not args.build_dir.is_dir():
        parser.error(f"build directory does not exist: {args.build_dir}")
    changed = pin_historical_versions(args.build_dir)
    print(f"Pinned rolling references in {changed} historical documentation files.")


if __name__ == "__main__":
    main()
