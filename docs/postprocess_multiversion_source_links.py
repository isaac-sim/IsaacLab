# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-process multi-version docs source links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ENVIRONMENTS_PAGE = "source/overview/environments.html"
_LEGACY_TASK_SOURCE_LINK = re.compile(
    r'href="(?P<href>(?:\.\./)+(?P<target>source/isaaclab_tasks/[^"#?]*)(?P<suffix>[#?][^"]*)?)"'
)
_GITHUB_TASK_SOURCE_LINK = re.compile(
    r'href="https://github\.com/isaac-sim/IsaacLab/(?:blob|tree)/(?P<ref>.+?)/'
    r'(?P<target>source/isaaclab_tasks/[^"#?]*)(?P<suffix>[#?][^"]*)?"'
)
_DIGIT_TRACKING_LOCO_MANIP_UNDERSCORE_TAGS = {"v2.1.1", "v2.2.0", "v2.2.1"}
_DIGIT_TRACKING_LOCO_MANIP_COMPACT_TAGS = {"v2.3.0", "v2.3.1"}


def _github_url(version: str, target: str, suffix: str) -> str:
    """Return the GitHub URL for a repository source target."""
    source_type = "blob" if Path(target).suffix else "tree"
    return f"https://github.com/isaac-sim/IsaacLab/{source_type}/{version}/{target}{suffix}"


def _normalize_target(version: str, target: str) -> str:
    """Return the historical source target for generated legacy links."""
    if version in _DIGIT_TRACKING_LOCO_MANIP_UNDERSCORE_TAGS:
        target = target.replace(
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/tracking/config/digit/",
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/loco_manipulation/tracking/config/digit/",
        )
    if version in _DIGIT_TRACKING_LOCO_MANIP_COMPACT_TAGS:
        target = target.replace(
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/loco_manipulation/tracking/config/digit/",
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/",
        )
    return target


def _version_name(page: Path, build_dir: Path) -> str:
    """Return the Sphinx-multiversion output version for an environments page."""
    relative_page = page.relative_to(build_dir).as_posix()
    return relative_page[: -len(f"/{_ENVIRONMENTS_PAGE}")]


def _rewrite_page(page: Path, build_dir: Path) -> int:
    """Rewrite legacy relative task source links in an environments page."""
    version = _version_name(page, build_dir)
    text = page.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        target = _normalize_target(version, match.group("target"))
        suffix = match.group("suffix") or ""
        return f'href="{_github_url(version, target, suffix)}"'

    updated_text, replacement_count = _LEGACY_TASK_SOURCE_LINK.subn(replace, text)
    updated_text, github_replacement_count = _GITHUB_TASK_SOURCE_LINK.subn(replace, updated_text)
    if replacement_count or github_replacement_count:
        page.write_text(updated_text, encoding="utf-8")
    return replacement_count + github_replacement_count


def main() -> int:
    """Rewrite legacy source links in generated multi-version environments pages."""
    build_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_build").resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"Build directory does not exist: {build_dir}")

    total_count = 0
    for page in build_dir.glob(f"**/{_ENVIRONMENTS_PAGE}"):
        total_count += _rewrite_page(page, build_dir)

    print(f"Rewrote {total_count} legacy task source links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
