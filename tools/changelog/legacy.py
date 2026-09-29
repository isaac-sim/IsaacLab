# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WAR: split pre-towncrier fragments into towncrier fragments.

A legacy fragment ``<slug>[.minor|.major].rst`` holds ``^``-underlined Added/Changed/Deprecated/
Removed/Fixed sections. ``cli.py`` still accepts and compiles it so open PRs need no migration.
Delete this module, and its uses in ``cli.py``, once no open PR carries the legacy format.
"""

from __future__ import annotations

import re
from pathlib import Path

SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed")
LEGACY_RE = re.compile(r"^(?P<slug>[^./]+)(?:\.(?P<tier>minor|major))?\.rst$")
HEADING_RE = re.compile(r"^(\S[^\n]*)\n\^+[ \t]*\n", re.MULTILINE)


def split_legacy(path: Path) -> dict[str, str]:
    """Return the towncrier fragments ``{name: text}`` equivalent to a legacy fragment.

    Each section becomes ``<slug>.<section>.rst`` and the tier an empty ``<slug>.minor``/``.major``.

    Raises:
        ValueError: If the file has no sections, or a section is unknown or repeated.
    """
    match = LEGACY_RE.match(path.name)
    parts = HEADING_RE.split(path.read_text(encoding="utf-8"))
    if parts[0].strip() or len(parts) < 3:
        raise ValueError(f"expected sections {', '.join(SECTIONS)} underlined with ^")
    fragments = {f"{match['slug']}.{match['tier']}": ""} if match["tier"] else {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        name = f"{match['slug']}.{heading.lower()}.rst"
        if heading not in SECTIONS or name in fragments:
            raise ValueError(f"unknown or repeated section {heading!r}")
        fragments[name] = body.strip("\n") + "\n"
    return fragments
