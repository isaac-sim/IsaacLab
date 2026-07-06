# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Choose whether Testmon can safely select affected tests for a change."""

from __future__ import annotations

import re
import sys

_RELEVANT = re.compile(
    r"^(?:source|docker|tools|apps|scripts)/"
    r"|^\.github/(?:workflows/[^/]+\.ya?ml|actions/)"
    r"|^\.gitmodules$"
    r"|^[^/]+\.(?:toml|yaml|yml|json|ini|cfg|conf|lock|sh|bat|ps1)$"
)
_IGNORED_SUFFIXES = (".md", ".rst", ".skip")


def _is_tracked_python(path: str) -> bool:
    """Return whether ``path`` is a Python source file whose dependencies Testmon tracks.

    Python files under ``.github/`` (e.g. action helper scripts) run outside the pytest
    process, so Testmon has no dependency data for them; changes to those files must fall
    back to a full collection rather than the Python-only fast path.
    """
    return path.endswith(".py") and not path.startswith(".github/")


def select_testmon_mode(paths: list[str]) -> str:
    """Return ``select`` for tracked Python-only changes, otherwise ``collect``."""
    relevant = [path for path in paths if _RELEVANT.search(path) and not path.endswith(_IGNORED_SUFFIXES)]
    return "select" if not relevant or all(_is_tracked_python(path) for path in relevant) else "collect"


if __name__ == "__main__":
    print(select_testmon_mode([line.strip() for line in sys.stdin if line.strip()]))
