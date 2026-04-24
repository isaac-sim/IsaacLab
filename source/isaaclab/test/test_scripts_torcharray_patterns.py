# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static scanner ensuring scripts/ does not regress the ProxyArray migration.

Every public ``.data.<field>`` on an asset or sensor now returns a
:class:`~isaaclab.utils.warp.ProxyArray` (or, for CameraData, a raw
``torch.Tensor``) — **never** a ``wp.array``. Wrapping such an access
in :func:`wp.to_torch` therefore fails at runtime with a confusing
``TypeError``. This test regex-scans ``scripts/`` for the pattern and
flags it before it can reach users running tutorials or demos.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Matches:
#   wp.to_torch(<chain>.data.<field>)
# where <chain> can be <name>, <name>[...], <name>.<name>, <name>.<name>[...], etc.
# Anchored so we don't match tool scripts that document the pattern as prose.
_WP_TO_TORCH_DOT_DATA = re.compile(
    r"wp\.to_torch\(\s*"
    r"[a-zA-Z_][a-zA-Z_0-9]*"  # first name
    r"(?:\[[^\]\[]*\]|\.[a-zA-Z_][a-zA-Z_0-9]*)*"  # chain of [..] or .name
    r"\.data\."  # .data.
    r"[a-zA-Z_][a-zA-Z_0-9]*"  # field
    r"\s*\)"
)

# Matches:
#   wp.to_torch(<name>_data.<field>)
# Catches the aliased pattern like ``object_data: RigidObjectData = ...`` followed
# by ``wp.to_torch(object_data.root_pos_w)``.
_WP_TO_TORCH_NAME_DATA = re.compile(
    r"wp\.to_torch\(\s*"
    r"[a-zA-Z_][a-zA-Z_0-9]*_data\."
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    r"\s*\)"
)

# scripts/tools/wrap_warp_to_torch.py is the migration utility and documents
# the old pattern inside strings. Exclude it from the scan.
_EXCLUDE = {"tools/wrap_warp_to_torch.py"}


def _repo_root() -> Path:
    # this test lives at source/isaaclab/test/test_scripts_torcharray_patterns.py
    # parents[0]=test, [1]=isaaclab, [2]=source, [3]=repo root
    return Path(__file__).resolve().parents[3]


def _scripts_files() -> list[Path]:
    scripts = _repo_root() / "scripts"
    return sorted(p for p in scripts.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _scripts_files(), ids=lambda p: str(p.relative_to(_repo_root())))
def test_no_wp_to_torch_on_torcharray_data(path: Path) -> None:
    """No ``wp.to_torch(<x>.data.<field>)`` / ``wp.to_torch(<x>_data.<field>)`` in scripts/.

    Post-migration, ``<asset>.data.<field>`` returns a ``ProxyArray``
    (or ``torch.Tensor`` for CameraData); passing either to ``wp.to_torch``
    raises ``TypeError`` because it expects ``wp.array``. Use the
    ``.torch`` accessor instead (or omit the wrap entirely for
    torch-native fields).
    """
    rel = path.relative_to(_repo_root()).as_posix()
    if any(rel.endswith(suffix) for suffix in _EXCLUDE):
        pytest.skip(f"{rel} is excluded from the ProxyArray hygiene scan")

    text = path.read_text(encoding="utf-8")

    offenders: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _WP_TO_TORCH_DOT_DATA.search(line) or _WP_TO_TORCH_NAME_DATA.search(line):
            offenders.append(f"{rel}:{i}: {line.rstrip()}")

    if offenders:
        pytest.fail(
            "Found wp.to_torch(...) calls on a migrated ProxyArray data accessor. "
            "Use .torch instead of wp.to_torch(...) (see isaaclab 4.6.15 CHANGELOG).\n" + "\n".join(offenders)
        )
