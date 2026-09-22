# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static scanner ensuring scripts/source do not regress the ProxyArray migration.

Every public ``.data.<field>`` on an asset or sensor now returns a
:class:`~isaaclab.utils.warp.ProxyArray` (or, for CameraData, a raw
``torch.Tensor``). Legacy conversion or tensor-method callsites on these
properties should migrate to explicit ``.torch`` or ``.warp`` access. These
tests regex-scan user-facing scripts/source files and flag regressions before
they reach users running tutorials or demos.
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

# Matches:
#   <chain>.data.<field>.clone()
#   <chain>.data.<field>[...].clone()
#   <chain>.data.<field>.assign(...)
# These are tensor/wp.array instance methods that ProxyArray intentionally does
# not forward. ``data.output[...]`` is camera data and remains torch-native.
_PROXYARRAY_DIRECT_METHOD_DOT_DATA = re.compile(
    r"\.data\."
    r"(?!_)"  # ignore private backing buffers such as data._sim_bind_...
    r"(?!output\b)"  # camera output dict is torch-native
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    r"(?:\[[^\]\[]*\])?"
    r"\.(?:clone|assign)\s*\("
)

_EXCLUDE_PREFIXES = ("source/isaaclab_contrib/",)


_REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def _python_files(*roots: str) -> list[Path]:
    """Return the Python files below the given repository directories, excluding the contrib package."""
    return sorted(
        path
        for root in roots
        for path in (_REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts and not path.relative_to(_REPO_ROOT).as_posix().startswith(_EXCLUDE_PREFIXES)
    )


def _offending_lines(files: list[Path], patterns: tuple[re.Pattern, ...]) -> list[str]:
    """Return ``path:line: text`` entries for every line matching one of the patterns."""
    offenders = []
    for path in files:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                offenders.append(f"{rel}:{number}: {line.rstrip()}")
    return offenders


def test_no_wp_to_torch_on_torcharray_data() -> None:
    """No ``wp.to_torch(<x>.data.<field>)`` / ``wp.to_torch(<x>_data.<field>)`` in scripts/.

    Post-migration, ``<asset>.data.<field>`` returns a ``ProxyArray`` (or ``torch.Tensor`` for
    CameraData). The temporary ``wp.to_torch`` shim is deprecated, so use the ``.torch`` accessor
    instead (or omit the wrap entirely for torch-native fields).
    """
    offenders = _offending_lines(_python_files("scripts"), (_WP_TO_TORCH_DOT_DATA, _WP_TO_TORCH_NAME_DATA))
    assert not offenders, (
        "Found wp.to_torch(...) calls on a migrated ProxyArray data accessor. "
        "Use .torch instead of wp.to_torch(...) (see isaaclab 4.6.15 CHANGELOG).\n" + "\n".join(offenders)
    )


def test_no_direct_proxyarray_data_methods() -> None:
    """No direct tensor/wp.array methods on migrated ``<x>.data.<field>`` accessors in scripts/ and source/."""
    offenders = _offending_lines(_python_files("scripts", "source"), (_PROXYARRAY_DIRECT_METHOD_DOT_DATA,))
    assert not offenders, (
        "Found direct tensor/wp.array methods on migrated ProxyArray data accessors. "
        "Use .torch.clone() for tensor copies or .warp.assign(...) for warp writes.\n" + "\n".join(offenders)
    )
