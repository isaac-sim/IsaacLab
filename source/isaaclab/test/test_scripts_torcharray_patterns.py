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


def _find_offenders(root: Path, scan_dirs: tuple[str, ...], patterns: tuple[re.Pattern, ...]) -> list[str]:
    """Return ``path:line: text`` for every in-scope line matching any pattern."""
    # A missing directory would make the scan pass vacuously.
    assert all((root / d).is_dir() for d in scan_dirs), f"scan directories missing under {root}: {scan_dirs}"
    offenders: list[str] = []
    for path in sorted(p for d in scan_dirs for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(_EXCLUDE_PREFIXES):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                offenders.append(f"{rel}:{i}: {line.rstrip()}")
    return offenders


def test_no_wp_to_torch_on_torcharray_data(source_checkout_root: Path) -> None:
    """No ``wp.to_torch(<x>.data.<field>)`` / ``wp.to_torch(<x>_data.<field>)`` in scripts/.

    Post-migration, ``<asset>.data.<field>`` returns a ``ProxyArray``
    (or ``torch.Tensor`` for CameraData). The temporary ``wp.to_torch``
    shim is deprecated, so use the ``.torch`` accessor instead (or omit
    the wrap entirely for torch-native fields).
    """
    offenders = _find_offenders(source_checkout_root, ("scripts",), (_WP_TO_TORCH_DOT_DATA, _WP_TO_TORCH_NAME_DATA))

    assert not offenders, (
        "Found wp.to_torch(...) calls on a migrated ProxyArray data accessor. "
        "Use .torch instead of wp.to_torch(...) (see isaaclab 4.6.15 CHANGELOG).\n" + "\n".join(offenders)
    )


def test_no_direct_proxyarray_data_methods(source_checkout_root: Path) -> None:
    """No direct tensor/wp.array methods on migrated ``<x>.data.<field>`` accessors."""
    offenders = _find_offenders(source_checkout_root, ("scripts", "source"), (_PROXYARRAY_DIRECT_METHOD_DOT_DATA,))

    assert not offenders, (
        "Found direct tensor/wp.array methods on migrated ProxyArray data accessors. "
        "Use .torch.clone() for tensor copies or .warp.assign(...) for warp writes.\n" + "\n".join(offenders)
    )
