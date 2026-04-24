# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detect ABI-clash risk from multiple distributions shipping overlapping ``pxr/`` files.

When two distributions (e.g. ``usd-core`` and ``usd-exchange``) both ship files
under ``pxr/``, the later-installed package silently overwrites the earlier
one's files on disk while each dist-info's RECORD keeps its own entries. Which
``_*.so`` "wins" on disk depends on install order, wheel selection, and the
pip resolver version, so the same version pins can produce a working import
tree on one machine and crash with ``Tf_PyEnumWrapper has not been created
yet`` on another.

A strict, packaging-level detector is more reliable than trying to observe the
runtime crash: any ``pxr/`` path claimed by more than one distribution is a
potential ABI mismatch, regardless of whether the current env happens to have
picked a working "winner".

Regression for GitHub PR #5386 / issue #5025: ``usd-core==25.8.0`` and
``usd-exchange==2.2.2`` both ship the entire ``pxr/`` tree.
"""

from __future__ import annotations

import importlib.metadata as md
from collections import defaultdict

import pytest


def _pxr_file_owners() -> dict[str, list[str]]:
    """Return a map of ``pxr/`` paths to the distributions that claim to own them."""
    owners: dict[str, set[str]] = defaultdict(set)
    for dist in md.distributions():
        if dist.name is None or dist.files is None:
            continue
        for f in dist.files:
            path = str(f)
            if path == "pxr" or path.startswith("pxr/"):
                owners[path].add(dist.name)
    return {p: sorted(ns) for p, ns in owners.items()}


def test_no_pxr_file_owner_overlap():
    """No path under ``pxr/`` should be claimed by more than one installed distribution."""
    try:
        md.distribution("usd-core")
    except md.PackageNotFoundError:
        pytest.skip("usd-core is not installed in this environment")

    owners = _pxr_file_owners()
    clashes = {p: ns for p, ns in owners.items() if len(ns) > 1}

    if clashes:
        lines = [f"Found {len(clashes)} pxr/ path(s) claimed by multiple distributions:"]
        for path, dist_names in sorted(clashes.items()):
            lines.append(f"  {path} -> {dist_names}")
        lines.append(
            "\nWhichever distribution pip installs last overwrites the other's files on"
            " disk, while each dist-info still references its own binaries. If the two"
            " were built against different USD ABIs, the resulting mix crashes at"
            " import time (e.g. 'Tf_PyEnumWrapper has not been created yet'). See"
            " GitHub PR #5386 / issue #5025 for the usd-core/usd-exchange case."
        )
        raise AssertionError("\n".join(lines))
