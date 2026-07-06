# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Sim-free cProfile parsing for startup benchmarks.

Filter to IsaacLab + first-level external calls and return per-function
own/cum time and call counts.
"""

from __future__ import annotations

import cProfile
import fnmatch
import io
import logging
import os
import pstats

logger = logging.getLogger(__name__)


def parse_cprofile_stats(
    profile: cProfile.Profile,
    isaaclab_prefixes: list[str],
    top_n: int = 30,
    whitelist: list[str] | None = None,
) -> list[tuple[str, float, float, int]]:
    """Parse cProfile stats, filtering to IsaacLab + first-level external calls.

    Walks the pstats data and keeps functions that are either (a) inside an
    IsaacLab source directory, or (b) directly called by an IsaacLab function.
    Results are sorted by own-time (tottime) descending.

    When *whitelist* is provided, only functions whose labels match at least one
    ``fnmatch`` pattern are returned. Patterns that match no profiled function
    emit a ``(pattern, 0.0, 0.0, 0)`` placeholder so dashboards always receive
    consistent keys. The *top_n* parameter is ignored in whitelist mode.

    Args:
        profile: A completed cProfile.Profile instance (after .disable()).
        isaaclab_prefixes: Absolute file path prefixes identifying IsaacLab source
            (e.g. ["/home/user/IsaacLab/source/isaaclab", ...]).
        top_n: Maximum number of functions to return. Ignored when
            *whitelist* is provided.
        whitelist: Optional list of ``fnmatch`` patterns to select specific
            functions (e.g. ``["isaaclab.cloner.*:usd_replicate"]``).

    Returns:
        List of (label, tottime_ms, cumtime_ms, ncalls) tuples sorted by
        tottime descending.
    """
    stats = pstats.Stats(profile, stream=io.StringIO())

    def _is_isaaclab(filename: str) -> bool:
        return any(filename.startswith(prefix) for prefix in isaaclab_prefixes)

    def _make_label(filename: str, funcname: str) -> str:
        # For builtins/C-extensions the filename is something like "~" or "<frozen ...>"
        if not filename or filename.startswith("<") or filename == "~":
            return funcname
        # Convert absolute path to dotted module-style label
        for prefix in sorted(isaaclab_prefixes, key=len, reverse=True):
            if filename.startswith(prefix):
                rel = os.path.relpath(filename, prefix)
                # Strip .py, replace os.sep with dot
                rel = rel.replace(os.sep, ".").removesuffix(".py")
                return f"{rel}:{funcname}"
        # External function — try to find the top-level package name
        # e.g. ".../site-packages/torch/nn/modules/linear.py" -> "torch.nn.modules.linear"
        parts = filename.replace(os.sep, "/").removesuffix(".py").split("/")
        # Find "site-packages" anchor or fall back to last 3 components
        try:
            sp_idx = parts.index("site-packages")
            short = ".".join(parts[sp_idx + 1 :])
        except ValueError:
            short = ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts)
        return f"{short}:{funcname}"

    # NOTE: stats.stats is an internal CPython dict, not part of the public pstats API.
    # The public get_stats_profile() (Python 3.9+) doesn't expose caller info, which
    # we need for the first-level external call filter. If a future Python release
    # breaks this, switch to get_stats_profile() and drop the caller-based filtering.
    results = []
    for func_key, (_pcalls, ncalls, tottime, cumtime, callers) in stats.stats.items():
        filename, _, funcname = func_key
        if _is_isaaclab(filename):
            label = _make_label(filename, funcname)
            results.append((label, tottime * 1000.0, cumtime * 1000.0, ncalls))
        else:
            # Check if any direct caller is an IsaacLab function
            for caller_key in callers:
                caller_filename = caller_key[0]
                if _is_isaaclab(caller_filename):
                    label = _make_label(filename, funcname)
                    results.append((label, tottime * 1000.0, cumtime * 1000.0, ncalls))
                    break

    # Sort by tottime (own-time) descending
    results.sort(key=lambda x: x[1], reverse=True)

    if whitelist is None:
        return results[:top_n]

    # Whitelist mode: filter by fnmatch patterns, emit placeholders for unmatched patterns
    matched: dict[str, tuple[str, float, float, int]] = {}
    matched_patterns: set[str] = set()
    for label, tottime, cumtime, ncalls in results:
        for pattern in whitelist:
            if fnmatch.fnmatch(label, pattern):
                if label not in matched:
                    matched[label] = (label, tottime, cumtime, ncalls)
                matched_patterns.add(pattern)

    # Add 0 placeholders for patterns that matched nothing
    for pattern in whitelist:
        if pattern not in matched_patterns:
            logger.warning(
                "Whitelist pattern '%s' matched no profiled functions. "
                "Check for typos or verify the function ran during this phase.",
                pattern,
            )
            matched[pattern] = (pattern, 0.0, 0.0, 0)

    filtered = list(matched.values())
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
