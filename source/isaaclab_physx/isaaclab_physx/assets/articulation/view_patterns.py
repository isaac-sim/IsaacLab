# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of articulation root expressions into physics-view glob patterns."""

from __future__ import annotations

from collections.abc import Callable


def resolve_view_path_patterns(
    root_prim_path_expr: str, find_matching_prim_paths: Callable[[str], list[str]]
) -> str | list[str]:
    """Resolve an articulation root expression into view-compatible glob pattern(s).

    A root expression with a single wildcard (the environment dimension) maps directly to one
    glob pattern. An expression with additional wildcards (e.g. one articulation per
    sub-asset, ``env_.*/Rig/parts/part_.*``) cannot form a single articulation view, so the
    extra dimensions are expanded into one single-wildcard pattern per distinct sub-asset
    path, keeping only the leading (environment) wildcard. The sub-asset set must be
    identical across environments; the patterns are returned in sorted order so the view
    layout is deterministic.

    Args:
        root_prim_path_expr: Regex expression matching the articulation root prims.
        find_matching_prim_paths: Callable resolving a prim path expression to the list of
            matching prim paths on the stage.

    Returns:
        A single glob pattern, or a list of single-wildcard glob patterns, one per distinct
        sub-asset path.

    Raises:
        RuntimeError: If the expression matches no prims, or the sub-asset set differs
            between environments.
    """
    view_glob = root_prim_path_expr.replace(".*", "*")
    if view_glob.count("*") <= 1:
        return view_glob

    matches = find_matching_prim_paths(root_prim_path_expr)
    if not matches:
        raise RuntimeError(f"No prims match the articulation root expression: {root_prim_path_expr}")

    # split every match at the leading (environment) wildcard and key by the remainder
    glob_tokens = view_glob.split("/")
    env_token_index = next(i for i, token in enumerate(glob_tokens) if "*" in token)
    per_env: dict[str, set[str]] = {}
    for path in matches:
        parts = path.split("/")
        env_name = parts[env_token_index]
        remainder = "/".join(parts[env_token_index + 1 :])
        per_env.setdefault(env_name, set()).add(remainder)

    remainders = sorted(next(iter(per_env.values())))
    for env_name, env_remainders in per_env.items():
        if sorted(env_remainders) != remainders:
            raise RuntimeError(
                "Multi-wildcard articulation root expression resolves to different sub-asset"
                f" sets across environments (environment '{env_name}' differs). All"
                " environments must contain the same articulations for a shared view:"
                f" {root_prim_path_expr}"
            )

    env_glob = "/".join(glob_tokens[: env_token_index + 1])
    return [f"{env_glob}/{remainder}" for remainder in remainders]
