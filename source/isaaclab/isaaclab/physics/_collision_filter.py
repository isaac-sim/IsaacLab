# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral compilation of declarative collision-filter semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .physics_manager_cfg import CollisionGroupCfg


@dataclass(frozen=True)
class _CompiledGroup:
    selectors: tuple[re.Pattern[str], ...]
    filtered_groups: frozenset[str]
    invert_filtered_groups: bool


class CompiledCollisionFilter:
    """Resolve path membership and symmetric deny-wins group policy."""

    def __init__(self, groups: dict[str, CollisionGroupCfg], env_template: str):
        """Compile group selectors using the clone plan's environment template."""
        try:
            env_regex_ns = env_template.format("[^/]+")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid environment template {env_template!r}.") from exc
        self.groups = {
            name: _CompiledGroup(
                selectors=tuple(
                    re.compile(selector.replace("{ENV_REGEX_NS}", env_regex_ns))
                    for selector in group.prim_path_exprs
                ),
                filtered_groups=frozenset(group.filtered_groups),
                invert_filtered_groups=group.invert_filtered_groups,
            )
            for name, group in groups.items()
        }

    def memberships(self, prim_path: str) -> tuple[str, ...]:
        """Return every group whose selector fully matches *prim_path*."""
        return tuple(
            name
            for name, group in self.groups.items()
            if any(selector.fullmatch(prim_path) is not None for selector in group.selectors)
        )

    def filters(self, first: tuple[str, ...], second: tuple[str, ...]) -> bool:
        """Return whether either membership set denies the symmetric pair."""
        return self._side_filters(first, second) or self._side_filters(second, first)

    def _side_filters(self, first: tuple[str, ...], second: tuple[str, ...]) -> bool:
        for group_name in first:
            group = self.groups[group_name]
            if not second and group.invert_filtered_groups:
                return True
            if any(
                (other_name in group.filtered_groups) != group.invert_filtered_groups for other_name in second
            ):
                return True
        return False


__all__ = ["CompiledCollisionFilter"]
