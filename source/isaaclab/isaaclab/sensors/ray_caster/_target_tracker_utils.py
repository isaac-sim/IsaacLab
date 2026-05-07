# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared multi-mesh ray-caster target-tracker prep.

Both backends need: the covering :class:`~isaaclab.cloner.ClonePlan`, one rigid-body
ancestor + body→mesh offset per prototype, and the per-env prototype assignment from
``clone_mask``. Backends differ only in what they do with the result (PhysX:
``RigidObjectView``; Newton: sites).
"""

from __future__ import annotations

import re

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan

_ENV_NAMESPACE_RE = re.compile(r"^/World/envs/env_(\d+)/(.+)$")


def resolve_rigid_body_anchor(prim) -> tuple[object, list[float]] | None:
    """Return ``(rigid_body_ancestor, [px,py,pz,qx,qy,qz,qw])`` or ``None``.

    Offset is ``prim`` resolved relative to ``ancestor`` (the constant body→prim
    transform). Sensor body tracker uses ``None`` as a static-parent signal; target
    tracker treats ``None`` as a contract violation (tracked targets need rigid-body
    anchors).
    """
    ancestor = sim_utils.get_first_matching_ancestor_prim(
        prim.GetPath(), predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if ancestor is None:
        return None
    pos, quat = sim_utils.resolve_prim_pose(prim, ancestor)
    return ancestor, [*pos, *quat]


def split_env_path(path: str) -> tuple[int | None, str]:
    """Split ``/World/envs/env_<N>/<rest>`` → ``(N, <rest>)``; ``(None, path)`` for globals.

    PhysX rebuilds the full env path (``f"/World/envs/env_{e}/{rest}"``) for the
    ``RigidObjectView``; Newton uses ``rest`` as the proto-local body label for
    :meth:`~isaaclab_newton.physics.NewtonManager.cl_register_site`.
    """
    m = _ENV_NAMESPACE_RE.match(path)
    return (int(m.group(1)), m.group(2)) if m else (None, path)


def walk_target_prototypes(
    target_prim_expr: str,
    plans: list[ClonePlan],
    num_envs: int,
    ctx_path: str,
) -> tuple[list[list[tuple[str, list[float]]]], list[int]]:
    """Walk one cloned env per prototype; return ``(per_proto_entries, env_proto_idx)``.

    ``per_proto_entries[p]`` is a list of ``(body_full_path_in_first_env, [px..qw])``
    for prototype ``p`` (empty if unused). ``env_proto_idx[e]`` is the prototype index
    env ``e`` uses. Plan resolution: global targets (no ``/env_`` token) need no plan;
    env-replicated targets must be covered (raises otherwise — env-0 fallback would
    silently mis-track heterogeneous scenes).
    """
    plan = _find_covering_plan(target_prim_expr, plans, ctx_path)

    if plan is not None:
        num_protos = plan.clone_mask.size(0)
        first_env_per_proto = [int(plan.clone_mask[p].nonzero(as_tuple=False)[0].item()) for p in range(num_protos)]
        env_proto_idx = [int(plan.clone_mask[:, e].nonzero(as_tuple=False)[0].item()) for e in range(num_envs)]
    else:
        first_env_per_proto = [0]
        env_proto_idx = [0] * num_envs

    per_proto_entries: list[list[tuple[str, list[float]]]] = [[] for _ in range(len(first_env_per_proto))]
    used_proto_indices = set(env_proto_idx)
    for proto_idx, first_env in enumerate(first_env_per_proto):
        if proto_idx not in used_proto_indices:
            continue
        env_pattern = target_prim_expr.replace("/env_.*/", f"/env_{first_env}/")
        prims = sim_utils.find_matching_prims(env_pattern)
        if not prims and env_pattern == target_prim_expr:
            prims = sim_utils.find_matching_prims(target_prim_expr)  # global, no env tag
        for prim in prims:
            anchor = resolve_rigid_body_anchor(prim)
            if anchor is None:
                raise RuntimeError(
                    f"MultiMeshRayCaster '{ctx_path}': tracked target prim '{prim.GetPath()}'"
                    " has no rigid-body ancestor. Disable track_mesh_transforms if static."
                )
            ancestor, offset = anchor
            per_proto_entries[proto_idx].append((str(ancestor.GetPath()), offset))
    return per_proto_entries, env_proto_idx


def _find_covering_plan(prim_expr: str, plans: list[ClonePlan], ctx_path: str) -> ClonePlan | None:
    """Most-specific plan (longest ``dest_template`` prefix) covering ``prim_expr``, or ``None``.

    ``None`` for globals (no ``/env_`` token, no plan needed). Raises for env-replicated
    targets with no covering plan — silent fallback would mis-track heterogeneous scenes.
    """
    if "/env_" not in prim_expr:
        return None
    best: ClonePlan | None = None
    best_len = -1
    for plan in plans:
        prefix = plan.dest_template.split("{}")[0]
        if prim_expr.startswith(prefix) and len(prefix) > best_len:
            best, best_len = plan, len(prefix)
    if best is None:
        raise RuntimeError(
            f"MultiMeshRayCaster '{ctx_path}': tracked target '{prim_expr}' is env-replicated"
            " (contains '/env_') but no ClonePlan covers it. Tracked env-replicated targets"
            " must be produced by the cloner."
        )
    return best
