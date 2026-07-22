# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static gate for the mask-native execution path.

The warp frontend's contract is that mask-native code performs no data-dependent
GPU->host synchronization outside sanctioned boundaries. Instead of inline marker
comments, the gate is structural:

* **Scope** — every function in the experimental production tree that sits on the
  step/reset pipeline, plus every ``*mask*``-named function in the core and Newton
  trees (the functions that *are* the warp path there). Torch-first core code is
  deliberately out of scope.
* **Sanctioned boundaries** — small, named helper functions listed in
  ``SANCTIONED_BOUNDARIES``; the list is exact (a stale entry fails the gate).
* **Runtime backstop** — ``ISAACLAB_SYNC_DEBUG=1`` runs every warp stage under
  ``torch.cuda.set_sync_debug_mode("error")`` (see ``WarpGraphCache``), trapping
  syncs this static scan cannot see.

The companion inventory test pins every ``@WarpCapturable(False)`` opt-out so
non-capturable terms are documented in exactly one reviewable place.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Attribute calls that synchronize (or copy) device memory to the host.
SYNC_ATTRS = {"nonzero", "argwhere", "item", "cpu", "tolist", "numpy"}

# Trees scanned in full (production only; every function is in scope unless excluded).
FULL_SCAN_ROOTS = [
    "source/isaaclab_experimental/isaaclab_experimental/envs",
    "source/isaaclab_experimental/isaaclab_experimental/managers",
    "source/isaaclab_experimental/isaaclab_experimental/utils",
]

# Trees where only ``*mask*``-named functions are in scope (the warp path inside
# otherwise torch-first packages).
MASK_SCAN_ROOTS = [
    "source/isaaclab/isaaclab/actuators",
    "source/isaaclab/isaaclab/scene",
    "source/isaaclab/isaaclab/sensors",
    "source/isaaclab/isaaclab/terrains",
    "source/isaaclab_newton/isaaclab_newton",
]

# Functions that never run on the per-step path: construction, reporting, and
# debug surfaces. Host syncs there are init-time or human-facing.
NON_STEP_FUNCTIONS = {
    "__init__",
    "__post_init__",
    "__str__",
    "__repr__",
    "__del__",
    "_prepare_terms",
    "serialize",
    "get_active_iterable_terms",
    "_debug_vis_callback",
    "_set_debug_vis_impl",
    # reporting / config-access surfaces (export- or user-facing, not per-step)
    "IO_descriptor",
    "get_IO_descriptors",
    "get_term_cfg",
}

# The sanctioned mask->ID / host-sync boundaries, by (path suffix, function name).
# Every entry must exist and contain a sync call; anything else that syncs fails.
SANCTIONED_BOUNDARIES = {
    ("managers/curriculum_manager.py", "_compact_legacy_env_ids"),
    ("envs/mdp/events.py", "_mask_to_env_ids"),
    ("envs/manager_based_env_warp.py", "reset_to"),
    # Coarse-grained by review preference: the recorder / legacy-term compactions
    # stay inline in their reset functions rather than one-line helpers; the
    # ISAACLAB_SYNC_DEBUG runtime trap is the fine-grained net inside them.
    ("envs/manager_based_env_warp.py", "reset"),
    ("envs/manager_based_rl_env_warp.py", "_reset_terminated_envs"),
    ("envs/manager_based_rl_env_warp.py", "_reset_mask"),
    ("utils/warp/utils.py", "any_env_set"),
    # camera's empty-reset predicate (the sensor-side analogue of any_env_set)
    ("isaaclab/sensors/camera/camera.py", "_env_mask_has_any"),
    # joint-limit writes mutate the solver model (host-side by nature, event-driven)
    ("isaaclab_newton/assets/articulation/articulation.py", "write_joint_position_limit_to_sim_mask"),
    # legacy-actuator compatibility fallback: mask-native actuator models override
    # this; legacy models compact once per reset (event-driven, not per-step).
    ("isaaclab/actuators/actuator_base.py", "reset_mask"),
}


def _sync_calls(func_node: ast.AST) -> list[str]:
    """Names of synchronizing attribute calls inside a function body."""
    hits = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SYNC_ATTRS:
            hits.append(node.func.attr)
    return hits


def _iter_functions(tree: ast.Module):
    """Yield (function node, name) for every def in the module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, node.name


def _scan(root: Path, mask_only: bool) -> dict[tuple[str, str], list[str]]:
    """Map (relative path, function) -> sync hits for in-scope functions under root."""
    findings: dict[tuple[str, str], list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if "/test/" in rel:
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        for func, name in _iter_functions(tree):
            if name in NON_STEP_FUNCTIONS:
                continue
            if mask_only and "mask" not in name:
                continue
            hits = _sync_calls(func)
            if hits:
                findings[(rel, name)] = hits
    return findings


def _is_sanctioned(rel: str, name: str) -> bool:
    return any(rel.endswith(suffix) and name == func for suffix, func in SANCTIONED_BOUNDARIES)


def test_mask_native_code_has_no_unsanctioned_host_syncs():
    """Every sync in mask-native scope is one of the named, sanctioned boundaries."""
    findings: dict[tuple[str, str], list[str]] = {}
    for root in FULL_SCAN_ROOTS:
        findings.update(_scan(_REPO_ROOT / root, mask_only=False))
    for root in MASK_SCAN_ROOTS:
        findings.update(_scan(_REPO_ROOT / root, mask_only=True))

    violations = {k: v for k, v in findings.items() if not _is_sanctioned(*k)}
    assert not violations, "Unsanctioned host syncs on the mask-native path:\n" + "\n".join(
        f"  {rel}::{name} -> {hits}" for (rel, name), hits in sorted(violations.items())
    )

    # The sanctioned list is exact: every entry must still exist and still sync.
    matched = {k for k in findings if _is_sanctioned(*k)}
    stale = {
        (suffix, func)
        for suffix, func in SANCTIONED_BOUNDARIES
        if not any(rel.endswith(suffix) and name == func for rel, name in matched)
    }
    assert not stale, f"Stale sanctioned-boundary entries (function gone or no longer syncs): {sorted(stale)}"


# Expected capturability opt-outs — ``@WarpCapturable(False)`` decorations or
# ``_warp_capturable = False`` class attributes: (path suffix, name).
EXPECTED_NON_CAPTURABLE = {
    ("isaaclab_experimental/envs/mdp/events.py", "randomize_rigid_body_com"),
    ("isaaclab_experimental/envs/mdp/observations.py", "height_scan"),
    # Reads the Python-owned common step counter; captured replay would bake it.
    ("isaaclab_tasks_experimental/core/reach/mdp/curriculums.py", "modify_reward_weight"),
}

NON_CAPTURABLE_SCAN_ROOTS = [
    "source/isaaclab_experimental/isaaclab_experimental",
    "source/isaaclab_tasks_experimental/isaaclab_tasks_experimental",
]


def _warp_capturable_false_targets(tree: ast.Module) -> list[str]:
    """Names annotated non-capturable: ``@WarpCapturable(False, ...)`` decorations
    or ``_warp_capturable = False`` class attributes."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for deco in node.decorator_list:
            if (
                isinstance(deco, ast.Call)
                and isinstance(deco.func, ast.Name)
                and deco.func.id == "WarpCapturable"
                and deco.args
                and isinstance(deco.args[0], ast.Constant)
                and deco.args[0].value is False
            ):
                out.append(node.name)
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_warp_capturable" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is False
                ):
                    out.append(node.name)
    return out


def test_non_capturable_terms_are_inventoried():
    """Every ``@WarpCapturable(False)`` opt-out is documented here, and only these."""
    found = set()
    for scan_root in NON_CAPTURABLE_SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / scan_root).rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "/test/" in rel:
                continue
            for name in _warp_capturable_false_targets(ast.parse(path.read_text(), filename=rel)):
                found.add((rel, name))
    expected = {(f"source/{suffix.split('/')[0]}/{suffix}", name) for suffix, name in EXPECTED_NON_CAPTURABLE}
    assert found == expected, (
        "Non-capturable inventory drift.\n"
        f"  unexpected: {sorted(found - expected)}\n"
        f"  missing:    {sorted(expected - found)}"
    )
