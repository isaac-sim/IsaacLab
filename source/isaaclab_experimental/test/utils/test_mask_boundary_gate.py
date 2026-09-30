# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static gate for the mask-first execution path.

Warp frontend code performs no data-dependent GPU-to-host synchronization outside named boundaries:

* **Scope**: every function in the experimental production tree, plus every ``*mask*``-named function in
  the core and Newton trees (the functions that are the Warp path there). Torch-first core code is out of
  scope.
* **Sanctioned boundaries**: the functions in :data:`SANCTIONED_BOUNDARIES`. The list is exact; a stale
  entry fails the gate.
* **Runtime backstop**: ``ISAACLAB_SYNC_DEBUG=1`` runs eager Warp stages under
  ``torch.cuda.set_sync_debug_mode("error")``, trapping synchronizations this scan cannot see.

A second test pins every ``@WarpCapturable(False)`` opt-out, so non-capturable terms are listed in one place.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Attribute calls that synchronize with, or copy from, device memory.
SYNC_ATTRS = {"nonzero", "argwhere", "item", "cpu", "tolist", "numpy"}

# Trees scanned in full (production code only).
FULL_SCAN_ROOTS = [
    "source/isaaclab_experimental/isaaclab_experimental/envs",
    "source/isaaclab_experimental/isaaclab_experimental/managers",
    "source/isaaclab_experimental/isaaclab_experimental/utils",
]

# Trees where only ``*mask*``-named functions are in scope.
MASK_SCAN_ROOTS = [
    "source/isaaclab/isaaclab/actuators",
    "source/isaaclab/isaaclab/scene",
    "source/isaaclab/isaaclab/sensors",
    "source/isaaclab/isaaclab/terrains",
    "source/isaaclab_newton/isaaclab_newton",
]

# Functions that never run on the per-step path: construction, reporting, and debug surfaces.
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
    # deprecated IO-descriptor export surfaces
    "IO_descriptor",
    "_collect_io_descriptors",
    "get_term_cfg",
}

# Sanctioned host-synchronization boundaries, by (path suffix, function name).
SANCTIONED_BOUNDARIES = {
    # one host predicate per step decides whether the reset pipeline runs (W7)
    ("envs/manager_based_rl_env_warp.py", "_reset_terminated_envs"),
    ("envs/direct_rl_env_warp.py", "step"),
    # mask to index compaction for the managers that reset by index (W11)
    ("envs/manager_based_env_warp.py", "_reset_env_ids"),
    # startup-only stable randomization terms take environment indices
    ("envs/mdp/events.py", "_mask_to_env_ids"),
    # the camera's empty-reset predicate
    ("isaaclab/sensors/camera/camera.py", "_env_mask_has_any"),
    # backends without a mask-native actuator reset fall back to indices; Newton and OVPhysX override it
    ("isaaclab/actuators/actuator_control.py", "reset_native_actuators_mask"),
    # joint-limit writes mutate the solver model (event-driven, not per-step)
    ("isaaclab_newton/assets/articulation/articulation.py", "write_joint_position_limit_to_sim_mask"),
}


def _sync_calls(func_node: ast.AST) -> list[str]:
    """Names of synchronizing attribute calls inside a function body."""
    return [
        node.func.attr
        for node in ast.walk(func_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SYNC_ATTRS
    ]


def _scan(root: Path, mask_only: bool) -> dict[tuple[str, str], list[str]]:
    """Map (relative path, function) to synchronizing calls for the in-scope functions under ``root``."""
    findings: dict[tuple[str, str], list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if "/test/" in rel:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=rel)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in NON_STEP_FUNCTIONS:
                continue
            if mask_only and "mask" not in node.name:
                continue
            hits = _sync_calls(node)
            if hits:
                findings[(rel, node.name)] = hits
    return findings


def _is_sanctioned(rel: str, name: str) -> bool:
    return any(rel.endswith(suffix) and name == func for suffix, func in SANCTIONED_BOUNDARIES)


def test_mask_first_code_has_no_unsanctioned_host_syncs():
    """Every synchronization in scope is one of the named, sanctioned boundaries."""
    findings: dict[tuple[str, str], list[str]] = {}
    for root in FULL_SCAN_ROOTS:
        findings.update(_scan(_REPO_ROOT / root, mask_only=False))
    for root in MASK_SCAN_ROOTS:
        findings.update(_scan(_REPO_ROOT / root, mask_only=True))

    violations = {key: hits for key, hits in findings.items() if not _is_sanctioned(*key)}
    assert not violations, "Unsanctioned host syncs on the mask-first path:\n" + "\n".join(
        f"  {rel}::{name} -> {hits}" for (rel, name), hits in sorted(violations.items())
    )

    matched = {key for key in findings if _is_sanctioned(*key)}
    stale = {
        (suffix, func)
        for suffix, func in SANCTIONED_BOUNDARIES
        if not any(rel.endswith(suffix) and name == func for rel, name in matched)
    }
    assert not stale, f"Stale sanctioned boundaries (function gone or no longer syncs): {sorted(stale)}"


# ``@WarpCapturable(False)`` opt-outs, by (path suffix, name).
EXPECTED_NON_CAPTURABLE = {
    ("isaaclab_experimental/envs/mdp/events.py", "randomize_rigid_body_com"),
    ("isaaclab_experimental/envs/mdp/events.py", "randomize_rigid_body_mass"),
    ("isaaclab_experimental/envs/mdp/events.py", "randomize_rigid_body_material"),
}

NON_CAPTURABLE_SCAN_ROOTS = [
    "source/isaaclab_experimental/isaaclab_experimental",
    "source/isaaclab_tasks_experimental/isaaclab_tasks_experimental",
]


def _non_capturable_targets(tree: ast.Module) -> list[str]:
    """Names of functions and classes decorated with ``@WarpCapturable(False, ...)``."""
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "WarpCapturable"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and decorator.args[0].value is False
            ):
                targets.append(node.name)
    return targets


def test_non_capturable_terms_are_inventoried():
    """Every ``@WarpCapturable(False)`` opt-out is listed here, and only these."""
    found = set()
    for scan_root in NON_CAPTURABLE_SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / scan_root).rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "/test/" in rel:
                continue
            for name in _non_capturable_targets(ast.parse(path.read_text(), filename=rel)):
                found.add((rel, name))
    expected = {(f"source/{suffix.split('/')[0]}/{suffix}", name) for suffix, name in EXPECTED_NON_CAPTURABLE}
    assert found == expected, (
        f"Non-capturable inventory drift.\n  unexpected: {sorted(found - expected)}\n"
        f"  missing:    {sorted(expected - found)}"
    )
