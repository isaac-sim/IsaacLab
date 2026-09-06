# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for FeatherPGS task presets."""

import ast
from pathlib import Path

_TASK_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "isaaclab_tasks"
_FIXED_TENDON_TASK_ROOTS = (Path("core/handover"), Path("core/reorient/config/shadow_hand"))


def _base_name(base: ast.expr) -> str:
    """Return the final name in a class base expression."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _classes() -> list[tuple[Path, ast.ClassDef]]:
    """Parse task classes with their source paths."""
    classes = []
    for path in _TASK_SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes.extend((path, node) for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return classes


def _defines_name(node: ast.ClassDef, name: str) -> bool:
    """Return whether a class body directly assigns a given field."""
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return True
    return False


def _assigned_value(node: ast.ClassDef, name: str) -> ast.expr | None:
    """Return the value directly assigned to a class field."""
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return statement.value
    return None


def test_physics_preset_classes_are_not_subclassed() -> None:
    """Forbid robot-specific subclasses of an existing physics preset class."""
    offenders = [
        f"{path.relative_to(_TASK_SOURCE_ROOT)}:{node.lineno} {node.name}"
        for path, node in _classes()
        if any(_base_name(base).endswith("PhysicsCfg") for base in node.bases)
    ]

    assert offenders == []


def test_feather_pgs_event_subclasses_are_not_defined() -> None:
    """Forbid FeatherPGS-specific event inheritance and event preset wrappers."""
    offenders = []
    for path, node in _classes():
        base_names = {_base_name(base) for base in node.bases}
        is_event_class = node.name.endswith("EventCfg") or any(name.endswith("EventCfg") for name in base_names)
        if is_event_class and ("FeatherPGS" in node.name or _defines_name(node, "feather_pgs")):
            offenders.append(f"{path.relative_to(_TASK_SOURCE_ROOT)}:{node.lineno} {node.name}")

    assert offenders == []


def test_feather_pgs_presets_are_owned_by_existing_composition_roots() -> None:
    """Require direct FeatherPGS fields to live on existing top-level preset containers."""
    offenders = []
    for path, node in _classes():
        if not _defines_name(node, "feather_pgs"):
            continue
        if {_base_name(base) for base in node.bases} != {"PresetCfg"}:
            offenders.append(f"{path.relative_to(_TASK_SOURCE_ROOT)}:{node.lineno} {node.name}")

    assert offenders == []


def test_feather_pgs_presets_select_physics_or_required_backend_assets() -> None:
    """Forbid FeatherPGS variants outside physics and explicitly audited backend assets."""
    allowed_backend_assets = {
        (Path("contrib/keyboard/so101_env_cfg.py"), "KeyboardAssetCfg"): "newton_mjwarp",
    }
    offenders = []
    for path, node in _classes():
        value = _assigned_value(node, "feather_pgs")
        if value is None:
            continue
        if isinstance(value, ast.Call) and _base_name(value.func) == "NewtonCfg":
            continue
        relative_path = path.relative_to(_TASK_SOURCE_ROOT)
        expected_name = allowed_backend_assets.get((relative_path, node.name))
        if not isinstance(value, ast.Name) or value.id != expected_name:
            offenders.append(f"{relative_path}:{node.lineno} {node.name}")

    assert offenders == []


def test_fixed_tendon_tasks_do_not_advertise_feather_pgs() -> None:
    """Forbid FeatherPGS presets on task families that require fixed tendons."""
    offenders = [
        f"{path.relative_to(_TASK_SOURCE_ROOT)}:{node.lineno} {node.name}"
        for path, node in _classes()
        if any(path.relative_to(_TASK_SOURCE_ROOT).is_relative_to(root) for root in _FIXED_TENDON_TASK_ROOTS)
        and _defines_name(node, "feather_pgs")
    ]

    assert offenders == []


def test_feather_pgs_presets_do_not_disable_cuda_graphs() -> None:
    """Forbid task presets from disabling the default FeatherPGS graph capture path."""
    offenders = []
    for path, node in _classes():
        value = _assigned_value(node, "feather_pgs")
        if not isinstance(value, ast.Call) or _base_name(value.func) != "NewtonCfg":
            continue
        for keyword in value.keywords:
            if (
                keyword.arg == "use_cuda_graph"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            ):
                offenders.append(f"{path.relative_to(_TASK_SOURCE_ROOT)}:{keyword.value.lineno} {node.name}")

    assert offenders == []
