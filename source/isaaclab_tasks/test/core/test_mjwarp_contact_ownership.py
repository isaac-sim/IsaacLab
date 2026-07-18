# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for first-party MJWarp contact ownership."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[4]
_SOURCE_ROOT = _REPO_ROOT / "source"
_SOURCE_PACKAGES = tuple(
    package_root for package_root in sorted(_SOURCE_ROOT.iterdir()) if (package_root / package_root.name).is_dir()
)
_PACKAGE_ROOTS = tuple(package_root / package_root.name for package_root in _SOURCE_PACKAGES)
_TEST_ROOTS = tuple(package_root / "test" for package_root in _SOURCE_PACKAGES if (package_root / "test").is_dir())
_CONFIG_ROOTS = (
    *_PACKAGE_ROOTS,
    *_TEST_ROOTS,
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "tools",
)
_COMPATIBILITY_DEFAULT_PATH = _SOURCE_ROOT / "isaaclab_newton" / "isaaclab_newton" / "physics" / "newton_manager_cfg.py"
_MJWARP_MANAGER_PATH = _SOURCE_ROOT / "isaaclab_newton" / "isaaclab_newton" / "physics" / "mjwarp_manager.py"
_COUPLED_COMPATIBILITY_DEFAULT_PATH = (
    _SOURCE_ROOT / "isaaclab_contrib" / "isaaclab_contrib" / "deformable" / "newton_manager_cfg.py"
)
_PHYSICS_TEST_UTILS_PATH = _SOURCE_ROOT / "isaaclab_newton" / "test" / "physics" / "physics_test_utils.py"
_MANAGER_ABSTRACTION_PATH = _SOURCE_ROOT / "isaaclab_newton" / "test" / "physics" / "test_newton_manager_abstraction.py"
_COUPLED_OWNERSHIP_PATH = (
    _SOURCE_ROOT / "isaaclab_contrib" / "test" / "deformable" / "test_coupled_mjwarp_contact_ownership.py"
)


def _call_name(func: ast.expr) -> str:
    """Return the dotted name for a call target."""
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _is_compatibility_default(path: Path, tree: ast.Module, call: ast.Call) -> bool:
    """Return whether this is an exact retained public compatibility default."""
    if path == _COMPATIBILITY_DEFAULT_PATH:
        newton_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonCfg")
        post_init = next(
            node for node in newton_cfg.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        return not call.args and not call.keywords and call in ast.walk(post_init)
    if path != _COUPLED_COMPATIBILITY_DEFAULT_PATH or call.args or call.keywords:
        return False
    coupled_cfg = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CoupledMJWarpVBDSolverCfg"
    )
    return any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "rigid_solver_cfg"
        and node.value is call
        for node in coupled_cfg.body
    )


def _imported_aliases(tree: ast.Module, symbol: str) -> set[str]:
    """Return local aliases importing an Isaac Lab Newton physics symbol."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("isaaclab_newton.physics")
        for alias in node.names
        if alias.name == symbol
    }


def _subclass_names(parsed: list[tuple[Path, ast.Module]], symbol: str) -> set[str]:
    """Return local class names that transitively derive from an imported physics symbol."""
    subclasses: set[str] = set()
    changed = True
    while changed:
        changed = False
        for _, tree in parsed:
            base_names = {symbol, *_imported_aliases(tree, symbol), *subclasses}
            for node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
                if node.name in subclasses:
                    continue
                if any(_call_name(base).rsplit(".", maxsplit=1)[-1] in base_names for base in node.bases):
                    subclasses.add(node.name)
                    changed = True
    return subclasses


def _location(path: Path, node: ast.AST, message: str) -> str:
    """Format a repository-relative architecture violation."""
    return f"{path.relative_to(_REPO_ROOT)}:{node.lineno} ({message})"


def _is_false(node: ast.expr | None) -> bool:
    """Return whether an expression is the literal ``False``."""
    return isinstance(node, ast.Constant) and node.value is False


def _is_contact_key(node: ast.expr) -> bool:
    """Return whether a string expression addresses the MJWarp contact-mode field."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    return node.value.strip("\"'").rsplit(".", maxsplit=1)[-1] == "use_mujoco_contacts"


def _is_hydra_true_override(value: str) -> bool:
    """Return whether a dot-list string enables MuJoCo contacts."""
    normalized = "".join(value.lower().split()).strip("\"'")
    separator = "=" if "=" in normalized else ":" if ":" in normalized else None
    if separator is None:
        return False
    key, setting = normalized.rsplit(separator, maxsplit=1)
    return key.lstrip("+~").rsplit(".", maxsplit=1)[-1] == "use_mujoco_contacts" and setting == "true"


def _class_field_value(node: ast.ClassDef, field: str) -> tuple[bool, ast.expr | None]:
    """Return whether a class body defines a field and its value expression."""
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == field:
                return True, statement.value
        elif isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == field for target in statement.targets
        ):
            return True, statement.value
    return False, None


def _is_solver_matrix_compatibility_default(path: Path, tree: ast.Module, call: ast.Call) -> bool:
    """Return whether this is the exact matrix row covering MJWarp's public legacy default."""
    if path != _MANAGER_ABSTRACTION_PATH or call.args or call.keywords:
        return False
    solver_matrix = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SOLVER_MATRIX" for target in node.targets)
        ),
        None,
    )
    if solver_matrix is None:
        return False
    for param in (node for node in ast.walk(solver_matrix) if isinstance(node, ast.Call)):
        if _call_name(param.func) != "pytest.param" or call not in ast.walk(param):
            continue
        id_kw = next((keyword for keyword in param.keywords if keyword.arg == "id"), None)
        return (
            id_kw is not None
            and isinstance(id_kw.value, ast.Constant)
            and id_kw.value.value == "mjwarp_internal_contacts"
        )
    return False


def _private_newton_import_violations(path: Path, tree: ast.Module) -> list[str]:
    """Reject cross-package imports of private ``isaaclab_newton`` symbols."""
    try:
        path.relative_to(_SOURCE_ROOT / "isaaclab_newton")
        return []
    except ValueError:
        pass
    return [
        _location(path, node, f"cross-package import of private {alias.name}")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("isaaclab_newton")
        for alias in node.names
        if alias.name.startswith("_")
    ]


def _config_validation_ownership_violations(path: Path, tree: ast.Module) -> list[str]:
    """Keep MJWarp contact validation on its public configuration owner."""
    if path != _MJWARP_MANAGER_PATH:
        return []
    return [
        _location(path, node, "contact-mode validation belongs to MJWarpSolverCfg")
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_validate_mjwarp_contact_mode"
    ]


def _is_legacy_test_call(path: Path, tree: ast.Module, call: ast.Call, contact_kw: ast.keyword | None) -> bool:
    """Return whether a call is exact, dedicated compatibility coverage."""
    if _is_solver_matrix_compatibility_default(path, tree, call):
        return True
    if contact_kw is None:
        return False
    if path == _PHYSICS_TEST_UTILS_PATH:
        return isinstance(contact_kw.value, ast.Name) and contact_kw.value.id == "use_mujoco_contacts"
    if path == _COUPLED_OWNERSHIP_PATH:
        if not isinstance(contact_kw.value, ast.Constant) or contact_kw.value.value is not True:
            return False
        return any(
            isinstance(node, ast.FunctionDef)
            and node.name == "test_coupled_mjwarp_legacy_contacts_warn"
            and call in ast.walk(node)
            for node in tree.body
        )
    if path != _MANAGER_ABSTRACTION_PATH or not isinstance(contact_kw.value, ast.Constant):
        return False
    if contact_kw.value.value is not True:
        return False
    return any(
        isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "test_mjwarp_cfg_owns_contact_mode_validation",
            "test_mjwarp_internal_contacts_with_collision_cfg_raises",
        }
        and call in ast.walk(node)
        for node in tree.body
    )


def _collect_contact_violations(
    path: Path,
    tree: ast.Module,
    newton_subclasses: set[str],
    mjwarp_subclasses: set[str],
) -> list[str]:
    """Collect statically provable violations of first-party Newton contact ownership."""
    violations = _private_newton_import_violations(path, tree) + _config_validation_ownership_violations(path, tree)

    defines_local_newton_cfg = any(isinstance(node, ast.ClassDef) and node.name == "NewtonCfg" for node in tree.body)
    newton_cfg_names = _imported_aliases(tree, "NewtonCfg") | newton_subclasses
    if not defines_local_newton_cfg:
        newton_cfg_names.add("NewtonCfg")
    mjwarp_cfg_names = _imported_aliases(tree, "MJWarpSolverCfg") | {"MJWarpSolverCfg"}

    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        if class_node.name not in mjwarp_subclasses:
            continue
        defines_contact_mode, contact_mode = _class_field_value(class_node, "use_mujoco_contacts")
        if not defines_contact_mode or not _is_false(contact_mode):
            violations.append(
                _location(path, class_node, "MJWarpSolverCfg subclass must default use_mujoco_contacts=False")
            )

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        call_name = _call_name(call.func)
        call_type = call_name.rsplit(".", maxsplit=1)[-1]
        contact_kw = next(
            (keyword for keyword in call.keywords if keyword.arg == "use_mujoco_contacts"),
            None,
        )
        if call_type in newton_cfg_names:
            solver_cfg_kw = next(
                (keyword for keyword in call.keywords if keyword.arg == "solver_cfg"),
                None,
            )
            if (
                call.args
                or solver_cfg_kw is None
                or (isinstance(solver_cfg_kw.value, ast.Constant) and solver_cfg_kw.value.value is None)
            ):
                violations.append(_location(path, call, "NewtonCfg must set solver_cfg explicitly"))
            continue
        if call_type in mjwarp_cfg_names:
            if _is_compatibility_default(path, tree, call):
                continue
            if not _is_legacy_test_call(path, tree, call, contact_kw) and (
                contact_kw is None or not _is_false(contact_kw.value)
            ):
                violations.append(_location(path, call, "must set use_mujoco_contacts=False"))
            if any(keyword.arg == "ccd_iterations" for keyword in call.keywords):
                violations.append(_location(path, call, "ccd_iterations is legacy-only"))
        elif (
            contact_kw is not None
            and not _is_false(contact_kw.value)
            and (call_type == "dict" or isinstance(contact_kw.value, ast.Constant) and contact_kw.value.value is True)
        ):
            violations.append(_location(path, call, "contact-mode override must be False"))

        if (
            call_type == "setattr"
            and len(call.args) >= 3
            and _is_contact_key(call.args[1])
            and not _is_false(call.args[2])
        ):
            violations.append(_location(path, call, "setattr contact-mode override must be False"))
        if (
            call_type == "update"
            and len(call.args) >= 3
            and _is_contact_key(call.args[1])
            and not _is_false(call.args[2])
        ):
            violations.append(_location(path, call, "mapping/Hydra contact-mode update must be False"))

    for mapping in (node for node in ast.walk(tree) if isinstance(node, ast.Dict)):
        for key, value in zip(mapping.keys, mapping.values):
            if key is not None and _is_contact_key(key) and not _is_false(value):
                violations.append(_location(path, mapping, "mapping contact-mode value must be False"))

    for container in (node for node in ast.walk(tree) if isinstance(node, (ast.List, ast.Tuple, ast.Set))):
        for item in container.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str) and _is_hydra_true_override(item.value):
                violations.append(_location(path, item, "Hydra contact-mode override must be false"))

    for assignment in (node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))):
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        value = assignment.value if not isinstance(assignment, ast.AugAssign) else None
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr == "use_mujoco_contacts" and not _is_false(value):
                violations.append(_location(path, assignment, "contact-mode assignment must be False"))
            if isinstance(target, ast.Subscript) and _is_contact_key(target.slice) and not _is_false(value):
                violations.append(_location(path, assignment, "mapping contact-mode assignment must be False"))
            if isinstance(target, ast.Attribute) and target.attr == "ccd_iterations":
                violations.append(_location(path, assignment, "ccd_iterations assignment"))

    return violations


def _synthetic_violations(source: str) -> list[str]:
    """Analyze an isolated source snippet with the same ownership rules."""
    path = _REPO_ROOT / "_synthetic_mjwarp_contact_cfg.py"
    tree = ast.parse(source, filename=str(path))
    parsed = [(path, tree)]
    return _collect_contact_violations(
        path,
        tree,
        _subclass_names(parsed, "NewtonCfg"),
        _subclass_names(parsed, "MJWarpSolverCfg"),
    )


def test_first_party_mjwarp_configs_use_newton_contacts():
    """First-party configs must not inherit, mutate, or request MJWarp's internal collision path."""
    paths = sorted({path for root in _CONFIG_ROOTS for path in root.rglob("*.py")})
    parsed = [(path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))) for path in paths]
    newton_subclasses = _subclass_names(parsed, "NewtonCfg")
    mjwarp_subclasses = _subclass_names(parsed, "MJWarpSolverCfg")
    violations = [
        violation
        for path, tree in parsed
        for violation in _collect_contact_violations(path, tree, newton_subclasses, mjwarp_subclasses)
    ]

    assert not violations, (
        "First-party MJWarp configs must use Newton contacts without legacy-only options; violations:\n"
        + "\n".join(violations)
    )


def test_gate_rejects_mutation_mapping_hydra_and_private_import_bypasses():
    """The architecture gate must reject non-constructor ways to restore internal contacts."""
    cases = {
        "attribute": "solver_cfg.use_mujoco_contacts = True",
        "mapping": 'cfg = {"use_mujoco_contacts": True}',
        "subscript": 'cfg["use_mujoco_contacts"] = True',
        "dict constructor": "cfg = dict(use_mujoco_contacts=True)",
        "Hydra dot list": 'overrides = ["physics.solver_cfg.use_mujoco_contacts=true"]',
        "OmegaConf update": 'OmegaConf.update(cfg, "physics.solver_cfg.use_mujoco_contacts", True)',
        "setattr": 'setattr(cfg, "use_mujoco_contacts", True)',
        "private import": "from isaaclab_newton.physics.mjwarp_manager import _validate_contact_mode",
    }
    for label, source in cases.items():
        assert _synthetic_violations(source), f"gate accepted unsafe {label} form"
    manager_tree = ast.parse("def _validate_mjwarp_contact_mode(): pass")
    assert _config_validation_ownership_violations(_MJWARP_MANAGER_PATH, manager_tree)


def test_gate_rejects_inherited_legacy_defaults():
    """MJWarp and Newton config subclasses must not silently inherit legacy defaults."""
    mjwarp_source = """
from isaaclab_newton.physics import MJWarpSolverCfg

class UnsafeMJWarpCfg(MJWarpSolverCfg):
    pass
"""
    newton_source = """
from isaaclab_newton.physics import NewtonCfg

class TaskNewtonCfg(NewtonCfg):
    pass

physics = TaskNewtonCfg()
"""
    assert _synthetic_violations(mjwarp_source)
    assert _synthetic_violations(newton_source)


def test_gate_accepts_explicit_external_contact_forms():
    """Equivalent explicit external-contact forms remain valid."""
    source = """
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

class ExternalMJWarpCfg(MJWarpSolverCfg):
    use_mujoco_contacts: bool = False

class TaskNewtonCfg(NewtonCfg):
    pass

solver_cfg = ExternalMJWarpCfg()
solver_cfg.use_mujoco_contacts = False
options = {"use_mujoco_contacts": False}
dict_options = dict(use_mujoco_contacts=False)
options["use_mujoco_contacts"] = False
overrides = ["physics.solver_cfg.use_mujoco_contacts=false"]
OmegaConf.update(options, "physics.solver_cfg.use_mujoco_contacts", False)
physics = TaskNewtonCfg(solver_cfg=solver_cfg)
"""
    assert not _synthetic_violations(source)
