# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Migration tool: wrap asset .data.* property accesses with wp.to_torch().

After the warp migration, all .data.* properties on asset objects return wp.array
instead of torch.Tensor. Downstream code (task envs, tests, MDP functions) that
consumes these properties in torch operations needs wp.to_torch() wrapping.
This tool automates that conversion for any Python file or directory.

Usage:
    # Dry-run (default) - show what would change
    python scripts/tools/wrap_warp_to_torch.py source/isaaclab_tasks/

    # Interactive apply - prompt per-change with context
    python scripts/tools/wrap_warp_to_torch.py source/isaaclab_tasks/ --apply

    # Bulk apply all without prompting
    python scripts/tools/wrap_warp_to_torch.py source/isaaclab_tasks/ --apply --force

    # Control context lines shown around each change (default: 2)
    python scripts/tools/wrap_warp_to_torch.py path/to/env.py --apply --context 4

    # Single file, verbose
    python scripts/tools/wrap_warp_to_torch.py path/to/env.py --apply --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known asset class names whose .data properties return wp.array
ASSET_CLASSES: frozenset[str] = frozenset(
    {
        "Articulation",
        "BaseArticulation",
        "RigidObject",
        "BaseRigidObject",
        "RigidObjectCollection",
        "BaseRigidObjectCollection",
        "DeformableObject",
        "AssetBase",
    }
)

# Properties on .data that do NOT return wp.array (strings, device, methods, etc.)
BLACKLISTED_PROPERTIES: frozenset[str] = frozenset(
    {
        "body_names",
        "joint_names",
        "fixed_tendon_names",
        "spatial_tendon_names",
        "device",
        "update",
        "reset",
    }
)

# Direct attributes on asset objects (not under .data) that are now wp.array
WARP_ASSET_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "_ALL_INDICES",
    }
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WrapTarget:
    """A source location that needs wp.to_torch() wrapping."""

    start_line: int  # 1-based
    start_col: int  # 0-based
    end_line: int  # 1-based
    end_col: int  # 0-based (exclusive)
    original_text: str = ""


@dataclass
class FileResult:
    """Result of processing a single file."""

    path: Path
    targets: list[WrapTarget] = field(default_factory=list)
    import_added: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_attr_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Extract an attribute chain as a tuple of names.

    ``self._robot`` → ``("self", "_robot")``
    ``robot``       → ``("robot",)``

    Returns ``None`` if the chain contains non-Name/Attribute nodes (e.g.
    subscripts).
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return None


def _is_asset_class_node(node: ast.AST) -> bool:
    """Check whether *node* refers to a known asset class name."""
    if isinstance(node, ast.Name):
        return node.id in ASSET_CLASSES
    if isinstance(node, ast.Attribute):
        return node.attr in ASSET_CLASSES
    return False


# ---------------------------------------------------------------------------
# Phase 1 – Collect asset variables
# ---------------------------------------------------------------------------


class AssetVariableCollector(ast.NodeVisitor):
    """Walk an AST and record variable names that hold asset instances."""

    def __init__(self) -> None:
        self.asset_vars: set[tuple[str, ...]] = set()

    # -- Pattern: self._robot = Articulation(cfg) -------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call) and _is_asset_class_node(node.value.func):
            for target in node.targets:
                chain = get_attr_chain(target)
                if chain:
                    self.asset_vars.add(chain)
        self.generic_visit(node)

    # -- Pattern: robot: Articulation  /  self.robot: Articulation = ... --

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target and _is_asset_class_node(node.annotation):
            chain = get_attr_chain(node.target)
            if chain:
                self.asset_vars.add(chain)
        # Also detect when the *value* is an asset constructor
        if node.target and node.value and isinstance(node.value, ast.Call) and _is_asset_class_node(node.value.func):
            chain = get_attr_chain(node.target)
            if chain:
                self.asset_vars.add(chain)
        self.generic_visit(node)

    # -- Pattern: def func(robot: Articulation): --------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation and _is_asset_class_node(arg.annotation):
                self.asset_vars.add((arg.arg,))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


# ---------------------------------------------------------------------------
# Phase 2 – Annotate parent references
# ---------------------------------------------------------------------------


def annotate_parents(node: ast.AST, parent: ast.AST | None = None) -> None:
    """Set ``_parent`` on every node in the tree."""
    node._parent = parent  # type: ignore[attr-defined]
    for child in ast.iter_child_nodes(node):
        annotate_parents(child, node)


# ---------------------------------------------------------------------------
# Phase 3 – Find unwrapped .data.PROPERTY accesses
# ---------------------------------------------------------------------------


def _is_wp_to_torch_call(node: ast.AST) -> bool:
    """Return True if *node* is ``wp.to_torch(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "to_torch"
        and isinstance(func.value, ast.Name)
        and func.value.id == "wp"
    )


def _is_already_wrapped(node: ast.AST) -> bool:
    """Check if *node* sits inside a ``wp.to_torch(...)`` call."""
    parent = getattr(node, "_parent", None)
    return _is_wp_to_torch_call(parent)


class DataAccessFinder(ast.NodeVisitor):
    """Find ``asset.data.PROPERTY`` accesses that need wrapping."""

    def __init__(self, asset_vars: set[tuple[str, ...]]) -> None:
        self.asset_vars = asset_vars
        self.targets: list[WrapTarget] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_unwrapped_data_property(node) or self._is_unwrapped_asset_attribute(node):
            self.targets.append(
                WrapTarget(
                    start_line=node.lineno,
                    start_col=node.col_offset,
                    end_line=node.end_lineno,
                    end_col=node.end_col_offset,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect ``getattr(asset.data, key)`` and ``asset.root_view.get_*()``."""
        if self._is_getattr_data_call(node) or self._is_root_view_getter(node):
            if not _is_already_wrapped(node):
                self.targets.append(
                    WrapTarget(
                        start_line=node.lineno,
                        start_col=node.col_offset,
                        end_line=node.end_lineno,
                        end_col=node.end_col_offset,
                    )
                )
        self.generic_visit(node)

    # -- internal ---------------------------------------------------------

    def _is_unwrapped_data_property(self, node: ast.Attribute) -> bool:
        """Return True if *node* is an unwrapped ``asset.data.PROPERTY``."""
        # 1. Leaf attribute must not be blacklisted
        if node.attr in BLACKLISTED_PROPERTIES:
            return False

        # 2. Immediate value must be .data
        data_node = node.value
        if not isinstance(data_node, ast.Attribute) or data_node.attr != "data":
            return False

        # 3. The chain before .data must be a known asset variable
        chain = get_attr_chain(data_node.value)
        if chain is None or chain not in self.asset_vars:
            return False

        # 4. Skip if this node is the func of a Call (method call, not property)
        parent = getattr(node, "_parent", None)
        if isinstance(parent, ast.Call) and parent.func is node:
            return False

        # 5. Skip if already wrapped in wp.to_torch()
        if _is_already_wrapped(node):
            return False

        return True

    def _is_unwrapped_asset_attribute(self, node: ast.Attribute) -> bool:
        """Return True if *node* is an unwrapped ``asset._ALL_INDICES`` etc."""
        if node.attr not in WARP_ASSET_ATTRIBUTES:
            return False

        # The value chain must be a known asset variable
        chain = get_attr_chain(node.value)
        if chain is None or chain not in self.asset_vars:
            return False

        if _is_already_wrapped(node):
            return False

        return True

    def _is_getattr_data_call(self, node: ast.Call) -> bool:
        """Return True if *node* is ``getattr(asset.data, key)``."""
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2):
            return False
        first_arg = node.args[0]
        if not (isinstance(first_arg, ast.Attribute) and first_arg.attr == "data"):
            return False
        chain = get_attr_chain(first_arg.value)
        return chain is not None and chain in self.asset_vars

    def _is_root_view_getter(self, node: ast.Call) -> bool:
        """Return True if *node* is ``asset.root_view.get_*(...)``."""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        if not func.attr.startswith("get_"):
            return False
        # func.value must be asset.root_view
        rv_node = func.value
        if not (isinstance(rv_node, ast.Attribute) and rv_node.attr == "root_view"):
            return False
        chain = get_attr_chain(rv_node.value)
        return chain is not None and chain in self.asset_vars


# ---------------------------------------------------------------------------
# Phase 4 – Text replacement
# ---------------------------------------------------------------------------


def _extract_text_span(lines: list[str], target: WrapTarget) -> str:
    """Extract the text covered by a WrapTarget from source lines."""
    if target.start_line == target.end_line:
        return lines[target.start_line - 1][target.start_col : target.end_col]
    # Multi-line span
    parts = [lines[target.start_line - 1][target.start_col :]]
    for line_idx in range(target.start_line, target.end_line - 1):
        parts.append(lines[line_idx])
    parts.append(lines[target.end_line - 1][: target.end_col])
    return "\n".join(parts)


def apply_wraps(source: str, targets: list[WrapTarget]) -> str:
    """Insert ``wp.to_torch(`` / ``)`` around every target span."""
    if not targets:
        return source

    lines = source.split("\n")

    # Sort bottom-right → top-left so earlier edits don't shift later offsets
    sorted_targets = sorted(targets, key=lambda t: (t.start_line, t.start_col), reverse=True)

    for target in sorted_targets:
        target.original_text = _extract_text_span(lines, target)

        if target.start_line == target.end_line:
            line = lines[target.start_line - 1]
            lines[target.start_line - 1] = (
                line[: target.start_col]
                + "wp.to_torch("
                + line[target.start_col : target.end_col]
                + ")"
                + line[target.end_col :]
            )
        else:
            # Multi-line: suffix on last line first, then prefix on first line
            last = lines[target.end_line - 1]
            lines[target.end_line - 1] = last[: target.end_col] + ")" + last[target.end_col :]
            first = lines[target.start_line - 1]
            lines[target.start_line - 1] = first[: target.start_col] + "wp.to_torch(" + first[target.start_col :]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 5 – Ensure ``import warp as wp``
# ---------------------------------------------------------------------------


def ensure_warp_import(source: str) -> tuple[str, bool]:
    """Add ``import warp as wp`` if not already present.

    Returns ``(new_source, was_added)``.
    """
    lines = source.split("\n")

    for line in lines:
        stripped = line.strip()
        if stripped == "import warp as wp" or stripped.startswith("import warp as wp"):
            return source, False

    # Find best insertion point
    torch_import_idx: int | None = None
    last_import_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i
            if stripped == "import torch" or stripped.startswith("import torch "):
                torch_import_idx = i

    insert_after = torch_import_idx if torch_import_idx is not None else last_import_idx
    lines.insert(insert_after + 1, "import warp as wp")
    return "\n".join(lines), True


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------


def prompt_user(message, default="y"):
    """Prompt user for yes/no/all/quit response."""
    valid = {"y": "yes", "n": "no", "a": "all", "q": "quit", "": default}
    prompt_str = f"{message} [Y/n/a/q]: "

    while True:
        choice = input(prompt_str).lower().strip()
        if choice in valid:
            return valid[choice] if choice else valid[default]
        print("Please respond with 'y' (yes), 'n' (no), 'a' (all), or 'q' (quit)")


def get_file_context(filepath, line_num, context_lines=2):
    """Get lines around the target line for context."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)

        context = []
        for i in range(start, end):
            prefix = ">>>" if i == line_num - 1 else "   "
            line_content = lines[i].rstrip()
            context.append(f"{prefix} {i + 1:4d} | {line_content}")

        return "\n".join(context)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def process_file(
    filepath: Path,
    *,
    apply: bool,
    verbose: bool,
    force: bool = False,
    context_lines: int = 2,
    apply_all: bool = False,
) -> tuple[FileResult, bool, bool]:
    """Run all five phases on a single file.

    Returns ``(result, apply_all, quit_requested)``.
    *apply_all* propagates across files when the user presses ``a``.
    """
    result = FileResult(path=filepath)
    quit_requested = False
    source = filepath.read_text()

    # -- Parse -----------------------------------------------------------
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        if verbose:
            print(f"  SKIP (syntax error): {filepath}")
        return result, apply_all, False

    # -- Phase 1: collect asset variables --------------------------------
    collector = AssetVariableCollector()
    collector.visit(tree)

    if not collector.asset_vars:
        return result, apply_all, False

    if verbose:
        print(f"  Asset variables in {filepath}: {collector.asset_vars}")

    # -- Phase 2: annotate parents ---------------------------------------
    annotate_parents(tree)

    # -- Phase 3: find unwrapped data accesses ---------------------------
    finder = DataAccessFinder(collector.asset_vars)
    finder.visit(tree)

    if not finder.targets:
        return result, apply_all, False

    # Extract original text for every target before any modifications
    lines = source.split("\n")
    for t in finder.targets:
        t.original_text = _extract_text_span(lines, t)

    # -- Dry-run path: just display --------------------------------------
    if not apply:
        new_source = apply_wraps(source, finder.targets)
        _, import_added = ensure_warp_import(new_source)
        result.targets = finder.targets
        result.import_added = import_added

        num = len(result.targets)
        print(f"{filepath} ({num} change{'s' if num != 1 else ''})")
        display_targets = sorted(result.targets, key=lambda t: (t.start_line, t.start_col))
        for t in display_targets:
            print(f"  L{t.start_line}: {t.original_text}")
            if verbose:
                print(f"       \u2192 wp.to_torch({t.original_text})")
        if import_added:
            print("  + import warp as wp")
        return result, apply_all, False

    # -- Force path: apply everything without prompting ------------------
    if force or apply_all:
        new_source = apply_wraps(source, finder.targets)
        new_source, import_added = ensure_warp_import(new_source)
        result.targets = finder.targets
        result.import_added = import_added

        num = len(result.targets)
        print(f"{filepath} ({num} change{'s' if num != 1 else ''})")
        display_targets = sorted(result.targets, key=lambda t: (t.start_line, t.start_col))
        for t in display_targets:
            print(f"  L{t.start_line}: {t.original_text}")
            if verbose:
                print(f"       \u2192 wp.to_torch({t.original_text})")
        if import_added:
            print("  + import warp as wp")

        filepath.write_text(new_source)
        print("  Applied.")
        return result, apply_all, False

    # -- Interactive path: prompt per-change -----------------------------
    display_targets = sorted(finder.targets, key=lambda t: (t.start_line, t.start_col))
    accepted: list[WrapTarget] = []

    for t in display_targets:
        print("\u2500" * 80)
        print(f"\U0001f4cd {filepath}:L{t.start_line}")
        ctx = get_file_context(filepath, t.start_line, context_lines)
        if ctx:
            print(ctx)
        print(f"  Change: {t.original_text} \u2192 wp.to_torch({t.original_text})")

        if apply_all:
            accepted.append(t)
            continue

        response = prompt_user("Apply this fix?")
        if response == "yes":
            accepted.append(t)
        elif response == "all":
            apply_all = True
            accepted.append(t)
        elif response == "quit":
            quit_requested = True
            break
        # "no" → skip this target

    if accepted:
        new_source = apply_wraps(source, accepted)
        new_source, import_added = ensure_warp_import(new_source)
        result.targets = accepted
        result.import_added = import_added
        filepath.write_text(new_source)
        num = len(accepted)
        print(f"  Applied {num} change{'s' if num != 1 else ''} to {filepath}.")
        if import_added:
            print("  + import warp as wp")
    else:
        print(f"  Skipped {filepath} (no changes accepted).")

    return result, apply_all, quit_requested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrap asset .data.* property accesses with wp.to_torch().")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes interactively (default is dry-run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply all changes without prompting (use with --apply)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Lines of context around each change (default: 2)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per transform")
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: {target_path} does not exist", file=sys.stderr)
        sys.exit(1)

    files: list[Path]
    if target_path.is_file():
        files = [target_path]
    else:
        files = sorted(target_path.rglob("*.py"))

    if not args.apply:
        print("DRY RUN (use --apply to modify files)\n")
    elif args.force:
        print("FORCE APPLY (all changes applied without prompting)\n")
    else:
        print("INTERACTIVE APPLY (prompting per change)\n")

    total_changes = 0
    total_imports = 0
    files_changed = 0
    apply_all = False

    for fp in files:
        res, apply_all, quit_requested = process_file(
            fp,
            apply=args.apply,
            verbose=args.verbose,
            force=args.force,
            context_lines=args.context,
            apply_all=apply_all,
        )
        n = len(res.targets)
        if n:
            total_changes += n
            files_changed += 1
        if res.import_added:
            total_imports += 1
        if quit_requested:
            print("\nQuitting early (user requested).")
            break

    print(f"\nSummary: {total_changes} wraps across {files_changed} file(s), {total_imports} import(s) added")


if __name__ == "__main__":
    main()
