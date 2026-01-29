#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool to find potential quaternion values that may need wxyz->xyzw conversion.

This script searches for 4-element tuples/lists in Python files and flags those
that haven't been modified compared to the main branch.

Usage:
    python tools/find_quaternions.py [--path PATH] [--show-all] [--check-identity]
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


def get_changed_files_from_base(base_ref="main"):
    """Get list of files that have been modified compared to base ref."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n"))
    except subprocess.CalledProcessError:
        print(f"Warning: Could not get git diff from {base_ref}. Showing all potential quaternions.")
        return set()


def get_diff_lines_for_file(filepath, base_ref="main"):
    """Get the line numbers that have been modified in a file compared to base ref."""
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", base_ref, "--", filepath],
            capture_output=True,
            text=True,
        )
        # Parse the unified diff to get changed line numbers
        changed_lines = set()
        for line in result.stdout.split("\n"):
            # Match lines like @@ -10,5 +12,7 @@
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                start_line = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                for i in range(count):
                    changed_lines.add(start_line + i)
        return changed_lines
    except subprocess.CalledProcessError:
        return set()


def is_potential_quaternion(values):
    """Check if 4 values look like a quaternion."""
    if len(values) != 4:
        return False

    # Skip boolean lists
    if any(isinstance(v, bool) for v in values):
        return False

    # Check if all values are numeric
    try:
        floats = [float(v) for v in values]
    except (ValueError, TypeError):
        return False

    # Check if it's roughly a unit quaternion (sum of squares ≈ 1)
    sum_sq = sum(v * v for v in floats)
    if 0.9 < sum_sq < 1.1:
        return True

    # Also catch identity-like patterns
    if floats in [[1, 0, 0, 0], [0, 0, 0, 1], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]:
        return True

    return False


def is_wxyz_identity(values):
    """Check if values look like wxyz identity quaternion [1, 0, 0, 0]."""
    try:
        # Skip boolean lists
        if any(isinstance(v, bool) for v in values):
            return False
        floats = [float(v) for v in values]
        # wxyz identity: w=1, x=0, y=0, z=0
        return abs(floats[0] - 1.0) < 0.01 and all(abs(v) < 0.01 for v in floats[1:])
    except (ValueError, TypeError):
        return False


def is_wxyz_format_likely(values):
    """Heuristic to check if quaternion might be in wxyz format.

    Returns True if the quaternion is LIKELY in wxyz format and needs conversion.
    Returns False if it looks like valid xyzw (last element is the w component).
    """
    try:
        floats = [float(v) for v in values]
        w_first, x, y, z_last = floats

        # Common cos(theta/2) values for typical rotations (0, 30, 45, 60, 90, 120, 135, 150, 180 degrees)
        common_cos_values = [1.0, 0.9659, 0.866, 0.707, 0.5, 0.2588, 0.0, -0.2588, -0.5, -0.707, -0.866, -0.9659, -1.0]

        def looks_like_w(val):
            """Check if value looks like a w component (cos of half-angle)."""
            return any(abs(val - c) < 0.02 for c in common_cos_values)

        # If last element looks like a valid w component (xyzw format), NOT wxyz
        if looks_like_w(z_last) and abs(z_last) > 0.1:
            return False

        # If first element looks like w and last is small, likely wxyz
        if looks_like_w(w_first) and abs(z_last) < 0.1:
            return True

        # wxyz identity [1,0,0,0] - first is 1, rest are 0
        if abs(w_first - 1.0) < 0.01 and all(abs(v) < 0.01 for v in [x, y, z_last]):
            return True

        # Check for pattern where first element is large and last is small (typical wxyz)
        if abs(w_first) > 0.5 and abs(z_last) < 0.2:
            return True

        # Pattern: [0, sin, 0, 0] or [0, 0, sin, 0] - rotation around single axis in wxyz
        # e.g., [0, 1, 0, 0] is 180° around X in wxyz format
        if abs(w_first) < 0.01 and abs(z_last) < 0.01:
            non_zero = sum(1 for v in [x, y] if abs(v) > 0.1)
            if non_zero == 1:
                return True

        return False
    except (ValueError, TypeError):
        return False


class QuaternionFinder(ast.NodeVisitor):
    """AST visitor to find potential quaternion values."""

    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.quaternions = []

    def visit_Tuple(self, node):
        self._check_sequence(node)
        self.generic_visit(node)

    def visit_List(self, node):
        self._check_sequence(node)
        self.generic_visit(node)

    def _check_sequence(self, node):
        if len(node.elts) != 4:
            return

        # Try to extract constant values
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                values.append(elt.value)
            elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                if isinstance(elt.operand, ast.Constant) and isinstance(elt.operand.value, (int, float)):
                    values.append(-elt.operand.value)
                else:
                    return
            else:
                return

        if is_potential_quaternion(values):
            # Get the source line for context
            line = self.source_lines[node.lineno - 1] if node.lineno <= len(self.source_lines) else ""
            is_identity = is_wxyz_identity(values)
            likely_wxyz = is_wxyz_format_likely(values)
            # Ambiguous: looks like a valid quaternion but we can't tell the format
            is_ambiguous = not is_identity and not likely_wxyz
            self.quaternions.append(
                {
                    "line": node.lineno,
                    "col": node.col_offset,
                    "values": values,
                    "source": line.strip(),
                    "is_wxyz_identity": is_identity,
                    "likely_wxyz": likely_wxyz,
                    "is_ambiguous": is_ambiguous,
                }
            )


def find_quaternions_in_file(filepath):
    """Find potential quaternions in a Python file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
            source_lines = source.split("\n")

        tree = ast.parse(source)
        finder = QuaternionFinder(source_lines)
        finder.visit(tree)
        return finder.quaternions
    except (SyntaxError, UnicodeDecodeError):
        return []


def find_quaternions_in_json(filepath):
    """Find potential quaternions in JSON files."""
    import json

    quaternions = []

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Simple regex to find 4-element arrays
        pattern = r"\[([^[\]]+)\]"
        for i, line in enumerate(lines, 1):
            for match in re.finditer(pattern, line):
                try:
                    values = [float(v.strip()) for v in match.group(1).split(",")]
                    if is_potential_quaternion(values):
                        is_identity = is_wxyz_identity(values)
                        likely_wxyz = is_wxyz_format_likely(values)
                        quaternions.append(
                            {
                                "line": i,
                                "col": match.start(),
                                "values": values,
                                "source": line.strip(),
                                "is_wxyz_identity": is_identity,
                                "likely_wxyz": likely_wxyz,
                                "is_ambiguous": not is_identity and not likely_wxyz,
                            }
                        )
                except ValueError:
                    pass
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    return quaternions


def find_quaternions_in_rst(filepath):
    """Find potential quaternions in RST documentation files."""
    quaternions = []

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        # Patterns to find quaternions in RST files
        # Look for tuples like (1.0, 0.0, 0.0, 0.0) or lists like [1.0, 0.0, 0.0, 0.0]
        tuple_pattern = r"\(([^()]+)\)"
        list_pattern = r"\[([^[\]]+)\]"

        for i, line in enumerate(lines, 1):
            for pattern in [tuple_pattern, list_pattern]:
                for match in re.finditer(pattern, line):
                    try:
                        # Try to parse as comma-separated floats
                        parts = match.group(1).split(",")
                        if len(parts) != 4:
                            continue
                        values = [float(v.strip()) for v in parts]
                        if is_potential_quaternion(values):
                            is_identity = is_wxyz_identity(values)
                            likely_wxyz = is_wxyz_format_likely(values)
                            quaternions.append(
                                {
                                    "line": i,
                                    "col": match.start(),
                                    "values": values,
                                    "source": line.strip()[:100],
                                    "is_wxyz_identity": is_identity,
                                    "likely_wxyz": likely_wxyz,
                                    "is_ambiguous": not is_identity and not likely_wxyz,
                                }
                            )
                    except ValueError:
                        pass
    except UnicodeDecodeError:
        pass

    return quaternions


def convert_wxyz_to_xyzw(values):
    """Convert quaternion from wxyz [w,x,y,z] to xyzw [x,y,z,w] format."""
    w, x, y, z = values
    return [x, y, z, w]


def format_quaternion(values, use_tuple=False, use_float=True):
    """Format quaternion values as a string."""
    if use_float:
        # Preserve the original precision
        formatted = [f"{v}" for v in values]
    else:
        formatted = [str(v) for v in values]

    if use_tuple:
        return f"({', '.join(formatted)})"
    else:
        return f"[{', '.join(formatted)}]"


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


def apply_fix(filepath, line_num, old_values, new_values):
    """Apply a fix to a specific line in a file. Replaces ALL occurrences on the line.

    Returns (success, new_line_content).
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        if line_num > len(lines):
            return False, None

        line = lines[line_num - 1]
        modified = False

        # Detect if tuple or list format - replace ALL occurrences
        for use_tuple in [True, False]:
            for use_float in [True, False]:
                old_str = format_quaternion(old_values, use_tuple, use_float)
                if old_str in line:
                    new_str = format_quaternion(new_values, use_tuple, use_float)
                    line = line.replace(old_str, new_str)  # Replace ALL
                    modified = True

        # Try without spaces - replace ALL
        for use_tuple in [True, False]:
            old_str = f"{'(' if use_tuple else '['}{','.join(str(v) for v in old_values)}{')' if use_tuple else ']'}"
            if old_str in line:
                new_str = (
                    f"{'(' if use_tuple else '['}{', '.join(str(v) for v in new_values)}{')' if use_tuple else ']'}"
                )
                line = line.replace(old_str, new_str)  # Replace ALL
                modified = True

        if modified:
            lines[line_num - 1] = line
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True, line.rstrip()

        return False, None
    except Exception as e:
        print(f"  Error applying fix: {e}")
        return False, None


def preview_fix(filepath, line_num, old_values, new_values):
    """Preview what a fix would look like without applying it. Shows ALL replacements."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        if line_num > len(lines):
            return None, 0

        line = lines[line_num - 1]
        count = 0

        # Try to find and replace the old pattern - ALL occurrences
        for use_tuple in [True, False]:
            for use_float in [True, False]:
                old_str = format_quaternion(old_values, use_tuple, use_float)
                occurrences = line.count(old_str)
                if occurrences > 0:
                    new_str = format_quaternion(new_values, use_tuple, use_float)
                    line = line.replace(old_str, new_str)
                    count += occurrences

        # Try without spaces
        for use_tuple in [True, False]:
            old_str = f"{'(' if use_tuple else '['}{','.join(str(v) for v in old_values)}{')' if use_tuple else ']'}"
            occurrences = line.count(old_str)
            if occurrences > 0:
                new_str = (
                    f"{'(' if use_tuple else '['}{', '.join(str(v) for v in new_values)}{')' if use_tuple else ']'}"
                )
                line = line.replace(old_str, new_str)
                count += occurrences

        if count > 0:
            return line.rstrip(), count
        return None, 0
    except Exception:
        return None, 0


def prompt_user(message, default="y"):
    """Prompt user for yes/no/all/quit response."""
    valid = {"y": "yes", "n": "no", "a": "all", "q": "quit", "": default}
    prompt_str = f"{message} [Y/n/a/q]: "

    while True:
        choice = input(prompt_str).lower().strip()
        if choice in valid:
            return valid[choice] if choice else valid[default]
        print("Please respond with 'y' (yes), 'n' (no), 'a' (all), or 'q' (quit)")


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Find potential quaternion values that may need conversion")
    parser.add_argument("--path", default="source", help="Path to search (default: source)")
    parser.add_argument("--show-all", action="store_true", help="Show all quaternions, not just unchanged ones")
    parser.add_argument("--check-identity", action="store_true", help="Only show wxyz identity quaternions [1,0,0,0]")
    parser.add_argument("--likely-wxyz", action="store_true", help="Only show quaternions likely in wxyz format")
    parser.add_argument("--fix", action="store_true", help="Interactively fix quaternions (converts wxyz to xyzw)")
    parser.add_argument("--fix-identity-only", action="store_true", help="Only fix identity quaternions [1,0,0,0]")
    parser.add_argument("--force", action="store_true", help="Apply fixes without confirmation (use with --fix)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--context", type=int, default=2, help="Lines of context to show (default: 2)")
    parser.add_argument(
        "--all-quats", "--all", action="store_true", help="Show ALL potential quaternions (ignore format heuristics)"
    )
    parser.add_argument("--base", default="main", help="Git ref to compare against (default: main)")
    args = parser.parse_args()

    # Get changed files
    changed_files = get_changed_files_from_base(args.base)

    # Find all Python, JSON, and RST files
    search_path = Path(args.path)
    if not search_path.exists():
        print(f"Error: Path '{args.path}' does not exist")
        sys.exit(1)

    py_files = list(search_path.rglob("*.py"))
    json_files = list(search_path.rglob("*.json"))
    rst_files = list(search_path.rglob("*.rst"))

    print(f"Searching {len(py_files)} Python, {len(json_files)} JSON, and {len(rst_files)} RST files...")
    print(f"Comparing against: {args.base}")
    print(f"Found {len(changed_files)} files changed from {args.base}\n")

    findings = []

    for filepath in py_files:
        rel_path = str(filepath)
        quaternions = find_quaternions_in_file(filepath)

        if not quaternions:
            continue

        # Get changed lines for this file
        changed_lines = get_diff_lines_for_file(rel_path, args.base)

        for q in quaternions:
            # Filter based on options
            if args.check_identity and not q["is_wxyz_identity"]:
                continue
            if args.likely_wxyz and not q["likely_wxyz"]:
                continue
            # --all-quats shows everything regardless of heuristics
            # (no filtering applied)

            # Check if this line was changed
            is_changed = q["line"] in changed_lines

            if args.show_all or not is_changed:
                findings.append({"file": rel_path, "changed": is_changed, **q})

    for filepath in json_files:
        rel_path = str(filepath)
        quaternions = find_quaternions_in_json(filepath)

        if not quaternions:
            continue

        changed_lines = get_diff_lines_for_file(rel_path, args.base)

        for q in quaternions:
            if args.check_identity and not q["is_wxyz_identity"]:
                continue
            if args.likely_wxyz and not q["likely_wxyz"]:
                continue

            is_changed = q["line"] in changed_lines

            if args.show_all or not is_changed:
                findings.append({"file": rel_path, "changed": is_changed, **q})

    for filepath in rst_files:
        rel_path = str(filepath)
        quaternions = find_quaternions_in_rst(filepath)

        if not quaternions:
            continue

        changed_lines = get_diff_lines_for_file(rel_path, args.base)

        for q in quaternions:
            if args.check_identity and not q["is_wxyz_identity"]:
                continue
            if args.likely_wxyz and not q["likely_wxyz"]:
                continue

            is_changed = q["line"] in changed_lines

            if args.show_all or not is_changed:
                findings.append({"file": rel_path, "changed": is_changed, **q})

    # Sort by priority: unchanged wxyz identity first, then unchanged likely wxyz, then others
    def sort_key(f):
        priority = 0
        if not f["changed"]:
            priority -= 100
        if f["is_wxyz_identity"]:
            priority -= 50
        if f["likely_wxyz"]:
            priority -= 25
        return priority

    findings.sort(key=sort_key)

    # Print results
    if not findings:
        print("No potential quaternions found that need review!")
        return

    print(f"Found {len(findings)} potential quaternions to review:\n")
    print("=" * 100)

    fixed_count = 0
    skipped_count = 0
    apply_all = args.force  # If --force, apply all without prompting

    for f in findings:
        status = "✓ CHANGED" if f["changed"] else "⚠ UNCHANGED"
        flags = []
        if f["is_wxyz_identity"]:
            flags.append("WXYZ_IDENTITY")
        if f["likely_wxyz"]:
            flags.append("LIKELY_WXYZ")
        if f.get("is_ambiguous"):
            flags.append("AMBIGUOUS")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        # Handle fixing mode
        if (args.fix or args.fix_identity_only) and not f["changed"]:
            # Skip non-identity if fix-identity-only
            if args.fix_identity_only and not f["is_wxyz_identity"]:
                print(f"\n{f['file']}:{f['line']}:{f['col']} {status}{flag_str}")
                print(f"  Values: {f['values']}")
                print("  ⏭ Skipped: not an identity quaternion")
                skipped_count += 1
                continue

            # Skip ambiguous unless --all-quats is set
            if not args.all_quats and not f["likely_wxyz"] and not f["is_wxyz_identity"]:
                print(f"\n{f['file']}:{f['line']}:{f['col']} {status}{flag_str}")
                print(f"  Values: {f['values']}")
                print("  ⏭ Skipped: ambiguous format (use --all to include)")
                skipped_count += 1
                continue

            # Skip boolean lists (false positive)
            if any(isinstance(v, bool) for v in f["values"]):
                print(f"\n{f['file']}:{f['line']}:{f['col']} {status}{flag_str}")
                print(f"  Values: {f['values']}")
                print("  ⏭ Skipped: boolean list, not a quaternion")
                skipped_count += 1
                continue

            new_values = convert_wxyz_to_xyzw(f["values"])

            # Show context
            print("\n" + "─" * 80)
            print(f"📍 {f['file']}:{f['line']}{flag_str}")
            print("─" * 80)

            context = get_file_context(f["file"], f["line"], args.context)
            if context:
                print(context)

            print("─" * 80)
            print(f"  Change: {f['values']} → {new_values}")

            # Preview the fix
            preview, count = preview_fix(f["file"], f["line"], f["values"], new_values)
            if preview:
                count_str = f" ({count} occurrence{'s' if count > 1 else ''})" if count > 1 else ""
                print(f"  Result{count_str}: {preview.strip()}")

            if args.dry_run:
                print("  [DRY RUN - no changes made]")
                continue

            # Prompt user unless force mode or apply_all
            if not apply_all:
                response = prompt_user("Apply this fix?")
                if response == "quit":
                    print("\n⛔ Aborted by user")
                    break
                if response == "all":
                    apply_all = True
                elif response == "no":
                    print("  ⏭ Skipped")
                    skipped_count += 1
                    continue

            # Apply the fix
            success, new_line = apply_fix(f["file"], f["line"], f["values"], new_values)
            if success:
                print("  ✅ Fixed!")
                fixed_count += 1
            else:
                print("  ❌ Failed to fix (pattern not found exactly)")
                skipped_count += 1
        else:
            # Just display mode (no fixing)
            print(f"\n{f['file']}:{f['line']}:{f['col']} {status}{flag_str}")
            print(f"  Values: {f['values']}")
            print(f"  Source: {f['source'][:80]}...")

    print("\n" + "=" * 100)

    # Summary
    unchanged = sum(1 for f in findings if not f["changed"])
    wxyz_identity = sum(1 for f in findings if f["is_wxyz_identity"] and not f["changed"])
    likely_wxyz = sum(1 for f in findings if f["likely_wxyz"] and not f["changed"])
    ambiguous = sum(1 for f in findings if f.get("is_ambiguous") and not f["changed"])

    print("\nSummary:")
    print(f"  Total potential quaternions: {len(findings)}")
    print(f"  Unchanged from main: {unchanged}")
    print(f"  Unchanged wxyz identity [1,0,0,0]: {wxyz_identity}")
    print(f"  Unchanged likely wxyz format: {likely_wxyz}")
    print(f"  Unchanged ambiguous (review manually): {ambiguous}")

    if args.fix or args.fix_identity_only:
        print("\nFix Results:")
        print(f"  Fixed: {fixed_count}")
        print(f"  Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
