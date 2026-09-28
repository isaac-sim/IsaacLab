# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check and compile per-package changelog fragments with towncrier and uv.

Each PR adds ``source/<pkg>/changelog.d/<slug>.<type>.rst`` per entry type (``added``, ``changed``,
``deprecated``, ``removed``, ``fixed``) holding ``* `` bullets, or an empty ``<slug>.skip`` for no
entry. An empty ``<slug>.minor`` or ``<slug>.major`` raises the version bump above patch.

Usage::

    cli.py check [<base-branch>] [--include-worktree]   # PR gate, run by pre-commit and CI
    cli.py compile --all                                # nightly: bump, build CHANGELOG.rst, re-lock
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "towncrier.toml"
TOWNCRIER = ("uvx", "--from", "towncrier==26.9.0", "towncrier")
TYPES = ("added", "changed", "deprecated", "removed", "fixed")
ENTRY_RE = re.compile(rf"^[^./]+\.({'|'.join(TYPES)})\.rst$")
MARKER_RE = re.compile(r"^[^./]+\.(skip|minor|major)$")


def run(*cmd: str | Path, env: dict[str, str] | None = None) -> str:
    """Run ``cmd`` in the repository root and return its stdout; raise on a non-zero exit."""
    return subprocess.run(
        [str(arg) for arg in cmd], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout


def packages() -> list[Path]:
    """Return the package directories that keep a ``docs/CHANGELOG.rst``."""
    return sorted(changelog.parent.parent for changelog in (REPO_ROOT / "source").glob("*/docs/CHANGELOG.rst"))


# WAR: pre-towncrier fragments ``<slug>[.minor|.major].rst`` with ``^``-underlined sections are still
# accepted by ``check`` and split by ``compile``, so open PRs need no migration. Delete this block and
# its two uses once no open PR carries them.
LEGACY_RE = re.compile(r"^(?P<slug>[^./]+)(?:\.(?P<tier>minor|major))?\.rst$")
HEADING_RE = re.compile(r"^(\S[^\n]*)\n\^+[ \t]*\n", re.MULTILINE)
ORPHAN_RE = re.compile(r"^(?!\*|\s|$)", re.MULTILINE)


def split_legacy(path: Path) -> dict[str, str]:
    """Return the towncrier fragments ``{name: text}`` equivalent to a legacy fragment.

    Raises:
        ValueError: If a section is unknown, repeated, or not a ``* `` bullet list.
    """
    match = LEGACY_RE.match(path.name)
    parts = HEADING_RE.split(path.read_text(encoding="utf-8"))
    if parts[0].strip() or len(parts) < 3:
        raise ValueError(f"expected sections {', '.join(t.title() for t in TYPES)} underlined with ^")
    fragments = {f"{match['slug']}.{match['tier']}": ""} if match["tier"] else {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        name = f"{match['slug']}.{heading.lower()}.rst"
        if heading.lower() not in TYPES or name in fragments:
            raise ValueError(f"unknown or repeated section {heading!r}")
        if not re.search(r"^\s*\*", body, re.MULTILINE) or ORPHAN_RE.search(body):
            raise ValueError(f"section {heading!r} must be a ``* `` bullet list")
        fragments[name] = body.strip("\n") + "\n"
    return fragments


def evaluate(changed: set[str], added: set[str]) -> list[str]:
    """Return the fragment errors for a branch that changed, and newly added, the given repo paths."""
    errors = []
    for pkg in packages():
        rel = pkg.relative_to(REPO_ROOT).as_posix()
        fragment_dir = f"{rel}/changelog.d/"
        fragments = sorted(f for f in added if f.startswith(fragment_dir) and not f.endswith("/.gitkeep"))
        for fragment in fragments:
            name = fragment.removeprefix(fragment_dir)
            if LEGACY_RE.match(name):  # WAR
                try:
                    split_legacy(REPO_ROOT / fragment)
                except ValueError as e:
                    errors.append(f"{fragment}: {e}")
            elif not (ENTRY_RE.match(name) or MARKER_RE.match(name)):
                errors.append(f"{fragment}: name it <slug>.<type>.rst, <slug>.skip, <slug>.minor or <slug>.major")
        entries = [f for f in fragments if not f.endswith((".minor", ".major"))]
        if not entries and any(f.startswith(f"{rel}/") and not f.startswith(fragment_dir) for f in changed):
            errors.append(
                f"{rel}: changed without a changelog fragment; add {fragment_dir}<slug>.<type>.rst or <slug>.skip"
            )
    return errors


def cmd_check(args: argparse.Namespace) -> int:
    base = run("git", "merge-base", f"origin/{args.base_ref}", "HEAD").strip()
    target = [] if args.include_worktree else ["HEAD"]
    changed = run("git", "diff", "--name-only", "--no-renames", base, *target).splitlines()
    added = run("git", "diff", "--name-only", "--no-renames", "--diff-filter=A", base, *target).splitlines()
    errors = evaluate(set(changed), set(added))
    for error in errors:
        print(f"::error::{error}")
    return int(bool(errors))


def compile_package(pkg: Path) -> str | None:
    """Build ``pkg``'s pending fragments into its CHANGELOG.rst and bump its version.

    Returns:
        The new version, or ``None`` if the package had no entries to release.
    """
    fragments = pkg / "changelog.d"
    if not fragments.is_dir():
        return None
    for path in sorted(fragments.glob("*.rst")):  # WAR
        if LEGACY_RE.match(path.name):
            for name, text in split_legacy(path).items():
                (fragments / name).write_text(text, encoding="utf-8")
            path.unlink()
    names = [path.name for path in fragments.iterdir()]
    if not any(ENTRY_RE.match(name) for name in names):
        for name in filter(MARKER_RE.match, names):  # stale skip and tier files
            (fragments / name).unlink()
        return None
    bump = next((tier for tier in ("major", "minor") if any(n.endswith(f".{tier}") for n in names)), "patch")
    version = run("uv", "version", "--project", pkg, "--bump", bump, "--frozen", "--short").strip()
    run(*TOWNCRIER, "build", "--yes", "--config", CONFIG, "--dir", pkg, "--version", version)
    return version


def sync_lock() -> None:
    """Re-lock ``uv.lock`` for the bumped versions, refusing any other change.

    Another uv release can re-serialize the whole lockfile; unlocked dependency edits can move pins.
    """
    run("uv", "lock", env={k: v for k, v in os.environ.items() if k != "UV_FROZEN"})
    diff = run("git", "diff", "--no-color", "-U0", "--", "uv.lock")
    moved = [line for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if any(not line[1:].startswith("version = ") for line in moved):
        run("git", "checkout", "--", "uv.lock")
        raise RuntimeError("uv lock changed more than package versions; uv.lock left unchanged")


def cmd_compile(args: argparse.Namespace) -> int:
    failed = False
    for pkg in packages():
        try:
            version = compile_package(pkg)
        except (subprocess.CalledProcessError, ValueError) as e:
            # Per-package isolation: roll this package back; the nightly commits the others.
            run("git", "checkout", "HEAD", "--", pkg / "pyproject.toml", pkg / "docs/CHANGELOG.rst")
            print(f"::error::{pkg.name}: {getattr(e, 'stderr', None) or e}", file=sys.stderr)
            failed = True
        else:
            if version:
                print(f"{pkg.name} -> {version}")
    try:
        sync_lock()
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"::error::uv.lock: {getattr(e, 'stderr', None) or e}", file=sys.stderr)
        failed = True
    return int(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="PR gate: every changed package adds a valid fragment.")
    check.add_argument("base_ref", nargs="?", default=os.environ.get("ISAACLAB_CHANGELOG_BASE_REF", "develop"))
    check.add_argument("--include-worktree", action="store_true", help="Include uncommitted tracked changes.")
    check.set_defaults(func=cmd_check)
    compile_parser = sub.add_parser("compile", help="Bump versions, build CHANGELOG.rst entries and re-lock.")
    # The nightly passes ``--all`` to every branch; pre-towncrier branches still require it.
    compile_parser.add_argument("--all", action="store_true", help="Compile every package (the only mode).")
    compile_parser.set_defaults(func=cmd_compile)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
