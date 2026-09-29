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
    cli.py compile                                      # nightly: bump, build CHANGELOG.rst, re-lock
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import legacy
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "towncrier.toml"
TOWNCRIER = ("uvx", "--from", "towncrier==26.9.0", "towncrier")
TYPES = ("added", "changed", "deprecated", "removed", "fixed")
ENTRY_RE = re.compile(rf"^[^./]+\.({'|'.join(TYPES)})\.rst$")
MARKER_RE = re.compile(r"^[^./]+\.(skip|minor|major)$")
RELEASE_NOTES_MARKER = ".. towncrier release notes start"


def run(*cmd: str | Path, env: dict[str, str] | None = None) -> str:
    """Run ``cmd`` in the repository root and return its stdout; raise on a non-zero exit."""
    return subprocess.run(
        [str(arg) for arg in cmd], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout


def packages() -> list[Path]:
    """Return the package directories that keep a ``docs/CHANGELOG.rst``."""
    return sorted(changelog.parent.parent for changelog in (REPO_ROOT / "source").glob("*/docs/CHANGELOG.rst"))


# ---------------------------------------------------------------------------------------------------
# check: the PR gate
# ---------------------------------------------------------------------------------------------------


def check_bullet_list(text: str) -> str | None:
    """Return why a fragment body is not a ``* `` bullet list, or ``None``.

    A flush-left line after a bullet ends the RST list and splits the entry.
    """
    if not re.search(r"^\s*\*", text, re.MULTILINE):
        return "has no ``* `` bullet"
    if re.search(r"^(?!\*|\s|$)", text, re.MULTILINE):
        return "has a line that is neither a ``* `` bullet nor an indented continuation"
    return None


def check_fragment(path: Path) -> list[str]:
    """Return the errors of one added fragment, named relative to the repository."""
    rel = path.relative_to(REPO_ROOT)
    if MARKER_RE.match(path.name):
        return []
    if ENTRY_RE.match(path.name):
        bodies = [path.read_text(encoding="utf-8")]
    elif legacy.LEGACY_RE.match(path.name):  # WAR
        try:
            bodies = [text for name, text in legacy.split_legacy(path).items() if name.endswith(".rst")]
        except ValueError as e:
            return [f"{rel}: {e}"]
    else:
        return [f"{rel}: name it <slug>.<type>.rst, <slug>.skip, <slug>.minor or <slug>.major"]
    return [f"{rel}: {error}" for body in bodies if (error := check_bullet_list(body))]


def check_changed_packages(changed: set[str], added: set[str]) -> list[str]:
    """Return the fragment errors for a branch that changed, and newly added, the given repo paths."""
    errors = []
    for pkg in packages():
        rel = pkg.relative_to(REPO_ROOT).as_posix()
        fragment_dir = f"{rel}/changelog.d/"
        fragments = sorted(f for f in added if f.startswith(fragment_dir) and not f.endswith("/.gitkeep"))
        for fragment in fragments:
            errors += check_fragment(REPO_ROOT / fragment)
        entries = [f for f in fragments if not f.endswith((".minor", ".major"))]
        if not entries and any(f.startswith(f"{rel}/") and not f.startswith(fragment_dir) for f in changed):
            errors.append(
                f"{rel}: changed without a changelog fragment; add {fragment_dir}<slug>.<type>.rst or <slug>.skip"
            )
    return errors


def check_markers() -> list[str]:
    """Return an error for each ``CHANGELOG.rst`` missing the line towncrier writes new entries after."""
    return [
        f"{pkg.relative_to(REPO_ROOT)}/docs/CHANGELOG.rst: missing the {RELEASE_NOTES_MARKER!r} line"
        for pkg in packages()
        if RELEASE_NOTES_MARKER not in (pkg / "docs/CHANGELOG.rst").read_text(encoding="utf-8")
    ]


def check_lock_pins() -> list[str]:
    """Return an error for each local package whose ``uv.lock`` version differs from its ``pyproject.toml``.

    Catches a PR that edits a version or takes its side of a ``uv.lock`` conflict, which would make the next
    ``uv run`` rewrite ``uv.lock``. Only these pins are compared; the rest of ``uv.lock`` is not resolved.
    """
    errors = []
    for package in tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]:
        path = package.get("source", {}).get("editable")
        if path:
            project = tomllib.loads((REPO_ROOT / path / "pyproject.toml").read_text(encoding="utf-8"))["project"]
            if project.get("version") != package["version"]:
                errors.append(
                    f"uv.lock pins {package['name']} {package['version']}, but {path}/pyproject.toml is"
                    f" {project.get('version')}; run `uv lock`"
                )
    return errors


def cmd_check(args: argparse.Namespace) -> int:
    """PR gate: check the fragments added since the merge base, the release-notes markers and the lock pins."""
    base = run("git", "merge-base", f"origin/{args.base_ref}", "HEAD").strip()
    target = [] if args.include_worktree else ["HEAD"]
    changed = run("git", "diff", "--name-only", "--no-renames", base, *target).splitlines()
    added = run("git", "diff", "--name-only", "--no-renames", "--diff-filter=A", base, *target).splitlines()
    errors = check_changed_packages(set(changed), set(added)) + check_markers() + check_lock_pins()
    for error in errors:
        print(f"::error::{error}")
    return int(bool(errors))


# ---------------------------------------------------------------------------------------------------
# compile: the nightly
# ---------------------------------------------------------------------------------------------------


def outputs(pkg: Path) -> list[Path]:
    """Return the paths ``compile`` may change in ``pkg``."""
    return [pkg / "pyproject.toml", pkg / "docs/CHANGELOG.rst", pkg / "changelog.d"]


def restore(pkg: Path) -> None:
    """Roll ``pkg``'s compile outputs back to ``HEAD``."""
    run("git", "checkout", "HEAD", "--", *outputs(pkg))
    run("git", "clean", "-fdq", "--", pkg / "changelog.d")


def compile_package(pkg: Path) -> str | None:
    """Build ``pkg``'s pending fragments into its CHANGELOG.rst and bump its version.

    Returns:
        The new version, or ``None`` if the package had no entries to release.
    """
    fragments = pkg / "changelog.d"
    if not fragments.is_dir():
        return None
    # WAR: split legacy multi-section fragments into towncrier's one-type-per-file fragments, so open PRs
    # need no migration. Remove with legacy.py once they have merged.
    for path in sorted(fragments.glob("*.rst")):
        if legacy.LEGACY_RE.match(path.name):
            for name, text in legacy.split_legacy(path).items():
                # Another fragment with the same slug would otherwise lose its entry.
                if (fragments / name).exists():
                    raise ValueError(f"splitting {path.name} would overwrite {name}")
                (fragments / name).write_text(text, encoding="utf-8")
            path.unlink()
    names = [path.name for path in fragments.iterdir()]
    # No entries means no release: drop the .skip and tier files, so a stale tier cannot bump a later release.
    if not any(ENTRY_RE.match(name) for name in names):
        for name in filter(MARKER_RE.match, names):
            (fragments / name).unlink()
        return None
    bump = next((tier for tier in ("major", "minor") if any(n.endswith(f".{tier}") for n in names)), "patch")
    version = run("uv", "version", "--project", pkg, "--bump", bump, "--frozen", "--short").strip()
    run(*TOWNCRIER, "build", "--yes", "--config", CONFIG, "--dir", pkg, "--version", version)
    return version


def without_local_versions(lock_text: str) -> dict:
    """Return a parsed ``uv.lock`` without the versions of the local (editable) packages."""
    lock = tomllib.loads(lock_text)
    for package in lock["package"]:
        if "editable" in package.get("source", {}):
            package.pop("version", None)
    return lock


def sync_lock() -> None:
    """Re-lock ``uv.lock`` for the bumped versions; restore it and raise on any other change."""
    before = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    # Under UV_FROZEN, which developer shells may export, `uv lock` only validates and keeps the old pins.
    run("uv", "lock", env={k: v for k, v in os.environ.items() if k != "UV_FROZEN"})
    after = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    diff = run("git", "diff", "--no-color", "-U0", "--", "uv.lock")
    moved = [line for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    # Only version lines may move, which rejects a uv release that re-serializes the file, and only local
    # packages' versions may change, which rejects a moved dependency pin.
    if any(not line[1:].startswith("version = ") for line in moved) or (
        without_local_versions(before) != without_local_versions(after)
    ):
        run("git", "checkout", "--", "uv.lock")
        raise RuntimeError("uv lock changed more than package versions; uv.lock left unchanged")


def cmd_compile(args: argparse.Namespace) -> int:
    """Nightly: bump and build every package with pending entries, re-lock ``uv.lock`` and stage the outputs.

    Returns 1 if anything failed; whatever succeeded stays staged.
    """
    # Rolling back a failed package restores HEAD, so start clean to never discard local work.
    if run("git", "status", "--porcelain"):
        print("::error::compile needs a clean checkout; commit or stash local changes first", file=sys.stderr)
        return 1
    failed = False
    bumped = []
    for pkg in packages():
        try:
            version = compile_package(pkg)
        except (subprocess.CalledProcessError, ValueError) as e:
            # Per-package isolation: roll this package back; the nightly commits the others.
            restore(pkg)
            print(f"::error::{pkg.name}: {getattr(e, 'stderr', None) or e}", file=sys.stderr)
            failed = True
        else:
            if version:
                bumped.append(pkg)
                print(f"{pkg.name} -> {version}")
    try:
        sync_lock()
    except (subprocess.CalledProcessError, RuntimeError) as e:
        # Versions must not move without their lock pins, so hold every bump until the lock is fixed.
        for pkg in bumped:
            restore(pkg)
        print(f"::error::uv.lock: {getattr(e, 'stderr', None) or e}", file=sys.stderr)
        failed = True
    # Stage exactly the compile outputs; anything else changed in the checkout is reported, not committed.
    run("git", "add", "-A", "--", "uv.lock", *(path for pkg in packages() for path in outputs(pkg) if path.exists()))
    unexpected = run("git", "ls-files", "--modified", "--others", "--exclude-standard").splitlines()
    if unexpected:
        print("::error::compile changed files outside its outputs:\n" + "\n".join(unexpected), file=sys.stderr)
        failed = True
    return int(failed)


# ---------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="PR gate: every changed package adds a valid fragment.")
    check.add_argument("base_ref", nargs="?", default=os.environ.get("ISAACLAB_CHANGELOG_BASE_REF", "develop"))
    check.add_argument("--include-worktree", action="store_true", help="Include uncommitted tracked changes.")
    check.set_defaults(func=cmd_check)
    compile_parser = sub.add_parser("compile", help="Bump versions, build CHANGELOG.rst entries and re-lock.")
    compile_parser.set_defaults(func=cmd_compile)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
