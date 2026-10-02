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

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "towncrier.toml"
TOWNCRIER = ("uvx", "--from", "towncrier==26.9.0", "towncrier")
TYPES = ("added", "changed", "deprecated", "removed", "fixed")
ENTRY_RE = re.compile(rf"^[^./]+\.({'|'.join(TYPES)})\.rst$")
MARKER_RE = re.compile(r"^[^./]+\.(skip|minor|major)$")
RELEASE_NOTES_MARKER = ".. towncrier release notes start"


# ---------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------


def run(*cmd: str | Path, env: dict[str, str] | None = None) -> str:
    """Run ``cmd`` in the repository root and return its stdout; raise on a non-zero exit."""
    return subprocess.run(
        [str(arg) for arg in cmd], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout


def packages() -> list[Path]:
    """Return the package directories that keep a ``docs/CHANGELOG.rst``."""
    return sorted(changelog.parent.parent for changelog in (REPO_ROOT / "source").glob("*/docs/CHANGELOG.rst"))


# ---------------------------------------------------------------------------------------------------
# legacy fragments (WAR): the check rejects the pre-towncrier format, but fragments merged before it, or
# without re-running CI, still reach the nightly, so compile splits them. Delete this section and its
# ``# WAR`` caller once no pending fragment uses the old format.
# ---------------------------------------------------------------------------------------------------

LEGACY_RE = re.compile(r"^(?P<slug>[^./]+)(?:\.(?P<tier>minor|major))?\.rst$")
LEGACY_HEADING_RE = re.compile(r"^(\S[^\n]*)\n\^+[ \t]*\n", re.MULTILINE)


def split_legacy(path: Path) -> dict[str, str]:
    """Return the towncrier fragments ``{name: text}`` equivalent to a legacy fragment.

    A legacy ``<slug>[.minor|.major].rst`` holds ``^``-underlined Added/Changed/Deprecated/Removed/Fixed
    sections. Each section becomes ``<slug>.<type>.rst`` and the tier an empty ``<slug>.minor``/``.major``.

    Raises:
        ValueError: If the file has no sections, or a section is unknown or repeated.
    """
    match = LEGACY_RE.match(path.name)
    parts = LEGACY_HEADING_RE.split(path.read_text(encoding="utf-8"))
    headings = [fragment_type.capitalize() for fragment_type in TYPES]
    if parts[0].strip() or len(parts) < 3:
        raise ValueError(f"expected sections {', '.join(headings)} underlined with ^")
    fragments = {f"{match['slug']}.{match['tier']}": ""} if match["tier"] else {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        name = f"{match['slug']}.{heading.lower()}.rst"
        if heading not in headings or name in fragments:
            raise ValueError(f"unknown or repeated section {heading!r}")
        fragments[name] = body.strip("\n") + "\n"
    return fragments


def split_legacy_fragments(fragments: Path) -> None:
    """Replace each legacy fragment in the ``fragments`` directory with its towncrier fragments.

    Raises:
        ValueError: If a legacy fragment is malformed or its split would overwrite another fragment.
    """
    for path in sorted(fragments.glob("*.rst")):
        if LEGACY_RE.match(path.name):
            for name, text in split_legacy(path).items():
                # Another fragment with the same slug would otherwise lose its entry.
                if (fragments / name).exists():
                    raise ValueError(f"splitting {path.name} would overwrite {name}")
                (fragments / name).write_text(text, encoding="utf-8")
            path.unlink()


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
    if not ENTRY_RE.match(path.name):
        return [
            f"{rel}: expected <slug>.<{'|'.join(TYPES)}>.rst holding * bullets, or an empty <slug>.skip,"
            " <slug>.minor or <slug>.major; see skills/_internal/changelog-fragments/SKILL.md"
        ]
    error = check_bullet_list(path.read_text(encoding="utf-8"))
    return [f"{rel}: {error}"] if error else []


def check_changed_packages(changed: set[str], added: set[str]) -> list[str]:
    """Return the fragment errors for a branch that changed, and newly added, the given repo paths."""
    errors = []
    for pkg in packages():
        rel = pkg.relative_to(REPO_ROOT).as_posix()
        fragment_dir = f"{rel}/changelog.d/"
        touched = {f for f in changed if f.startswith(fragment_dir) and not f.endswith("/.gitkeep")}
        # Pending fragments are immutable: an edit could rewrite another PR's entry.
        errors += [f"{f}: changes a pending fragment; add a new one instead" for f in sorted(touched - added)]
        fragments = sorted(touched & added)
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
        # towncrier matches the whole line; otherwise it writes the release above the file header.
        if f"{RELEASE_NOTES_MARKER}\n" not in (pkg / "docs/CHANGELOG.rst").read_text(encoding="utf-8")
    ]


def check_lock_pins() -> list[str]:
    """Return an error for each local package missing from ``uv.lock`` or pinned there at a stale version.

    Catches a PR that adds a package, edits a version or takes its side of a ``uv.lock`` conflict without
    re-locking, which would make the next ``uv run`` rewrite ``uv.lock``. Only local packages are compared;
    the rest of ``uv.lock`` is not resolved.
    """
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {
        package["source"]["editable"]: package for package in lock["package"] if "editable" in package.get("source", {})
    }
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = root.get("tool", {}).get("uv", {}).get("sources", {}).values()
    errors = [
        f"uv.lock has no entry for {source['path']}; run `uv lock`"
        for source in sources
        if isinstance(source, dict) and source.get("editable") and source["path"] not in locked
    ]
    for path, package in locked.items():
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
    split_legacy_fragments(fragments)  # WAR
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
    """Re-lock ``uv.lock`` for the bumped versions; raise if anything else changed."""
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
        raise RuntimeError("uv lock changed more than package versions")


def cmd_compile(args: argparse.Namespace) -> int:
    """Nightly: bump and build every package with pending entries, re-lock ``uv.lock`` and stage the outputs.

    Returns 1 if anything failed. A failed package is rolled back while the others stay staged; a failed
    re-lock rolls everything back and stages nothing.
    """
    # Rolling back restores HEAD, so start clean to never discard local work.
    if run("git", "status", "--porcelain"):
        print("::error::compile needs a clean checkout; commit or stash local changes first", file=sys.stderr)
        return 1
    failed = False
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
                print(f"{pkg.name} -> {version}")
    try:
        sync_lock()
    except (subprocess.CalledProcessError, RuntimeError) as e:
        # Versions must not move without their lock pins: abort, leaving the checkout at HEAD.
        run("git", "checkout", "HEAD", "--", "uv.lock")
        for pkg in packages():
            restore(pkg)
        print(f"::error::uv.lock: {getattr(e, 'stderr', None) or e}", file=sys.stderr)
        return 1
    # Stage exactly the compile outputs; anything else changed in the checkout is reported, not committed.
    run("git", "add", "-A", "--", "uv.lock", *(path for pkg in packages() for path in outputs(pkg) if path.exists()))
    unexpected = run("git", "ls-files", "--modified", "--others", "--exclude-standard").splitlines()
    if unexpected:
        print("::error::compile changed files outside its outputs:\n" + "\n".join(unexpected), file=sys.stderr)
        failed = True
    return int(failed)


# ---------------------------------------------------------------------------------------------------
# entry point
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
