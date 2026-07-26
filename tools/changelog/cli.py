# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manage changelog fragments — single entry point for the whole lifecycle.

Each PR drops a fragment under ``source/<package>/changelog.d/<slug>.rst``.
The slug is any short, unique name — the contributor's branch name (with
``/`` replaced by ``-``) is the recommended default. The file mirrors
the RST that will appear in the changelog — one or more section headings
(``Added``, ``Changed``, ``Deprecated``, ``Removed``, ``Fixed``) each
underlined with ``^``. The **filename suffix** declares the bump tier:

- ``<slug>.rst`` — patch bump.
- ``<slug>.minor.rst`` — minor bump.
- ``<slug>.major.rst`` — major bump.
- ``<slug>.skip`` — no entry, no bump.

When a batch compiles together, the highest declared bump wins for the
package (one ``.major.rst`` anywhere → major).

Subcommands:

  check      PR gate. Verifies every modified package has a valid fragment.
  compile    Roll accumulated fragments into ``CHANGELOG.rst`` and bump each
             package's version metadata file. Run by maintainers when
             cutting a release.
  auto-bump  The whole nightly lifecycle: compile every package, sync
             ``uv.lock``, stage exactly what was written, commit as the bot,
             and push to the target branch with retry-on-conflict. Run by
             ``.github/workflows/nightly-changelog.yml`` on a cron.
  sync-lock  Re-point ``uv.lock``'s workspace-member versions at their
             manifests. ``auto-bump`` does this automatically; run it
             directly to repair a lock by hand or, with ``--check``, as a
             gate.

Which file holds a package's version is the branch's business, not this
tool's — :attr:`Package.toml_path` resolves it, so the same ``cli.py`` is
correct on branches that keep versions in ``pyproject.toml`` and on release
branches that still keep them in ``config/extension.toml``. That matters
because the nightly workflow lives only on the default branch and invokes
whatever ``cli.py`` each target branch carries.

Usage:

    # ── check ─────────────────────────────────────────────────────
    # CI invocation on every pull_request:
    cli.py check <base-branch>

    # ── compile ───────────────────────────────────────────────────
    # Normal release-time invocation — bump every managed package
    # from accumulated fragments, write entries, delete fragments:
    cli.py compile --all

    # Preview only (no writes, no deletes):
    cli.py compile --all --dry-run

    # Pin one package to a specific version (single-package only —
    # each managed package has its own version trajectory):
    cli.py compile --package isaaclab --version 4.7.0

    # Preview against a worked example without touching real packages:
    cli.py compile --package isaaclab --dry-run \\
        --fragments-dir tools/changelog/test/integration/02_minor_bump/fragments

    # ── auto-bump ─────────────────────────────────────────────────
    # What the nightly cron runs (the workflow supplies the branch):
    cli.py auto-bump --branch develop --remote origin \\
        --event-name schedule

    # ── sync-lock ─────────────────────────────────────────────────
    # Repair a lock whose member versions drifted from their manifests:
    cli.py sync-lock

    # Report drift and exit non-zero, without writing:
    cli.py sync-lock --check

For big version jumps (e.g. ``2.1`` → ``4.7``) edit the package's version
metadata file directly and prepend a manual entry to
``source/<pkg>/docs/CHANGELOG.rst``. The compiler is for fragment-driven
incremental bumps, not for jumps.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from autobump import AutoBumpRun
from lockfile import LockFile
from packages import CHANGELOG_HEADER_RE, REPO_ROOT, FragmentFilename, Package, PRDiff, RootPackage, Version

# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_compile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.fragments_dir is not None and args.all:
        parser.error("--fragments-dir requires --package (it cannot apply to all packages at once)")
    if args.version is not None and args.all:
        parser.error(
            "--version requires --package (each managed package has its own version trajectory; "
            "pin one with --package <name>)"
        )
    # Validate ``--version`` shape up front so a typo like ``--version 4.7``
    # fails at argument parsing instead of silently writing ``4.7`` into
    # ``CHANGELOG.rst`` and ``pyproject.toml``.
    explicit_version: Version | None = None
    if args.version is not None:
        try:
            explicit_version = Version(args.version)
        except ValueError as e:
            parser.error(f"--version: {e}")

    if args.package:
        pkg = Package.from_name(args.package)
        if not pkg.root.is_dir():
            parser.error(f"--package {args.package!r}: directory not found at {pkg.root}")
        if not pkg.is_managed:
            parser.error(
                f"--package {args.package!r} is not managed: missing {pkg.toml_path.name} or "
                f"docs/CHANGELOG.rst at {pkg.root}. Run with --all to see the discovered list."
            )
        packages = [pkg]
    else:
        packages = Package.discover()

    # Per-package isolation: one package's failure must not abort the batch.
    # The nightly workflow commits and pushes whatever compiled successfully,
    # so a malformed file in one package only loses that package's release
    # notes for this cycle — the rest still ships.
    any_compiled = False
    failures: list[tuple[str, str]] = []
    for pkg in packages:
        try:
            compiled, _ = pkg.compile(
                fragments_dir=args.fragments_dir,
                explicit_version=explicit_version,
                dry_run=args.dry_run,
            )
        # ``CompileFailed`` wraps a failure that happened after the compile had
        # already written; this command has nothing to stage, so it is reported
        # exactly like a failure that happened before the first write.
        except (Package.CompileFailed, OSError, ValueError) as e:
            print(f"  ERROR ({pkg.name}): {e}", file=sys.stderr)
            failures.append((pkg.name, str(e)))
            continue
        any_compiled = any_compiled or compiled

    if failures:
        print(file=sys.stderr)
        print(f"::error::{len(failures)} package(s) failed to compile:", file=sys.stderr)
        for name, reason in failures:
            print(f"  • {name}: {reason}", file=sys.stderr)
        return 1

    if not any_compiled:
        print("No fragments found in any package.")
    return 0


def cmd_check(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    try:
        diff = PRDiff.from_git(args.base_ref)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git diff failed: {e.stderr}", file=sys.stderr)
        return 1

    packages = Package.discover()

    # Header invariant — every managed package's ``CHANGELOG.rst`` must
    # contain a parseable header. ``write_changelog_entry`` self-heals the
    # narrow "missing trailing blank line" case, but every other broken
    # shape (no ``Changelog`` header, no underline, wrong underline char,
    # leading whitespace, etc.) still raises at compile time and would
    # wedge the next nightly. Block those at PR time with a clear error.
    malformed_headers: list[str] = []
    for pkg in packages:
        text = pkg.changelog_path.read_text(encoding="utf-8")
        # Apply the same normalization compile would apply, then check.
        text = re.sub(r"^(Changelog\n-+)\n(?!\n)", r"\1\n\n", text, count=1, flags=re.MULTILINE)
        if CHANGELOG_HEADER_RE.search(text) is None:
            malformed_headers.append(str(pkg.changelog_path.relative_to(REPO_ROOT)))

    missing, invalid_fragments = diff.evaluate(packages)

    if invalid_fragments:
        print("::error::Invalid changelog fragment(s) in this PR:")
        for path, reason in invalid_fragments:
            print(f"  • {path}")
            print(f"    → {reason}")
        print()

    if missing:
        print("::error::Missing changelog fragments for the following packages:")
        for pkg_name in missing:
            print(f"  • {pkg_name}")
            for line in FragmentFilename.help_lines_for_package(pkg_name):
                print(f"    → {line}")
        print()
        print("Slug = your branch name with `/` replaced by `-` (or any short, unique name).")
        print()
        print("Fragment format (source/<pkg>/changelog.d/<slug>[.minor|.major].rst):")
        print()
        print("    Added")
        print("    ^^^^^")
        print()
        print("    * Added :class:`~pkg.Foo` for feature X.")
        print()
        print("    Fixed")
        print("    ^^^^^")
        print()
        print("    * Fixed edge case in :meth:`~pkg.Foo.bar`.")
        print()
        print("See AGENTS.md ## Changelog for full guidance.")

    if malformed_headers:
        print("::error::Malformed CHANGELOG.rst — header must contain ``Changelog\\n---------\\n\\n``")
        print("(header line, underline, then a blank line — the anchor the nightly compile prepends to):")
        for path in malformed_headers:
            print(f"  • {path}")
        print()
        print("Seed the file with at minimum:")
        print()
        print("    Changelog")
        print("    ---------")
        print()
        print("    0.1.0 (YYYY-MM-DD)")
        print("    ~~~~~~~~~~~~~~~~~~")
        print()
        print("    Added")
        print("    ^^^^^")
        print()
        print("    * Initial release.")
        print()

    if invalid_fragments or missing or malformed_headers:
        return 1

    print("✓ All modified packages have valid changelog fragments.")
    return 0


def cmd_auto_bump(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    return AutoBumpRun(
        branch=args.branch,
        remote=args.remote,
        event_name=args.event_name,
        dry_run=args.dry_run,
    ).run()


def cmd_sync_lock(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    """Sync ``uv.lock`` on its own, outside a nightly run.

    ``auto-bump`` already does this as part of the lifecycle; this
    subcommand exists for the two cases that sit outside it — a maintainer
    repairing a lock by hand after a manual version edit, and ``--check``
    as a gate that fails when the lock has drifted.
    """
    lock = LockFile(RootPackage(REPO_ROOT))
    try:
        if not args.check:
            lock.sync()
            return 0
        if not lock.exists:
            # ``check`` is silent by contract; say so here where a human
            # is watching. ``sync`` prints its own no-op notice.
            print(f"No {LockFile.LOCK_NAME} on this branch — nothing to sync.")
            return 0
        changes = lock.check()
    except LockFile.Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not changes:
        print(f"{LockFile.LOCK_NAME} is in sync with the workspace versions.")
        return 0
    print(f"{LockFile.LOCK_NAME} is out of sync with the workspace versions:")
    for drift in changes:
        print(f"  {drift.package}: {drift.old} -> {drift.new}")
    # Flush before writing to stderr so the summary cannot overtake the list
    # above when both streams land in the same terminal or CI log.
    sys.stdout.flush()
    print("\nRun `cli.py sync-lock` to fix.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # The module docstring carries the full usage walkthrough — surfacing
        # it as the parser description means ``cli.py --help`` shows the same
        # guidance someone reading the source would see.
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{compile,check,auto-bump,sync-lock}")

    p_compile = sub.add_parser(
        "compile",
        help="Compile fragments into CHANGELOG.rst (maintainer release-time tool).",
        description=(
            "Compile accumulated fragments into per-package CHANGELOG.rst entries and bump each "
            "package's version metadata file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compile.set_defaults(func=cmd_compile)

    # ── Target: which packages to compile (required, mutually exclusive) ──
    target = p_compile.add_argument_group("target", "Which package(s) to compile (required, mutually exclusive)")
    target_group = target.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--package", metavar="NAME", help="Compile a single package.")
    target_group.add_argument("--all", action="store_true", help="Compile all managed packages.")

    # ── Version source: by default inferred from filename suffixes ────────
    version_group = p_compile.add_argument_group(
        "version (optional)",
        "By default the new version is inferred from the filename suffixes of the consumed fragments.",
    )
    version_group.add_argument(
        "--version",
        metavar="X.Y.Z",
        help=(
            "Pin the package to an explicit version, skipping the per-fragment bump inference. "
            "Requires --package — each managed package has its own version trajectory and "
            "applying a single version to all of them would corrupt their independent histories."
        ),
    )

    # ── Execution mode: preview vs apply, where to read fragments from ────
    exec_group = p_compile.add_argument_group("execution")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview only — no files are written or deleted. Without this flag, "
            "the compile writes the new entry, bumps the version, and deletes "
            "the consumed fragments."
        ),
    )
    exec_group.add_argument(
        "--fragments-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override the directory to read fragments from "
            "(default: source/<pkg>/changelog.d/). "
            "Useful for previewing against example fragments without touching real ones. "
            "Only valid with --package."
        ),
    )

    p_check = sub.add_parser(
        "check",
        help="Verify each modified package has a valid fragment (PR gate).",
        description="Verify each modified package has a valid changelog fragment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_check.set_defaults(func=cmd_check)
    p_check.add_argument(
        "base_ref",
        help=(
            "Base branch to diff against (e.g. 'main' or 'develop'). "
            "The diff is taken against ``origin/<base_ref>...HEAD``."
        ),
    )

    p_auto_bump = sub.add_parser(
        "auto-bump",
        help="Compile + commit + push for the nightly cron (replaces the inline workflow shell).",
        description=(
            "End-to-end nightly auto-bump: compile every managed package's fragments, stage the "
            "files cli.py actually wrote, build a bot-attributed commit, and push to the target "
            "branch with retry-on-conflict. Replaces the inline shell in nightly-changelog.yml so "
            "the workflow stays free of changelog-tooling knowledge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_auto_bump.set_defaults(func=cmd_auto_bump)
    p_auto_bump.add_argument(
        "--branch",
        required=True,
        metavar="REF",
        help="Target branch to push the auto-commit to (e.g. develop, release/3.0.0-beta2).",
    )
    p_auto_bump.add_argument(
        "--remote",
        default="origin",
        metavar="NAME",
        help="Remote to push to. Default: origin.",
    )
    p_auto_bump.add_argument(
        "--event-name",
        # GitHub Actions exports GITHUB_EVENT_NAME into every step, so the
        # workflow does not have to pass it through. Anything the runner
        # already knows is better read here than restated in YAML that has to
        # be kept in step with this file.
        default=os.environ.get("GITHUB_EVENT_NAME", "manual"),
        metavar="NAME",
        help=(
            "GitHub event that triggered this run (e.g. 'schedule', 'workflow_dispatch'). "
            "Surfaces in the commit message's parenthetical suffix. Defaults to "
            "$GITHUB_EVENT_NAME when set, else 'manual' for local invocations."
        ),
    )
    p_auto_bump.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only — compile in dry-run mode and skip commit/push entirely.",
    )

    p_sync_lock = sub.add_parser(
        "sync-lock",
        help="Re-point uv.lock's workspace-member versions at their manifests.",
        description=(
            "Rewrite the ``version`` line of each uv workspace member's own ``[[package]]`` block "
            "in uv.lock so it matches that package's manifest — nothing else is touched. This is "
            "NOT a resolve: third-party pins, hashes, and markers are left alone, and a lock whose "
            "membership has changed is refused because only a real `uv lock` can repair it. "
            "``auto-bump`` runs this automatically; use it directly to repair a lock by hand, or "
            "with --check as a gate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sync_lock.set_defaults(func=cmd_sync_lock)
    p_sync_lock.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of writing uv.lock.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args, parser))


if __name__ == "__main__":
    main()
