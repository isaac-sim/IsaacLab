# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Pull-request diffs and changelog fragment policy."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from fragments import Fragment, FragmentFilename
from packages import Package
from paths import REPO_ROOT


@dataclass(frozen=True)
class PRDiff:
    """A snapshot of "what this PR changed against its base branch."

    Wraps two views from the same git diff: ``changed`` (any file modified
    or added) and ``added`` (the strict subset that's new on this branch).
    Tests construct ``PRDiff`` directly with synthetic sets;
    :meth:`from_git` runs the real ``git diff`` for production use.
    """

    changed: set[str]
    added: set[str]

    @classmethod
    def from_git(cls, base_ref: str, *, include_worktree: bool = False) -> PRDiff:
        """Run ``git diff`` against ``origin/<base_ref>`` to populate the diff.

        Args:
            base_ref: Base branch to diff against.
            include_worktree: Whether to also count staged and unstaged
                changes to tracked files. The pre-commit hook needs this
                because the fragment it is gating is not committed yet;
                CI diffs the branch as pushed and leaves it off.
        """

        remote_base = f"origin/{base_ref}"
        if include_worktree:
            # ``git diff <merge-base>`` (two-dot against the merge base)
            # spans the worktree; the three-dot form used by CI compares
            # two commits and cannot see uncommitted work.
            diff_target = subprocess.run(
                ["git", "merge-base", remote_base, "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
        else:
            diff_target = f"{remote_base}...HEAD"

        def _diff(extra_args: list[str]) -> set[str]:
            result = subprocess.run(
                ["git", "diff", "--name-only", *extra_args, diff_target],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
            return {f for f in result.stdout.splitlines() if f}

        return cls(changed=_diff([]), added=_diff(["--diff-filter=A"]))

    def evaluate(
        self,
        packages: list[Package],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Apply the PR-gate rules and return ``(missing_packages, invalid_fragments)``.

        Rules:

        1. **Immutability** — every fragment file in the diff must be in
           ``added`` (added on this branch). Modifying or renaming an existing
           fragment is rejected with a hint to add a new one instead.

        2. **Content validity** — every added ``*.rst`` fragment must parse
           (recognised section headings + at least one bullet). ``.skip`` and
           ``.gitkeep`` are exempt.

        3. **Slug uniqueness** — within a package's ``changelog.d/``, no two
           fragments may share the same slug. If an added fragment's slug
           collides with an existing or co-added fragment, fail with a hint
           to rename (e.g. append ``-2``).

        4. **Required fragment per touched package** — for each managed
           package the PR touches in ``source/`` (outside ``changelog.d/``),
           the PR must *add* at least one valid fragment to that package's
           ``changelog.d/``. Chained PRs (parent PR's fragment shows up in
           the child's diff) naturally satisfy this — slug uniqueness is
           the only constraint that matters.
        """
        missing: list[str] = []
        invalid_fragments: list[tuple[str, str]] = []

        for pkg in packages:
            pkg_prefix = Package.package_prefix(pkg.name)
            changelog_dir = Package.fragment_dir_prefix(pkg.name)
            source_changed = [f for f in self.changed if f.startswith(pkg_prefix) and not f.startswith(changelog_dir)]
            fragment_changes = [f for f in self.changed if f.startswith(changelog_dir)]

            invalid_fragments.extend(self._check_fragments(pkg, changelog_dir, fragment_changes))
            if source_changed and not self._has_owned_fragment(fragment_changes):
                missing.append(pkg.name)

        return missing, invalid_fragments

    # ---- Internals: one method per documented rule -----------------------

    def _check_fragments(self, pkg: Package, changelog_dir: str, fragment_changes: list[str]) -> list[tuple[str, str]]:
        """Apply rules 1–3 to each fragment the PR touched in one package.

        The three run in order and short-circuit per file: a fragment that
        fails immutability is not then also reported as malformed, which
        would bury the actionable message under a derived one.
        """
        existing_slugs = self._existing_slugs(pkg, changelog_dir)
        added_slugs: dict[str, str] = {}
        problems: list[tuple[str, str]] = []

        for f in fragment_changes:
            path = Path(f)
            if FragmentFilename(path.name).is_ignorable:
                continue
            if (err := self._check_immutability(f)) is not None:
                problems.append((f, err))
                continue
            if (err := self._check_content(f, path)) is not None:
                problems.append((f, err))
                continue
            slug, err = self._check_slug_uniqueness(path, existing_slugs, added_slugs)
            if err is not None:
                problems.append((f, err))
                continue
            added_slugs[slug] = path.name
        return problems

    def _check_immutability(self, changed_path: str) -> str | None:
        """Rule 1 — a fragment already on the base branch may not be edited."""
        if changed_path in self.added:
            return None
        return "fragments are immutable — add a new fragment with a different slug instead of editing an existing one"

    @staticmethod
    def _check_content(changed_path: str, path: Path) -> str | None:
        """Rule 2 — an added ``*.rst`` fragment must parse. ``*.skip`` is exempt."""
        if FragmentFilename(path.name).is_skip:
            return None
        return Fragment(REPO_ROOT / changed_path).validate()

    @staticmethod
    def _check_slug_uniqueness(
        path: Path,
        existing_slugs: dict[str, str],
        added_slugs: dict[str, str],
    ) -> tuple[str, str | None]:
        """Rule 3 — no two fragments in one ``changelog.d/`` may share a slug.

        Returns ``(slug, error)``; ``error`` is ``None`` when the slug is
        free. Collisions are reported against both pre-existing fragments and
        others added by the same PR.
        """
        slug = Fragment.parse_slug(path.name)
        if slug is None:
            # Filename validation already flagged this for ``*.rst``, but a
            # malformed ``*.skip`` would otherwise slip through.
            return "", ("invalid filename — must be <slug>.rst, <slug>.minor.rst, <slug>.major.rst, or <slug>.skip")
        if slug in existing_slugs and existing_slugs[slug] != path.name:
            return slug, (
                f"slug {slug!r} collides with existing fragment "
                f"{existing_slugs[slug]!r} — rename to {slug}-2 (or any unused slug)"
            )
        if slug in added_slugs and added_slugs[slug] != path.name:
            return slug, (
                f"slug {slug!r} collides with another added fragment "
                f"{added_slugs[slug]!r} — rename one to {slug}-2 (or any unused slug)"
            )
        return slug, None

    def _has_owned_fragment(self, fragment_changes: list[str]) -> bool:
        """Rule 4 — did this PR *add* a recognisable fragment for the package?

        Chained PRs naturally satisfy this: the parent's fragment shows up in
        the child's diff as added, so only slug uniqueness constrains them.
        """
        return any(f in self.added and FragmentFilename(Path(f).name).is_valid for f in fragment_changes)

    def _existing_slugs(self, pkg: Package, changelog_dir: str) -> dict[str, str]:
        """Map slug → filename for fragments already on the base branch.

        The CI checkout holds base-branch fragments and the PR's additions
        side by side, so added files are excluded explicitly: otherwise an
        added file overwrites the entry for a pre-existing fragment sharing
        its slug, hiding the very collision rule 3 exists to catch.
        """
        added_basenames = {Path(f).name for f in self.added if f.startswith(changelog_dir)}
        existing: dict[str, str] = {}
        directory = pkg.default_fragment_dir
        if not directory.is_dir():
            return existing
        for p in directory.iterdir():
            if p.is_dir() or FragmentFilename(p.name).is_ignorable or p.name in added_basenames:
                continue
            if (slug := Fragment.parse_slug(p.name)) is not None:
                existing[slug] = p.name
        return existing
