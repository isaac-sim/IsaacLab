# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The nightly auto-bump lifecycle: compile, sync, stage, commit, push.

Sits above the domain model and below the command line. The whole reason
this is Python rather than inline workflow shell is the staging set: the
files to commit come from what :meth:`Package.compile` reports it changed,
an in-process list, rather than from a glob duplicated in YAML that has to
be updated in lockstep every time a write site is added.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Keep these imports available for callers of the original autobump module.
from gitrepo import GitError as GitError
from gitrepo import GitRepo as GitRepo
from lockfile import LockFile
from packages import Package, RootPackage
from paths import REPO_ROOT


class AutoBumpRun:
    """One-shot orchestrator for the nightly auto-commit lifecycle.

    Compiles every managed package, stages whatever the compile actually
    wrote (no external glob — the touched-paths list comes from each
    :meth:`Package.compile` return value), builds a bot-attributed commit,
    and pushes it to the target branch. Retries the push on
    non-fast-forward by fetching and rebasing the auto-commit onto the new
    tip, so a human commit landing mid-run doesn't waste the batch.

    Identity is hardcoded — the bot's user name and email are deterministic
    public values derived from the GitHub App registration (anyone reading
    a nightly commit can see them). The App credential itself stays in repo
    secrets and is consumed by the workflow, not this class.
    """

    # ---- Class constants ------------------------------------------------

    AUTHOR_NAME = "isaaclab-bot[bot]"
    AUTHOR_EMAIL = "282401363+isaaclab-bot[bot]@users.noreply.github.com"
    PUSH_RETRIES = 3
    COMMIT_PREFIX = "[CI][Auto Version Bump]"

    # ---- Construction ---------------------------------------------------

    def __init__(
        self,
        *,
        branch: str,
        remote: str,
        event_name: str = "manual",
        dry_run: bool = False,
        repo_root: Path = REPO_ROOT,
    ):
        """Configure one nightly run.

        Args:
            branch: Target branch to push the auto-commit to.
            remote: Remote to push to.
            event_name: GitHub event that triggered the run; surfaces in the
                commit message's parenthetical suffix.
            dry_run: Preview only — compile without writing, and skip
                commit and push entirely.
            repo_root: Working tree to operate on. Defaults to this
                checkout; tests point it at a tempdir.
        """
        # Configuration and collaborators only. The products of a run --
        # which files changed, what compiled, what failed -- are returned by
        # the phases that produce them rather than accumulated here, so
        # ``run`` reads as a pipeline instead of a sequence of side effects.
        # ``failures`` is the one exception: it is fed by two separate phases
        # and consumed only at the very end.
        self.branch = branch
        self.remote = remote
        self.event_name = event_name
        self.dry_run = dry_run
        self.repo_root = repo_root
        self.repo = GitRepo(repo_root)
        self.packages = Package.discover(packages_root=repo_root / "source")
        self.lock = LockFile(RootPackage(repo_root))
        self.failures: list[tuple[str, str]] = []

    # ---- Public API -----------------------------------------------------

    def run(self) -> int:
        """Execute the whole nightly lifecycle. Returns a process exit code.

        Raises:
            GitError: The working tree was not clean at entry.
        """
        # This command owns the working tree for the duration of a run, which
        # is what the nightly workflow always hands it: a fresh checkout.
        # Asserting that once is cheaper and clearer than scoping every git
        # operation to survive a tree it will never see, and it turns local
        # misuse into one obvious error instead of a subtly wrong commit.
        if not self.dry_run:
            self.repo.assert_clean()
        touched, any_compiled = self._compile_all()
        # The lock is reconciled on every run, before the "nothing to do" exit
        # rather than after it. A lock that cannot be repaired is a standing
        # inconsistency on the branch, and checking it only on nights that
        # happened to compile a fragment would report it once and then go
        # quiet -- the failure would look fixed simply because no package had
        # pending work that night.
        touched += self._sync_lock()
        if self.dry_run:
            self._report_dry_run(any_compiled)
            return self._exit_code()
        if not touched:
            # Nothing was written, so nothing compiled either: outside dry-run
            # a compile that processes fragments always writes.
            print("No fragments found in any package.")
            return self._exit_code()
        self._stage_and_commit(touched)
        self._push_with_retry()
        return self._exit_code()

    # ---- Internals: the phases of ``run``, in order ----------------------

    def _compile_all(self) -> tuple[list[Path], bool]:
        """Compile every managed package.

        Returns ``(touched, any_compiled)`` — the paths written across all
        packages, and whether any package had fragments to process. One
        package's failure is recorded and skipped so the rest still ship.
        """
        touched: list[Path] = []
        any_compiled = False
        for pkg in self.packages:
            try:
                compiled, pkg_touched = pkg.compile(dry_run=self.dry_run)
            except Package.CompileFailed as e:
                # Raised after the compile already wrote. Those writes are
                # rolled back rather than committed: half a compile is a
                # changelog entry announcing a version the manifest never
                # received, over a fragment that was never consumed — which
                # the next run would compile into a second identical entry.
                # Restoring also leaves the tree clean, which the rebase in
                # the push-retry loop requires.
                print(f"  ERROR ({pkg.name}): {e}", file=sys.stderr)
                self.failures.append((pkg.name, str(e)))
                if not self.dry_run:
                    self.repo.restore(e.written, original_contents=e.original_contents)
                continue
            except (OSError, ValueError) as e:
                # Matches what ``Package.compile`` raises when it fails before
                # writing. A narrower tuple would let e.g. PermissionError
                # escape and take the whole batch down with it — the opposite
                # of the per-package isolation this loop exists for.
                print(f"  ERROR ({pkg.name}): {e}", file=sys.stderr)
                self.failures.append((pkg.name, str(e)))
                continue
            any_compiled = any_compiled or compiled
            touched.extend(pkg_touched)
        return touched, any_compiled

    def _sync_lock(self) -> list[Path]:
        """Re-point ``uv.lock`` at the versions the compile just wrote.

        Returns the paths written, which join the compile's own so the lock
        is staged through one manifest — ``uv.lock`` is a single repo-level
        artifact pinning all members at once, so the sync is global rather
        than per-package, and no separate ``git add`` exists to drift.

        A failure is recorded like a failed package compile — non-zero exit,
        red tile — but deliberately does not block the commit. A lock that
        needs a full ``uv lock`` was already stale before this run started,
        and wedging the nightly over it would strand every package's
        changelog for a problem the auto-commit did not cause.
        """
        try:
            return self.lock.sync(dry_run=self.dry_run)
        except LockFile.Error as e:
            print(f"  ERROR ({LockFile.LOCK_NAME}): {e}", file=sys.stderr)
            self.failures.append((LockFile.LOCK_NAME, str(e)))
            return []

    def _report_dry_run(self, any_compiled: bool) -> None:
        """Print what a real run would have done. Writes nothing."""
        if any_compiled:
            print(f"DRY RUN — compile complete; would commit/push to {self.branch}.")
        else:
            print("DRY RUN — no fragments found in any package.")

    def _stage_and_commit(self, touched: list[Path]) -> None:
        """Stage exactly ``touched`` and commit it as the bot."""
        self.repo.add(touched)
        if not self.repo.has_staged_changes():
            print("Nothing actually staged after compile — skipping commit.")
            return
        # Author identity belongs in-process: the workflow YAML stops carrying
        # changelog-tool knowledge, so cli.py-only PRs need no paired YAML edit.
        self.repo.commit(
            self._build_commit_message(touched),
            author_name=self.AUTHOR_NAME,
            author_email=self.AUTHOR_EMAIL,
        )
        print(f"Committed {len(touched)} file(s).")

    def _build_commit_message(self, touched: list[Path]) -> str:
        """Compose the auto-commit subject and its per-package bump list."""
        # The set of version files is taken from the packages themselves
        # (:attr:`Package.toml_path`) rather than a hardcoded filename, so
        # this produces a correct message both here — where versions live in
        # ``pyproject.toml`` — and on release branches that still keep them
        # in ``config/extension.toml``. Files that carry no ``version`` line
        # (``uv.lock``, ``CHANGELOG.rst``) are staged but not enumerated.
        version_files = {pkg.toml_path: pkg.name for pkg in self.packages}
        bumped = sorted(p for p in touched if p in version_files)
        if not bumped:
            # No package bumped, so "Compile changelog fragments" over an empty
            # list would misdescribe the commit. Name what actually changed:
            # a lock re-point and a stale-skip sweep are both ordinary nights,
            # and a run may be either or both.
            return f"{self.COMMIT_PREFIX} {self._no_bump_subject(touched)} ({self.event_name})\n"
        lines = [
            f"{self.COMMIT_PREFIX} Compile changelog fragments ({self.event_name})",
            "",
            "Bumped packages:",
        ]
        for path in bumped:
            diff = self.repo.staged_diff(self._relative(path))
            old = self._version_from_diff(diff, "-")
            new = self._version_from_diff(diff, "+")
            lines.append(f"- {version_files[path]}: {old} → {new}")
        return "\n".join(lines) + "\n"

    def _no_bump_subject(self, touched: list[Path]) -> str:
        """Subject for a commit that changed something but bumped nothing.

        Two causes, independently possible: the lock was re-pointed, and/or
        stale ``.skip`` markers were swept. Naming only one of them — or
        naming the lock on a branch that has none — turns the commit log
        into a misleading record of what the nightly did.
        """
        parts = []
        if any(p.name == LockFile.LOCK_NAME for p in touched):
            parts.append(f"Sync {LockFile.LOCK_NAME} with workspace versions")
        if any(p.suffix == ".skip" for p in touched):
            parts.append("clean stale skip files")
        if not parts:
            return "Compile changelog fragments"
        subject = parts[0]
        for extra in parts[1:]:
            subject += f" and {extra}"
        return subject

    def _relative(self, path: Path) -> Path:
        """``path`` relative to the repo root, or unchanged if it lies outside."""
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            return path

    def _push_with_retry(self) -> None:
        """Push the auto-commit, rebasing onto the branch tip if it moved.

        A human commit landing between checkout and push (a window of a
        couple of minutes on a branch taking ~8 commits a day) rejects the
        push as non-fast-forward. Without the retry the whole night's batch
        would wait for the next run.

        The retry deliberately does not re-compile: a fragment added by the
        racing commit is left for the next night rather than amending the
        commit currently being pushed.
        """
        refspec = f"HEAD:refs/heads/{self.branch}"
        last_err: GitRepo.NonFastForward | None = None
        for attempt in range(self.PUSH_RETRIES):
            try:
                self.repo.push(self.remote, refspec)
                if attempt > 0:
                    print(f"Push succeeded on attempt {attempt + 1}.")
                else:
                    print(f"Push succeeded → {self.remote} {refspec}.")
                return
            except GitRepo.NonFastForward as e:
                last_err = e
                if attempt + 1 == self.PUSH_RETRIES:
                    break
                print(
                    f"  push rejected (attempt {attempt + 1}/{self.PUSH_RETRIES}); "
                    f"fetching and rebasing onto {self.branch}: {e}",
                    file=sys.stderr,
                )
                self.repo.fetch(self.remote, f"refs/heads/{self.branch}")
                self.repo.rebase("FETCH_HEAD")
        # All retries exhausted.
        assert last_err is not None  # the loop only exits here when it raised at least once
        raise GitError(f"push failed after {self.PUSH_RETRIES} attempts; last error: {last_err}")

    def _exit_code(self) -> int:
        """Return 1 if anything failed this run, 0 otherwise.

        Failures are reported but never block the commit, so the exit code
        is what keeps the job tile red while the healthy packages still ship.
        """
        if not self.failures:
            return 0
        print(file=sys.stderr)
        print(f"::error::{len(self.failures)} item(s) failed:", file=sys.stderr)
        for name, reason in self.failures:
            print(f"  • {name}: {reason}", file=sys.stderr)
        return 1

    # ---- Pure helpers ---------------------------------------------------

    @staticmethod
    def _version_from_diff(diff_text: str, prefix: str) -> str:
        """Pull the version string out of a staged version-metadata diff line.

        The diff contains lines like ``-version = "1.2.3"`` (old) and
        ``+version = "1.3.0"`` (new); ``prefix`` selects which side to read.
        Returns ``"?"`` if no match — the commit message is informational,
        not machine-parsed, so a missing value shouldn't fail the run.
        """
        for line in diff_text.splitlines():
            if line.startswith(f"{prefix}version"):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1)
        return "?"
