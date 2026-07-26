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
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from lockfile import LockFile
from packages import REPO_ROOT, Package


class GitError(Exception):
    """Base class for failures raised by :class:`GitRepo`."""


class GitRepo:
    """Thin subprocess wrapper around the ``git`` CLI scoped to one working tree.

    Owns only the working directory and the policy decisions that need
    typed errors. All other behavior delegates straight to ``git``, so
    the unit tests can run against a real tempdir repo + bare-repo remote
    instead of mocking subprocess.
    """

    class NonFastForward(GitError):
        """Raised when ``git push`` is rejected because the remote advanced."""

    def __init__(self, cwd: Path):
        self.cwd = cwd

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        # ``-c color.ui=false`` because this output is parsed, not displayed.
        # git suppresses colour for a pipe by default, but a developer or
        # runner with ``color.ui = always`` configured overrides that, and the
        # resulting ANSI escapes silently break line-prefix matching: a diff
        # line arrives as ``\x1b[31m-version = "1.2.3"``, so the version
        # extraction reports "?" and the non-fast-forward detection stops
        # recognising a rejected push. Neither failure is loud.
        return subprocess.run(
            ["git", "-c", "color.ui=false", *args],
            cwd=self.cwd,
            text=True,
            capture_output=True,
            check=check,
        )

    def config(self, key: str, value: str) -> None:
        self._run("config", key, value)

    def add(self, paths: Iterable[Path | str]) -> None:
        """Stage ``paths``, including deletions.

        ``git add`` on a path that is gone *and* was never tracked aborts the
        whole invocation with ``pathspec ... did not match any files``, which
        would take the entire staging step down over one stray file. Deleted
        paths are therefore filtered to the ones git actually knows about;
        a deleted-but-tracked path stages its removal, as intended.
        """
        path_strs = [str(p) for p in paths]
        vanished = [p for p in path_strs if not Path(p).exists()]
        if vanished:
            known = set(self._tracked(vanished))
            path_strs = [p for p in path_strs if Path(p).exists() or p in known]
        if not path_strs:
            return
        self._run("add", "--", *path_strs)

    def _tracked(self, paths: Iterable[Path | str]) -> list[str]:
        """Return the subset of ``paths`` git has under version control.

        ``-z`` is load-bearing, not a style choice. With ``core.quotePath`` at
        its default, ``git ls-files`` C-quotes any path containing a non-ASCII
        byte -- ``josé-fix.rst`` comes back as ``"jos\\303\\251-fix.rst"``.
        Fragment slugs allow those characters, so the quoted form would fail
        the caller's comparison, its deletion would be dropped from the staging
        set, and the fragment would survive on the branch to be compiled a
        second time. NUL-delimited output is emitted verbatim.
        """
        listed = self._run("ls-files", "-z", "--", *[str(p) for p in paths]).stdout.split("\0")
        # ``ls-files`` reports repo-relative paths; callers hold absolute ones.
        return [str(self.cwd / line) for line in listed if line]

    def has_staged_changes(self) -> bool:
        return self._run("diff", "--staged", "--quiet", check=False).returncode != 0

    def staged_diff(self, path: Path | str) -> str:
        return self._run("diff", "--staged", "--", str(path)).stdout

    def commit(self, message: str) -> None:
        self._run("commit", "-m", message)

    def fetch(self, remote: str, ref: str) -> None:
        """Fetch ``ref`` from ``remote``.

        Callers pass a fully qualified ``refs/heads/<branch>``: an unqualified
        name is ambiguous, and a tag sharing the branch's name would win the
        lookup and land in ``FETCH_HEAD``, sending the retry rebase onto the
        wrong commit.
        """
        self._run("fetch", remote, ref)

    def rebase(self, onto: str) -> None:
        self._run("rebase", onto)

    def push(self, remote: str, refspec: str) -> None:
        result = self._run("push", "--porcelain", remote, refspec, check=False)
        if result.returncode == 0:
            return
        combined = ((result.stdout or "") + (result.stderr or "")).strip()
        # ``--porcelain`` emits one machine-readable status line per ref, with
        # a flag character in column zero: ``!`` marks a rejected ref. Testing
        # that flag rather than scanning the human summary for "[rejected]"
        # keeps the retry decision working under any locale -- the prose is
        # translated, the flag is not.
        if any(line.startswith("!") for line in (result.stdout or "").splitlines()):
            raise self.NonFastForward(combined)
        raise GitError(f"git push failed: {combined}")


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

    AUTHOR_NAME = "isaaclab-bot[bot]"
    AUTHOR_EMAIL = "282401363+isaaclab-bot[bot]@users.noreply.github.com"
    PUSH_RETRIES = 3
    COMMIT_PREFIX = "[CI][Auto Version Bump]"

    def __init__(
        self,
        *,
        branch: str,
        remote: str,
        event_name: str = "manual",
        dry_run: bool = False,
        repo_root: Path = REPO_ROOT,
    ):
        self.branch = branch
        self.remote = remote
        self.event_name = event_name
        self.dry_run = dry_run
        self.repo_root = repo_root
        self.repo = GitRepo(repo_root)
        self.packages = Package.discover(packages_root=repo_root / "source")
        self.lock = LockFile(repo_root, Package.declared_version)
        self.touched: list[Path] = []
        self.failures: list[tuple[str, str]] = []
        self.any_compiled = False

    def run(self) -> int:
        self._compile_all()
        # The lock is reconciled on every run, before the "nothing to do" exit
        # rather than after it. A lock that cannot be repaired is a standing
        # inconsistency on the branch, and checking it only on nights that
        # happened to compile a fragment would report it once and then go
        # quiet -- the failure would look fixed simply because no package had
        # pending work that night.
        self._sync_lock()
        if self.dry_run:
            self._report_dry_run()
            return self._exit_code()
        if not self.touched:
            if not self.any_compiled:
                print("No fragments found in any package.")
            else:
                print("All compiles ran but produced no on-disk writes (already up to date).")
            return self._exit_code()
        self._stage_and_commit()
        self._push_with_retry()
        return self._exit_code()

    def _compile_all(self) -> None:
        for pkg in self.packages:
            try:
                compiled, touched = pkg.compile(dry_run=self.dry_run)
            except Package.CompileFailed as e:
                # Raised after the compile already wrote to disk. Those writes
                # are real and must still be staged: an unstaged half-applied
                # compile leaves the working tree dirty, and the rebase in the
                # push-retry loop refuses to run against a dirty tree.
                print(f"  ERROR ({pkg.name}): {e}", file=sys.stderr)
                self.failures.append((pkg.name, str(e)))
                self.touched.extend(e.touched)
                continue
            except (FileNotFoundError, ValueError) as e:
                print(f"  ERROR ({pkg.name}): {e}", file=sys.stderr)
                self.failures.append((pkg.name, str(e)))
                continue
            self.any_compiled = self.any_compiled or compiled
            self.touched.extend(touched)

    def _sync_lock(self) -> None:
        """Re-point ``uv.lock`` at the versions the compile just wrote.

        Runs once, after every package has compiled — ``uv.lock`` is a
        single repo-level artifact pinning all members at once, so there is
        nothing per-package about it. The written path joins
        :attr:`touched`, which is what gets the lock staged through the same
        manifest as every other compiler output; no separate ``git add``
        exists to drift.

        A failure is recorded like a failed package compile — non-zero exit,
        red tile — but deliberately does not block the commit. A lock that
        needs a full ``uv lock`` was already stale before this run started,
        and wedging the nightly over it would strand every package's
        changelog for a problem the auto-commit did not cause.
        """
        try:
            self.touched.extend(self.lock.sync(dry_run=self.dry_run))
        except LockFile.Error as e:
            print(f"  ERROR ({LockFile.LOCK_NAME}): {e}", file=sys.stderr)
            self.failures.append((LockFile.LOCK_NAME, str(e)))

    def _report_dry_run(self) -> None:
        if self.any_compiled:
            print(f"DRY RUN — compile complete; would commit/push to {self.branch}.")
        else:
            print("DRY RUN — no fragments found in any package.")

    def _stage_and_commit(self) -> None:
        # Author identity belongs in-process: the workflow YAML stops
        # carrying changelog-tool knowledge so cli.py-only PRs don't need
        # paired YAML edits to stay consistent.
        self.repo.config("user.name", self.AUTHOR_NAME)
        self.repo.config("user.email", self.AUTHOR_EMAIL)
        self.repo.add(self.touched)
        if not self.repo.has_staged_changes():
            print("Nothing actually staged after compile — skipping commit.")
            return
        self.repo.commit(self._build_commit_message())
        print(f"Committed bump for {len(self.touched)} file(s).")

    def _relative(self, path: Path) -> Path:
        """``path`` relative to the repo root, or unchanged if it lies outside."""
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            return path

    def _build_commit_message(self) -> str:
        # Derive the per-package "old → new" lines from the staged version
        # metadata diffs. The set of version files is taken from the packages
        # themselves (:attr:`Package.toml_path`) rather than a hardcoded
        # filename, so this produces a correct message both here — where
        # versions live in ``pyproject.toml`` — and on release branches that
        # still keep them in ``config/extension.toml``. Files touched by a
        # future write site that carries no ``version`` line (``uv.lock``,
        # ``CHANGELOG.rst``) are staged but not enumerated here.
        version_files = {pkg.toml_path: pkg.name for pkg in self.packages}
        bumped = sorted(p for p in self.touched if p in version_files)
        if not bumped:
            # Reachable now that the lock is reconciled on every run: a night
            # with no pending fragments can still have a lock to re-point.
            # Saying "Compile changelog fragments" over an empty package list
            # would misdescribe the commit.
            return f"{self.COMMIT_PREFIX} Sync {LockFile.LOCK_NAME} with workspace versions ({self.event_name})\n"
        lines = [
            f"{self.COMMIT_PREFIX} Compile changelog fragments ({self.event_name})",
            "",
            "Bumped packages:",
        ]
        for path in bumped:
            diff = self.repo.staged_diff(self._relative(path))
            old = _extract_version_from_diff(diff, "-")
            new = _extract_version_from_diff(diff, "+")
            lines.append(f"- {version_files[path]}: {old} → {new}")
        return "\n".join(lines) + "\n"

    def _push_with_retry(self) -> None:
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
        if self.failures:
            print(file=sys.stderr)
            print(f"::error::{len(self.failures)} package(s) failed to compile:", file=sys.stderr)
            for name, reason in self.failures:
                print(f"  • {name}: {reason}", file=sys.stderr)
            return 1
        return 0


def _extract_version_from_diff(diff_text: str, prefix: str) -> str:
    """Pull the version string out of a staged version-metadata diff line.

    The diff contains lines like ``-version = "1.2.3"`` (old) and
    ``+version = "1.3.0"`` (new). ``prefix`` selects which side to read.
    Returns ``"?"`` if no match — the commit message is informational, not
    machine-parsed, so a missing value shouldn't fail the run.
    """
    for line in diff_text.splitlines():
        if line.startswith(f"{prefix}version"):
            m = re.search(r'"([^"]+)"', line)
            if m:
                return m.group(1)
    return "?"
