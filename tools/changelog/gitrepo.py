# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Git operations shared by manual compilation and the nightly lifecycle."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


class GitError(Exception):
    """Base class for failures raised by :class:`GitRepo`."""


class GitRepo:
    """Thin subprocess wrapper around the ``git`` CLI scoped to one working tree.

    Owns only the working directory and the policy decisions that need
    typed errors. All other behavior delegates straight to ``git``, so
    the unit tests can run against a real tempdir repo + bare-repo remote
    instead of mocking subprocess.
    """

    # ---- Nested types ---------------------------------------------------

    class NonFastForward(GitError):
        """Raised when ``git push`` is rejected because the remote advanced."""

    # ---- Construction ---------------------------------------------------

    def __init__(self, cwd: Path):
        self.cwd = cwd

    # ---- Public API -----------------------------------------------------

    def add(self, paths: Iterable[Path | str]) -> list[str]:
        """Stage ``paths``, including deletions. Returns the ones staged.

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
            return []
        self._run("add", "--", *self._literal(path_strs))
        return path_strs

    def assert_clean(self) -> None:
        """Raise unless the working tree and index are clean.

        Raises:
            GitError: Something is modified or staged.
        """
        dirty = self._run("status", "--porcelain").stdout.strip()
        if dirty:
            raise GitError("working tree is not clean; auto-bump needs sole ownership of it:\n" + dirty)

    def restore(self, paths: Iterable[Path | str]) -> None:
        """Undo working-tree changes to ``paths``, including deletions.

        Used to roll back a compile that failed part-way. Only paths git
        already tracks can be restored; anything the compile created fresh is
        removed instead, so a half-applied compile leaves nothing behind
        either way.
        """
        path_strs = [str(p) for p in paths]
        if not path_strs:
            return
        tracked = set(self._tracked(path_strs))
        if restorable := [p for p in path_strs if p in tracked]:
            self._run("checkout", "--", *self._literal(restorable))
        for p in path_strs:
            if p not in tracked and Path(p).exists():
                Path(p).unlink()

    def has_staged_changes(self) -> bool:
        """Whether anything is staged for commit."""
        return self._run("diff", "--staged", "--quiet", check=False).returncode != 0

    def staged_diff(self, path: Path | str) -> str:
        """Return the staged diff for one path.

        ``--no-color`` rather than relying on the ``color.ui`` override in
        :meth:`_run`: ``color.diff`` is more specific and wins over it, so a
        config carrying ``color.diff = always`` would still wrap these lines
        in ANSI escapes and break the caller's prefix matching.
        """
        return self._run("diff", "--staged", "--no-color", "--", str(path)).stdout

    def commit(self, message: str, *, author_name: str, author_email: str) -> None:
        """Commit the staged changes under a one-off identity.

        Commits the index rather than a pathspec, which is exact because
        :meth:`assert_clean` established that the index holds nothing but
        what this run staged.

        The identity is passed per-invocation with ``-c`` rather than written
        via ``git config``, which would permanently rewrite the identity of
        whatever clone this ran in — including a maintainer's own.
        """
        self._run(
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "commit",
            "-m",
            message,
        )

    def fetch(self, remote: str, ref: str) -> None:
        """Fetch ``ref`` from ``remote``.

        Callers pass a fully qualified ``refs/heads/<branch>``: an unqualified
        name is ambiguous, and a tag sharing the branch's name would win the
        lookup and land in ``FETCH_HEAD``, sending the retry rebase onto the
        wrong commit.
        """
        self._run("fetch", remote, ref)

    def rebase(self, onto: str) -> None:
        """Replay the current branch's commits onto ``onto``."""
        self._run("rebase", onto)

    def push(self, remote: str, refspec: str) -> None:
        """Push ``refspec`` to ``remote``.

        Raises:
            NonFastForward: The remote moved; the caller may fetch, rebase
                and retry.
            GitError: Any other failure, including a rejection by a remote
                hook or ruleset, which retrying cannot fix.
        """
        result = self._run("push", "--porcelain", remote, refspec, check=False)
        if result.returncode == 0:
            return
        combined = ((result.stdout or "") + (result.stderr or "")).strip()
        # ``--porcelain`` emits one machine-readable status line per ref with a
        # flag character in column zero; ``!`` marks a rejected ref. Reading
        # the flag rather than scanning the human summary keeps this working
        # under any locale — the prose is translated, the flag is not.
        #
        # ``!`` alone is not enough to retry on, though: it covers both "the
        # remote moved" and "a hook or ruleset said no". Only the former is
        # recoverable, and retrying the latter costs two pointless fetch and
        # rebase cycles while logging a misleading reason.
        rejected = [line for line in (result.stdout or "").splitlines() if line.startswith("!")]
        if rejected and not any("remote rejected" in line for line in rejected):
            raise self.NonFastForward(combined)
        raise GitError(f"git push failed: {combined}")

    # ---- Internals ------------------------------------------------------

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run git in this working tree and capture its output.

        ``-c color.ui=false`` because this output is parsed, not displayed.
        git suppresses colour for a pipe by default, but a developer or runner
        configured with ``color.ui = always`` overrides that, and the escapes
        silently break line-prefix matching. Commands whose output is parsed
        line-by-line pass ``--no-color`` as well, since the per-command colour
        settings outrank this one.
        """
        return subprocess.run(
            ["git", "-c", "color.ui=false", *args],
            cwd=self.cwd,
            text=True,
            capture_output=True,
            check=check,
        )

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
        listed = self._run("ls-files", "-z", "--", *self._literal(paths)).stdout.split("\0")
        # ``ls-files`` reports repo-relative paths; callers hold absolute ones.
        return [str(self.cwd / line) for line in listed if line]

    @staticmethod
    def _literal(paths: Iterable[Path | str]) -> list[str]:
        """Mark ``paths`` as literal pathspecs.

        Git reads a bare pathspec as a glob, and fragment slugs may contain
        ``*``, ``?`` or ``[``. Without this, ``feat[Z].rst`` would also match
        ``featZ.rst`` and could stage or restore an unrelated file.
        """
        return [f":(literal){p}" for p in paths]
