# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate automated release backports before they are pushed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_BACKPORT_MARKER = "backport-active-release"
_BACKPORT_PATTERN = re.compile(
    rf"^\s*-\s*\[\s*[xX]\s*\]\s*<!--\s*{re.escape(_BACKPORT_MARKER)}\s*-->",
    flags=re.MULTILINE,
)


class BackportValidationError(RuntimeError):
    """Raised when a candidate backport violates the source PR contract."""


def backport_requested(body: str | None) -> bool:
    """Return whether a pull request body contains the checked backport marker."""
    return bool(body and _BACKPORT_PATTERN.search(body))


def changed_paths(base: str, head: str | None = None, *, cached: bool = False) -> set[str]:
    """Return paths changed between two revisions or between a revision and the index."""
    command = ["git", "diff", "--no-renames", "--name-only", "-z"]
    if cached:
        command.extend(["--cached", base])
    else:
        if head is None:
            raise ValueError("head is required unless cached=True")
        command.extend([base, head])
    command.append("--")
    output = _run(command, text=False).stdout
    return {item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item}


def validate_source(
    source_parent: str,
    source: str,
    pr_base: str,
    pr_head: str,
    expected_files_json: Path,
) -> set[str]:
    """Verify that the merged source commit covers every file reported by the PR API."""
    expected_paths = _expected_pr_paths(expected_files_json)
    source_paths = changed_paths(source_parent, source)
    if source_paths != expected_paths:
        raise BackportValidationError(
            "the merged commit does not represent the complete PR file set"
            f"\nexpected: {_format_paths(expected_paths)}"
            f"\nsource:   {_format_paths(source_paths)}"
        )
    pr_merge_base = _run(["git", "merge-base", pr_base, pr_head]).stdout.strip()
    if _change_digest(source_parent, source) != _change_digest(pr_merge_base, pr_head):
        raise BackportValidationError("the merged commit content differs from the complete PR patch")
    return source_paths


def validate_candidate(
    source_parent: str,
    source: str,
    target: str,
    candidate: str | None,
    *,
    require_exact_patch: bool,
) -> dict[str, object]:
    """Validate a committed candidate or the currently staged conflict resolution."""
    source_paths = changed_paths(source_parent, source)
    if candidate is None:
        _validate_staged_worktree()
        candidate_paths = changed_paths(target, cached=True)
    else:
        candidate_paths = changed_paths(target, candidate)

    unexpected_paths = candidate_paths - source_paths
    if unexpected_paths:
        raise BackportValidationError(
            "the candidate changes files outside the original PR: " + _format_paths(unexpected_paths)
        )

    missing_paths = source_paths - candidate_paths
    nonmatching_missing_paths = {
        path for path in missing_paths if _tree_entry(source, path) != _candidate_entry(candidate, path)
    }
    if nonmatching_missing_paths:
        raise BackportValidationError(
            "the candidate omits source paths that do not already match the source result: "
            + _format_paths(nonmatching_missing_paths)
        )

    if require_exact_patch:
        if missing_paths:
            raise BackportValidationError(
                "an exact automatic backport must replay every source path: " + _format_paths(missing_paths)
            )
        if _change_digest(source_parent, source) != _change_digest(target, candidate):
            raise BackportValidationError("the candidate added or removed content differs from the original PR")

    return {
        "candidate_paths": sorted(candidate_paths),
        "exact_patch": require_exact_patch,
        "missing_paths_already_present": sorted(missing_paths),
        "source_paths": sorted(source_paths),
    }


def _run(command: list[str], *, input_bytes: bytes | None = None, text: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return its completed process."""
    return subprocess.run(command, input=input_bytes, check=True, capture_output=True, text=text)


def _expected_pr_paths(path: Path) -> set[str]:
    """Read the paginated GitHub PR-files response and include both sides of renames."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload if payload and isinstance(payload[0], list) else [payload]
    paths: set[str] = set()
    for page in pages:
        for item in page:
            paths.add(item["filename"])
            if item.get("status") == "renamed" and item.get("previous_filename"):
                paths.add(item["previous_filename"])
    return paths


def _validate_staged_worktree():
    """Require a fully resolved, fully staged conflict-resolution worktree."""
    unmerged = _run(["git", "diff", "--name-only", "--diff-filter=U", "-z"], text=False).stdout
    if unmerged:
        raise BackportValidationError("the candidate still contains unresolved merge conflicts")
    unstaged = _run(["git", "diff", "--name-only", "-z"], text=False).stdout
    if unstaged:
        raise BackportValidationError("the candidate contains unstaged tracked changes")
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], text=False).stdout
    if untracked:
        raise BackportValidationError("the candidate contains untracked files")


def _tree_entry(revision: str, path: str) -> tuple[str, str] | None:
    """Return a revision's ``(mode, blob)`` entry for a path, or ``None`` when absent."""
    result = subprocess.run(
        ["git", "ls-tree", "-z", revision, "--", path], check=True, capture_output=True, text=False
    ).stdout
    if not result:
        return None
    metadata = result.split(b"\t", maxsplit=1)[0].decode("ascii")
    mode, _object_type, object_id = metadata.split()
    return mode, object_id


def _candidate_entry(candidate: str | None, path: str) -> tuple[str, str] | None:
    """Return a committed candidate or index entry for a path."""
    if candidate is not None:
        return _tree_entry(candidate, path)
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", path], check=True, capture_output=True, text=False
    ).stdout
    if not result:
        return None
    metadata = result.split(b"\t", maxsplit=1)[0].decode("ascii")
    mode, object_id, stage = metadata.split()
    if stage != "0":
        raise BackportValidationError(f"the index entry for {path!r} is still unmerged")
    return mode, object_id


def _change_digest(base: str, head: str | None) -> str:
    """Hash exact added and removed bytes while ignoring base-dependent diff context."""
    if head is None:
        raise ValueError("head is required when validating an exact patch")
    patch = _run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-color",
            "--no-ext-diff",
            "--no-renames",
            "--unified=0",
            base,
            head,
            "--",
        ],
        text=False,
    ).stdout
    retained: list[bytes] = []
    in_hunk = False
    in_binary_patch = False
    for line in patch.splitlines(keepends=True):
        if line.startswith(b"diff --git "):
            in_hunk = False
            in_binary_patch = False
            retained.append(line)
        elif line.startswith((b"new file mode ", b"deleted file mode ", b"old mode ", b"new mode ")) or line.startswith(
            (b"--- ", b"+++ ")
        ):
            retained.append(line)
        elif line.startswith(b"@@"):
            in_hunk = True
        elif line == b"GIT binary patch\n":
            in_binary_patch = True
            retained.append(line)
        elif in_binary_patch or in_hunk and line.startswith((b"+", b"-", b"\\")):
            retained.append(line)
    return hashlib.sha256(b"".join(retained)).hexdigest()


def _format_paths(paths: set[str]) -> str:
    """Format paths deterministically for diagnostics."""
    return ", ".join(repr(path) for path in sorted(paths)) or "<none>"


def _create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    requested = subparsers.add_parser("requested", help="Read the backport checkbox from a GitHub event")
    requested.add_argument("--event", type=Path, required=True)

    source = subparsers.add_parser("validate-source", help="Verify that a merge commit represents the full PR")
    source.add_argument("--source_parent", required=True)
    source.add_argument("--source", required=True)
    source.add_argument("--pr_base", required=True)
    source.add_argument("--pr_head", required=True)
    source.add_argument("--expected_files_json", type=Path, required=True)

    candidate = subparsers.add_parser("validate-candidate", help="Verify a candidate backport")
    candidate.add_argument("--source_parent", required=True)
    candidate.add_argument("--source", required=True)
    candidate.add_argument("--target", required=True)
    candidate.add_argument("--candidate")
    candidate.add_argument("--exact_patch", action="store_true")
    return parser


def main() -> int:
    """Run the requested validation command."""
    args = _create_parser().parse_args()
    try:
        if args.command == "requested":
            event = json.loads(args.event.read_text(encoding="utf-8"))
            print(str(backport_requested(event.get("pull_request", {}).get("body"))).lower())
        elif args.command == "validate-source":
            paths = validate_source(
                args.source_parent, args.source, args.pr_base, args.pr_head, args.expected_files_json
            )
            print(json.dumps({"source_paths": sorted(paths)}, sort_keys=True))
        elif args.command == "validate-candidate":
            result = validate_candidate(
                args.source_parent,
                args.source,
                args.target,
                args.candidate,
                require_exact_patch=args.exact_patch,
            )
            print(json.dumps(result, sort_keys=True))
    except (BackportValidationError, subprocess.CalledProcessError, ValueError) as error:
        print(f"backport validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
