# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin subprocess wrappers around the ``osmo`` CLI.

One module, one public class :class:`OsmoClient`. Each method shells out
to a single ``osmo`` invocation and parses the output. Errors are typed:

- :class:`OsmoAuthError` — auth/credential failure; caller surfaces.
- :class:`OsmoTransientError` — retryable (HTTP 5xx, connection reset).
- :class:`OsmoCliError` — anything else (bad spec, parse failure).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OsmoAuthError",
    "OsmoCliError",
    "OsmoClient",
    "OsmoTransientError",
    "TaskSnapshot",
    "WorkflowSnapshot",
]


class OsmoCliError(RuntimeError):
    """Generic ``osmo`` CLI failure (non-zero exit, parse failure, etc.)."""


class OsmoAuthError(OsmoCliError):
    """Auth failure (HTTP 401/403). Not retried."""


class OsmoTransientError(OsmoCliError):
    """Retryable failure (HTTP 5xx, connection issues)."""


@dataclass(frozen=True)
class TaskSnapshot:
    """One task's state in an OSMO workflow snapshot."""

    name: str
    status: str  # COMPLETED | FAILED | RUNNING | etc.
    exit_code: int | None


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Snapshot of a workflow + per-task states from one ``osmo workflow query`` call."""

    workflow_id: str
    status: str
    tasks: list[TaskSnapshot]


_AUTH_PATTERN = re.compile(r"(HTTP|status code) 40[13]|unauthori[sz]ed", re.IGNORECASE)
_TRANSIENT_PATTERN = re.compile(
    r"(HTTP|status code) 5\d\d|connection (reset|refused|timed?\s+out|timeout)",
    re.IGNORECASE,
)
_WORKFLOW_ID_PATTERN = re.compile(r"^Workflow ID\s+-\s+(\S+)", re.MULTILINE)


def _classify(stderr: str) -> type[OsmoCliError]:
    if _AUTH_PATTERN.search(stderr):
        return OsmoAuthError
    if _TRANSIENT_PATTERN.search(stderr):
        return OsmoTransientError
    return OsmoCliError


class OsmoClient:
    """Subprocess-based wrapper around the ``osmo`` CLI.

    Args:
        profile: OSMO profile name. Passed via ``OSMO_PROFILE`` env var on
            every invocation.
        executable: ``osmo`` binary path. Defaults to ``"osmo"`` (relies
            on ``$PATH``).
    """

    def __init__(self, *, profile: str, executable: str = "osmo") -> None:
        self._profile = profile
        self._exe = executable

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OSMO_PROFILE"] = self._profile
        return env

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def submit(self, yaml_path: Path, *, pool: str | None = None, priority: str | None = None) -> str:
        """Submit a workflow YAML and return the workflow_id.

        Args:
            yaml_path: Path to the rendered workflow YAML.
            pool: Target OSMO pool, or ``None`` to use the profile's default.
            priority: Scheduling priority (``HIGH``, ``NORMAL``, ``LOW``), or
                ``None`` to use OSMO's default.

        Returns:
            The OSMO workflow ID parsed from stdout.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cmd: list[str] = [self._exe, "workflow", "submit", str(yaml_path)]
        if pool is not None:
            cmd.extend(["--pool", pool])
        if priority is not None:
            cmd.extend(["--priority", priority])
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow submit` failed: {cp.stderr.strip()}")
        m = _WORKFLOW_ID_PATTERN.search(cp.stdout)
        if not m:
            raise OsmoCliError(f"could not parse Workflow ID from osmo stdout: {cp.stdout!r}")
        return m.group(1)

    def status(self, workflow_id: str) -> WorkflowSnapshot:
        """Fetch the workflow snapshot.

        Args:
            workflow_id: The OSMO workflow ID returned by :meth:`submit`.

        Returns:
            A :class:`WorkflowSnapshot` populated from the OSMO response.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cp = self._run([self._exe, "workflow", "query", workflow_id, "-t", "json"])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow query` failed: {cp.stderr.strip()}")
        return self._parse_status_json(cp.stdout, workflow_id)

    @staticmethod
    def _parse_status_json(stdout: str, workflow_id: str) -> WorkflowSnapshot:
        """Parse ``osmo workflow query -t json`` output.

        The document names the workflow under ``name`` and nests tasks one level
        down, under ``groups[].tasks[]`` — Odin puts every task in its own group,
        but the nesting is OSMO's regardless.
        """
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise OsmoCliError(f"could not parse JSON status: {e}") from e
        tasks = [
            TaskSnapshot(
                name=str(task["name"]),
                status=str(task["status"]),
                exit_code=(None if task.get("exit_code") in (None, "-") else int(task["exit_code"])),
            )
            for group in data.get("groups") or []
            for task in group.get("tasks") or []
        ]
        return WorkflowSnapshot(
            workflow_id=str(data.get("name") or workflow_id),
            status=str(data["status"]),
            tasks=tasks,
        )

    def validate(self, yaml_path: Path) -> None:
        """Validate a workflow YAML server-side without submitting it.

        Catches schema drift that local rendering cannot: OSMO rejects unknown
        task fields outright, and validates pool, platform, credential names,
        and image reachability.

        Args:
            yaml_path: Path to the rendered workflow YAML.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cp = self._run([self._exe, "workflow", "validate", str(yaml_path)])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow validate` failed for {yaml_path.name}: {cp.stderr.strip()}")

    def data_check(self, remote_uri: str) -> bool:
        """Return whether the profile can write to *remote_uri*.

        Used as a submit-time preflight. A bucket can be flipped read-only
        server-side — that is how OSMO retired datasets, which surfaces as
        ``Bucket <name> mode is read-only`` — and discovering it after a
        dispatch has burned GPU hours is expensive.

        Args:
            remote_uri: Backend URI, e.g. ``swift://host/AUTH_x/prefix``.

        Returns:
            ``True`` when the check reports a pass.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cp = self._run([self._exe, "data", "check", remote_uri, "-a", "WRITE"])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo data check` failed: {cp.stderr.strip()}")
        return '"pass"' in cp.stdout or "'pass'" in cp.stdout

    def data_probe_write(self, remote_uri: str) -> str | None:
        """Upload a few bytes to *remote_uri* and return the failure reason, if any.

        :meth:`data_check` reports only permission, so it passes on a bucket
        whose account is over quota, where every upload then fails with
        ``EntityTooLarge ... Upload exceeds quota``. OSMO also exits 0 and
        prints ``Data has been uploaded`` in that case, so the outcome has to
        be read out of the output.

        Args:
            remote_uri: Backend URI to probe under.

        Returns:
            ``None`` when the probe uploaded, else the reported reason.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "odin-preflight-probe"
            probe.write_text("odin preflight\n")
            cp = self._run([self._exe, "data", "upload", f"{remote_uri.rstrip('/')}/_preflight/", str(probe)])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo data upload` failed: {cp.stderr.strip()}")
        combined = f"{cp.stdout}\n{cp.stderr}"
        if "Upload Failed on files" not in combined:
            return None
        for line in combined.splitlines():
            if "An error occurred" in line:
                # Trailing "Retrying N more times. Request ID: None" is noise.
                return line.split("An error occurred", 1)[1].split(". Retrying", 1)[0].strip()
        return "upload reported a failure"

    def data_download(self, remote_uri: str, dest_dir: Path) -> None:
        """Download a bucket URI into a local directory.

        OSMO datasets were retired; ``osmo data`` against a backend URI is the
        replacement.

        Args:
            remote_uri: Backend URI, e.g. ``s3://bucket/prefix``.
            dest_dir: Local destination, created with parents if missing.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        cp = self._run([self._exe, "data", "download", remote_uri, str(dest_dir)])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo data download` failed: {cp.stderr.strip()}")
