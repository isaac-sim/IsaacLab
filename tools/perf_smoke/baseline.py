# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read exact GitHub benchmark artifacts, preserving incomplete workloads and source evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_REQUEST_TIMEOUT_SECONDS = 60


class EvidenceError(ValueError):
    """An explicit evidence failure, with a safe machine-readable reason."""

    def __init__(self, code: str, message: str, identity: dict | None = None):
        super().__init__(message)
        self.code = code
        self.identity = identity or {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubClient:
    """Small stdlib REST client. Authentication is confined to the API origin."""

    def __init__(self, repository: str, token: str | None = None):
        if re.fullmatch(r"[\w.-]+/[\w.-]+", repository) is None:
            raise EvidenceError("invalid_repository", "Expected an owner/repository name")
        self.repository = repository
        self._token = token if token is not None else os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, path: str, *, download: bool = False) -> bytes:
        url = "https://api.github.com" + path
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            try:
                response = self._opener.open(request, timeout=_REQUEST_TIMEOUT_SECONDS)
            except urllib.error.HTTPError as exc:
                if not download or exc.code not in (301, 302, 303, 307, 308):
                    raise
                destination = exc.headers.get("Location", "")
                if urllib.parse.urlsplit(destination).scheme != "https":
                    raise EvidenceError("download_redirect", "GitHub artifact redirect is not HTTPS") from None
                # Start a new request so the signed download does not receive GitHub credentials.
                response = urllib.request.urlopen(urllib.request.Request(destination), timeout=_REQUEST_TIMEOUT_SECONDS)
            with response:
                return response.read()
        except EvidenceError:
            raise
        except urllib.error.HTTPError as exc:
            code = "expired" if exc.code == 410 else "inaccessible"
            raise EvidenceError(code, f"GitHub evidence request failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError) as exc:
            # Exception text can include credential-bearing signed URLs.
            if isinstance(exc, TimeoutError) or (
                isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)
            ):
                raise EvidenceError("timeout", "GitHub evidence request timed out") from None
            raise EvidenceError("inaccessible", f"GitHub evidence request failed ({type(exc).__name__})") from None

    def get(self, path: str) -> dict:
        data = self._request(path)
        try:
            value = json.loads(data)
        except (UnicodeError, ValueError):
            raise EvidenceError("invalid_api_response", "GitHub response was not JSON") from None
        if not isinstance(value, dict):
            raise EvidenceError("invalid_api_response", "GitHub response was not an object")
        return value

    def paginate(self, path: str, key: str, **params: Any) -> list[dict]:
        """Read every API page; detect an incomplete API result instead of truncating."""
        output = []
        page = 1
        total = None
        while True:
            query = urllib.parse.urlencode({**params, "per_page": 100, "page": page})
            value = self.get(f"{path}?{query}")
            batch = value.get(key)
            if not isinstance(batch, list) or not all(isinstance(item, dict) for item in batch):
                raise EvidenceError("invalid_api_response", f"GitHub response has no {key} array")
            if isinstance(value.get("total_count"), int):
                total = value["total_count"]
            output.extend(batch)
            if not batch or len(batch) < 100 or (total is not None and len(output) >= total):
                if total is not None and len(output) < total:
                    raise EvidenceError("incomplete_history", "GitHub did not return the complete requested history")
                return output
            page += 1

    def run_attempt(self, run_id: int, attempt: int) -> dict:
        return self.get(f"/repos/{self.repository}/actions/runs/{run_id}/attempts/{attempt}")

    def jobs(self, run_id: int, attempt: int) -> list[dict]:
        return self.paginate(f"/repos/{self.repository}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")

    def artifacts(self, run_id: int) -> list[dict]:
        return self.paginate(f"/repos/{self.repository}/actions/runs/{run_id}/artifacts", "artifacts")

    def download(self, artifact: dict) -> bytes:
        if artifact.get("expired"):
            raise EvidenceError("expired", "Selected GitHub artifact has expired", {"artifact_id": artifact.get("id")})
        return self._request(
            f"/repos/{self.repository}/actions/artifacts/{artifact['id']}/zip",
            download=True,
        )


@dataclass
class Evidence:
    identity: dict[str, Any]
    context: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]]
    statuses: dict[str, str]
    issues: list[str]
    measurement_start: str | None
    measurement_end: str | None
    files: dict[str, bytes] = field(repr=False)


def parse_timestamp(value: Any) -> datetime | None:
    """Read a timezone-aware ISO timestamp as UTC, or return None."""
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def is_commit_sha(value: Any) -> bool:
    """Whether the value identifies a full Git commit SHA."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def valid_fps(value: Any) -> float | None:
    """Read a finite, nonnegative numeric FPS value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def has_usable_runtime(bundle: dict) -> bool:
    """Whether a completed runtime sample has a finite, nonnegative FPS value."""
    fps = bundle.get("runtime", {}).get("total_fps")
    value = fps.get("mean") if isinstance(fps, dict) else None
    return bundle.get("run", {}).get("status") == "completed" and valid_fps(value) is not None


def workload_key(bundle: dict) -> str | None:
    """Shared workload identity, independent of metric formula and runtime versions."""
    run = bundle.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("config"), dict):
        return None
    config = run["config"]
    task, physics, envs = run.get("task"), config.get("physics_backend"), run.get("num_envs")
    renderer = config.get("rendering_backend")
    presets = config.get("presets")
    if (
        not isinstance(task, str)
        or not task
        or not isinstance(physics, str)
        or not physics
        or not isinstance(renderer, str)
        or not renderer.strip()
        or not isinstance(envs, int)
        or isinstance(envs, bool)
        or envs <= 0
        or not isinstance(presets, list)
        or not all(isinstance(item, str) for item in presets)
    ):
        return None
    return json.dumps(
        [task, physics, renderer, envs, presets],
        separators=(",", ":"),
    )


def download_artifact(client: GitHubClient, artifact: dict) -> tuple[bytes, dict[str, bytes]]:
    """Download and validate an artifact, retaining its exact ZIP and decoded files."""
    data = client.download(artifact)
    digest = hashlib.sha256(data).hexdigest()
    if artifact.get("digest") and artifact["digest"].lower() != f"sha256:{digest}":
        raise EvidenceError(
            "corrupt", "GitHub artifact digest does not match downloaded ZIP", {"artifact_id": artifact["id"]}
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = {}
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if entry.filename in files:
                    raise EvidenceError("corrupt", "Artifact ZIP contains duplicate file names")
                files[entry.filename] = archive.read(entry)
            return data, files
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise EvidenceError("corrupt", f"Artifact ZIP could not be read ({type(exc).__name__})") from None


def _object(data: bytes, name: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError):
        raise EvidenceError("corrupt", f"Artifact {name} is not valid JSON") from None
    if not isinstance(value, dict):
        raise EvidenceError("corrupt", f"Artifact {name} is not an object")
    return value


def artifact_json(files: dict[str, bytes], name: str) -> dict:
    """Read a named JSON object from the validated artifact contents."""
    if name not in files:
        raise EvidenceError("missing_source_evidence", f"The artifact has no {name}.")
    return _object(files[name], name)


def _github_links(identity: dict) -> dict[str, str]:
    repository, run_id, attempt = identity["repository"], identity["run_id"], identity["run_attempt"]
    return {
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{attempt}",
        "commit_url": f"https://github.com/{repository}/commit/{identity['source_commit']}",
        "artifact_url": f"https://github.com/{repository}/actions/runs/{run_id}/artifacts/{identity['artifact_id']}",
    }


def read_evidence(
    client: GitHubClient,
    run: dict,
    artifact: dict,
    attempt: int,
    *,
    artifact_name: str | None = None,
    contents: tuple[bytes, dict[str, bytes]] | None = None,
) -> Evidence:
    """Download one exact producing attempt; do not discard failed or missing legs."""
    run_id = run["id"]
    name = artifact_name or f"performance-smoke-{run_id}-{attempt}"
    if artifact.get("name") != name or (artifact.get("workflow_run") or {}).get("id", run_id) != run_id:
        raise EvidenceError("identity_mismatch", "Artifact does not belong to the requested producing attempt")
    try:
        data, files = contents if contents is not None else download_artifact(client, artifact)
    except EvidenceError as exc:
        known = {
            "repository": client.repository,
            "run_id": run_id,
            "run_attempt": attempt,
            "artifact_id": artifact.get("id"),
            "artifact_name": name,
            "source_commit": run.get("head_sha"),
            "github_head_commit": run.get("head_sha"),
            "tested_commit": None,
            "source_provenance": "github_run_metadata",
            "event": run.get("event"),
            "branch": run.get("head_branch"),
        }
        known.update(_github_links(known))
        raise EvidenceError(exc.code, str(exc), known) from None
    context = _object(files["build-context.json"], "build-context.json") if "build-context.json" in files else {}
    execution = context.get("execution", {})
    source = context.get("source", {})
    if not isinstance(execution, dict) or not isinstance(source, dict):
        raise EvidenceError("corrupt", "Artifact build context source/execution is not an object")
    if context and (execution.get("run_id") != run_id or execution.get("run_attempt") != attempt):
        raise EvidenceError("identity_mismatch", "Artifact build context identifies another producing attempt")
    commit = source.get("commit") or run.get("head_sha")
    if not is_commit_sha(commit):
        raise EvidenceError("identity_mismatch", "Evidence source commit is not a full Git SHA")
    samples: dict[str, list[dict]] = {}
    statuses = {}
    issues = []
    starts = []
    ends = []
    missing_start = False
    missing_end = False
    for path, content in sorted(files.items()):
        parts = path.split("/")
        if len(parts) == 2 and parts[1] == "status":
            try:
                statuses[parts[0]] = content.decode("utf-8").strip()
            except UnicodeError:
                raise EvidenceError("corrupt", "Workload status is not UTF-8") from None
            samples.setdefault(parts[0], [])
        elif len(parts) >= 2 and parts[-1].startswith("benchmark_runtime_") and parts[-1].endswith(".json"):
            samples.setdefault(parts[0], [])
            try:
                bundle = _object(content, "runtime bundle")
                if not isinstance(bundle.get("run"), dict) or not isinstance(bundle.get("runtime"), dict):
                    raise EvidenceError("corrupt", "Runtime bundle run/runtime is not an object")
            except EvidenceError as exc:
                issues.append(f"{path}: {exc}")
                missing_start = missing_end = True
                continue
            samples.setdefault(parts[0], []).append({"path": path, "bundle": bundle})
            start = parse_timestamp(bundle["run"].get("start_time_utc"))
            end = parse_timestamp(bundle["run"].get("end_time_utc"))
            if start and end and end < start:
                issues.append(f"{path}: runtime ends before it starts")
                start = end = None
            if start:
                starts.append(start)
            else:
                missing_start = True
                issues.append(f"{path}: runtime has no valid start timestamp")
            if end:
                ends.append(end)
            else:
                missing_end = True
                issues.append(f"{path}: runtime has no valid end timestamp")
            if bundle["run"].get("status") == "completed" and not has_usable_runtime(bundle):
                issues.append(f"{path}: completed runtime has no finite nonnegative FPS value")
    if not context:
        issues.append("Build context and FPS formula provenance are unavailable in this artifact")
    capture_time = parse_timestamp(execution.get("measurement_not_before"))
    cutoff = min(starts) if starts and not missing_start else capture_time
    if cutoff and (missing_start or not starts):
        if starts and cutoff > min(starts):
            cutoff = None
            issues.append("Provenance capture time is later than a sample start; candidate cutoff is unknown")
        else:
            issues.append("Candidate cutoff uses the provenance capture time because sample start times are incomplete")
    completed = max(ends) if ends and not missing_end else None
    if missing_end:
        jobs = [job for job in client.jobs(run_id, attempt) if job.get("name") == "performance-smoke-benchmarks"]
        job_ends = [parse_timestamp(job.get("completed_at")) for job in jobs]
        if job_ends and all(end is not None for end in job_ends):
            bound = max(job_ends)
            if not ends or bound >= max(ends):
                completed = bound
                issues.append(
                    "Measurement end uses the GitHub benchmark-job completion bound because sample times are incomplete"
                )
    identity = {
        "repository": client.repository,
        "run_id": run_id,
        "run_attempt": attempt,
        "artifact_id": artifact["id"],
        "artifact_name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_commit": commit,
        "tested_commit": source.get("commit"),
        "github_head_commit": run.get("head_sha"),
        "requested_head_commit": source.get("requested_head_commit"),
        "source_provenance": "artifact_context" if source.get("commit") else "github_run_metadata",
        "event": run.get("event"),
        "branch": run.get("head_branch"),
        "workflow_id": run.get("workflow_id"),
    }
    identity.update(_github_links(identity))
    return Evidence(
        identity,
        context,
        samples,
        statuses,
        issues,
        cutoff.isoformat() if cutoff else None,
        completed.isoformat() if completed else None,
        files,
    )


def _exact_artifact(artifacts: list[dict], name: str) -> dict | None:
    matches = [artifact for artifact in artifacts if artifact.get("name") == name]
    if len(matches) > 1:
        raise EvidenceError("ambiguous", f"More than one GitHub artifact has the exact name {name}")
    return matches[0] if matches else None


def resolve_candidate(client: GitHubClient, run_id: int, report_attempt: int) -> Evidence:
    """Resolve the measured attempt, including a rerun of only the summary job."""
    artifacts = client.artifacts(run_id)
    for attempt in range(report_attempt, 0, -1):
        run = client.run_attempt(run_id, attempt)
        artifact = _exact_artifact(artifacts, f"performance-smoke-{run_id}-{attempt}")
        if artifact:
            result = read_evidence(client, run, artifact, attempt)
            result.identity["report_attempt"] = report_attempt
            return result
        started = parse_timestamp(run.get("run_started_at"))
        jobs = [job for job in client.jobs(run_id, attempt) if job.get("name") == "performance-smoke-benchmarks"]
        for job in jobs:
            if job.get("conclusion") == "skipped":
                continue
            job_started = parse_timestamp(job.get("started_at"))
            if started is None or job_started is None:
                raise EvidenceError("ambiguous_attempt", "Cannot establish when the benchmark job ran")
            if job_started >= started:
                raise EvidenceError(
                    "missing_candidate",
                    "Benchmark ran in this attempt but its evidence artifact is unavailable",
                    {"run_id": run_id, "run_attempt": attempt},
                )
        # Jobs carried over from an earlier attempt can have new IDs but retain their old start times.
    raise EvidenceError("missing_candidate", "No producing benchmark artifact is available for this workflow run")
