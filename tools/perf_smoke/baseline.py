# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve exact GitHub benchmark evidence and its historical branch baseline.

Selection uses ancestry and measurement time, never measured performance or the
conclusion of an unrelated workflow job. Original ZIPs and incomplete workloads
remain available to the report layer. This module does not execute benchmarks.
"""

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

    def _request(self, path: str, *, download: bool = False) -> tuple[bytes, Any]:
        url = "https://api.github.com" + path
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            try:
                response = self._opener.open(request)
            except urllib.error.HTTPError as exc:
                if not download or exc.code not in (301, 302, 303, 307, 308):
                    raise
                destination = exc.headers.get("Location", "")
                if urllib.parse.urlsplit(destination).scheme != "https":
                    raise EvidenceError("download_redirect", "GitHub artifact redirect is not HTTPS") from None
                # A fresh request/opener carries no GitHub authorization to the signed URL.
                response = urllib.request.urlopen(urllib.request.Request(destination))
            with response:
                return response.read(), response.headers
        except EvidenceError:
            raise
        except urllib.error.HTTPError as exc:
            code = "expired" if exc.code == 410 else "inaccessible"
            raise EvidenceError(code, f"GitHub evidence request failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError) as exc:
            # Exception text can include credential-bearing signed URLs.
            raise EvidenceError("inaccessible", f"GitHub evidence request failed ({type(exc).__name__})") from None

    def get(self, path: str) -> dict:
        data, _ = self._request(path)
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

    def runs(self, commit: str, branch: str, event: str, workflow_id: int | None = None) -> list[dict]:
        workflow = f"workflows/{workflow_id}/" if workflow_id else ""
        return self.paginate(
            f"/repos/{self.repository}/actions/{workflow}runs",
            "workflow_runs",
            head_sha=commit,
            branch=branch,
            event=event,
        )

    def commit(self, commit: str) -> dict:
        return self.get(f"/repos/{self.repository}/commits/{urllib.parse.quote(commit, safe='')}")

    def download(self, artifact: dict) -> bytes:
        if artifact.get("expired"):
            raise EvidenceError("expired", "Selected GitHub artifact has expired", {"artifact_id": artifact.get("id")})
        data, _ = self._request(
            f"/repos/{self.repository}/actions/artifacts/{artifact['id']}/zip",
            download=True,
        )
        return data


@dataclass
class Evidence:
    identity: dict[str, Any]
    context: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]]
    statuses: dict[str, str]
    issues: list[str]
    measurement_start: str | None
    measurement_end: str | None
    zip_bytes: bytes = field(repr=False)
    run: dict[str, Any] = field(repr=False)

    @property
    def has_completed_runtime(self) -> bool:
        return any(_usable(item["bundle"]) for items in self.samples.values() for item in items)


@dataclass
class Selection:
    evidence: Evidence | None
    metadata: dict[str, Any]


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def _usable(bundle: dict) -> bool:
    fps = bundle.get("runtime", {}).get("total_fps")
    value = fps.get("mean") if isinstance(fps, dict) else None
    if bundle.get("run", {}).get("status") != "completed" or isinstance(value, bool):
        return False
    try:
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


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


def _zip_files(client: GitHubClient, artifact: dict) -> tuple[bytes, dict[str, bytes]]:
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


def _github_links(identity: dict) -> dict[str, str]:
    repository, run_id, attempt = identity["repository"], identity["run_id"], identity["run_attempt"]
    return {
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{attempt}",
        "commit_url": f"https://github.com/{repository}/commit/{identity['source_commit']}",
        "artifact_url": f"https://github.com/{repository}/actions/runs/{run_id}/artifacts/{identity['artifact_id']}",
    }


def _saved_baseline_identity(value: Any) -> dict:
    """Validate fields needed to resolve a saved pin and rebuild its public links."""
    if not isinstance(value, dict):
        raise EvidenceError("corrupt", "Previous comparison baseline identity is not an object")
    repository = value.get("repository")
    if not isinstance(repository, str) or re.fullmatch(r"[\w.-]+/[\w.-]+", repository) is None:
        raise EvidenceError("corrupt", "Previous comparison baseline repository is invalid")
    for key in ("run_id", "run_attempt", "artifact_id"):
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise EvidenceError("corrupt", f"Previous comparison baseline {key} is not a positive integer")
    digest = value.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise EvidenceError("corrupt", "Previous comparison baseline SHA256 is invalid")
    if not _sha(value.get("source_commit")):
        raise EvidenceError("corrupt", "Previous comparison baseline source commit is not a full Git SHA")
    identity = {**value, "sha256": digest.lower(), "source_commit": value["source_commit"].lower()}
    identity["artifact_name"] = f"performance-smoke-{identity['run_id']}-{identity['run_attempt']}"
    identity.update(_github_links(identity))
    return identity


def read_evidence(client: GitHubClient, run: dict, artifact: dict, attempt: int) -> Evidence:
    """Download one exact producing attempt; do not discard failed or missing legs."""
    run_id = run["id"]
    name = f"performance-smoke-{run_id}-{attempt}"
    if artifact.get("name") != name or (artifact.get("workflow_run") or {}).get("id", run_id) != run_id:
        raise EvidenceError("identity_mismatch", "Artifact does not belong to the requested producing attempt")
    try:
        data, files = _zip_files(client, artifact)
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
    if not _sha(commit):
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
            start = _time(bundle["run"].get("start_time_utc"))
            end = _time(bundle["run"].get("end_time_utc"))
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
            if bundle["run"].get("status") == "completed" and not _usable(bundle):
                issues.append(f"{path}: completed runtime has no finite nonnegative FPS value")
    if not context:
        issues.append("Build context and FPS formula provenance are unavailable in this historical artifact")
    capture_time = _time(execution.get("measurement_not_before"))
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
        job_ends = [_time(job.get("completed_at")) for job in jobs]
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
        "run_url": f"https://github.com/{client.repository}/actions/runs/{run_id}/attempts/{attempt}",
        "commit_url": f"https://github.com/{client.repository}/commit/{commit}",
        "artifact_url": f"https://github.com/{client.repository}/actions/runs/{run_id}/artifacts/{artifact['id']}",
    }
    return Evidence(
        identity,
        context,
        samples,
        statuses,
        issues,
        cutoff.isoformat() if cutoff else None,
        completed.isoformat() if completed else None,
        data,
        run,
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
        started = _time(run.get("run_started_at"))
        jobs = [job for job in client.jobs(run_id, attempt) if job.get("name") == "performance-smoke-benchmarks"]
        for job in jobs:
            if job.get("conclusion") == "skipped":
                continue
            job_started = _time(job.get("started_at"))
            if started is None or job_started is None:
                raise EvidenceError("ambiguous_attempt", "Cannot establish when the benchmark job ran")
            if job_started >= started:
                raise EvidenceError(
                    "missing_candidate",
                    "Benchmark ran in this attempt but its evidence artifact is unavailable",
                    {"run_id": run_id, "run_attempt": attempt},
                )
        # Carried-forward jobs can have fresh IDs/attempt numbers but old start times.
    raise EvidenceError("missing_candidate", "No producing benchmark artifact is available for this workflow run")


def load_previous_selection(client: GitHubClient, candidate: Evidence) -> dict | None:
    """Recover a prior A only when the report identifies the identical candidate ZIP."""
    identity = candidate.identity
    report_attempt = identity.get("report_attempt", identity["run_attempt"])
    prefix = f"performance-build-comparison-{identity['run_id']}-"
    reports = []
    for artifact in client.artifacts(identity["run_id"]):
        name = artifact.get("name", "")
        suffix = name.removeprefix(prefix)
        if name.startswith(prefix) and suffix.isdigit() and int(suffix) < report_attempt:
            reports.append((int(suffix), artifact))
    for _, artifact in sorted(reports, key=lambda item: item[0], reverse=True):
        _, files = _zip_files(client, artifact)
        if "build-comparison.json" not in files:
            raise EvidenceError("corrupt", "Previous comparison artifact has no build-comparison.json")
        report = _object(files["build-comparison.json"], "build-comparison.json")
        previous = report.get("candidate")
        if previous is None:
            continue
        if not isinstance(previous, dict):
            raise EvidenceError("corrupt", "Previous comparison candidate identity is not an object or null")
        if (previous.get("artifact_id"), previous.get("sha256")) == (identity["artifact_id"], identity["sha256"]):
            if report.get("baseline") is not None:
                return _saved_baseline_identity(report["baseline"])
            selection = report.get("selection")
            if selection is not None and not isinstance(selection, dict):
                raise EvidenceError("corrupt", "Previous comparison selection is not an object or null")
            if selection and selection.get("pinned") is True and selection.get("unavailable_side") == "baseline":
                return _saved_baseline_identity(selection.get("unavailable_evidence"))
    return None


def select_baseline(client: GitHubClient, candidate: Evidence, pinned_identity: dict | None = None) -> Selection:
    """Find one measured branch execution on the nearest first-parent ancestor."""
    if pinned_identity is not None:
        pinned_identity = _saved_baseline_identity(pinned_identity)
    source = candidate.context.get("source", {})
    branch = source.get("reference_branch")
    anchor = source.get("reference_commit")
    if not branch and candidate.identity["event"] in ("push", "workflow_dispatch"):
        branch = candidate.identity["branch"]
    metadata: dict[str, Any] = {
        "reference_branch": branch,
        "reference_commit": anchor,
        "candidate_measurement_start": candidate.measurement_start,
        "visited_commits": [],
        "issues": [],
        "pinned": bool(pinned_identity),
    }
    candidate_keys = {
        key
        for items in candidate.samples.values()
        for item in items
        if (key := workload_key(item["bundle"])) is not None
    }
    if not candidate_keys:
        metadata.update(
            reason_code="no_readable_candidate_workload",
            reason="Candidate evidence has no readable workload identity to match against historical measurements",
        )
        return Selection(None, metadata)
    if not anchor and candidate.identity["event"] in ("push", "workflow_dispatch"):
        parents = client.commit(candidate.identity["source_commit"]).get("parents", [])
        anchor = parents[0].get("sha") if parents else None
        metadata["reference_commit"] = anchor
    if pinned_identity:
        if pinned_identity.get("repository") != client.repository:
            raise EvidenceError("identity_mismatch", "Pinned baseline belongs to another repository")
        run_id, attempt = pinned_identity["run_id"], pinned_identity["run_attempt"]
        artifact = _exact_artifact(client.artifacts(run_id), f"performance-smoke-{run_id}-{attempt}")
        if artifact is None or artifact["id"] != pinned_identity.get("artifact_id"):
            raise EvidenceError(
                "missing_pinned_baseline", "Previously selected baseline artifact is unavailable", pinned_identity
            )
        baseline = read_evidence(client, client.run_attempt(run_id, attempt), artifact, attempt)
        if (
            baseline.identity["sha256"] != pinned_identity["sha256"]
            or baseline.identity["source_commit"] != pinned_identity["source_commit"]
        ):
            raise EvidenceError(
                "identity_mismatch",
                "Previously selected baseline artifact bytes or source identity changed",
                pinned_identity,
            )
        metadata.update(
            reason="Reused the prior report's exact baseline for the same candidate artifact",
            selected=baseline.identity,
        )
        return Selection(baseline, metadata)
    cutoff = _time(candidate.measurement_start)
    if not branch or not _sha(anchor) or cutoff is None:
        metadata["reason"] = "Baseline reference branch, tested parent, or candidate measurement start is unavailable"
        return Selection(None, metadata)
    while anchor:
        if anchor in metadata["visited_commits"]:
            raise EvidenceError("invalid_ancestry", "GitHub commit ancestry contains a cycle")
        metadata["visited_commits"].append(anchor)
        eligible = []
        for event in ("push", "workflow_dispatch"):
            for listed in client.runs(anchor, branch, event, candidate.identity.get("workflow_id")):
                if (
                    listed.get("head_sha") != anchor
                    or listed.get("head_branch") != branch
                    or listed.get("event") != event
                ):
                    continue
                run_id = listed["id"]
                for artifact in client.artifacts(run_id):
                    match = re.fullmatch(rf"performance-smoke-{run_id}-(\d+)", artifact.get("name", ""))
                    if not match:
                        continue
                    attempt = int(match[1])
                    run = client.run_attempt(run_id, attempt)
                    attempt_start = _time(run.get("run_started_at"))
                    if attempt_start and attempt_start >= cutoff:
                        continue
                    baseline = read_evidence(client, run, artifact, attempt)
                    if baseline.identity["source_commit"] != anchor:
                        metadata["issues"].append(
                            {
                                "artifact_id": artifact["id"],
                                "reason": "Measured checkout differs from the branch commit",
                            }
                        )
                        continue
                    end = _time(baseline.measurement_end)
                    if not baseline.has_completed_runtime:
                        metadata["issues"].append(
                            {"artifact_id": artifact["id"], "reason": "No usable completed runtime sample"}
                        )
                    elif not any(
                        baseline.statuses.get(leg) == "ok"
                        and _usable(item["bundle"])
                        and workload_key(item["bundle"]) in candidate_keys
                        for leg, items in baseline.samples.items()
                        for item in items
                    ):
                        metadata["issues"].append(
                            {
                                "artifact_id": artifact["id"],
                                "reason": "No shared workload with status ok and usable completed evidence",
                            }
                        )
                    elif end is None:
                        metadata["issues"].append(
                            {"artifact_id": artifact["id"], "reason": "Measurement end time is unavailable"}
                        )
                    elif end < cutoff:
                        eligible.append((end, baseline))
        if eligible:
            eligible.sort(key=lambda item: (item[0], item[1].identity["run_id"], item[1].identity["run_attempt"]))
            baseline = eligible[-1][1]
            metadata.update(
                reason=(
                    "Latest measured branch execution before the candidate, "
                    "on the nearest measured first-parent ancestor"
                ),
                selected=baseline.identity,
                baseline_measurement_end=baseline.measurement_end,
            )
            return Selection(baseline, metadata)
        parents = client.commit(anchor).get("parents", [])
        anchor = parents[0].get("sha") if parents else None
        if anchor is not None and not _sha(anchor):
            raise EvidenceError("invalid_ancestry", "GitHub returned an invalid first-parent commit")
    metadata["reason"] = "No available measured target-branch ancestor completed before the candidate"
    return Selection(None, metadata)
