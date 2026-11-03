# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run, reuse, and compare verified PR benchmark measurements."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import io
import json
import os
import re
import socket
import statistics
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .contract import normalize_gpu_model, workload_identity
from .metrics import PerfSmokeError, nested_number
from .report import render_build_comparison
from .source_revision import RUNTIME_MODULE, measurement_digest, prepare_manifest

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

    def __init__(self, repository: str):
        if re.fullmatch(r"[\w.-]+/[\w.-]+", repository) is None:
            raise EvidenceError("invalid_repository", "Expected an owner/repository name")
        self.repository = repository
        self._token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
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
                # Keep the GitHub token out of the artifact download request.
                response = urllib.request.urlopen(urllib.request.Request(destination), timeout=_REQUEST_TIMEOUT_SECONDS)
            with response:
                return response.read()
        except urllib.error.HTTPError as exc:
            code = "expired" if exc.code == 410 else "inaccessible"
            raise EvidenceError(code, f"GitHub evidence request failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # Error messages may contain secret download URLs.
            if isinstance(exc, TimeoutError) or (
                isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)
            ):
                raise EvidenceError("timeout", "GitHub evidence request timed out") from None
            raise EvidenceError("inaccessible", f"GitHub evidence request failed ({type(exc).__name__})") from None

    def get(self, path: str) -> dict:
        return _json_object(self._request(path), "GitHub response", code="invalid_api_response")

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
        return self._request(f"/repos/{self.repository}/actions/artifacts/{artifact['id']}/zip", download=True)


@dataclasses.dataclass
class Evidence:
    identity: dict[str, Any]
    context: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]]
    statuses: dict[str, str]
    issues: list[str]
    files: dict[str, bytes] = dataclasses.field(repr=False)


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
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def runtime_fps(bundle: dict) -> float | None:
    """Read a sample's finite, nonnegative numeric FPS value, excluding booleans."""
    try:
        value = nested_number(bundle, ("runtime", "total_fps", "mean"))
        return value if value is not None and value >= 0 else None
    except PerfSmokeError:
        return None


def workload_key(bundle: dict) -> str | None:
    """Shared workload identity, independent of metric formula and runtime versions."""
    try:
        return json.dumps(workload_identity(bundle), sort_keys=True, separators=(",", ":"))
    except PerfSmokeError:
        return None


def download_artifact(
    client: GitHubClient, artifact: dict, comparison_dir: Path | None = None
) -> tuple[str, dict[str, bytes]]:
    """Download and validate an artifact, retaining its ZIP digest and decoded files."""
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
            if comparison_dir is not None:
                for name in files:
                    if Path(name).name == "comparison.json":
                        archive.extract(name, comparison_dir)
            return digest, files
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise EvidenceError("corrupt", f"Artifact ZIP could not be read ({type(exc).__name__})") from None


def _json_object(data: bytes, description: str, code: str = "corrupt") -> dict:
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError):
        raise EvidenceError(code, f"{description} is not valid JSON") from None
    if not isinstance(value, dict):
        raise EvidenceError(code, f"{description} is not an object")
    return value


def artifact_json(files: dict[str, bytes], name: str) -> dict:
    if name not in files:
        raise EvidenceError("missing_source_evidence", f"The artifact has no {name}.")
    return _json_object(files[name], f"Artifact {name}")


def read_evidence(
    client: GitHubClient,
    run: dict,
    artifact: dict,
    attempt: int,
    *,
    artifact_name: str | None = None,
    contents: tuple[str, dict[str, bytes]] | None = None,
    comparison_dir: Path | None = None,
) -> Evidence:
    """Download one exact producing attempt; do not discard failed or missing legs."""
    run_id = run["id"]
    name = artifact_name or f"performance-smoke-{run_id}-{attempt}"
    if artifact.get("name") != name or (artifact.get("workflow_run") or {}).get("id", run_id) != run_id:
        raise EvidenceError("identity_mismatch", "Artifact does not belong to the requested producing attempt")
    identity = {
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
    try:
        digest, files = contents if contents is not None else download_artifact(client, artifact, comparison_dir)
    except EvidenceError as exc:
        raise EvidenceError(exc.code, str(exc), identity) from None
    context = artifact_json(files, "build-context.json") if "build-context.json" in files else {}
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
                bundle = _json_object(content, "Artifact runtime bundle")
                if not isinstance(bundle.get("run"), dict) or not isinstance(bundle.get("runtime"), dict):
                    raise EvidenceError("corrupt", "Runtime bundle run/runtime is not an object")
            except EvidenceError as exc:
                issues.append(f"{path}: {exc}")
                continue
            samples[parts[0]].append({"path": path, "bundle": bundle})
            if bundle["run"].get("status") == "completed" and runtime_fps(bundle) is None:
                issues.append(f"{path}: completed runtime has no finite nonnegative FPS value")
    if not context:
        issues.append("Build context and FPS formula provenance are unavailable in this artifact")
    identity.update(
        sha256=digest,
        source_commit=commit,
        tested_commit=source.get("commit"),
        requested_head_commit=source.get("requested_head_commit"),
        source_provenance="artifact_context" if source.get("commit") else "github_run_metadata",
        workflow_id=run.get("workflow_id"),
    )
    return Evidence(identity, context, samples, statuses, issues, files)


def resolve_candidate(
    client: GitHubClient, run_id: int, report_attempt: int, comparison_dir: Path | None = None
) -> Evidence:
    """Read the measured attempt shared by the paired report and rolling-history gate."""
    artifacts = client.artifacts(run_id)
    for attempt in range(report_attempt, 0, -1):
        run = client.run_attempt(run_id, attempt)
        name = f"performance-smoke-{run_id}-{attempt}"
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        if len(matches) > 1:
            raise EvidenceError("ambiguous", f"More than one GitHub artifact has the exact name {name}")
        if matches:
            return read_evidence(client, run, matches[0], attempt, comparison_dir=comparison_dir)
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
        # Reruns can give old jobs new IDs, so compare their start times.
    raise EvidenceError("missing_candidate", "No producing benchmark artifact is available for this workflow run")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def baseline_name(pr_number: int, commit: str, attempt: int = 1) -> str:
    return f"performance-pr-baseline-{pr_number}-{commit}-{attempt}"


def _checkout_source(root: Path) -> tuple[str, list[str]]:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    # Read the commit object because shallow checkouts hide parents from rev-list.
    header = subprocess.check_output(["git", "-C", str(root), "cat-file", "-p", commit], text=True)
    parents = [line.split()[1] for line in header.split("\n\n", 1)[0].splitlines() if line.startswith("parent ")]
    return commit, parents


def resolve_base(commit: str, parents: list[str], event: dict) -> str:
    """Resolve the base commit from the tested PR merge."""
    tested = os.environ.get("GITHUB_SHA")
    head = event["pull_request"]["head"]["sha"]
    if not is_commit_sha(tested) or commit != tested:
        raise ValueError("Checked-out PR revision does not match the immutable tested merge GITHUB_SHA.")
    if len(parents) != 2 or parents[1] != head:
        raise ValueError("Tested PR merge does not have the event's head commit as its second parent.")
    return parents[0]


def _resolved_base() -> str:
    commit = os.environ.get("PERF_BASE_COMMIT")
    if not is_commit_sha(commit):
        detail = os.environ.get("PERF_PAIR_ERROR")
        raise ValueError(detail or "The tested PR merge's first parent was not resolved; PERF_BASE_COMMIT is missing.")
    return commit


def _benchmark_protocol(legs: Path, measurement_manifest: dict | None = None) -> dict[str, str]:
    """Identify the matrix and preserved controller files that actually launch both builds."""
    paths = {
        "matrix_sha256": legs,
        "benchmark_launcher_sha256": Path(__file__).with_name("run_benchmarks.sh"),
        "source_launcher_sha256": Path(__file__).with_name("source_revision.py"),
    }
    protocol = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    if measurement_manifest is not None:
        protocol["measurement_sha256"] = measurement_manifest["digest"]
    return protocol


def source_issues(evidence: Evidence) -> list[str]:
    """Check each result against its same-process source proof, retaining per-leg failures."""
    files, commit = evidence.files, evidence.identity["source_commit"]
    try:
        manifest = artifact_json(files, "source-manifest.json")
        if manifest.get("schema_version") != 4 or manifest.get("commit") != commit or not manifest.get("files"):
            raise EvidenceError("source_mismatch", "Source manifest does not identify the selected commit.")
        measurement = manifest.get("measurement")
        if measurement is None and (
            evidence.context.get("measurement") is not None
            or str(evidence.context.get("metric_definition", {}).get("total_fps", "")).startswith("measurement-v1:")
        ):
            raise EvidenceError("source_mismatch", "Shared measurement source manifest is missing.")
        if measurement is not None and (
            not measurement.get("files")
            or measurement_digest(measurement["files"]) != measurement.get("digest")
            or evidence.context.get("measurement") != {key: measurement[key] for key in ("commit", "digest")}
            or evidence.context.get("metric_definition", {}).get("total_fps")
            != f"measurement-v1:{measurement['digest']}"
        ):
            raise EvidenceError("source_mismatch", "Measurement source identity was not verified.")
    except (EvidenceError, KeyError, TypeError, AttributeError) as exc:
        return [f"{leg}/source: {exc}" for leg in evidence.samples.keys() | evidence.statuses.keys()]
    issues = []
    for leg, samples in evidence.samples.items():
        for sample in samples:
            path = sample["path"]
            sidecar_path = str(PurePosixPath(path).parent / "source-revision.json")
            try:
                proof = artifact_json(files, sidecar_path)
                modules = proof.get("modules") or []
                if (
                    proof.get("schema_version") != 4
                    or proof.get("status") != "verified"
                    or proof.get("bytecode_policy") != "fresh_process_cache"
                    or proof.get("commit") != commit
                    or proof.get("mismatches") != []
                    or not any(module.get("name") == RUNTIME_MODULE for module in modules)
                ):
                    raise EvidenceError("source_mismatch", "Executing source was not verified for this commit.")
                if measurement is not None and proof.get("measurement_digest") != measurement["digest"]:
                    raise EvidenceError("source_mismatch", "Measurement source proof does not match.")
                for module in modules:
                    relative = module.get("relative_path")
                    if relative not in manifest["files"] and (
                        measurement is None or relative not in measurement["files"]
                    ):
                        raise EvidenceError("source_mismatch", "Loaded source does not match its manifest.")
                matches = [item for item in proof.get("outputs", []) if item.get("path") == PurePosixPath(path).name]
                content = files[path]
                if (
                    len(matches) != 1
                    or matches[0].get("sha256") != hashlib.sha256(content).hexdigest()
                    or matches[0].get("bytes") != len(content)
                ):
                    raise EvidenceError("source_mismatch", "Result bytes do not match their source verification.")
            except (EvidenceError, KeyError, TypeError, AttributeError) as exc:
                issues.append(f"{leg}/source: {path}: {exc}")
    return issues


def _read_baseline(client: GitHubClient, artifact: dict, pr_number: int, base_commit: str) -> Evidence:
    contents = download_artifact(client, artifact)
    context = artifact_json(contents[1], "build-context.json")
    execution, source = context.get("execution", {}), context.get("source", {})
    if not isinstance(execution, dict) or not isinstance(source, dict):
        raise EvidenceError("corrupt", "Baseline source/execution context is not an object.")
    run_id, attempt = execution.get("run_id"), execution.get("run_attempt")
    if (
        source.get("commit") != base_commit
        or source.get("reference_commit") != base_commit
        or source.get("pull_request_number") != pr_number
        or source.get("benchmark_role") != "baseline"
        or not isinstance(run_id, int)
        or not isinstance(attempt, int)
    ):
        raise EvidenceError("identity_mismatch", "Baseline does not belong to this PR and exact base commit.")
    run = client.run_attempt(run_id, attempt)
    if run.get("event") != "pull_request":
        raise EvidenceError("identity_mismatch", "Baseline was not produced by a PR workflow.")
    evidence = read_evidence(
        client,
        run,
        artifact,
        attempt,
        artifact_name=baseline_name(pr_number, base_commit, attempt),
        contents=contents,
    )
    for leg in execution.get("expected_legs", []):
        evidence.samples.setdefault(leg, [])
    evidence.issues.extend(source_issues(evidence))
    return evidence


def _initial_selection(event: dict, base_commit: str) -> dict:
    pr = event["pull_request"]
    return {
        "schema_version": 1,
        "pull_request_number": pr["number"],
        "reference_branch": pr["base"]["ref"],
        "reference_commit": base_commit,
        "event_base_commit": pr["base"]["sha"],
        "requested_head_commit": pr["head"]["sha"],
        "baseline_reused": False,
        "baseline_origin": None,
        "issues": [],
    }


def restore_baseline(
    client: GitHubClient,
    checkout_root: Path,
    selection_path: Path,
    event: dict,
    run_id: int,
    run_attempt: int,
    *,
    legs: Path,
    measurement_manifest: dict | None = None,
) -> dict:
    """Select a complete verified measurement for this PR/base, or request a fresh one."""
    pr = event["pull_request"]
    pr_number, base_commit = pr["number"], _resolved_base()
    prepare_manifest(checkout_root, measurement_manifest, expected_commit=base_commit)
    selection = _initial_selection(event, base_commit)
    selection["reason"] = "No reusable verified baseline exists for this PR and base commit; measure it before the PR."
    # Save the fresh-run request in case the lookup is interrupted.
    _write(selection_path, selection)
    protocol = _benchmark_protocol(legs, measurement_manifest)
    prefix = f"performance-pr-baseline-{pr_number}-{base_commit}-"
    try:
        workflow_id = client.run_attempt(run_id, run_attempt).get("workflow_id")
        if not isinstance(workflow_id, int) or workflow_id <= 0:
            raise EvidenceError("identity_mismatch", "The producing workflow identity is unavailable.")
        runs = client.paginate(
            f"/repos/{client.repository}/actions/workflows/{workflow_id}/runs",
            "workflow_runs",
            event="pull_request",
            branch=pr["head"]["ref"],
        )
        artifacts = []
        for run in runs:
            try:
                artifacts.extend(
                    artifact for artifact in client.artifacts(run["id"]) if artifact.get("name", "").startswith(prefix)
                )
            except EvidenceError as exc:
                selection["issues"].append(f"Run {run['id']} artifacts were unavailable: {exc}")
        for artifact in sorted(artifacts, key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True):
            if artifact.get("expired"):
                continue
            try:
                evidence = _read_baseline(client, artifact, pr_number, base_commit)
                if evidence.identity.get("workflow_id") != workflow_id:
                    raise EvidenceError("identity_mismatch", "Baseline belongs to another workflow.")
                if evidence.context["execution"].get("benchmark_protocol") != protocol:
                    raise EvidenceError(
                        "protocol_mismatch",
                        "Baseline benchmark protocol is missing or differs from the current matrix, launchers,"
                        " or measurement code.",
                    )
                expected = evidence.context["execution"].get("expected_samples")
                planned = evidence.context["execution"].get("expected_legs")
                complete = (
                    isinstance(expected, int)
                    and expected > 0
                    and isinstance(planned, list)
                    and bool(planned)
                    and set(planned) == evidence.samples.keys() | evidence.statuses.keys()
                    and not evidence.issues
                    and all(
                        evidence.statuses.get(leg) == "ok"
                        and len(evidence.samples.get(leg, [])) == expected
                        and all(item["bundle"]["run"].get("status") == "completed" for item in evidence.samples[leg])
                        for leg in evidence.samples.keys() | evidence.statuses.keys()
                    )
                )
                if not complete:
                    raise EvidenceError("unverified_baseline", "Baseline is incomplete or its source proof is invalid.")
                if evidence.identity["run_id"] == run_id and evidence.identity["run_attempt"] >= run_attempt:
                    continue
                selection.update(
                    baseline_reused=True,
                    baseline_origin=evidence.identity,
                    reason="Reused the verified baseline measurement for this PR's unchanged base commit.",
                )
                break
            except EvidenceError as exc:
                selection["issues"].append(f"Artifact {artifact['id']} was not reused: {exc}")
    except EvidenceError as exc:
        selection["issues"].append(f"Baseline lookup unavailable; measuring a fresh baseline: {exc}")
    _write(selection_path, selection)
    return selection


def capture_context(
    root: Path,
    output_dir: Path,
    image_ref: str,
    role: str,
    event: dict,
    legs: Path,
    measurement_manifest: dict | None = None,
) -> dict:
    """Prepare verified source, then record optional comparison metadata."""
    commit, parents = _checkout_source(root)
    if measurement_manifest is not None and (
        not isinstance(measurement_manifest, dict) or measurement_manifest.get("commit") != os.environ.get("GITHUB_SHA")
    ):
        raise ValueError("Measurement source does not identify the tested PR merge GITHUB_SHA.")
    pr = event.get("pull_request", {})
    if pr and role == "current":
        reference_commit = resolve_base(commit, parents, event)
    else:
        reference_commit = _resolved_base() if pr else None
    expected_commit = reference_commit if pr and role == "baseline" else os.environ.get("GITHUB_SHA")
    manifest = prepare_manifest(root, measurement_manifest, expected_commit=expected_commit)
    _write(output_dir / "source-manifest.json", manifest)
    if not pr:
        return {}
    context = {
        "schema_version": 1,
        "source": {
            "commit": commit,
            "requested_head_commit": pr.get("head", {}).get("sha") or os.environ["GITHUB_SHA"],
            "event": os.environ["GITHUB_EVENT_NAME"],
            "reference_branch": pr.get("base", {}).get("ref") or os.environ["GITHUB_REF_NAME"],
            "reference_commit": reference_commit,
            "commit_parents": parents,
            "event_base_commit": pr.get("base", {}).get("sha"),
            "pull_request_number": pr.get("number"),
            "benchmark_role": role,
            "image_ref": image_ref,
            "image_digest": None,
            "image_id": None,
        },
        "execution": {
            "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "job": os.environ["GITHUB_JOB"],
            "runner_name": os.environ.get("RUNNER_NAME"),
            "hostname": socket.gethostname(),
            "measurement_not_before": datetime.now(timezone.utc).isoformat(),
            "expected_samples": 3,
            "expected_legs": [line.split("|", 1)[0] for line in legs.read_text().splitlines() if line.strip()],
            "benchmark_protocol": _benchmark_protocol(legs, measurement_manifest),
        },
        "metric_definition": {"total_fps": None},
    }
    if measurement_manifest is not None:
        context["measurement"] = {key: measurement_manifest[key] for key in ("commit", "digest")}
        context["metric_definition"]["total_fps"] = f"measurement-v1:{measurement_manifest['digest']}"
    try:
        inspected = subprocess.run(["docker", "image", "inspect", image_ref], capture_output=True, text=True)
        image = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else {}
        context["source"].update(image_digest=next(iter(image.get("RepoDigests", [])), None), image_id=image.get("Id"))
    except (OSError, ValueError, TypeError, AttributeError, IndexError, KeyError, subprocess.SubprocessError) as exc:
        reason = f"Optional build provenance capture failed: {exc}"
        context["metric_definition"]["reason"] = reason
        print(f"::warning::{reason}")
    try:
        _write(output_dir / "build-context.json", context)
    except OSError as exc:
        print(f"::warning::Optional build provenance could not be saved: {exc}")
    return context


def bind_baseline(
    selection_path: Path,
    artifact_id: str | None,
    output_dir: Path,
    repository: str,
    baseline_issue: str = "",
    *,
    reuse_confirmed: bool | None = None,
) -> None:
    """Pin the uploaded baseline before the current PR is measured."""
    selection = json.loads(selection_path.read_text())
    if selection.get("reference_commit") != _resolved_base():
        raise ValueError("Baseline selection does not match the tested PR merge's resolved first parent.")
    if artifact_id or reuse_confirmed is False:
        # A fresh baseline clears any earlier reuse decision.
        selection["baseline_reused"] = False
        selection["baseline_origin"] = None
    if not selection["baseline_reused"]:
        if artifact_id:
            selection["baseline_origin"] = {
                "repository": repository,
                "artifact_id": int(artifact_id),
                "artifact_name": baseline_name(
                    selection["pull_request_number"],
                    selection["reference_commit"],
                    int(os.environ["GITHUB_RUN_ATTEMPT"]),
                ),
                "run_id": int(os.environ["GITHUB_RUN_ID"]),
                "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                "source_commit": selection["reference_commit"],
            }
            selection["baseline_finished_before"] = datetime.now(timezone.utc).isoformat()
            selection["reason"] = baseline_issue or (
                "Measured the PR's exact base commit before its tested revision on the same runner."
            )
        else:
            selection["reason"] = baseline_issue or (
                "The base benchmark artifact was not produced; baseline FPS is unavailable."
            )
        if baseline_issue:
            selection.setdefault("issues", []).append(baseline_issue)
    selection["tested_commit"] = os.environ["GITHUB_SHA"]
    _write(output_dir / "pr-comparison.json", selection)


def select_pr_baseline(client: GitHubClient, candidate: Evidence) -> tuple[Evidence | None, dict]:
    """Read the exact base measurement pinned by this PR benchmark job, without ancestry search."""
    source = candidate.context.get("source", {})
    for leg in candidate.context.get("execution", {}).get("expected_legs", []):
        candidate.samples.setdefault(leg, [])
    candidate.issues.extend(source_issues(candidate))
    try:
        selection = artifact_json(candidate.files, "pr-comparison.json")
    except EvidenceError as exc:
        return None, {"reason": str(exc), "reason_code": exc.code}
    base_commit, pr_number = source.get("reference_commit"), source.get("pull_request_number")
    if (
        not is_commit_sha(base_commit)
        or source.get("commit_parents") != [base_commit, source.get("requested_head_commit")]
        or selection.get("reference_commit") != base_commit
        or selection.get("pull_request_number") != pr_number
        or selection.get("tested_commit") != candidate.identity["source_commit"]
        or selection.get("requested_head_commit") != source.get("requested_head_commit")
    ):
        raise EvidenceError("identity_mismatch", "The pinned comparison does not identify this tested PR revision.")
    origin = selection.get("baseline_origin")
    if not origin:
        return None, selection
    try:
        artifact = client.get(f"/repos/{client.repository}/actions/artifacts/{origin['artifact_id']}")
        baseline = _read_baseline(client, artifact, pr_number, base_commit)
        for field in ("artifact_id", "run_id", "run_attempt", "source_commit", "sha256"):
            if field in origin and origin[field] != baseline.identity[field]:
                raise EvidenceError("identity_mismatch", "Pinned baseline identity or bytes changed.")
        if not selection.get("baseline_reused"):
            left, right = baseline.context["execution"], candidate.context["execution"]
            for field in ("run_id", "run_attempt", "job", "runner_name", "hostname"):
                if not left.get(field) or left.get(field) != right.get(field):
                    raise EvidenceError("runner_mismatch", "Fresh base and PR were not measured on the same runner.")
        selection["baseline_origin"] = baseline.identity
        return baseline, selection
    except EvidenceError as exc:
        selection.update(
            reason=str(exc), reason_code=exc.code, unavailable_evidence=origin, unavailable_side="baseline"
        )
        return None, selection


def _execute(args: argparse.Namespace, event: dict) -> None:
    measurement = (
        json.loads(args.measurement_manifest.read_text())
        if getattr(args, "measurement_manifest", None) is not None
        else None
    )
    if args.command == "resolve":
        commit, parents = _checkout_source(args.checkout_root)
        base_commit = resolve_base(commit, parents, event)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"base_commit={base_commit}\n")
        print(f"Resolved tested PR merge {commit} first parent: {base_commit}")
    elif args.command == "restore":
        selection = restore_baseline(
            GitHubClient(os.environ["GITHUB_REPOSITORY"]),
            args.checkout_root,
            args.selection,
            event,
            int(os.environ["GITHUB_RUN_ID"]),
            int(os.environ["GITHUB_RUN_ATTEMPT"]),
            legs=args.legs,
            measurement_manifest=measurement,
        )
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"reused={str(selection['baseline_reused']).lower()}\n")
        print(selection["reason"])
        for issue in selection["issues"]:
            print(issue)
    elif args.command == "capture":
        capture_context(
            args.checkout_root,
            args.output_dir,
            args.image,
            args.role,
            event,
            args.legs,
            measurement_manifest=measurement,
        )
    else:
        if not args.selection.exists():
            _write(args.selection, _initial_selection(event, _resolved_base()))
            if not args.baseline_issue:
                args.baseline_issue = "Baseline selection metadata was not produced; baseline FPS is unavailable."
        bind_baseline(
            args.selection,
            args.artifact_id,
            args.output_dir,
            os.environ["GITHUB_REPOSITORY"],
            args.baseline_issue,
            reuse_confirmed=os.environ.get("PERF_BASELINE_REUSED", "true") == "true",
        )


def _record_failure(args: argparse.Namespace, event: dict, reason: str) -> None:
    if args.command == "resolve" and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"reason={' '.join(reason.split())}\n")
    if not args.output_dir:
        return
    failure_path = args.output_dir / "paired-failure.json"
    pr = event.get("pull_request") or {}
    role = getattr(args, "role", None)
    tested = os.environ.get("GITHUB_SHA")
    reference = os.environ.get("PERF_BASE_COMMIT") or None
    source = {
        "commit": None,
        "intended_commit": reference if role == "baseline" or args.command == "restore" else tested,
        "tested_commit": tested,
        "reference_commit": reference,
        "event_base_commit": pr.get("base", {}).get("sha"),
        "requested_head_commit": pr.get("head", {}).get("sha"),
    }
    if getattr(args, "checkout_root", None):
        with suppress(OSError, subprocess.SubprocessError):
            source["commit"], source["commit_parents"] = _checkout_source(args.checkout_root)
    if failure_path.exists():
        failure = json.loads(failure_path.read_text())
        original = failure.setdefault("source", {})
        if (
            source["commit"]
            and original.get("intended_commit") == source["intended_commit"]
            and original.get("commit") in (None, source["commit"])
        ):
            original["commit"] = source["commit"]
            original.setdefault("commit_parents", source["commit_parents"])
            _write(failure_path, failure)
        return
    execution = {}
    for field in ("run_id", "run_attempt"):
        value = os.environ.get(f"GITHUB_{field.upper()}", "")
        execution[field] = int(value) if value.isdecimal() else None
    _write(
        failure_path,
        {
            "schema_version": 1,
            "stage": f"capture_{role}" if role else args.command,
            "reason": reason,
            "source": source,
            "execution": execution,
        },
    )


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _formula(evidence: Evidence | None) -> str | None:
    definition = _object(evidence.context.get("metric_definition")) if evidence else {}
    value = definition.get("total_fps")
    if not isinstance(value, str):
        return None
    return value if value.strip().lower() not in ("", "unknown", "aggregate_frames_over_measured_seconds") else None


def _groups(evidence: Evidence | None, aliases: dict[str, set[str]]) -> dict:
    groups = {}
    if evidence is None:
        return groups
    for leg in sorted(evidence.samples.keys() | evidence.statuses.keys()):
        samples = evidence.samples.get(leg, [])
        for sample in samples or [None]:
            key = workload_key(sample["bundle"]) if sample else None
            unknown = key is None
            if unknown:
                possibilities = aliases.get(leg, set())
                key = next(iter(possibilities)) if len(possibilities) == 1 else "unknown:" + leg
            group = groups.setdefault(key, {"legs": set(), "samples": [], "unknown_identity": False})
            group["legs"].add(leg)
            group["unknown_identity"] |= unknown
            if sample:
                group["samples"].append(sample)
    return groups


def _summary(evidence: Evidence | None, group: dict | None) -> tuple[dict, list[str]]:
    items = group["samples"] if group else []
    measured = [(item["path"], value) for item in items if (value := runtime_fps(item["bundle"])) is not None]
    values = [value for _, value in measured]
    expected = _object(evidence.context.get("execution")).get("expected_samples") if evidence else None
    known_expected = isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
    reasons = []
    unknown, partial = not known_expected, False
    if not group or not items:
        reasons.append("No runtime samples identify this workload.")
    elif unknown:
        reasons.append("The expected per-leg sample count is not recorded.")
    if group and evidence:
        if group["unknown_identity"]:
            reasons.append(
                "Workload identity is unavailable for some output; the leg name is only a display association."
            )
            unknown = True
        for leg in sorted(group["legs"]):
            leg_status = evidence.statuses.get(leg)
            leg_items = evidence.samples.get(leg, [])
            if any(issue.startswith(leg + "/") for issue in evidence.issues):
                reasons.append(f"Leg {leg} has unresolved sample evidence issues; see the evidence notes.")
                unknown = True
            if leg_status != "ok":
                reasons.append(f"Leg {leg} status is {leg_status or 'not recorded'}.")
                partial = True
            if known_expected and len(leg_items) != expected:
                reasons.append(f"Leg {leg} has {len(leg_items)} runtime samples; expected {expected}.")
                partial = True
            identities = {workload_key(item["bundle"]) for item in leg_items}
            if len(identities) > 1:
                reasons.append(
                    f"Leg {leg} contains different workload identities; per-workload completeness is unknown."
                )
                unknown = True
        if any(_object(item["bundle"].get("run")).get("status") != "completed" for item in items):
            reasons.append("Some runtime samples did not complete.")
            partial = True
        if len(values) != len(items):
            reasons.append("Some FPS values are missing, non-numeric, non-finite or negative.")
            partial = True
    return {
        "count": len(values),
        "observed_count": len(items),
        "expected_count": expected * len(group["legs"]) if known_expected and group else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "samples": values,
        "sample_paths": [path for path, _ in measured],
        "status": "missing" if not items else "partial" if partial else "unknown" if unknown else "complete",
    }, reasons


def _protocol(bundle: dict) -> dict:
    runtime = _object(bundle.get("runtime"))
    timing = _object(runtime.get("environment_step_timing"))
    return {
        "seed": _object(bundle.get("run")).get("seed"),
        "measured_steps": runtime.get("iterations_completed"),
        "steps_per_iteration": runtime.get("steps_per_iteration"),
        "warmup_steps": timing.get("warmup_steps"),
        "measurement_mode": timing.get("measurement_mode"),
    }


def _flatten(value: Any, prefix: str = "") -> dict:
    if isinstance(value, dict) and value:
        result = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def _observations(group: dict | None, extractor) -> dict:
    observations = defaultdict(dict)
    for item in group["samples"] if group else []:
        for field, value in extractor(item["bundle"]).items():
            observations[field][json.dumps(value, sort_keys=True)] = value
    return {field: list(values.values()) for field, values in observations.items()}


def _display(values: list | None) -> Any:
    if not values:
        return None
    return values[0] if len(values) == 1 else {"sample_values": values}


def _differences(a: dict, b: dict, *, missing: bool = False) -> list[dict]:
    return [
        {"field": field, "baseline": _display(a.get(field)), "candidate": _display(b.get(field))}
        for field in sorted(a.keys() | b.keys())
        if a.get(field) != b.get(field)
        or len(a.get(field, [])) > 1
        or len(b.get(field, [])) > 1
        or (missing and (None in a.get(field, []) or None in b.get(field, [])))
    ]


def _context(evidence: Evidence | None, group: dict | None) -> dict:
    observed = _observations(
        group,
        lambda bundle: _flatten(
            {
                "hardware": bundle.get("hardware"),
                "versions": bundle.get("versions"),
                "config": _object(bundle.get("run")).get("config"),
            }
        ),
    )
    source = _object(evidence.context.get("source")) if evidence else {}
    for field in ("image_ref", "image_digest", "image_id"):
        observed["source." + field] = [source.get(field)]
    return observed


def _hardware(bundle: dict) -> dict:
    """Identify recorded CPU/GPU hardware without treating host metadata as hardware."""
    hardware = _object(bundle.get("hardware"))
    identity = {"cpu_name": None, "cpu_count": None, "gpu_devices": None}
    cpu = hardware.get("cpu_name")
    if isinstance(cpu, str) and cpu.strip() and cpu.strip().lower() != "unknown":
        identity["cpu_name"] = cpu.strip().casefold()
    cores = hardware.get("cpu_count")
    if isinstance(cores, int) and not isinstance(cores, bool) and cores > 0:
        identity["cpu_count"] = cores
    devices = hardware.get("gpu_devices")
    if isinstance(devices, list) and devices:
        models = [normalize_gpu_model(_object(device).get("name")) for device in devices]
        if all(model not in ("unknown", "unknown-gpu") for model in models):
            # Preserve device order because benchmarks use cuda:0.
            identity["gpu_devices"] = models
    return identity


def _label(key: str) -> str:
    if key.startswith("unknown:"):
        return key.removeprefix("unknown:") + " (workload identity unavailable)"
    workload = json.loads(key)
    label = (
        f"{workload['task']} · {workload['physics_backend']} / {workload['render_backend'] or 'none'}"
        f" · {workload['num_envs']} envs"
    )
    if workload["presets"]:
        label += f" · presets: {', '.join(workload['presets'])}"
    if "physics_solvers" in workload:
        label += f" · solvers: {', '.join(workload['physics_solvers'])}"
        label += f" · coupling: {workload['physics_coupling'] or 'none'}"
    return label


def _row(
    key: str,
    a: Evidence | None,
    b: Evidence,
    left: dict | None,
    right: dict | None,
) -> dict:
    before, left_reasons = _summary(a, left)
    after, right_reasons = _summary(b, right)
    reasons = ["Baseline: " + reason for reason in left_reasons] + ["Candidate: " + reason for reason in right_reasons]
    formulas = {"baseline": _formula(a), "candidate": _formula(b)}
    pa, pb = _observations(left, _protocol), _observations(right, _protocol)
    protocol_differences = _differences(pa, pb, missing=True)
    context_differences = _differences(_context(a, left), _context(b, right))
    ha, hb = _observations(left, _hardware), _observations(right, _hardware)
    hardware_differences = _differences(ha, hb, missing=True)
    statuses = [value for value in ("missing", "partial", "unknown") if value in (before["status"], after["status"])]
    if not all(formulas.values()):
        reasons.append("FPS formula identity is unknown for one or both selections.")
        statuses.append("unknown")
    elif formulas["baseline"] != formulas["candidate"]:
        reasons.append("FPS formula identities differ.")
        statuses.append("incompatible")
    if protocol_differences:
        reasons.append("Measurement protocol differs, varies within a selection, or has unrecorded fields.")
        statuses.append("unknown" if any(None in values for values in (*pa.values(), *pb.values())) else "incompatible")
    if hardware_differences:
        unavailable = any(None in values for values in (*ha.values(), *hb.values()))
        reasons.append(
            "CPU/GPU hardware identity is unavailable for one or more samples."
            if unavailable
            else "CPU/GPU hardware differs or varies within a selection."
        )
        statuses.append("unknown" if unavailable else "incompatible")
    status = next(iter(statuses), "compared")
    absolute_change = change_pct = None
    notes = []
    if status == "compared":
        absolute_change = after["median"] - before["median"]
        if before["median"] != 0:
            change_pct = 100 * absolute_change / before["median"]
        else:
            notes.append("Percentage change is undefined because the baseline median is zero.")
    if context_differences:
        notes.append(
            "Execution context differs or varies; this observed build comparison does not isolate a commit effect."
        )
    for side, evidence in (("Baseline", a), ("Candidate", b)):
        if evidence and not _object(evidence.context.get("source")).get("image_digest"):
            notes.append(f"{side} image digest is not recorded.")
    return {
        "workload_key": None if key.startswith("unknown:") else key,
        "label": _label(key),
        "legs": {"baseline": sorted(left["legs"]) if left else [], "candidate": sorted(right["legs"]) if right else []},
        "status": status,
        "reasons": reasons,
        "baseline": before,
        "candidate": after,
        "formula": formulas,
        "absolute_change": absolute_change,
        "change_pct": change_pct,
        "protocol_differences": protocol_differences,
        "context_differences": context_differences,
        "notes": notes,
    }


def compare_evidence(baseline: Evidence | None, candidate: Evidence) -> dict:
    """Return dynamically discovered workload rows and descriptive FPS deltas."""
    aliases = defaultdict(set)
    for evidence in (baseline, candidate):
        for leg, samples in evidence.samples.items() if evidence else []:
            for item in samples:
                key = workload_key(item["bundle"])
                if key is not None:
                    aliases[leg].add(key)
    left, right = _groups(baseline, aliases), _groups(candidate, aliases)
    notes = [
        "FPS deltas describe these samples; they are not statistical significance, causal attribution or a CI verdict."
    ]
    if baseline is None:
        notes.append("No baseline evidence was selected.")
    for side, evidence in (("Baseline", baseline), ("Candidate", candidate)):
        if evidence:
            notes.extend(f"{side}: {issue}" for issue in evidence.issues)
            definition = _object(evidence.context.get("metric_definition"))
            if definition.get("reason"):
                notes.append(f"{side} FPS definition: {definition['reason']}")
    if not left and not right:
        notes.append("No runtime workloads or leg statuses were present in the selected evidence.")
    return {
        "baseline": baseline.identity if baseline else None,
        "candidate": candidate.identity,
        "measurement_sources": {
            side: _object(evidence.context.get("measurement"))
            for side, evidence in (("baseline", baseline), ("candidate", candidate))
            if evidence and evidence.context.get("measurement")
        },
        "rows": [
            _row(key, baseline, candidate, left.get(key), right.get(key)) for key in sorted(left.keys() | right.keys())
        ],
        "notes": notes,
    }


def _paired_failure(candidate: Evidence) -> dict | None:
    """Recover the benchmark job's concrete failure before checking its incomplete pin."""
    files = candidate.files
    if "paired-failure.json" not in files:
        return None
    try:
        failure = artifact_json(files, "paired-failure.json")
    except EvidenceError as exc:
        candidate.issues.append(f"Paired failure diagnostics are unreadable: {exc}")
        return None
    execution = _object(failure.get("execution"))
    if (
        failure.get("schema_version") != 1
        or not isinstance(failure.get("reason"), str)
        or not failure["reason"].strip()
        or not isinstance(failure.get("source"), dict)
        or any(execution.get(key) != candidate.identity[key] for key in ("run_id", "run_attempt"))
    ):
        candidate.issues.append("The paired failure record does not identify this producing attempt.")
        return None
    return failure


def _report(args: argparse.Namespace) -> int:
    candidate = selected = paired_failure = None
    selection = {}
    try:
        client = GitHubClient(args.repository)
        candidate = resolve_candidate(client, args.run_id, args.run_attempt, args.comparison_dir)
        print(f"Using benchmark results from attempt {candidate.identity['run_attempt']}.")
        if not args.gate_only:
            paired_failure = _paired_failure(candidate)
            selected, selection = select_pr_baseline(client, candidate)
    except EvidenceError as exc:
        selection = {
            "reason": str(exc),
            "reason_code": exc.code,
            "unavailable_evidence": exc.identity,
            "unavailable_side": "candidate" if candidate is None else "baseline",
        }

    if args.gate_only:
        if selection.get("reason"):
            print(f"::warning::{selection['reason']}")
        return 0

    if paired_failure:
        selection.update(reason=paired_failure["reason"], reason_code="paired_failure", paired_failure=paired_failure)
        if paired_failure.get("stage") in ("resolve", "capture_current"):
            selection["unavailable_side"] = "candidate"

    payload = (
        compare_evidence(selected, candidate)
        if candidate is not None
        else {"candidate": None, "baseline": None, "rows": [], "notes": ["Candidate evidence is unavailable."]}
    )
    payload.update(
        schema_version=1,
        repository=args.repository,
        run_id=args.run_id,
        report_attempt=args.run_attempt,
        selection=selection,
        candidate_kind="merge" if candidate and candidate.identity.get("tested_commit") else None,
        source_context=_object(candidate.context.get("source")) if candidate else {},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build-comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "build-comparison.md").write_text(render_build_comparison(payload), encoding="utf-8")
    # The rolling-history gate sets the pass/fail result.
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "restore", "capture", "bind"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=name != "resolve")
        if name != "bind":
            command.add_argument("--checkout-root", type=Path, required=True)
        if name in ("restore", "bind"):
            command.add_argument("--selection", type=Path, required=True)
        if name in ("restore", "capture"):
            command.add_argument("--legs", type=Path, required=True)
            command.add_argument(
                "--measurement-manifest", type=Path, default=os.environ.get("PERF_MEASUREMENT_MANIFEST")
            )
        if name == "capture":
            command.add_argument("--image", required=True)
            command.add_argument("--role", choices=("baseline", "current"), required=True)
        if name == "bind":
            command.add_argument("--artifact-id")
            command.add_argument("--baseline-issue", default="")
    report = commands.add_parser("report", help="compare PR measurements and collect rolling-history results")
    report.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    report.add_argument("--run_id", type=int, default=os.environ.get("GITHUB_RUN_ID"))
    report.add_argument("--run_attempt", type=int, default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    report.add_argument("--output_dir", type=Path, default=Path("build-comparison"))
    report.add_argument("--comparison_dir", type=Path, help="Save rolling-history reports from the same artifact")
    report.add_argument(
        "--gate-only", action="store_true", help="Collect rolling-history reports without a PR comparison"
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        if not args.repository or not args.run_id or not args.run_attempt or min(args.run_id, args.run_attempt) < 1:
            report.error("Repository, run ID and report attempt are needed to resolve current benchmark evidence")
        return _report(args)
    event = {}
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        _execute(args, event)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
        _record_failure(args, event, reason)
        print(f"Paired PR {args.command} failed: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import contextlib
import copy
import sys
from unittest.mock import Mock, patch

from . import source_revision
from .report import render_aggregate

REPO = "isaac-sim/IsaacLab"
HEAD, PARENT = "a" * 40, "b" * 40


def stamp(hour, minute=0):
    return f"2026-10-01T{hour:02d}:{minute:02d}:00+00:00"


def bundle(fps=100, task="Task", *, start=stamp(10), end=stamp(10, 5), status="completed", seed=42):
    return {
        "run": {
            "task": task,
            "num_envs": 512,
            "status": status,
            "seed": seed,
            "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": []},
            "start_time_utc": start,
            "end_time_utc": end,
        },
        "runtime": {
            "total_fps": {"mean": fps},
            "iterations_completed": 200,
            "steps_per_iteration": 512,
            "environment_step_timing": {"warmup_steps": 100, "measurement_mode": "host_return"},
        },
        "hardware": {
            "hostname": "worker-a",
            "cpu_name": "Intel(R) Xeon(R) 6975P-C",
            "cpu_count": 16,
            "gpu_devices": [{"name": "NVIDIA RTX PRO 4500 Blackwell Server Edition"}],
        },
        "versions": {"warp": "1.0"},
    }


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, value in files.items():
            archive.writestr(path, value if isinstance(value, (str, bytes)) else json.dumps(value))
    return output.getvalue()


class FixtureClient(GitHubClient):
    repository = REPO

    def __init__(self):
        self.attempts = {}
        self.run_artifacts = {}
        self.contents = {}
        self.run_jobs = {}
        self.downloaded = []
        self.pages_requested = []
        self.attempts_requested = []

    def add(
        self,
        run_id,
        commit,
        *,
        attempt=1,
        start=stamp(8),
        event="push",
        branch="develop",
        samples=None,
        artifact=True,
        statuses=None,
        conclusion="success",
    ):
        run = {
            "id": run_id,
            "run_attempt": attempt,
            "head_sha": commit,
            "head_branch": branch,
            "event": event,
            "workflow_id": 7,
            "run_started_at": start,
            "created_at": start,
            "status": "completed",
            "conclusion": conclusion,
        }
        self.attempts[run_id, attempt] = run
        self.run_jobs[run_id, attempt] = [
            {
                "name": "performance-smoke-benchmarks",
                "started_at": start,
                "run_attempt": attempt,
                "conclusion": conclusion,
            }
        ]
        self.run_artifacts.setdefault(run_id, [])
        if artifact:
            files = {
                f"{leg}/sample-{index}/benchmark_runtime_fixture.json": value
                for leg, values in (samples or {"leg": [bundle()]}).items()
                for index, value in enumerate(values, 1)
            }
            files.update({f"{leg}/status": value for leg, value in (statuses or {"leg": "ok"}).items()})
            files["build-context.json"] = {
                "source": {
                    "commit": commit,
                },
                "execution": {
                    "run_id": run_id,
                    "run_attempt": attempt,
                    "expected_samples": 3,
                    "measurement_not_before": start,
                },
                "metric_definition": {"total_fps": "source-fps-v2:" + "a" * 64},
            }
            self.add_artifact(run_id, f"performance-smoke-{run_id}-{attempt}", files)
        return run

    def add_artifact(self, run_id, name, files):
        data = zipped(files)
        artifact_id = len(self.contents) + 1
        item = {
            "id": artifact_id,
            "name": name,
            "workflow_run": {"id": run_id},
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "expired": False,
        }
        self.contents[artifact_id] = data
        self.run_artifacts.setdefault(run_id, []).append(item)
        return item

    def run_attempt(self, run_id, attempt):
        self.attempts_requested.append((run_id, attempt))
        return self.attempts[run_id, attempt]

    def jobs(self, run_id, attempt):
        return self.run_jobs[run_id, attempt]

    def artifacts(self, run_id):
        return self.run_artifacts[run_id]

    def download(self, artifact):
        self.downloaded.append(artifact["id"])
        with patch.object(self, "_request", return_value=self.contents[artifact["id"]]):
            return super().download(artifact)

    def paginate(self, path, key, **params):
        self.pages_requested.append((path, key, params))
        if key == "workflow_runs":
            latest = {}
            for (run_id, attempt), run in self.attempts.items():
                if attempt >= latest.get(run_id, {}).get("run_attempt", 0):
                    latest[run_id] = run
            return list(latest.values())
        raise AssertionError(f"Unexpected fixture API request: {path}, {key}, {params}")

    def get(self, path):
        artifact_id = int(path.rsplit("/", 1)[1])
        matches = [item for items in self.run_artifacts.values() for item in items if item["id"] == artifact_id]
        if not matches:
            raise EvidenceError("inaccessible", "Fixture artifact is unavailable")
        return matches[0]


def check_evidence(subtests):
    import pytest

    with subtests.test(evidence="rerun with an incomplete leg"):
        client = FixtureClient()
        client.add(10, PARENT)
        client.add(
            10,
            PARENT,
            attempt=2,
            start=stamp(11),
            samples={"renamed-leg": [bundle(start=stamp(11, 1), end=stamp(11, 5))]},
            statuses={"renamed-leg": "ok", "missing-leg": "failed"},
            conclusion="failure",
        )
        evidence = resolve_candidate(client, 10, 2)
        assert evidence.identity["run_attempt"] == 2
        assert evidence.samples["missing-leg"] == []
        assert evidence.statuses["missing-leg"] == "failed"

    with subtests.test(evidence="valid, corrupt, unusable and failed samples"):
        malformed = bundle()
        malformed["runtime"]["total_fps"] = []
        client = FixtureClient()
        client.add(10, PARENT, artifact=False)
        client.add_artifact(
            10,
            "performance-smoke-10-1",
            {
                "leg/sample-1/benchmark_runtime_valid.json": bundle(),
                "leg/status": "ok",
                "broken/sample-1/benchmark_runtime_broken.json": "{not json",
                "broken/status": "failed",
                "unusable/sample-1/benchmark_runtime_malformed.json": malformed,
                "unusable/sample-2/benchmark_runtime_overflow.json": bundle(fps=10**1000),
                "unusable/status": "ok",
                "failed/sample-1/benchmark_runtime_failed.json": bundle(status="failed"),
                "failed/status": "failed",
            },
        )
        evidence = resolve_candidate(client, 10, 1)
        assert evidence.identity["run_id"] == 10
        assert evidence.context == {}
        assert evidence.identity["tested_commit"] is None
        assert any("provenance" in issue for issue in evidence.issues)
        assert len(evidence.samples["leg"]) == 1
        assert evidence.samples["broken"] == []
        assert "broken/sample-1/benchmark_runtime_broken.json" in str(evidence.issues)
        assert len(evidence.samples["unusable"]) == 2
        assert sum("FPS value" in issue for issue in evidence.issues) == 2
        assert evidence.statuses["failed"] == "failed"
        assert evidence.samples["failed"][0]["bundle"]["run"]["status"] == "failed"

    for mutation, code, message in (
        ("missing_job_time", "ambiguous_attempt", None),
        ("digest", "corrupt", "digest does not match"),
        ("zip", "corrupt", "ZIP could not be read"),
        ("expired", "expired", None),
    ):
        with subtests.test(evidence=mutation):
            client = FixtureClient()
            client.add(20, HEAD)
            artifact = client.run_artifacts[20][0]
            attempt = 1
            if mutation == "missing_job_time":
                attempt = 2
                client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
                client.run_jobs[20, 2][0]["started_at"] = None
            elif mutation == "digest":
                artifact["digest"] = "sha256:" + "0" * 64
            elif mutation == "zip":
                artifact.pop("digest")
                client.contents[artifact["id"]] = b"not a zip"
            else:
                artifact["expired"] = True
            with pytest.raises(EvidenceError, match=message) as raised:
                resolve_candidate(client, 20, attempt)
            assert raised.value.code == code
            if mutation == "expired":
                identity = raised.value.identity
                assert (identity["repository"], identity["run_id"], identity["run_attempt"]) == (REPO, 20, 1)
                assert identity["artifact_id"] == artifact["id"]
                assert identity["source_commit"] == identity["github_head_commit"] == HEAD
                assert identity["source_provenance"] == "github_run_metadata"
                assert identity["tested_commit"] is None

    sample = bundle()
    sample["run"]["config"].pop("rendering_backend")
    assert workload_key(sample) is None
    for renderer in (None, 42, "", " "):
        with subtests.test(renderer=renderer):
            sample["run"]["config"]["rendering_backend"] = renderer
            assert workload_key(sample) is None


def check_transport(subtests):
    import pytest

    with subtests.test(transport="pagination"):
        client = GitHubClient(REPO)
        pages = [{"id": i} for i in range(100)], [{"id": 100}]
        client.get = Mock(side_effect=[{"total_count": 101, "artifacts": page} for page in pages])
        assert len(client.artifacts(10)) == 101
        assert [
            urllib.parse.parse_qs(urllib.parse.urlsplit(call.args[0]).query)["page"]
            for call in client.get.call_args_list
        ] == [["1"], ["2"]]
        client.get = Mock(return_value={"total_count": 1001, "artifacts": []})
        with pytest.raises(EvidenceError) as raised:
            client.artifacts(10)
        assert raised.value.code == "incomplete_history"

    secret_url = "https://storage.example.invalid/artifact?sig=secret"
    for kind, open_error, read_error, code in (
        ("download", None, None, None),
        ("download", None, TimeoutError("read stalled: ?sig=secret"), "timeout"),
        ("download", None, "truncated", "inaccessible"),
        ("api_connect", urllib.error.URLError(TimeoutError("connect stalled: ?sig=secret")), None, "timeout"),
        ("download", urllib.error.HTTPError(secret_url, 403, "secret", {}, None), None, "inaccessible"),
    ):
        with subtests.test(transport=kind, open_error=open_error, read_error=read_error):
            with patch.dict(os.environ, GH_TOKEN="memory-only" if open_error is None else "", GITHUB_TOKEN=""):
                client = GitHubClient(REPO)
            response = io.BytesIO(b"original zip")
            if read_error == "truncated":
                body = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
                response = http.client.HTTPResponse(Mock(makefile=Mock(return_value=body)))
                response.begin()
                read_error = response.read
            client._opener = Mock()
            redirect = kind == "download" and open_error is None
            client._opener.open.return_value = response
            client._opener.open.side_effect = (
                urllib.error.HTTPError(
                    "https://api.github.com/artifact", 302, "redirect", {"Location": secret_url}, None
                )
                if redirect
                else open_error
            )
            with (
                patch("urllib.request.urlopen", return_value=response) as opened,
                patch.object(response, "read", side_effect=read_error, return_value=b"original zip"),
            ):
                request = (
                    (lambda: client.download({"id": 7})) if kind == "download" else (lambda: client.run_attempt(10, 1))
                )
                if code is None:
                    assert request() == b"original zip"
                else:
                    with pytest.raises(EvidenceError) as raised:
                        request()
                    assert raised.value.code == code
                    assert "secret" not in str(raised.value)
                    if isinstance(open_error, urllib.error.HTTPError):
                        assert "403" in str(raised.value)
            if read_error:
                assert response.closed
                assert client._opener.open.call_args.kwargs["timeout"] == 60
            if redirect:
                assert opened.call_args.kwargs["timeout"] == 60
                assert opened.call_args.args[0].get_header("Authorization") is None
                assert client._opener.open.call_args.args[0].get_header("Authorization") == "Bearer memory-only"


def evidence(samples=None, *, fps=(90, 100, 110), expected=3, formula="measurement-v1:" + "a" * 64, statuses=None):
    if samples is None:
        samples = {"leg": [bundle(value) for value in (fps if isinstance(fps, tuple) else (fps,) * 3)]}
    return Evidence(
        identity={"run_id": 1, "run_attempt": 1, "source_commit": "a" * 40},
        context={
            "source": {"image_digest": "sha256:fixture"},
            "execution": {"expected_samples": expected},
            "metric_definition": {"total_fps": formula},
        },
        samples={
            leg: [
                {"path": f"{leg}/sample-{index}/benchmark_runtime.json", "bundle": value}
                for index, value in enumerate(values, 1)
            ]
            for leg, values in samples.items()
        },
        statuses={leg: "ok" for leg in samples} if statuses is None else statuses,
        issues=[],
        files={},
    )


def set_sample_field(selection, path, value, *, index=None, remove=False):
    """Change or remove one field in all repeats or one selected repeat."""
    samples = selection.samples["leg"]
    for sample in samples if index is None else [samples[index]]:
        target = sample["bundle"]
        *parents, key = path.split(".")
        for parent in parents:
            target = target[parent]
        if remove:
            target.pop(key)
        else:
            target[key] = copy.deepcopy(value)


def assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected


def run_report(client, output, *, attempt=1, comparison_dir=None, gate_only=False):
    arguments = [
        "--repository",
        client.repository,
        "--run_id",
        "20",
        "--run_attempt",
        str(attempt),
        "--output_dir",
        str(output),
    ]
    if comparison_dir is not None:
        arguments += ["--comparison_dir", str(comparison_dir)]
    if gate_only:
        arguments += ["--gate-only"]
    with patch.object(sys.modules[__name__], "GitHubClient", return_value=client):
        status = main(["report", *arguments])
    if gate_only:
        assert not output.exists()
        return (status, None, None)
    return (
        status,
        json.loads((output / "build-comparison.json").read_text()),
        (output / "build-comparison.md").read_text(),
    )


def check_comparisons(subtests):
    with subtests.test(msg="complete fps samples use medians and conventional delta"):
        a = evidence({"leg": [bundle(90), bundle(100), bundle(1100)]})
        b = evidence({"leg": [bundle(93), bundle(96), bundle(999)]})
        report = compare_evidence(a, b)
        row = report["rows"][0]
        assert json.loads(row["workload_key"]) == {
            "task": "Task",
            "physics_backend": "newton_mjwarp",
            "render_backend": None,
            "presets": [],
            "num_envs": 512,
        }
        assert_fields(row["baseline"], samples=[90, 100, 1100], median=100, min=90, max=1100, expected_count=3)
        assert row["candidate"]["median"] == 96
        assert_fields(row, absolute_change=-4, change_pct=-4, status="compared")
        assert "verdict" not in row
        assert any("not statistical significance" in note for note in report["notes"])
    with subtests.test(msg="dynamic workloads include every shared and missing identity"):
        a = evidence({"unfamiliar-leg": [bundle(100, "Shared")] * 3, "removed": [bundle(50, "Removed")] * 3})
        b = evidence({"renamed-leg": [bundle(110, "Shared")] * 3, "new": [bundle(60, "New")] * 3})
        rows = compare_evidence(a, b)["rows"]
        assert len(rows) == 3
        compared = [row for row in rows if row["status"] == "compared"]
        assert len(compared) == 1
        assert compared[0]["change_pct"] == 10
        assert sum(row["status"] == "missing" for row in rows) == 2
    with subtests.test(msg="status only leg is associated without claiming identity"):
        for status in ("ok", "failed"):
            with subtests.test(status=status):
                b = evidence({"leg": []}, statuses={"leg": status})
                rows = compare_evidence(evidence(), b)["rows"]
                assert len(rows) == 1
                assert_fields(rows[0], status="missing", change_pct=None)
                assert rows[0]["candidate"]["median"] is None
                assert any("display association" in reason for reason in rows[0]["reasons"])
        unknown = compare_evidence(b, b)["rows"][0]
        assert unknown["workload_key"] is None
        assert "identity unavailable" in unknown["label"]
    with subtests.test(msg="incomplete or unknown sample counts and failed runs preserve values"):
        variants = [
            ("partial", evidence({"leg": [bundle(100), bundle(110)]})),
            ("partial", evidence(statuses={"leg": "failed"})),
            ("partial", evidence(statuses={})),
        ]
        runtime_failed = evidence()
        runtime_failed.samples["leg"][1]["bundle"]["run"]["status"] = "failed"
        variants.append(("partial", runtime_failed))
        variants.extend(("unknown", evidence(expected=value)) for value in (None, True, "3", 0))
        for status, b in variants:
            with subtests.test(statuses=b.statuses, execution=b.context["execution"], count=len(b.samples["leg"])):
                report = compare_evidence(evidence(), b)
                row = report["rows"][0]
                assert_fields(row, status=status, absolute_change=None, change_pct=None)
                assert row["candidate"]["samples"]
                if status == "unknown":
                    assert_fields(row["candidate"], count=3, expected_count=None)
                elif len(b.samples["leg"]) != 3:
                    assert "runtime samples; expected 3." in render_build_comparison(report)
    with subtests.test(msg="invalid fps is not zero or a valid repeat"):
        for value in (None, True, "100", float("nan"), float("inf"), -1):
            with subtests.test(value=value):
                b = evidence({"leg": [bundle(90), bundle(value), bundle(110)]})
                row = compare_evidence(evidence(), b)["rows"][0]
                assert_fields(row["candidate"], samples=[90, 110], observed_count=3, count=2)
                assert_fields(row, status="partial", change_pct=None)
    with subtests.test(msg="unknown different and legacy fps formulas are not equivalent"):
        known = "measurement-v1:" + "a" * 64
        legacy = "aggregate_frames_over_measured_seconds"
        for case, baseline_formula, candidate_formula, candidate_fps, expected in (
            ("missing", known, None, 100, "unknown"),
            ("unknown", known, "unknown", 100, "unknown"),
            ("padded_legacy", known, " AGGREGATE_FRAMES_OVER_MEASURED_SECONDS ", 100, "unknown"),
            ("different", known, "mean_instantaneous_fps", 100, "incompatible"),
            ("legacy_baseline", legacy, known, 110, "unknown"),
            ("legacy_candidate", known, legacy, 110, "unknown"),
            ("legacy_both", legacy, legacy, 110, "unknown"),
        ):
            with subtests.test(case=case):
                a = evidence(formula=baseline_formula)
                b = evidence(fps=(90, 100, 110) if candidate_fps == 100 else candidate_fps, formula=candidate_formula)
                report = compare_evidence(a, b)
                row = report["rows"][0]
                assert row["status"] == expected
                assert (row["baseline"]["median"], row["candidate"]["median"]) == (100, candidate_fps)
                if case == "different":
                    assert "FPS formula identities differ." in render_build_comparison(report)
                assert_fields(row, absolute_change=None, change_pct=None)
                if case.startswith("legacy_"):
                    assert "FPS formula identity is unknown for one or both selections." in row["reasons"]
    with subtests.test(msg="protocol changes and missing values are explicit"):
        for field, path, value in (
            ("seed", "run.seed", 7),
            ("warmup_steps", "runtime.environment_step_timing.warmup_steps", 20),
            ("measured_steps", "runtime.iterations_completed", 100),
            ("steps_per_iteration", "runtime.steps_per_iteration", 256),
            ("measurement_mode", "runtime.environment_step_timing.measurement_mode", "synchronized"),
            ("measurement_mode", "runtime.environment_step_timing.measurement_mode", None),
        ):
            with subtests.test(field=field, value=value):
                b = evidence()
                set_sample_field(b, path, value)
                row = compare_evidence(evidence(), b)["rows"][0]
                assert row["status"] == ("unknown" if value is None else "incompatible")
                assert field in [item["field"] for item in row["protocol_differences"]]
                assert row["change_pct"] is None
    with subtests.test(msg="identical protocol variation on both sides is not silently pooled"):
        a = evidence({"leg": [bundle(90, seed=1), bundle(100, seed=2), bundle(110, seed=3)]})
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        assert row["status"] == "incompatible"
        assert row["protocol_differences"][0]["baseline"] == {"sample_values": [1, 2, 3]}
        assert row["absolute_change"] is None
    with subtests.test(msg="hardware packages and image qualify without causal verdict"):
        for image, device_metadata in (
            ("sha256:other", {}),
            ("sha256:fixture", {"uuid": "different-device", "index": 7, "driver_version": "other-driver"}),
        ):
            with subtests.test(image=image, device_metadata=device_metadata):
                b = evidence(fps=110)
                b.context["source"]["image_digest"] = image
                for sample in b.samples["leg"]:
                    hardware = sample["bundle"]["hardware"]
                    hardware["hostname"] = "worker-b"
                    hardware["gpu_devices"][0].update(device_metadata)
                    sample["bundle"]["versions"]["warp"] = "2.0"
                report = compare_evidence(evidence(), b)
                row = report["rows"][0]
                assert_fields(row, status="compared", change_pct=10)
                fields = {item["field"] for item in row["context_differences"]}
                expected = {"hardware.hostname", "versions.warp"}
                if image != "sha256:fixture":
                    expected.add("source.image_digest")
                assert expected.issubset(fields)
                assert any("does not isolate a commit effect" in note for note in row["notes"])
                assert "does not isolate a commit effect" in render_build_comparison(report)
    with subtests.test(msg="changed or unavailable hardware preserves fps without comparing them"):
        cases = (
            ("incompatible", "candidate", "cpu_name", "AMD EPYC 9554", None),
            ("incompatible", "candidate", "cpu_count", 32, None),
            ("incompatible", "candidate", "gpu_devices", [{"name": "NVIDIA H100"}], None),
            (
                "incompatible",
                "candidate",
                "gpu_devices",
                [{"name": "NVIDIA RTX PRO 4500 Blackwell Server Edition"}] * 2,
                None,
            ),
            ("unknown", "both", None, None, None),
            ("unknown", "baseline", "cpu_name", None, None),
            ("unknown", "candidate", "cpu_count", None, None),
            ("unknown", "both", "gpu_devices", None, None),
            ("unknown", "candidate", "cpu_name", " unknown ", None),
            ("unknown", "baseline", "cpu_name", 7, None),
            ("unknown", "baseline", "cpu_count", True, None),
            ("unknown", "candidate", "cpu_count", 0, None),
            ("unknown", "candidate", "gpu_devices", [], None),
            ("unknown", "baseline", "gpu_devices", "GPU", None),
            ("unknown", "both", "gpu_devices", [{"name": "unknown"}], None),
            ("unknown", "candidate", "gpu_devices", [None], None),
            ("unknown", "baseline", "cpu_count", None, 1),
            ("unknown", "candidate", "gpu_devices", [], 1),
        )
        for status, side, field, value, index in cases:
            with subtests.test(status=status, side=side, field=field, value=value, index=index):
                a, b = (evidence(), evidence(fps=110))
                for selection in (a, b) if side == "both" else (a if side == "baseline" else b,):
                    set_sample_field(
                        selection, "hardware" + (f".{field}" if field else ""), value, index=index, remove=value is None
                    )
                report = compare_evidence(a, b)
                row = report["rows"][0]
                assert row["status"] == status
                assert (row["baseline"]["median"], row["candidate"]["median"]) == (100, 110)
                assert_fields(row, absolute_change=None, change_pct=None)
                primary, details = render_build_comparison(report).split("<details>", 1)
                assert "| ⚪ Not comparable | leg | 100 | 110 | — |" in primary
                if status == "incompatible":
                    assert "CPU/GPU hardware differs or varies within a selection." in row["reasons"]
                    assert "+10.00%" not in primary
                    assert "CPU/GPU hardware differs" in details
                else:
                    assert any("hardware" in reason for reason in row["reasons"])
    with subtests.test(msg="hardware variation and heterogeneous gpu order are not pooled"):
        a = evidence()
        set_sample_field(a, "hardware.cpu_count", 32, index=1)
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        assert_fields(row, status="incompatible", change_pct=None)
        a, b = (evidence(), evidence())
        for selection, models in ((a, ("A100", "H100")), (b, ("H100", "A100"))):
            set_sample_field(selection, "hardware.gpu_devices", [{"name": model} for model in models])
        row = compare_evidence(a, b)["rows"][0]
        assert_fields(row, status="incompatible", change_pct=None)
    with subtests.test(msg="unavailable hardware preserves partial sample status"):
        b = evidence({"leg": [bundle(110)]})
        set_sample_field(b, "hardware", None, remove=True)
        row = compare_evidence(evidence(), b)["rows"][0]
        assert row["status"] == "partial"
        assert row["candidate"]["median"] == 110
        assert row["change_pct"] is None
    with subtests.test(msg="presets and environment count use validated workload identity"):
        for path, value in (("run.num_envs", 1024), ("run.config.presets", ["new_preset"])):
            b = evidence()
            set_sample_field(b, path, value)
            rows = compare_evidence(evidence(), b)["rows"]
            assert len(rows) == 2
            assert all(row["status"] == "missing" for row in rows)
    with subtests.test(msg="solver and coupling changes or variation preserve fps without comparing workloads"):
        config = {"physics_solvers": ["newton_mjwarp", "newton_vbd"], "physics_coupling": "proxy"}
        for field, value, index in (
            ("physics_solvers", ["newton_mjwarp"], None),
            ("physics_solvers", None, None),
            ("physics_solvers", ["newton_vbd"], None),
            ("physics_coupling", "admm", None),
            ("physics_coupling", None, None),
            ("physics_coupling", "admm", 1),
        ):
            with subtests.test(field=field, value=value, index=index):
                a, b = (evidence(), evidence(fps=110))
                for selection in (a, b):
                    for key, setting in config.items():
                        set_sample_field(selection, f"run.config.{key}", setting)
                set_sample_field(b, f"run.config.{field}", value, index=index, remove=value is None)
                report = compare_evidence(a, b)
                rows = report["rows"]
                assert all(row["change_pct"] is None for row in rows)
                for side in ("baseline", "candidate"):
                    assert sum(row[side]["count"] for row in rows) == 3
                primary, details = render_build_comparison(report).split("<details>", 1)
                assert "⚪ Not comparable" in primary
                assert "+10.00%" not in primary
                assert "solvers: newton_mjwarp, newton_vbd · coupling: proxy" in details
                if index is not None:
                    assert len(rows) == 2
                    shared = next(row for row in rows if row["baseline"]["count"])
                    assert shared["status"] == "unknown"
                    assert any("different workload identities" in reason for reason in shared["reasons"])
    check_rendering(subtests)


def check_rendering(subtests):
    from . import compare

    with subtests.test(msg="headline counts and escaped artifact text preserve tables and sources"):
        a = evidence({name: [bundle(100, name)] * 3 for name in ("Higher", "Lower", "Same", "Tiny")})
        b = evidence(
            {
                name: [bundle(fps, name)] * 3
                for name, fps in (("Higher", 120), ("Lower", 80), ("Same", 100), ("Tiny", 100.0001), ("New", 130))
            }
        )
        report = compare_evidence(a, b)
        report["baseline"].update(repository="isaac-sim/IsaacLab", run_url="https://example.invalid/untrusted")
        row = report["rows"][0]
        unsafe_leg = "leg | </details><script>"
        row["legs"] = {"baseline": [unsafe_leg], "candidate": [unsafe_leg]}
        row["label"] = "Task | [click](https://example.invalid)\n<script>bad()</script>"
        row["notes"].append(unsafe_leg)
        before = copy.deepcopy(report)
        markdown = render_build_comparison(report)
        primary, diagnostics = markdown.split("<details>", 1)
        assert "**🟢 Improved 2 · 🔴 Regressed 1 · ⚪ Not comparable 1 · ⚪ Unchanged 1**" in primary
        assert "Status | Workload | Baseline FPS | PR FPS | Change %" in primary
        assert "PASS" not in markdown
        assert "FAIL" not in markdown
        rows = [line for line in primary.splitlines() if line.startswith("|")]
        assert len(rows) == 7
        assert all(len(line.split("|")) == 7 for line in rows)
        assert "Baseline: No runtime samples identify this workload." in diagnostics
        assert "samples A / B (valid/expected)" in diagnostics
        assert "3/3 / 3/3" in diagnostics
        assert "build-comparison.json" in diagnostics
        assert report == before
        assert "<script>" not in markdown
        assert "[click]" not in markdown
        assert "&#91;click&#93;" in diagnostics
        assert "&#124;" in markdown
        assert "&lt;/details&gt;&lt;script&gt;" in markdown
        assert markdown.count("<details>") == 1
        assert markdown.count("</details>") == 1
        assert "https://example.invalid/untrusted" not in markdown
        assert "https://github.com/isaac-sim/IsaacLab/actions/runs/1/attempts/1" in markdown
    with subtests.test(msg="zero baseline and no rows are not misreported"):
        report = compare_evidence(evidence({"leg": [bundle(0)] * 3}), evidence({"leg": [bundle(10)] * 3}))
        row = report["rows"][0]
        assert_fields(row, absolute_change=10, change_pct=None)
        assert "zero" in row["notes"][0]
        primary = render_build_comparison(report).split("<details>", 1)[0]
        assert "🟢 Improved 1 · 🔴 Regressed 0 · ⚪ Not comparable 0" in primary
        assert "N/A (baseline is zero)" in primary
        empty = render_build_comparison({"rows": [], "selection": {"reason": "Current result artifact is missing."}})
        assert "⚪ Not comparable: no workload results available" in empty
        assert "Current result artifact is missing" in empty.split("<details>", 1)[0]
        assert "Not comparable 0" not in empty
    with subtests.test(msg="selection reason is visible when no baseline was found without an error code"):
        report = compare_evidence(None, evidence())
        report["selection"] = {"reason": "No verified baseline was available."}
        primary = render_build_comparison(report).split("<details>", 1)[0]
        assert "No verified baseline was available" in primary
        assert "| ⚪ Not comparable | leg | — | 100 | — |" in primary
    with subtests.test(msg="duplicate short labels do not collide with existing numbered label"):
        a = evidence({"leg": [bundle(100, "Before")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        b = evidence({"leg": [bundle(100, "Task")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        primary, diagnostics = render_build_comparison(compare_evidence(a, b)).split("<details>", 1)
        labels = [line.split("|")[2].strip() for line in primary.splitlines() if line.startswith("| ⚪")]
        assert len(labels) == 3
        assert len(set(labels)) == 3
        assert "leg (1)" in labels
        assert "leg (2)" in labels
        assert "newton_mjwarp" not in primary
        assert "Before · newton_mjwarp" in diagnostics
        assert "Task · newton_mjwarp" in diagnostics
        for side in ("Baseline", "Candidate"):
            assert f"{side}: No runtime samples identify this workload." in diagnostics
    with subtests.test(msg="paired pr labels exact sources and reused baseline inside details"):
        a, b = (evidence(), evidence(fps=110))
        a.context["measurement"] = {"commit": "d" * 40, "digest": "e" * 64}
        b.context["measurement"] = {"commit": HEAD, "digest": "e" * 64}
        report = compare_evidence(a, b)
        for side, commit, run in (("baseline", PARENT, 12), ("candidate", HEAD, 15)):
            report[side].update(
                repository="isaac-sim/IsaacLab", source_commit=commit, run_id=run, artifact_id=run + 100
            )
        report["candidate"].update(requested_head_commit="c" * 40, event="pull_request")
        report.update(candidate_kind="merge", report_attempt=1)
        report["source_context"] = {"reference_commit": PARENT, "event_base_commit": HEAD}
        report["selection"] = {
            "reason": "Exact PR base and merge result.",
            "baseline_reused": True,
            "baseline_origin": report["baseline"],
        }
        primary, details = render_build_comparison(report).split("<details>", 1)
        assert "| 🟢 Improved | leg | 100 | 110 | +10.00% |" in primary
        assert "A — PR base" in details
        assert "B — PR merge result" in details
        for commit in (HEAD, PARENT, "c" * 40, "d" * 40):
            assert f"https://github.com/isaac-sim/IsaacLab/commit/{commit}" in details
            assert commit not in primary
        for run in (12, 15):
            assert details.count(f"https://github.com/isaac-sim/IsaacLab/actions/runs/{run}/attempts/1") == 1
            assert f"https://github.com/isaac-sim/IsaacLab/actions/runs/{run}/artifacts/{run + 100}" in details
        assert "Baseline measurements: reused." in details
        assert f"Resolved PR base: `{PARENT}`" in details
        assert f"The PR event reported `{HEAD}`" in details
        assert "Measurement sources (commit · SHA-256)" in details
        assert "e" * 64 in details
    with subtests.test(msg="paired baseline measurement status reflects available evidence"):
        for baseline, reused, expected in (
            (None, False, "unavailable."),
            (None, True, "unavailable."),
            (evidence(), False, "run for this PR."),
        ):
            with subtests.test(baseline_available=baseline is not None, reused=reused):
                report = compare_evidence(baseline, evidence())
                report["candidate_kind"] = "merge"
                report["selection"] = {"baseline_reused": reused}
                markdown = render_build_comparison(report)
                assert f"Baseline measurements: {expected}" in markdown
    with subtests.test(msg="rolling gate exposes recorded skip and error causes"):
        missing_credential = "No baseline store credential is available for this run"
        reports = [
            (
                "cartpole-newton",
                compare.Report(
                    verdict=compare.SKIP,
                    message=missing_credential,
                    metrics=(compare.MetricResult("total_fps", "Total FPS", 1801565.067428147),),
                ),
            ),
            (
                "partial",
                compare.Report(
                    verdict=compare.SKIP,
                    message="No gating metric could be compared",
                    metrics=(
                        compare.MetricResult(
                            "total_fps",
                            "Total FPS",
                            100,
                            note="insufficient independent runs for ASV significance testing",
                        ),
                    ),
                ),
            ),
            ("failed", compare.errored("comparison artifact could not be read: invalid JSON")),
        ]
        snapshot = [report.as_dict() for _, report in reports]
        markdown = render_aggregate(reports)
        assert "### Rolling-history CI gate: 🚫 ERROR" in markdown
        assert "<details>" not in markdown
        assert "| Combination | Total FPS | Baseline | FPS regression %" in markdown
        assert f"- **cartpole-newton — SKIP:** {missing_credential}" in markdown
        assert "Total FPS: insufficient independent runs for ASV significance testing" in markdown
        assert "- **failed — ERROR:** comparison artifact could not be read: invalid JSON" in markdown
        assert [report.as_dict() for _, report in reports] == snapshot
        mixed = reports[1:2] + [
            (name, compare.Report(verdict=compare.PASS, message="No performance regression detected"))
            for name in ("g1", "camera-newton", "camera-ovphysx")
        ]
        markdown = render_aggregate(mixed)
        introduction = markdown.split("| Combination |", 1)[0]
        assert "### Rolling-history CI gate: ✅ PASS" in introduction
        assert "Workload checks: 3 passed, 1 skipped." in introduction
        assert "No gating metric could be compared" not in introduction
        assert "No performance regression detected" not in introduction
        assert "- **partial — SKIP:** No gating metric could be compared" in markdown
    with subtests.test(msg="rolling gate does not invent an absent reason"):
        markdown = render_aggregate([("unknown", compare.Report(verdict=compare.SKIP))])
        assert "No reason was recorded in this comparison artifact" in markdown
        assert "credential" not in markdown
        assert "### Rolling-history CI gate: no results" in render_aggregate([])


def report_client():
    client = FixtureClient()
    samples = [bundle(80, start=stamp(12, 1), end=stamp(12, 5)) for _ in range(3)]
    client.add(20, HEAD, start=stamp(12), samples={"leg": samples}, event="pull_request")
    return client


def check_reports(tmp_path, subtests):
    with subtests.test(msg="capture failure explains missing results before pin identity error"):
        client = FixtureClient()
        client.add(20, HEAD, start=stamp(12), event="pull_request", artifact=False)
        reason = "The tested merge checkout has a different first parent than the requested base."
        failure = {
            "schema_version": 1,
            "stage": "capture_current",
            "reason": reason,
            "source": {
                "commit": "d" * 40,
                "intended_commit": "d" * 40,
                "reference_commit": PARENT,
                "event_base_commit": "c" * 40,
                "requested_head_commit": HEAD,
                "commit_parents": [PARENT, HEAD],
            },
            "execution": {"run_id": 20, "run_attempt": 1},
        }
        client.add_artifact(
            20,
            "performance-smoke-20-1",
            {
                "paired-failure.json": failure,
                "pr-comparison.json": {"reference_commit": PARENT, "tested_commit": "d" * 40},
                "leg/status": "error",
            },
        )
        status, result, markdown = run_report(
            client, tmp_path / "test_capture_failure_explains_missing_results_before_pin_identity_error"
        )
        primary, details = markdown.split("<details>", 1)
        assert status == 0
        assert_fields(result["selection"], reason=reason, paired_failure=failure)
        assert "pinned comparison" not in primary
        assert "| Status |" not in primary
        assert "| ⚪ Not comparable |" not in primary
        assert primary.count(reason) == 1
        assert "⚪ Not comparable 1" in primary
        assert len(result["rows"]) == 1
        assert result["rows"][0]["change_pct"] is None
        assert result["candidate_kind"] is None
        assert "PR merge result" not in details
        assert "PR revision (not verified)" in details
        for commit in (HEAD, PARENT, "c" * 40, "d" * 40):
            assert commit in details
        assert "Actual checkout" in details
        assert "Intended checkout" in details
    with subtests.test(msg="baseline failure keeps available current fps and partial workloads"):
        client = report_client()
        candidate = resolve_candidate(client, 20, 1)
        files = candidate.files
        reason = "The base dependency image could not be pulled; baseline samples were not produced."
        files["paired-failure.json"] = json.dumps(
            {
                "schema_version": 1,
                "stage": "capture_baseline",
                "reason": reason,
                "source": {"commit": PARENT, "intended_commit": PARENT},
                "execution": {"run_id": 20, "run_attempt": 1},
            }
        ).encode()
        files["another-workload/status"] = b"error"
        client.run_artifacts[20] = []
        client.add_artifact(20, "performance-smoke-20-1", files)
        status, result, markdown = run_report(
            client, tmp_path / "test_baseline_failure_keeps_available_current_fps_and_partial_workloads"
        )
        primary = markdown.split("<details>", 1)[0]
        assert status == 0
        assert reason in primary
        assert "| Status | Workload | Baseline FPS | PR FPS | Change % |" in primary
        assert len(result["rows"]) == 2
        assert {row["candidate"]["median"] for row in result["rows"]} == {80, None}
        assert all(row["change_pct"] is None for row in result["rows"])
        assert "another-workload" in primary
    with subtests.test(msg="new benchmark attempt without artifact is not old measurements"):
        client = report_client()
        client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
        status, result, markdown = run_report(
            client, tmp_path / "test_new_benchmark_attempt_without_artifact_is_not_old_measurements", attempt=2
        )
        assert status == 0
        assert result["candidate"] is None
        assert result["selection"]["reason_code"] == "missing_candidate"
        assert "Candidate evidence is unavailable" in markdown


RUNTIME = "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py"


class PairedFixture:
    def __init__(self, root, stack):
        self.root = root.resolve()
        self.root.mkdir(parents=True)
        self.checkout = self.root / "base checkout"
        self.checkout.mkdir()
        self._git("init", "--quiet")
        source = self.checkout / RUNTIME
        source.parent.mkdir(parents=True)
        source.write_text('def run(argv):\n    """Selected base runtime."""\n    return argv\n')
        (self.checkout / "source/isaaclab/isaaclab/__init__.py").write_text("")
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", "Base")
        self.commit = self._git("rev-parse", "HEAD")
        stack.enter_context(patch.dict(os.environ, {"PERF_BASE_COMMIT": self.commit}))
        self.manifest = source_revision.prepare_manifest(self.checkout)
        measurement_files = {RUNTIME: hashlib.sha256(b"def run(argv): return 'PR measurement'\n").hexdigest()}
        self.measurement = {
            "schema_version": 1,
            "commit": HEAD,
            "files": measurement_files,
            "digest": source_revision.measurement_digest(measurement_files),
        }
        self.client = FixtureClient()
        self.output = self.root / "baseline output"
        self.selection_path = self.root / "pr-comparison.json"
        self.legs = self.root / "benchmark-legs.tsv"
        self.legs.write_text("first|first|512|600|physics=newton_mjwarp\nsecond|second|512|600|physics=newton_mjwarp\n")
        self.controller = self.root / "controller"
        self.controller.mkdir()
        for name in ("run_benchmarks.sh", "source_revision.py"):
            (self.controller / name).write_bytes(Path(source_revision.__file__).with_name(name).read_bytes())
        self.protocol = {
            "matrix_sha256": hashlib.sha256(self.legs.read_bytes()).hexdigest(),
            "benchmark_launcher_sha256": hashlib.sha256(
                (self.controller / "run_benchmarks.sh").read_bytes()
            ).hexdigest(),
            "source_launcher_sha256": hashlib.sha256((self.controller / "source_revision.py").read_bytes()).hexdigest(),
        }
        stack.enter_context(patch.object(sys.modules[__name__], "__file__", str(self.controller / "pr_comparison.py")))
        self.event = {
            "number": 42,
            "pull_request": {
                "number": 42,
                "base": {"sha": self.commit, "ref": "develop", "repo": {"full_name": REPO}},
                "head": {"sha": HEAD, "ref": "feature", "repo": {"full_name": REPO}},
            },
        }

    def _git(self, *args):
        identity = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
        return subprocess.check_output(["git", "-C", str(self.checkout), *identity, *args], text=True).strip()

    def _commit_file(self, name, content):
        (self.checkout / name).write_text(content)
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", name)
        return self._git("rev-parse", "HEAD")

    def _moving_merge(self):
        self._git("checkout", "--quiet", "-b", "fixture-pr")
        head = self._commit_file("source/isaaclab/isaaclab/pr.py", "VALUE = 'PR'\n")
        self._git("checkout", "--quiet", "-b", "fixture-target", self.commit)
        base = self._commit_file("source/isaaclab/isaaclab/base.py", "VALUE = 'Advanced base'\n")
        self._git("merge", "--quiet", "--no-ff", "-m", "Tested merge", head)
        merge = self._git("rev-parse", "HEAD")
        latest = self._commit_file("source/isaaclab/isaaclab/later.py", "VALUE = 'Later target'\n")
        self._git("checkout", "--quiet", "--detach", merge)
        self.event["pull_request"]["head"]["sha"] = head
        return (base, head, merge, latest)

    @contextlib.contextmanager
    def _workflow(self, **environment):
        event_path = self.root / "event.json"
        event_path.write_bytes(json.dumps(self.event).encode())
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_PATH": str(event_path),
                "GITHUB_OUTPUT": str(self.root / "github-output"),
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_RUN_ID": "20",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_JOB": "performance-smoke-benchmarks",
                "GITHUB_REPOSITORY": REPO,
                **environment,
            },
        ):
            yield

    def _cli(self, args, **environment):
        with self._workflow(**environment):
            return subprocess.run(
                [sys.executable, "-m", "tools.performance_regression.pr_comparison", *args],
                cwd=Path(source_revision.__file__).resolve().parents[2],
                text=True,
                capture_output=True,
            )

    def _capture_args(self, output, role="current", image="same-image", legs=None):
        return [
            "capture",
            "--checkout-root",
            str(self.checkout),
            "--output-dir",
            str(output),
            "--image",
            image,
            "--role",
            role,
            "--legs",
            str(self.legs if legs is None else legs),
        ]

    def _files(self, run_id=10, attempt=1, measurement=None, *, commit=None, **bundle_args):
        commit = self.commit if commit is None else commit
        manifest = dict(self.manifest, commit=commit)
        if measurement is not None:
            manifest["measurement"] = measurement
        context = {
            "source": {
                "commit": commit,
                "requested_head_commit": HEAD,
                "reference_commit": self.commit,
                "reference_branch": "develop",
                "event_base_commit": self.commit,
                "pull_request_number": 42,
                "benchmark_role": "baseline",
            },
            "execution": {
                "run_id": run_id,
                "run_attempt": attempt,
                "expected_samples": 3,
                "expected_legs": ["first", "second"],
                "benchmark_protocol": dict(self.protocol),
                "measurement_not_before": stamp(8),
                "job": "performance-smoke-benchmarks",
                "runner_name": "fixture-gpu-runner",
                "hostname": "fixture-gpu-host",
            },
            "metric_definition": {"total_fps": "source-fps-v2:fixture"},
        }
        if measurement is not None:
            context["measurement"] = {key: measurement[key] for key in ("commit", "digest")}
            context["metric_definition"] = {"total_fps": f"measurement-v1:{measurement['digest']}"}
            context["execution"]["benchmark_protocol"]["measurement_sha256"] = measurement["digest"]
        files = {"source-manifest.json": manifest, "build-context.json": context}
        measured_files = measurement["files"] if measurement is not None else {}
        modules = [
            {
                "name": "isaaclab.benchmark.entrypoints.runtime" if path == RUNTIME else "isaaclab",
                "relative_path": path,
            }
            for path in sorted(set(self.manifest["files"]) | measured_files.keys())
        ]
        for leg in ("first", "second"):
            files[f"{leg}/status"] = b"ok"
            for sample in range(1, 4):
                path = f"{leg}/sample-{sample}/benchmark_runtime_fixture.json"
                value = bundle(task=leg, **({"start": stamp(9, sample), "end": stamp(9, sample + 1)} | bundle_args))
                data = json.dumps(value).encode()
                files[path] = value
                files[f"{leg}/sample-{sample}/source-revision.json"] = {
                    "schema_version": manifest["schema_version"],
                    **({"measurement_digest": measurement["digest"]} if measurement else {}),
                    "status": "verified",
                    "bytecode_policy": "fresh_process_cache",
                    "commit": commit,
                    "modules": modules,
                    "outputs": [
                        {
                            "path": "benchmark_runtime_fixture.json",
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data),
                        }
                    ],
                    "mismatches": [],
                }
        # Keep each artifact document independent.
        return {name: copy.deepcopy(value) for name, value in files.items()}

    def _add_baseline(self, files=None, *, run_id=10, attempt=1):
        self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return self.client.add_artifact(
            run_id,
            baseline_name(42, self.commit, attempt),
            self._files(run_id, attempt) if files is None else files,
        )

    def _restore(self, *, run_id=20, attempt=1, measurement=None):
        if (run_id, attempt) not in self.client.attempts:
            self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return restore_baseline(
            self.client,
            self.checkout,
            self.selection_path,
            self.event,
            run_id,
            attempt,
            legs=self.legs,
            measurement_manifest=measurement,
        )

    def _candidate(
        self,
        origin,
        *,
        reused=False,
        hostname="fixture-gpu-host",
        start=stamp(10),
        fps=100,
        selection=None,
        measurement=None,
    ):
        files = self._files(run_id=20, measurement=measurement, commit=HEAD, fps=fps, start=start, end=stamp(11))
        context = files["build-context.json"]
        context["source"].update(benchmark_role="current", commit_parents=[self.commit, HEAD])
        context["execution"].update(hostname=hostname, measurement_not_before=start)
        files["pr-comparison.json"] = {
            "schema_version": 1,
            "pull_request_number": 42,
            "reference_commit": self.commit,
            "requested_head_commit": HEAD,
            "tested_commit": HEAD,
            "baseline_reused": reused,
            "baseline_origin": origin,
            **(selection or {}),
        }
        run = self.client.add(20, HEAD, event="pull_request", branch="feature", artifact=False)
        artifact = self.client.add_artifact(20, "performance-smoke-20-1", files)
        return read_evidence(self.client, run, artifact, 1)

    def _origin(self, artifact):
        return {
            "repository": REPO,
            "artifact_id": artifact["id"],
            "run_id": artifact["workflow_run"]["id"],
            "run_attempt": 1,
            "source_commit": self.commit,
        }

    def _run_report(self, attempt=1, *, gate_only=False):
        return run_report(
            self.client,
            self.root / ("gate-only" if gate_only else "report"),
            attempt=attempt,
            comparison_dir=self.root / "comparisons",
            gate_only=gate_only,
        )


@contextlib.contextmanager
def paired_fixture(root, subtests, name):
    with subtests.test(msg=name), contextlib.ExitStack() as stack:
        yield PairedFixture(root / name.replace(" ", "-"), stack)


def check_capture(tmp_path, subtests):
    import pytest

    with paired_fixture(tmp_path, subtests, "resolve and capture metadata") as pair:
        base, head, merge, latest = pair._moving_merge()
        assert base != pair.event["pull_request"]["base"]["sha"]
        assert base != latest
        result = pair._cli(["resolve", "--checkout-root", str(pair.checkout)], GITHUB_SHA=merge)
        assert result.returncode == 0, result.stderr + result.stdout
        outputs = dict(line.split("=", 1) for line in (pair.root / "github-output").read_text().splitlines())
        assert outputs == {"base_commit": base}
        original_run = subprocess.run
        original_write = _write
        failure = None

        def inspect_or_run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                if failure in ("image", "image_shape"):
                    return subprocess.CompletedProcess(command, 0, "not JSON" if failure == "image" else "{}", "")
                return subprocess.CompletedProcess(command, 0, '[{"Id":"same-image","RepoDigests":[]}]', "")
            return original_run(command, **kwargs)

        def write_context(path, value):
            if failure == "context_write" and path.name == "build-context.json":
                raise OSError("fixture context write failed")
            original_write(path, value)

        with (
            pair._workflow(PERF_BASE_COMMIT=base, GITHUB_SHA=merge),
            patch.object(subprocess, "run", side_effect=inspect_or_run),
            patch.object(sys.modules[__name__], "_write", side_effect=write_context),
            patch.object(socket, "gethostname", return_value="fixture-gpu-host"),
        ):
            for role, commit in (("baseline", base), ("current", merge)):
                pair._git("checkout", "--quiet", "--detach", commit)
                output = pair.root / role
                context = capture_context(pair.checkout, output, "same-image", role, pair.event, pair.legs)
                source = context["source"]
                assert source["commit"] == commit
                assert source["reference_commit"] == base
                assert source["event_base_commit"] == pair.commit
                assert source["requested_head_commit"] == head
                assert json.loads((output / "source-manifest.json").read_text())["commit"] == commit
                saved = json.loads((output / "build-context.json").read_text())
                assert_fields(saved["execution"], hostname="fixture-gpu-host", benchmark_protocol=pair.protocol)
                if role == "current":
                    assert source["commit_parents"] == [base, head]
            for failure in ("image", "image_shape", "context_write"):
                with subtests.test(optional_failure=failure):
                    output = pair.root / failure
                    result = main(pair._capture_args(output))
                    assert result == 0
                    manifest = json.loads((output / "source-manifest.json").read_text())
                    assert manifest["commit"] == merge
                    assert RUNTIME in manifest["files"]
                    assert not (output / "paired-failure.json").exists()
                    if failure != "context_write":
                        context = json.loads((output / "build-context.json").read_text())
                        assert context["execution"]["benchmark_protocol"] == pair.protocol
                        assert context["metric_definition"]["total_fps"] is None
                        assert "provenance capture failed" in context["metric_definition"]["reason"]
                    else:
                        assert not (output / "build-context.json").exists()
            failure = None
            snapshot = pair.root / "measurement"
            measurement = source_revision.snapshot_measurement(pair.checkout, snapshot)
            with patch.dict(os.environ, {"PERF_MEASUREMENT_MANIFEST": str(snapshot / "measurement-manifest.json")}):
                for role, commit in (("baseline", base), ("current", merge)):
                    pair._git("checkout", "--quiet", "--detach", commit)
                    output = pair.root / role
                    args = pair._capture_args(output, role=role, image="fixture")
                    assert main(args) == 0
                    manifest = json.loads((output / "source-manifest.json").read_text())
                    context = json.loads((output / "build-context.json").read_text())
                    assert_fields(manifest, commit=commit, measurement=measurement)
                    assert context["source"]["commit"] == commit
                    assert context["measurement"] == {"commit": merge, "digest": measurement["digest"]}
                    assert context["metric_definition"]["total_fps"] == f"measurement-v1:{measurement['digest']}"
                    assert context["execution"]["benchmark_protocol"]["measurement_sha256"] == measurement["digest"]
                (snapshot / "measurement-manifest.json").write_bytes(
                    json.dumps({**measurement, "commit": base}).encode()
                )
                assert main(args) == 1
                assert (
                    "Measurement source does not identify"
                    in json.loads((output / "paired-failure.json").read_text())["reason"]
                )
                (output / "paired-failure.json").unlink()
                (snapshot / "measurement-manifest.json").unlink()
                assert main(args) == 1
                assert "measurement-manifest.json" in json.loads((output / "paired-failure.json").read_text())["reason"]
    with paired_fixture(tmp_path, subtests, "non-PR capture") as pair:
        with (
            patch.dict(os.environ, {"GITHUB_SHA": pair.commit}),
            patch.object(subprocess, "run", wraps=subprocess.run) as commands,
        ):
            context = capture_context(pair.checkout, pair.output, "same-image", "current", {}, pair.legs)
            assert context == {}
            assert json.loads((pair.output / "source-manifest.json").read_text()) == pair.manifest
            assert not (pair.output / "build-context.json").exists()
            assert not any(call.args[0][0] == "docker" for call in commands.call_args_list)
            with patch.dict(os.environ, {"GITHUB_SHA": HEAD}), pytest.raises(ValueError, match="expected commit"):
                capture_context(pair.checkout, pair.output, "same-image", "current", {}, pair.legs)
    with paired_fixture(tmp_path, subtests, "invalid source") as pair:
        base, head, merge, _ = pair._moving_merge()
        for mismatch in ("checkout", "head", "nonmerge", "invalid_sha"):
            with subtests.test(mismatch=mismatch):
                commit = base if mismatch == "nonmerge" else merge
                tested = base if mismatch in ("checkout", "nonmerge") else merge
                if mismatch == "invalid_sha":
                    tested = "invalid"
                pair._git("checkout", "--quiet", "--detach", commit)
                pair.event["pull_request"]["head"]["sha"] = HEAD if mismatch == "head" else head
                resolved = pair._cli(["resolve", "--checkout-root", str(pair.checkout)], GITHUB_SHA=tested)
                assert resolved.returncode != 0
                outputs = (pair.root / "github-output").read_text()
                assert "reason=" in outputs
                assert "base_commit=" not in outputs
                output = pair.root / mismatch
                result = pair._cli(
                    pair._capture_args(output, image="unused", legs=pair.root / "unused.tsv"),
                    GITHUB_SHA=tested,
                    PERF_BASE_COMMIT="",
                )
                assert result.returncode != 0
                failure_path = output / "paired-failure.json"
                failure = json.loads(failure_path.read_text())
                assert failure["stage"] == "capture_current"
                expected_reason = "GITHUB_SHA" if mismatch in ("checkout", "invalid_sha") else "second parent"
                assert expected_reason in failure["reason"]
                assert failure["source"]["commit"] == commit
                assert failure["source"]["intended_commit"] == tested
                assert failure["source"]["tested_commit"] == tested
                assert failure["source"]["commit_parents"] == ([pair.commit] if commit == base else [base, head])
                assert failure["source"]["event_base_commit"] == pair.commit
                assert failure["execution"] == {"run_id": 20, "run_attempt": 1}
                assert not (output / "source-manifest.json").exists()
                previous = failure_path.read_bytes()
                pair._cli(
                    ["bind", "--selection", str(pair.selection_path), "--output-dir", str(output)],
                    GITHUB_SHA=tested,
                    PERF_BASE_COMMIT="",
                )
                assert failure_path.read_bytes() == previous
    with paired_fixture(tmp_path, subtests, "stale baseline metadata") as pair:
        base, head, merge, _ = pair._moving_merge()
        reason = "CPU could not resolve the tested PR merge."
        env = {"GITHUB_SHA": merge, "PERF_BASE_COMMIT": "", "PERF_PAIR_ERROR": reason}
        bound = pair._cli(["bind", "--selection", str(pair.selection_path), "--output-dir", str(pair.output)], **env)
        assert bound.returncode != 0
        path = pair.output / "paired-failure.json"
        original = json.loads(path.read_text())
        assert original["stage"] == "bind"
        assert original["reason"] == reason
        assert original["source"]["commit"] is None
        original_run = subprocess.run

        def inspect_or_run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0, '[{"Id":"same-image","RepoDigests":[]}]', "")
            return original_run(command, **kwargs)

        with pair._workflow(**env), patch.object(subprocess, "run", side_effect=inspect_or_run):
            for resolved in ("", pair.commit):
                with subtests.test(resolved=resolved), patch.dict(os.environ, {"PERF_BASE_COMMIT": resolved}):
                    context = capture_context(
                        pair.checkout, pair.output, "same-image", "current", pair.event, pair.legs
                    )
                    assert context["source"]["commit"] == merge
                    assert context["source"]["reference_commit"] == base
                    assert context["source"]["commit_parents"] == [base, head]
                    assert context["source"]["event_base_commit"] == pair.commit
                    manifest = json.loads((pair.output / "source-manifest.json").read_text())
                    assert manifest["commit"] == merge
                    for name in ("pr.py", "base.py"):
                        source_path = f"source/isaaclab/isaaclab/{name}"
                        assert source_path in manifest["files"]
                    assert json.loads(path.read_text()) == original
                    assert not (pair.output / "pr-comparison.json").exists()
                    with pytest.raises(ValueError):
                        capture_context(pair.checkout, pair.output, "same-image", "baseline", pair.event, pair.legs)
            (pair.checkout / RUNTIME).write_text("def run(argv): return 'modified'\n")
            invalid = pair.root / "modified-source"
            with pytest.raises(ValueError, match="Checked-out Python source differs"):
                capture_context(pair.checkout, invalid, "same-image", "current", pair.event, pair.legs)
            assert not (invalid / "source-manifest.json").exists()
            captured = main(pair._capture_args(pair.output))
            assert captured != 0
            expected = copy.deepcopy(original)
            expected["source"].update(commit=merge, commit_parents=[base, head])
            assert json.loads(path.read_text()) == expected


def check_reuse(tmp_path, subtests):
    import pytest

    with paired_fixture(tmp_path, subtests, "changed protocol") as pair:
        pair._add_baseline()
        for path in (pair.legs, pair.controller / "run_benchmarks.sh", pair.controller / "source_revision.py"):
            with subtests.test(protocol_file=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(
                        original.replace(b"physics=newton_mjwarp", b"physics=ovphysx")
                        if path == pair.legs
                        else original + b"\n# Changed controller launcher\n"
                    )
                    selection = pair._restore()
                    assert not selection["baseline_reused"]
                    assert "protocol" in " ".join(selection["issues"])
                finally:
                    path.write_bytes(original)
    with paired_fixture(tmp_path, subtests, "measurement snapshot reuse") as pair:
        pair._add_baseline(run_id=9)
        assert not pair._restore(measurement=pair.measurement)["baseline_reused"]
        artifact = pair._add_baseline(pair._files(measurement=pair.measurement))
        original = pair.client.contents[artifact["id"]]
        measurement = {**pair.measurement, "commit": "d" * 40}
        restored = pair._restore(measurement=measurement)
        assert restored["baseline_reused"]
        assert restored["baseline_origin"]["artifact_id"] == artifact["id"]
        with patch.dict(os.environ, {"GITHUB_SHA": HEAD}):
            bind_baseline(pair.selection_path, None, pair.output, REPO, "Base image config failed.")
        selection = json.loads((pair.output / "pr-comparison.json").read_text())
        assert selection["baseline_origin"] == restored["baseline_origin"]
        assert selection["reason"] == restored["reason"]
        assert selection["issues"] == restored["issues"]
        candidate = pair._candidate(
            restored["baseline_origin"], reused=True, selection=selection, measurement=measurement
        )
        evidence, _ = select_pr_baseline(pair.client, candidate)
        assert evidence.issues == []
        assert evidence.identity["sha256"] == hashlib.sha256(original).hexdigest()
        assert candidate.issues == []
        report = compare_evidence(evidence, candidate)
        assert all(row["status"] == "compared" for row in report["rows"])
        assert all(row["change_pct"] == 0 for row in report["rows"])
        assert pair.client.contents[artifact["id"]] == original
        pair.event["pull_request"]["head"]["sha"] = "d" * 40
        selection = pair._restore(measurement=measurement)
        assert selection["baseline_reused"]
        assert selection["requested_head_commit"] == "d" * 40
        assert_fields(selection["baseline_origin"], artifact_id=artifact["id"], source_commit=pair.commit)
        assert json.loads(pair.selection_path.read_text()) == selection
        changed_files = {**pair.measurement["files"], RUNTIME: "0" * 64}
        changed = {
            **pair.measurement,
            "files": changed_files,
            "digest": source_revision.measurement_digest(changed_files),
        }
        selection = pair._restore(measurement=changed)
        assert not selection["baseline_reused"]
        assert "measurement code" in str(selection["issues"])
    with paired_fixture(tmp_path, subtests, "resolved reference") as pair:
        event_base = "f" * 40
        files = pair._files()
        files["build-context.json"]["source"]["event_base_commit"] = event_base
        artifact = pair._add_baseline(files)
        pair.event["pull_request"]["base"]["sha"] = event_base
        restored = pair._restore()
        assert restored["baseline_reused"]
        assert_fields(restored, reference_commit=pair.commit, event_base_commit=event_base)
        candidate = pair._candidate(pair._origin(artifact), reused=True, hostname="different-gpu-host")
        candidate.context["source"]["event_base_commit"] = event_base
        evidence, metadata = select_pr_baseline(pair.client, candidate)
        assert_fields(evidence.identity, source_commit=pair.commit, run_id=10)
        assert metadata["baseline_reused"]
    with paired_fixture(tmp_path, subtests, "unresolved reference") as pair:
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": ""}):
            with pytest.raises(ValueError, match="first parent was not resolved"):
                restore_baseline(pair.client, pair.checkout, pair.selection_path, pair.event, 20, 1, legs=pair.legs)
        assert not pair.selection_path.exists()
        assert pair.client.pages_requested == []
    with paired_fixture(tmp_path, subtests, "workflow registration") as pair:
        workflow_id = 812345
        pair.client.add(20, HEAD, event="pull_request", branch="feature", artifact=False)["workflow_id"] = workflow_id
        expected = pair._add_baseline(run_id=10)
        pair.client.attempts[10, 1]["workflow_id"] = workflow_id
        obsolete = pair._add_baseline(run_id=11)
        pair.client.attempts[11, 1]["workflow_id"] = 700000
        selected = pair._restore()
        assert selected["baseline_reused"]
        assert selected["baseline_origin"]["artifact_id"] == expected["id"]
        assert pair.client.attempts_requested[0] == (20, 1)
        assert (
            f"/repos/{REPO}/actions/workflows/{workflow_id}/runs",
            "workflow_runs",
            {"event": "pull_request", "branch": "feature"},
        ) in pair.client.pages_requested
        assert not any("build.yaml" in path for path, _, _ in pair.client.pages_requested)
        assert any(f"Artifact {obsolete['id']}" in issue for issue in selected["issues"])
    with paired_fixture(tmp_path, subtests, "changed base") as pair:
        pair._add_baseline()
        source = pair.checkout / RUNTIME
        new_commit = pair._commit_file(RUNTIME, source.read_text().replace("Selected base", "Changed base"))
        pair.event["pull_request"]["base"]["sha"] = new_commit
        os.environ["PERF_BASE_COMMIT"] = new_commit
        selection = pair._restore()
        assert not selection["baseline_reused"]
        assert selection["reference_commit"] == new_commit
        assert pair.client.downloaded == []
    with paired_fixture(tmp_path, subtests, "inaccessible historical run") as pair:
        pair._add_baseline(run_id=10)
        expected = pair._add_baseline(run_id=11)
        pair.client.add(9, HEAD, event="pull_request", branch="feature", artifact=False)
        artifacts = pair.client.artifacts

        def artifacts_or_error(run_id):
            if run_id == 9:
                raise EvidenceError("inaccessible", "GitHub evidence request failed (HTTP 404)")
            return artifacts(run_id)

        with patch.object(pair.client, "artifacts", side_effect=artifacts_or_error):
            selection = pair._restore()
        assert selection["baseline_reused"]
        assert selection["baseline_origin"]["artifact_id"] == expected["id"]
        assert pair.client.downloaded == [expected["id"]]
        assert any("Run 9" in issue and "HTTP 404" in issue for issue in selection["issues"])
        assert json.loads(pair.selection_path.read_text()) == selection
    with paired_fixture(tmp_path, subtests, "different PR") as pair:
        pair._add_baseline()
        pair.event["number"] = pair.event["pull_request"]["number"] = 43
        selection = pair._restore()
        assert not selection["baseline_reused"]
        assert pair.client.downloaded == []
    with paired_fixture(tmp_path, subtests, "interrupted lookup") as pair:
        pair._add_baseline()
        with patch.object(pair.client, "run_attempt", side_effect=TimeoutError("lookup interrupted")):
            with pytest.raises(TimeoutError):
                pair._restore()
        selection = json.loads(pair.selection_path.read_text())
        assert not selection["baseline_reused"]
        assert selection["baseline_origin"] is None
        assert selection["reference_commit"] == pair.commit
    with paired_fixture(tmp_path, subtests, "invalid source proof") as pair:
        for mutation, sample, reason in (
            ("missing", "second/sample-3", "source-revision.json"),
            ("result", "first/sample-1", "Result bytes do not match their source verification"),
            ("module", "first/sample-1", "Loaded source does not match its manifest"),
            ("legacy-format", "first/sample-1", "Executing source was not verified for this commit"),
        ):
            with subtests.test(mutation=mutation):
                pair.client = FixtureClient()
                files = pair._files()
                path = f"{sample}/source-revision.json"
                if mutation == "missing":
                    del files[path]
                elif mutation == "result":
                    files[f"{sample}/benchmark_runtime_fixture.json"]["runtime"]["total_fps"]["mean"] = 900
                elif mutation == "legacy-format":
                    files[path]["schema_version"] = 2
                else:
                    files[path]["modules"][0]["relative_path"] = "source/isaaclab/isaaclab/missing.py"
                artifact = pair._add_baseline(files)
                candidate = pair._candidate(pair._origin(artifact), reused=True)
                evidence, _ = select_pr_baseline(pair.client, candidate)
                assert sample in str(evidence.issues)
                assert reason in str(evidence.issues)
                selection = pair._restore()
                assert not selection["baseline_reused"]
                assert selection["baseline_origin"] is None
                assert "Baseline is incomplete or its source proof is invalid." in str(selection["issues"])
    with paired_fixture(tmp_path, subtests, "mixed measurement proof") as pair:
        for mutation in (
            "failed_proof",
            "digest",
            "context",
            "missing_manifest",
            "missing_measurement_digest",
        ):
            with subtests.test(mutation=mutation):
                pair.client = FixtureClient()
                files = pair._files(measurement=pair.measurement)
                path = "first/sample-1/source-revision.json"
                proof = files[path]
                if mutation == "failed_proof":
                    proof["status"] = "failed"
                elif mutation == "digest":
                    files["source-manifest.json"]["measurement"]["digest"] = "0" * 64
                elif mutation == "missing_manifest":
                    del files["source-manifest.json"]["measurement"]
                elif mutation == "missing_measurement_digest":
                    del proof["measurement_digest"]
                else:
                    files["build-context.json"]["metric_definition"]["total_fps"] = "measurement-v1:" + "0" * 64
                artifact = pair._add_baseline(files)
                candidate = pair._candidate(pair._origin(artifact), reused=True, measurement=pair.measurement)
                evidence, _ = select_pr_baseline(pair.client, candidate)
                assert evidence.issues
                report = compare_evidence(evidence, candidate)
                assert all("Baseline: " + issue in report["notes"] for issue in evidence.issues)
                first = next(row for row in report["rows"] if "first" in row["legs"]["baseline"])
                assert first["change_pct"] is None
                assert not pair._restore(measurement=pair.measurement)["baseline_reused"]
    with paired_fixture(tmp_path, subtests, "unusable baseline") as pair:
        for mutation, reason in (
            ("missing-sample", ""),
            ("failed-leg", ""),
            ("missing-workload", ""),
            ("missing-protocol", "protocol"),
            ("current-role", "does not belong to this PR and exact base commit"),
        ):
            with subtests.test(mutation=mutation):
                pair.client = FixtureClient()
                files = pair._files()
                if mutation == "missing-sample":
                    del files["second/sample-3/benchmark_runtime_fixture.json"]
                elif mutation == "failed-leg":
                    files["second/status"] = b"failed"
                elif mutation == "missing-workload":
                    files = {name: data for name, data in files.items() if not name.startswith("second/")}
                elif mutation == "missing-protocol":
                    del files["build-context.json"]["execution"]["benchmark_protocol"]
                else:
                    files["build-context.json"]["source"]["benchmark_role"] = "current"
                artifact = pair._add_baseline(files)
                selection = pair._restore()
                assert not selection["baseline_reused"]
                assert selection["issues"]
                if reason:
                    assert reason in " ".join(selection["issues"])
                if mutation == "current-role":
                    assert pair.client.downloaded == [artifact["id"]]
    with paired_fixture(tmp_path, subtests, "baseline retry lifecycle") as pair:
        incomplete = pair._files()
        del incomplete["second/sample-3/source-revision.json"]
        first = pair._add_baseline(incomplete)
        first_bytes = pair.client.contents[first["id"]]
        second = pair._add_baseline(run_id=10, attempt=2)
        assert first["name"] != second["name"]
        with subtests.test(stage="valid_attempt_cannot_reuse_itself"):
            assert not pair._restore(run_id=10, attempt=2)["baseline_reused"]
        selection = pair._restore(run_id=10, attempt=3)
        assert selection["baseline_reused"]
        assert_fields(selection["baseline_origin"], artifact_id=second["id"], run_attempt=2)
        assert pair.client.contents[first["id"]] == first_bytes
        second["expired"] = True
        assert not pair._restore(run_id=10, attempt=3)["baseline_reused"]


def check_pair_reports(tmp_path, subtests):
    from . import cli

    with paired_fixture(tmp_path, subtests, "report lifecycle and gate") as pair:
        artifact = pair._add_baseline(pair._files(measurement=pair.measurement))
        candidate = pair._candidate(pair._origin(artifact), reused=True, measurement=pair.measurement, fps=80)
        files = {**candidate.files, "first/comparison.json": json.dumps({"verdict": "FAIL", "label": "first"}).encode()}
        pair.client.run_artifacts[20] = []
        current = pair.client.add_artifact(20, "performance-smoke-20-1", files)
        pair.client.downloaded.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            status, report, markdown = pair._run_report()
            gate_exit = cli.main(["aggregate", "--comparison_dir", str(pair.root / "comparisons")])
        assert (gate_exit, status) == (1, 0)
        assert pair.client.downloaded == [current["id"], artifact["id"]]
        assert report["baseline"]["run_id"] == 10
        assert report["candidate"]["run_id"] == 20
        assert report["candidate_kind"] == "merge"
        assert len(report["rows"]) == 2
        assert all(row["change_pct"] == -20 for row in report["rows"])
        assert "-20.00%" in markdown
        assert "Comparison unavailable" not in markdown
        assert "/actions/runs/10/attempts/1" in markdown
        assert "/actions/runs/20/attempts/1" in markdown
        assert "3/3 / 3/3" in markdown
        assert pair.client.pages_requested == []
        with subtests.test(msg="report retry and expiry"):
            first = report
            gate = files["first/comparison.json"]
            (pair.root / "comparisons/first/comparison.json").unlink()
            pair.client.add(20, HEAD, attempt=2, start=stamp(13), event="pull_request", artifact=False)
            pair.client.run_jobs[20, 2][0].update(started_at=stamp(12), id=999)
            _, second, rerun_markdown = pair._run_report(attempt=2)
            assert (pair.root / "comparisons/first/comparison.json").read_bytes() == gate
            assert second["baseline"] == first["baseline"]
            assert (second["candidate"]["run_attempt"], second["report_attempt"]) == (1, 2)
            assert "Measurements: attempt 1; report: attempt 2" in rerun_markdown
            artifact["expired"] = True
            pair._add_baseline(pair._files(run_id=11, measurement=pair.measurement), run_id=11)
            _, result, markdown = pair._run_report(attempt=2)
            assert result["baseline"] is None
            assert result["selection"]["reason_code"] == "expired"
            assert result["selection"]["unavailable_evidence"] == pair._origin(artifact)
            assert all(row["candidate"]["median"] == 80 for row in result["rows"])
            assert all(row["change_pct"] is None for row in result["rows"])
            assert "PR base (evidence unavailable)" in markdown
            assert "/actions/runs/10/attempts/1" in markdown
            assert "expired" in markdown
            assert pair.client.pages_requested == []
        files["build-context.json"] = b"{invalid JSON"
        pair.client.run_artifacts[20] = []
        current = pair.client.add_artifact(20, "performance-smoke-20-1", files)
        for gate_only in (False, True):
            with subtests.test(gate_only=gate_only), contextlib.redirect_stdout(io.StringIO()):
                (pair.root / "comparisons/first/comparison.json").unlink()
                pair.client.downloaded.clear()
                status, report, _ = pair._run_report(gate_only=gate_only)
                gate_exit = cli.main(["aggregate", "--comparison_dir", str(pair.root / "comparisons")])
                assert (gate_exit, status) == (1, 0)
                assert pair.client.downloaded == [current["id"]]
                if not gate_only:
                    assert report["selection"]["reason_code"] == "corrupt"


def check_binding(tmp_path, subtests):
    import pytest

    with paired_fixture(tmp_path, subtests, "parent mismatch") as pair:
        candidate = pair._candidate({"artifact_id": 123})
        downloaded = list(pair.client.downloaded)
        candidate.context["source"]["commit_parents"] = ["f" * 40, HEAD]
        with pytest.raises(EvidenceError, match="does not identify this tested PR revision"):
            select_pr_baseline(pair.client, candidate)
        assert pair.client.downloaded == downloaded
    with paired_fixture(tmp_path, subtests, "fresh runner and sample clocks") as pair:
        for case, arguments, reason in (
            ("pinned_digest", {}, "identity_mismatch"),
            ("different_runner", {"hostname": "different-gpu-host"}, "runner_mismatch"),
            ("sample_clock_overlap", {"start": stamp(9)}, None),
            (
                "sample_clock_disagrees_with_phase",
                {"start": stamp(9), "selection": {"baseline_finished_before": stamp(8, 59)}},
                None,
            ),
        ):
            with subtests.test(case=case):
                pair.client = FixtureClient()
                artifact = pair._add_baseline(run_id=20)
                origin = pair._origin(artifact)
                if case == "pinned_digest":
                    origin["sha256"] = "0" * 64
                candidate = pair._candidate(origin, **arguments)
                evidence, metadata = select_pr_baseline(pair.client, candidate)
                if reason:
                    assert evidence is None
                    assert metadata["reason_code"] == reason
                else:
                    assert_fields(evidence.identity, artifact_id=artifact["id"], source_commit=pair.commit)
                    assert pair.client.pages_requested == []
    with paired_fixture(tmp_path, subtests, "partial baseline") as pair:
        pair._restore()
        issue = "The baseline benchmark step failed after producing partial evidence."
        files = pair._files(run_id=20)
        files["second/sample-3/benchmark_runtime_fixture.json"] = b"{invalid JSON"
        artifact = pair._add_baseline(files, run_id=20)
        origin = pair._origin(artifact)
        with patch.dict(os.environ, {"GITHUB_SHA": HEAD, "GITHUB_RUN_ID": "20", "GITHUB_RUN_ATTEMPT": "1"}):
            bind_baseline(pair.selection_path, str(artifact["id"]), pair.output, REPO, issue)
        pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert pin["reason"] == issue
        assert issue in pin["issues"]
        assert pin["baseline_origin"]["artifact_id"] == artifact["id"]
        candidate = pair._candidate(origin, selection=pin)
        evidence, _ = select_pr_baseline(pair.client, candidate)
        assert evidence is not None
        report = compare_evidence(evidence, candidate)
        by_leg = {row["legs"]["baseline"][0]: row for row in report["rows"]}
        assert_fields(by_leg["first"], status="compared", change_pct=0)
        assert by_leg["second"]["status"] == "partial"
        assert by_leg["second"]["change_pct"] is None
    with paired_fixture(tmp_path, subtests, "bind wrong base") as pair:
        pair._restore()
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": "f" * 40, "GITHUB_SHA": HEAD}):
            with pytest.raises(ValueError, match="selection does not match"):
                bind_baseline(pair.selection_path, None, pair.output, REPO)
        assert not (pair.output / "pr-comparison.json").exists()
    with paired_fixture(tmp_path, subtests, "unconfirmed handoff") as pair:
        pair._add_baseline(run_id=10)
        pair._restore()
        requested = list(pair.client.pages_requested)
        fresh = pair._add_baseline(run_id=20)
        for uploaded in (True, False):
            with subtests.test(fresh_upload_succeeded=uploaded):
                with pair._workflow(GITHUB_SHA=HEAD, PERF_BASELINE_REUSED=""):
                    arguments = ["bind", "--selection", str(pair.selection_path), "--output-dir", str(pair.output)]
                    if uploaded:
                        arguments.extend(["--artifact-id", str(fresh["id"])])
                    assert main(arguments) == 0
                pin = json.loads((pair.output / "pr-comparison.json").read_text())
                candidate = pair._candidate(pin["baseline_origin"], selection=pin)
                evidence, metadata = select_pr_baseline(pair.client, candidate)
                if uploaded:
                    assert_fields(evidence.identity, artifact_id=fresh["id"], run_id=20)
                else:
                    assert evidence is None
                    assert metadata["baseline_origin"] is None
                    assert (
                        metadata["reason"]
                        == "The base benchmark artifact was not produced; baseline FPS is unavailable."
                    )
                    assert pair.client.pages_requested == requested
                assert not metadata["baseline_reused"]
    with paired_fixture(tmp_path, subtests, "missing selection") as pair:
        tested_merge = "e" * 40
        for issue in ("The PR base checkout failed; no baseline was measured.", None):
            with subtests.test(failure_detail=issue):
                pair.selection_path.unlink(missing_ok=True)
                arguments = ["bind", "--selection", str(pair.selection_path), "--output-dir", str(pair.output)]
                if issue:
                    arguments.extend(["--baseline-issue", issue])
                result = pair._cli(arguments, GITHUB_SHA=tested_merge)
                assert result.returncode == 0, result.stderr + result.stdout
                selection = json.loads((pair.output / "pr-comparison.json").read_text())
                if issue:
                    assert selection["reason"] == issue
                    assert issue in selection["issues"]
                else:
                    assert "Baseline selection metadata was not produced" in selection["reason"]
                assert selection["pull_request_number"] == 42
                assert selection["reference_commit"] == pair.commit
                assert selection["requested_head_commit"] == HEAD
                assert selection["tested_commit"] == tested_merge
                assert not selection["baseline_reused"]
                assert selection["baseline_origin"] is None


def test_pr_comparison(tmp_path, subtests):
    check_evidence(subtests)
    check_transport(subtests)
    check_comparisons(subtests)
    check_reports(tmp_path, subtests)
    check_capture(tmp_path, subtests)
    check_reuse(tmp_path, subtests)
    check_pair_reports(tmp_path, subtests)
    check_binding(tmp_path, subtests)
