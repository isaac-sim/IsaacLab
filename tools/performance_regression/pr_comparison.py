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

from . import cli as gate_cli
from .compare import errored
from .contract import normalize_gpu_model, workload_identity
from .metrics import PerformanceRegressionError, nested_number
from .report import render_build_comparison, write_json
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
    except PerformanceRegressionError:
        return None


def workload_key(bundle: dict) -> str | None:
    """Shared workload identity, independent of metric formula and runtime versions."""
    try:
        return json.dumps(workload_identity(bundle), sort_keys=True, separators=(",", ":"))
    except PerformanceRegressionError:
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
    history_dir: Path | None = None,
) -> Evidence:
    """Download one exact producing attempt; do not discard failed or missing legs."""
    run_id = run["id"]
    name = artifact_name or f"performance-regression-{run_id}-{attempt}"
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
        saved_reports = None if history_dir else comparison_dir
        digest, files = contents if contents is not None else download_artifact(client, artifact, saved_reports)
    except EvidenceError as exc:
        raise EvidenceError(exc.code, str(exc), identity) from None
    if history_dir is not None:
        _refresh_history(files, history_dir)
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
    client: GitHubClient,
    run_id: int,
    report_attempt: int,
    comparison_dir: Path | None = None,
    history_dir: Path | None = None,
) -> Evidence:
    """Read the measured attempt shared by the paired report and rolling-history gate."""
    artifacts = client.artifacts(run_id)
    for attempt in range(report_attempt, 0, -1):
        run = client.run_attempt(run_id, attempt)
        name = f"performance-regression-{run_id}-{attempt}"
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        if len(matches) > 1:
            raise EvidenceError("ambiguous", f"More than one GitHub artifact has the exact name {name}")
        if matches:
            return read_evidence(
                client, run, matches[0], attempt, comparison_dir=comparison_dir, history_dir=history_dir
            )
        started = parse_timestamp(run.get("run_started_at"))
        for job in client.jobs(run_id, attempt):
            if job.get("name") != "performance-regression-benchmarks" or job.get("conclusion") == "skipped":
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


def restore_baseline(args: argparse.Namespace, event: dict, measurement_manifest: dict | None) -> dict:
    """Select a complete verified measurement for this PR/base, or request a fresh one."""
    client = GitHubClient(os.environ["GITHUB_REPOSITORY"])
    run_id = int(os.environ["GITHUB_RUN_ID"])
    run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    pr = event["pull_request"]
    pr_number, base_commit = pr["number"], _resolved_base()
    prepare_manifest(args.checkout_root, measurement_manifest, expected_commit=base_commit)
    selection = _initial_selection(event, base_commit)
    selection["reason"] = "No reusable verified baseline exists for this PR and base commit; measure it before the PR."
    # Save the fresh-run request in case the lookup is interrupted.
    _write(args.selection, selection)
    protocol = _benchmark_protocol(args.legs, measurement_manifest)
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
                        for leg in planned
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
    _write(args.selection, selection)
    return selection


def capture_context(args: argparse.Namespace, event: dict, measurement_manifest: dict | None) -> dict:
    """Prepare verified source, then record optional comparison metadata."""
    commit, parents = _checkout_source(args.checkout_root)
    if measurement_manifest is not None and (
        not isinstance(measurement_manifest, dict) or measurement_manifest.get("commit") != os.environ.get("GITHUB_SHA")
    ):
        raise ValueError("Measurement source does not identify the tested PR merge GITHUB_SHA.")
    pr = event.get("pull_request", {})
    if pr and args.role == "current":
        reference_commit = resolve_base(commit, parents, event)
    else:
        reference_commit = _resolved_base() if pr else None
    expected_commit = reference_commit if pr and args.role == "baseline" else os.environ.get("GITHUB_SHA")
    manifest = prepare_manifest(args.checkout_root, measurement_manifest, expected_commit=expected_commit)
    _write(args.output_dir / "source-manifest.json", manifest)
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
            "benchmark_role": args.role,
            "image_ref": args.image,
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
            "expected_legs": [line.split("|", 1)[0] for line in args.legs.read_text().splitlines() if line.strip()],
            "benchmark_protocol": _benchmark_protocol(args.legs, measurement_manifest),
        },
        "metric_definition": {"total_fps": None},
    }
    if measurement_manifest is not None:
        context["measurement"] = {key: measurement_manifest[key] for key in ("commit", "digest")}
        context["metric_definition"]["total_fps"] = f"measurement-v1:{measurement_manifest['digest']}"
    try:
        inspected = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, text=True)
        image = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else {}
        context["source"].update(image_digest=next(iter(image.get("RepoDigests", [])), None), image_id=image.get("Id"))
    except (OSError, ValueError, TypeError, AttributeError, IndexError, KeyError, subprocess.SubprocessError) as exc:
        reason = f"Optional build provenance capture failed: {exc}"
        context["metric_definition"]["reason"] = reason
        print(f"::warning::{reason}")
    try:
        _write(args.output_dir / "build-context.json", context)
    except OSError as exc:
        print(f"::warning::Optional build provenance could not be saved: {exc}")
    return context


def bind_baseline(args: argparse.Namespace, event: dict) -> None:
    """Pin the uploaded baseline before the current PR is measured."""
    if not args.selection.exists():
        _write(args.selection, _initial_selection(event, _resolved_base()))
        if not args.baseline_issue:
            args.baseline_issue = "Baseline selection metadata was not produced; baseline FPS is unavailable."
    repository = os.environ["GITHUB_REPOSITORY"]
    selection = json.loads(args.selection.read_text())
    reuse_confirmed = os.environ.get("PERF_BASELINE_REUSED", "true") == "true"
    if selection.get("reference_commit") != _resolved_base():
        raise ValueError("Baseline selection does not match the tested PR merge's resolved first parent.")
    if args.artifact_id or not reuse_confirmed:
        # A fresh baseline clears any earlier reuse decision.
        selection.update(baseline_reused=False, baseline_origin=None)
    if not selection["baseline_reused"]:
        if args.artifact_id:
            selection["baseline_origin"] = {
                "repository": repository,
                "artifact_id": int(args.artifact_id),
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
        selection["reason"] = args.baseline_issue or (
            "Measured the PR's exact base commit before its tested revision on the same runner."
            if args.artifact_id
            else "The base benchmark artifact was not produced; baseline FPS is unavailable."
        )
        if args.baseline_issue:
            selection.setdefault("issues", []).append(args.baseline_issue)
    selection["tested_commit"] = os.environ["GITHUB_SHA"]
    _write(args.output_dir / "pr-comparison.json", selection)


def _unavailable(exc: EvidenceError, side: str, identity: dict) -> dict:
    return {"reason": str(exc), "reason_code": exc.code, "unavailable_evidence": identity, "unavailable_side": side}


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
        selection.update(_unavailable(exc, "baseline", origin))
        return None, selection


def _execute(args: argparse.Namespace, event: dict) -> None:
    if args.command == "resolve":
        commit, parents = _checkout_source(args.checkout_root)
        base_commit = resolve_base(commit, parents, event)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"base_commit={base_commit}\n")
        print(f"Resolved tested PR merge {commit} first parent: {base_commit}")
    elif args.command == "bind":
        bind_baseline(args, event)
    else:
        manifest = args.measurement_manifest
        measurement = json.loads(manifest.read_text()) if manifest is not None else None
        if args.command == "capture":
            capture_context(args, event, measurement)
        else:
            selection = restore_baseline(args, event, measurement)
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                output.write(f"reused={str(selection['baseline_reused']).lower()}\n")
            print(selection["reason"])
            for issue in selection["issues"]:
                print(issue)


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
            aliases[leg] |= {workload_key(item["bundle"]) for item in samples} - {None}
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


def _refresh_history(files: dict[str, bytes], directory: Path) -> None:
    """Recompute the existing history gate from saved measurements."""
    try:
        expected = _object(artifact_json(files, "build-context.json").get("execution")).get("expected_legs")
    except EvidenceError:
        expected = None
    if not isinstance(expected, list) or not expected or any(not isinstance(leg, str) for leg in expected):
        write_json(errored("The configured workload list is unavailable."), directory / "comparison.json")
        expected = []
    samples = defaultdict(list)
    for name, data in files.items():
        leg, _, rest = name.partition("/")
        if rest == "status":
            samples.setdefault(leg, [])
        elif rest and PurePosixPath(name).name.startswith("benchmark_runtime_") and name.endswith(".json"):
            samples[leg].append(data)
    try:
        for index, leg in enumerate(sorted(set(expected) | samples.keys())):
            output = directory / str(index) / "comparison.json"
            if files.get(f"{leg}/status", b"").strip() != b"ok" or len(samples[leg]) != 3:
                write_json(errored("Expected three samples from a successful benchmark run.", label=leg), output)
                continue
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    paths = [Path(temporary, f"sample-{number}.json") for number in range(3)]
                    for path, data in zip(paths, samples[leg]):
                        path.write_bytes(data)
                    gate_cli.main(
                        [
                            "compare",
                            "--benchmark_result",
                            *map(str, paths),
                            f"--label={leg}",
                            "--output_json",
                            str(output),
                        ]
                    )
            except Exception as exc:
                if not output.exists():
                    write_json(errored(f"History comparison failed ({type(exc).__name__}).", label=leg), output)
    finally:
        gate_exit = gate_cli.main(
            [
                "aggregate",
                "--comparison_dir",
                str(directory),
                "--output_markdown",
                str(directory / "rolling-history.md"),
            ]
        )
        _write(directory / "status.json", {"failed": gate_exit == 1})


def _report(args: argparse.Namespace) -> int:
    candidate = selected = paired_failure = None
    selection = {}
    try:
        client = GitHubClient(args.repository)
        candidate = resolve_candidate(client, args.run_id, args.run_attempt, args.comparison_dir, args.history_dir)
        print(f"Using benchmark results from attempt {candidate.identity['run_attempt']}.")
        if not args.gate_only:
            paired_failure = _paired_failure(candidate)
            selected, selection = select_pr_baseline(client, candidate)
    except EvidenceError as exc:
        selection = _unavailable(exc, "candidate" if candidate is None else "baseline", exc.identity)

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
    report.add_argument("--history_dir", type=Path, help="Compare saved measurements against Azure history")
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

import sys

# Load the tests only when pytest is running.
if "pytest" in sys.modules:
    import copy
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    import pytest

    from . import source_revision
    from .report import render_aggregate

    REPO, HEAD, PARENT = "isaac-sim/IsaacLab", "a" * 40, "b" * 40
    RUNTIME = "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py"

    def bundle(fps=100, task="Task"):
        return {
            "run": {
                "task": task,
                "num_envs": 512,
                "status": "completed",
                "seed": 42,
                "start_time_utc": "2026-10-01T10:00:00Z",
                "end_time_utc": "2026-10-01T10:05:00Z",
                "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": []},
            },
            "runtime": {
                "total_fps": {"mean": fps},
                "iterations_completed": 200,
                "steps_per_iteration": 512,
                "environment_step_timing": {"warmup_steps": 100, "measurement_mode": "host_return"},
            },
            "hardware": {
                "hostname": "worker",
                "cpu_name": "Xeon",
                "cpu_count": 16,
                "gpu_devices": [{"name": "NVIDIA H100"}],
            },
            "versions": {"warp": "1.0"},
        }

    def evidence(fps=(90, 100, 110), *, task="Task", expected=3):
        workloads = (
            fps
            if isinstance(fps, dict)
            else {"leg": [bundle(value, task) for value in (fps if isinstance(fps, tuple) else (fps,) * 3)]}
        )
        samples = {
            leg: [
                {"path": f"{leg}/sample-{i}/benchmark_runtime_fixture.json", "bundle": value}
                for i, value in enumerate(values)
            ]
            for leg, values in workloads.items()
        }
        return Evidence(
            {"repository": REPO, "run_id": 1, "run_attempt": 1, "source_commit": HEAD},
            {
                "source": {"image_digest": "sha256:fixture"},
                "execution": {"expected_samples": expected},
                "metric_definition": {"total_fps": "measurement-v1:" + "a" * 64},
            },
            samples,
            {leg: "ok" for leg in samples},
            [],
            {},
        )

    def assert_fields(actual, **expected):
        assert {key: actual[key] for key in expected} == expected

    def set_field(document, path, value, *, remove=False):
        for key in path[:-1]:
            document = document[key]
        if remove:
            document.pop(path[-1], None)
        else:
            document[path[-1]] = copy.deepcopy(value)

    def measurement_file(pair, **changes):
        path = pair.tmp / "measurement.json"
        path.write_text(json.dumps(pair.measurement | changes))
        return path

    class FixtureClient(GitHubClient):
        repository = REPO

        def __init__(self):
            self.runs, self.run_jobs, self.items, self.contents = {}, {}, {}, {}
            self.downloaded, self.queries = [], []

        def add_run(self, run_id, attempt=1, **fields):
            run = dict(
                id=run_id,
                run_attempt=attempt,
                head_sha=HEAD,
                event="pull_request",
                head_branch="feature",
                workflow_id=7,
                run_started_at="2026-10-01T10:00:00Z",
            )
            run.update(fields)
            self.runs[run_id, attempt] = run
            self.run_jobs[run_id, attempt] = [
                dict(name="performance-regression-benchmarks", started_at=run["run_started_at"], conclusion="success")
            ]
            return run

        def publish(self, run_id, name, files):
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as archive:
                for path, value in files.items():
                    archive.writestr(path, value if isinstance(value, (str, bytes)) else json.dumps(value))
            data, identity = output.getvalue(), len(self.items) + 1
            artifact = dict(
                id=identity,
                name=name,
                workflow_run={"id": run_id},
                expired=False,
                digest="sha256:" + hashlib.sha256(data).hexdigest(),
            )
            self.items[identity], self.contents[identity] = artifact, data
            return artifact

        def run_attempt(self, run_id, attempt):
            return self.runs[run_id, attempt]

        def jobs(self, run_id, attempt):
            return self.run_jobs[run_id, attempt]

        def artifacts(self, run_id):
            return [item for item in self.items.values() if item["workflow_run"]["id"] == run_id]

        def download(self, artifact):
            self.downloaded.append(artifact["id"])
            with patch.object(self, "_request", return_value=self.contents[artifact["id"]]):
                return super().download(artifact)

        def get(self, path):
            return self.items[int(path.rsplit("/", 1)[1])]

        def paginate(self, path, key, **params):
            self.queries.append((path, key, params))
            return list({run["id"]: run for run in self.runs.values()}.values())

    def command_args(command, **options):
        arguments = [command]
        for name, value in options.items():
            name = name if command == "report" and name != "gate_only" else name.replace("_", "-")
            if value is not None and value is not False:
                arguments += ["--" + name] + ([] if value is True else [str(value)])
        return arguments

    @pytest.fixture
    def pair(tmp_path, monkeypatch):
        """A tested PR merge in a real Git checkout, its Actions environment, and an in-memory GitHub API."""
        pair = SimpleNamespace(
            root=tmp_path / "checkout",
            tmp=tmp_path,
            output=tmp_path / "output",
            selection=tmp_path / "selection.json",
            legs=tmp_path / "legs.tsv",
            controller=tmp_path / "controller",
            client=FixtureClient(),
            image='[{"Id":"fixture","RepoDigests":[]}]',
        )

        def git(*args):
            identity = ["-c", "user.name=Fixture", "-c", "user.email=test@example.invalid"]
            return subprocess.check_output(["git", "-C", str(pair.root), *identity, *args], text=True).strip()

        def commit(name, text, *branch):
            if branch:
                git("checkout", "--quiet", "-b", *branch)
            (pair.root / name).parent.mkdir(parents=True, exist_ok=True)
            (pair.root / name).write_text(text)
            git("add", ".")
            git("commit", "--quiet", "-m", name)
            return git("rev-parse", "HEAD")

        pair.root.mkdir()
        git("init", "--quiet")
        pair.git, pair.checkout = git, lambda sha: git("checkout", "--quiet", "--detach", sha)
        pair.old = commit(RUNTIME, "VALUE = 'base'\n")
        pair.head = commit("head.txt", "head", "feature")
        pair.base = commit("base.txt", "advanced base", "target", pair.old)
        git("merge", "--quiet", "--no-ff", "-m", "tested merge", pair.head)
        pair.merge = git("rev-parse", "HEAD")
        pair.checkout(pair.base)
        pair.event = {
            "pull_request": {
                "number": 42,
                "base": {"sha": pair.old, "ref": "target"},
                "head": {"sha": pair.head, "ref": "feature"},
            }
        }
        for key, value in dict(
            GITHUB_SHA=pair.merge,
            PERF_BASE_COMMIT=pair.base,
            GITHUB_EVENT_NAME="pull_request",
            GITHUB_EVENT_PATH=str(tmp_path / "event.json"),
            GITHUB_OUTPUT=str(tmp_path / "github-output"),
            GITHUB_RUN_ID="20",
            GITHUB_RUN_ATTEMPT="1",
            GITHUB_JOB="benchmarks",
            GITHUB_REPOSITORY=REPO,
            RUNNER_NAME="runner",
        ).items():
            monkeypatch.setenv(key, value)
        pair.legs.write_text("first|Task|512|600|physics=newton_mjwarp\nsecond|Other|512|600|physics=newton_mjwarp\n")
        pair.controller.mkdir()
        for name in ("run_benchmarks.sh", "source_revision.py"):
            (pair.controller / name).write_bytes(Path(source_revision.__file__).with_name(name).read_bytes())
        monkeypatch.setattr(sys.modules[__name__], "__file__", str(pair.controller / "pr_comparison.py"))
        measured = {RUNTIME: "a" * 64}
        pair.measurement = dict(commit=pair.merge, files=measured, digest=measurement_digest(measured))
        pair.protocol = _benchmark_protocol(pair.legs, pair.measurement)
        for run_id in (10, 20):
            pair.client.add_run(run_id)
        monkeypatch.setattr(sys.modules[__name__], "GitHubClient", lambda repository: pair.client)
        original_run = subprocess.run

        def run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0, pair.image, "")
            return original_run(command, **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        paths = dict(checkout_root=pair.root, output_dir=pair.output, legs=pair.legs)
        defaults = {
            "resolve": dict(checkout_root=pair.root),
            "capture": paths | dict(image="fixture", role="current"),
            "restore": paths | dict(selection=pair.selection),
            "bind": dict(selection=pair.selection, output_dir=pair.output),
            "report": dict(repository=REPO, run_id=20, run_attempt=1, output_dir=pair.output),
        }

        def arguments(command, options):
            (tmp_path / "event.json").write_text(json.dumps(pair.event))
            return command_args(command, **(defaults[command] | options))

        def invoke(command, **options):
            return main(arguments(command, options))

        def cli(command, **options):
            """Run the command in a fresh interpreter, as the workflow does."""
            module = [sys.executable, "-m", "tools.performance_regression.pr_comparison"]
            cwd = Path(source_revision.__file__).resolve().parents[2]
            return subprocess.run([*module, *arguments(command, options)], cwd=cwd, capture_output=True, text=True)

        pair.invoke, pair.cli = invoke, cli
        return pair

    def artifact_files(pair, role="baseline", run_id=10, attempt=1):
        commit = pair.base if role == "baseline" else pair.merge
        context = {
            "source": dict(
                commit=commit,
                reference_commit=pair.base,
                event_base_commit=pair.old,
                pull_request_number=42,
                requested_head_commit=pair.head,
                benchmark_role=role,
                commit_parents=[pair.base, pair.head],
            ),
            "execution": dict(
                run_id=run_id,
                run_attempt=attempt,
                expected_samples=3,
                expected_legs=["first", "second"],
                benchmark_protocol=pair.protocol,
                job="benchmarks",
                runner_name="runner",
                hostname="host",
            ),
            "measurement": {key: pair.measurement[key] for key in ("commit", "digest")},
            "metric_definition": {"total_fps": "measurement-v1:" + pair.measurement["digest"]},
        }
        files = {
            "build-context.json": context,
            "source-manifest.json": dict(
                schema_version=4, commit=commit, files={RUNTIME: "b" * 64}, measurement=pair.measurement
            ),
        }
        for leg in ("first", "second"):
            files[f"{leg}/status"] = "ok"
            for repeat in range(1, 4):
                value = bundle(task=leg)
                data = json.dumps(value).encode()
                files[f"{leg}/sample-{repeat}/benchmark_runtime_fixture.json"] = value
                files[f"{leg}/sample-{repeat}/source-revision.json"] = dict(
                    schema_version=4,
                    status="verified",
                    bytecode_policy="fresh_process_cache",
                    commit=commit,
                    measurement_digest=pair.measurement["digest"],
                    mismatches=[],
                    modules=[dict(name=RUNTIME_MODULE, relative_path=RUNTIME)],
                    outputs=[
                        dict(
                            path="benchmark_runtime_fixture.json",
                            sha256=hashlib.sha256(data).hexdigest(),
                            bytes=len(data),
                        )
                    ],
                )
        return copy.deepcopy(files)

    def publish_pair(pair, baseline=None, *, reused=True, **pin_fields):
        baseline = (
            pair.client.publish(10, baseline_name(42, pair.base), artifact_files(pair))
            if baseline is None
            else baseline
        )
        files = artifact_files(pair, "current", 20)
        files["pr-comparison.json"] = dict(
            reference_commit=pair.base,
            pull_request_number=42,
            tested_commit=pair.merge,
            requested_head_commit=pair.head,
            baseline_reused=reused,
            baseline_origin=dict(
                artifact_id=baseline["id"],
                run_id=baseline["workflow_run"]["id"],
                run_attempt=1,
                source_commit=pair.base,
            ),
        )
        files["first/comparison.json"] = {"verdict": "FAIL", "label": "first"}
        files["pr-comparison.json"].update(pin_fields)
        current = pair.client.publish(20, "performance-regression-20-1", files)
        return baseline, current

    def restore(pair):
        assert pair.invoke("restore", measurement_manifest=measurement_file(pair)) == 0
        return json.loads(pair.selection.read_text())

    @pytest.mark.parametrize(
        "path,value,index,status",
        [
            ("run.seed", 7, None, "incompatible"),
            ("run.seed", 7, 1, "incompatible"),
            ("runtime.iterations_completed", 100, None, "incompatible"),
            ("runtime.steps_per_iteration", 256, None, "incompatible"),
            ("runtime.environment_step_timing.warmup_steps", 20, None, "incompatible"),
            ("runtime.environment_step_timing.measurement_mode", "sync", None, "incompatible"),
            ("runtime.environment_step_timing.measurement_mode", None, None, "unknown"),
            ("hardware.cpu_name", "AMD", None, "incompatible"),
            ("hardware.cpu_count", 32, 1, "incompatible"),
            ("hardware.gpu_devices", [{"name": "A100"}], None, "incompatible"),
            ("hardware.gpu_devices", [{"name": "H100"}] * 2, None, "incompatible"),
            ("hardware.cpu_name", None, None, "unknown"),
            ("hardware.cpu_name", " unknown ", None, "unknown"),
            ("hardware.cpu_name", 7, None, "unknown"),
            ("hardware.cpu_count", True, None, "unknown"),
            ("hardware.cpu_count", 0, None, "unknown"),
            ("hardware.cpu_count", None, 1, "unknown"),
            ("hardware.gpu_devices", [], None, "unknown"),
            ("hardware.gpu_devices", [], 1, "unknown"),
            ("hardware.gpu_devices", "GPU", None, "unknown"),
            ("hardware.gpu_devices", [None], None, "unknown"),
            ("hardware.gpu_devices", [{"name": "unknown"}], None, "unknown"),
            ("run.status", "failed", 1, "partial"),
        ],
    )
    def test_compatibility(path, value, index, status):
        a, b = evidence(), evidence((100, 110, 120))
        for sample in b.samples["leg"] if index is None else [b.samples["leg"][index]]:
            set_field(sample["bundle"], path.split("."), value)
        report = compare_evidence(a, b)
        row = report["rows"][0]
        assert (row["status"], row["absolute_change"], row["change_pct"]) == (status, None, None)
        assert (row["baseline"]["median"], row["candidate"]["median"]) == (100, 110)
        primary, details = render_build_comparison(report).split("<details>", 1)
        assert "| ⚪ Not comparable | leg | 100 | 110 | — |" in primary and "+10.00%" not in primary
        if path.startswith("hardware"):
            assert "hardware" in str(row["reasons"])
            if status == "incompatible":
                assert "CPU/GPU hardware differs" in details

    @pytest.mark.parametrize("value", [None, True, "100", float("nan"), float("inf"), -1])
    def test_invalid_fps(value):
        row = compare_evidence(evidence(), evidence((90, value, 110)))["rows"][0]
        assert row["candidate"]["samples"] == [90, 110]
        assert (row["candidate"]["count"], row["candidate"]["observed_count"], row["status"]) == (2, 3, "partial")
        assert row["change_pct"] is None

    @pytest.mark.parametrize(
        "formula,side",
        [
            (None, "candidate"),
            ("unknown", "candidate"),
            (" AGGREGATE_FRAMES_OVER_MEASURED_SECONDS ", "candidate"),
            ("different", "candidate"),
            ("aggregate_frames_over_measured_seconds", "baseline"),
            ("aggregate_frames_over_measured_seconds", "both"),
        ],
    )
    def test_formula(formula, side):
        a, b = evidence(), evidence((110,) * 3)
        for item in (a, b) if side == "both" else (a if side == "baseline" else b,):
            item.context["metric_definition"]["total_fps"] = formula
        report = compare_evidence(a, b)
        row = report["rows"][0]
        assert row["status"] == ("incompatible" if formula == "different" else "unknown")
        assert row["absolute_change"] is row["change_pct"] is None
        reason = "FPS formula identities differ." if formula == "different" else "FPS formula identity is unknown"
        assert reason in render_build_comparison(report)

    @pytest.mark.parametrize(
        "kind", ["few", "failed", "status_missing", "empty_ok", "empty_failed", None, True, "3", 0]
    )
    def test_incomplete(kind):
        b = evidence()
        if kind in ("few", "empty_ok", "empty_failed"):
            b.samples["leg"] = b.samples["leg"][:2] if kind == "few" else []
        if kind in ("failed", "empty_failed"):
            b.statuses["leg"] = "failed"
        if kind == "status_missing":
            b.statuses = {}
        if kind in (None, True, "3", 0):
            b.context["execution"]["expected_samples"] = kind
        report = compare_evidence(evidence(), b)
        row = report["rows"][0]
        expected = (
            "missing"
            if kind in ("empty_ok", "empty_failed")
            else "unknown"
            if kind in (None, True, "3", 0)
            else "partial"
        )
        assert row["status"] == expected and row["change_pct"] is None
        if expected == "missing":
            assert row["candidate"]["median"] is None and "display association" in str(row["reasons"])

    @pytest.mark.parametrize("path,index", [("hardware.cpu_name", None), ("hardware.cpu_count", 1)])
    def test_missing_baseline_hardware(path, index):
        a, b = evidence(), evidence(110)
        for sample in a.samples["leg"] if index is None else [a.samples["leg"][index]]:
            set_field(sample["bundle"], path.split("."), None)
        report = compare_evidence(a, b)
        assert_fields(report["rows"][0], status="unknown", absolute_change=None, change_pct=None)
        assert "CPU/GPU hardware identity is unavailable" in render_build_comparison(report)

    def test_medians_workloads_and_context():
        row = compare_evidence(evidence((90, 100, 1100)), evidence((93, 96, 999)))["rows"][0]
        assert (row["baseline"]["median"], row["absolute_change"], row["change_pct"]) == (100, -4, -4)
        assert row["baseline"]["min"] == 90 and row["baseline"]["max"] == 1100 and "verdict" not in row
        a, b = evidence(), evidence((110,) * 3)
        for sample in b.samples["leg"]:
            sample["bundle"]["hardware"]["hostname"] = "other"
            sample["bundle"]["hardware"]["gpu_devices"][0].update(uuid="other", index=7, driver_version="other")
            sample["bundle"]["versions"]["warp"] = "2.0"
        b.context["source"]["image_digest"] = "changed"
        report = compare_evidence(a, b)
        fields = {item["field"] for item in report["rows"][0]["context_differences"]}
        assert {"hardware.hostname", "versions.warp", "source.image_digest"}.issubset(fields)
        assert report["rows"][0]["change_pct"] == 10 and "does not isolate a commit effect" in render_build_comparison(
            report
        )
        for field, value in (
            ("num_envs", 1024),
            ("config", dict(physics_backend="newton_mjwarp", rendering_backend="none", presets=["x"])),
        ):
            changed = copy.deepcopy(b)
            for sample in changed.samples["leg"]:
                sample["bundle"]["run"][field] = value
            rows = compare_evidence(a, changed)["rows"]
            assert len(rows) == 2 and all(row["status"] == "missing" for row in rows)

    SAMPLE, PROOF = "first/sample-1/benchmark_runtime_fixture.json", "first/sample-1/source-revision.json"
    MANIFEST, CONTEXT, INCOMPLETE = "source-manifest.json", "build-context.json", "Baseline is incomplete"

    # name: (artifact field path, new value or None to delete, restore rejection, pinned leg and its source issue)
    REJECTED = {
        "missing_proof": (("second/sample-3/source-revision.json",), None, INCOMPLETE, ("second", "artifact has no")),
        "result": ((SAMPLE, "runtime", "total_fps", "mean"), 900, INCOMPLETE, ("first", "Result bytes")),
        "module": ((PROOF, "modules", 0, "relative_path"), "missing.py", INCOMPLETE, ("first", "Loaded source")),
        "legacy": ((PROOF, "schema_version"), 2, INCOMPLETE, ("first", "Executing source")),
        "failed_proof": ((PROOF, "status"), "failed", INCOMPLETE, ("first", "Executing source")),
        "digest": ((MANIFEST, "measurement", "digest"), "0" * 64, INCOMPLETE, ("first", "identity was not")),
        "formula": (
            (CONTEXT, "metric_definition", "total_fps"),
            "measurement-v1:" + "0" * 64,
            INCOMPLETE,
            ("first", "identity"),
        ),
        "measurement_missing": ((MANIFEST, "measurement"), None, INCOMPLETE, ("first", "manifest is missing")),
        "measurement_proof_missing": ((PROOF, "measurement_digest"), None, INCOMPLETE, ("first", "proof does not")),
        "missing_sample": (("second/sample-3/benchmark_runtime_fixture.json",), None, INCOMPLETE, None),
        "failed_leg": (("second/status",), "failed", INCOMPLETE, None),
        "missing_leg": ((), None, INCOMPLETE, None),
        "protocol": ((CONTEXT, "execution", "benchmark_protocol"), None, "protocol is missing or differs", None),
        "role": ((CONTEXT, "source", "benchmark_role"), "current", "does not belong to this PR", None),
    }

    @pytest.mark.parametrize("mutation", REJECTED)
    def test_rejected_baselines(pair, mutation):
        path, value, rejection, pinned = REJECTED[mutation]
        files = artifact_files(pair)
        if mutation == "missing_leg":
            files = {name: data for name, data in files.items() if not name.startswith("second/")}
        else:
            set_field(files, path, value, remove=value is None)
        artifact = pair.client.publish(10, baseline_name(42, pair.base), files)
        selection = restore(pair)
        assert not selection["baseline_reused"] and selection["baseline_origin"] is None
        assert f"Artifact {artifact['id']} was not reused: " in str(selection["issues"])
        assert rejection in str(selection["issues"])
        if pinned:
            leg, issue = pinned
            publish_pair(pair, artifact)
            candidate = resolve_candidate(pair.client, 20, 1)
            selected, _ = select_pr_baseline(pair.client, candidate)
            assert any(note.startswith(f"{leg}/source: ") and issue in note for note in selected.issues)
            if mutation in ("missing_proof", "result", "module", "legacy"):
                assert ("second/sample-3" if mutation == "missing_proof" else "first/sample-1") in str(selected.issues)
            report = compare_evidence(selected, candidate)
            assert all("Baseline: " + note in report["notes"] for note in selected.issues)
            assert next(row for row in report["rows"] if leg in row["legs"]["baseline"])["change_pct"] is None

    def test_reuse_binding_and_retries(pair, monkeypatch):
        baseline, current = publish_pair(pair)
        original = pair.client.contents[baseline["id"]]
        selection = restore(pair)
        assert selection["baseline_reused"] and selection["baseline_origin"]["artifact_id"] == baseline["id"]
        assert selection["reference_commit"] == pair.base and selection["event_base_commit"] == pair.old
        pair.measurement["commit"] = HEAD
        pair.event["pull_request"]["head"]["sha"] = HEAD
        updated = restore(pair)
        assert updated["baseline_reused"] and updated["requested_head_commit"] == HEAD
        assert updated["baseline_origin"]["artifact_id"] == baseline["id"]
        assert pair.invoke("bind", baseline_issue="Base image configuration failed.") == 0
        pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert_fields(pin, baseline_origin=selection["baseline_origin"], issues=updated["issues"])
        assert pin["reason"] == updated["reason"]
        assert pair.client.contents[baseline["id"]] == original
        fresh = pair.client.publish(20, baseline_name(42, pair.base), artifact_files(pair, run_id=20))
        assert pair.invoke("bind", artifact_id=fresh["id"]) == 0
        fresh_pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert not fresh_pin["baseline_reused"]
        assert (
            fresh_pin["reason"] == "Measured the PR's exact base commit before its tested revision on the same runner."
        )
        assert_fields(fresh_pin["baseline_origin"], artifact_id=fresh["id"], run_id=20)
        monkeypatch.setenv("PERF_BASELINE_REUSED", "")
        assert pair.invoke("bind") == 0
        assert json.loads((pair.output / "pr-comparison.json").read_text())["baseline_origin"] is None
        baseline["expired"] = True
        assert not restore(pair)["baseline_reused"]

    def test_cli_capture_and_failure(pair, monkeypatch):
        pair.checkout(pair.merge)
        assert pair.cli("resolve").returncode == 0
        assert f"base_commit={pair.base}" in (pair.tmp / "github-output").read_text()
        for role, commit in (("baseline", pair.base), ("current", pair.merge)):
            pair.checkout(commit)
            assert pair.invoke("capture", role=role, measurement_manifest=measurement_file(pair)) == 0
            context = json.loads((pair.output / "build-context.json").read_text())
            source, execution = context["source"], context["execution"]
            assert_fields(source, commit=commit, reference_commit=pair.base, event_base_commit=pair.old)
            assert_fields(source, requested_head_commit=pair.head, benchmark_role=role)
            assert context["measurement"] == {"commit": pair.merge, "digest": pair.measurement["digest"]}
            assert context["metric_definition"]["total_fps"] == "measurement-v1:" + pair.measurement["digest"]
            assert_fields(execution, benchmark_protocol=pair.protocol, expected_legs=["first", "second"])
            assert execution["benchmark_protocol"]["measurement_sha256"] == pair.measurement["digest"]
            assert_fields(execution, run_id=20, run_attempt=1, expected_samples=3)
            assert_fields(execution, job="benchmarks", runner_name="runner", hostname=socket.gethostname())
        monkeypatch.setenv("GITHUB_SHA", "invalid")
        assert pair.cli("capture", output_dir=pair.tmp / "failure").returncode == 1
        assert not (pair.tmp / "failure" / "source-manifest.json").exists()
        failure = json.loads((pair.tmp / "failure" / "paired-failure.json").read_text())
        assert failure["stage"] == "capture_current" and "GITHUB_SHA" in failure["reason"]
        assert failure["source"]["commit"] == pair.merge and failure["source"]["tested_commit"] == "invalid"

    def test_report_reruns_and_gate(pair):
        from . import cli

        baseline, current = publish_pair(pair)
        for attempt in (1, 2):
            if attempt == 2:
                pair.client.add_run(20, 2)
                pair.client.run_jobs[20, 2][0]["started_at"] = "2026-10-01T09:00:00Z"
            comparisons = pair.tmp / f"comparisons-{attempt}"
            assert pair.invoke("report", run_attempt=attempt, comparison_dir=comparisons) == 0
            report = json.loads((pair.output / "build-comparison.json").read_text())
            assert report["candidate"]["run_attempt"] == 1 and report["report_attempt"] == attempt
            assert [row["status"] for row in report["rows"]] == ["compared"] * 2 and report["candidate_kind"] == "merge"
            assert cli.main(["aggregate", "--comparison_dir", str(comparisons)]) == 1
        baseline["expired"] = True
        pair.client.add_run(11)
        pair.client.publish(11, baseline_name(42, pair.base), artifact_files(pair, run_id=11))
        assert pair.invoke("report", run_attempt=2) == 0
        report = json.loads((pair.output / "build-comparison.json").read_text())
        assert [row["candidate"]["median"] for row in report["rows"]] == [100, 100] and not pair.client.queries
        assert report["baseline"] is None and all(row["change_pct"] is None for row in report["rows"])
        assert_fields(report["selection"], reason_code="expired", unavailable_side="baseline")
        assert report["selection"]["unavailable_evidence"] == dict(
            artifact_id=baseline["id"], run_id=10, run_attempt=1, source_commit=pair.base
        )
        pair.client.run_jobs[20, 2][0]["started_at"] = "2026-10-01T10:00:00Z"
        with pytest.raises(EvidenceError, match="Benchmark ran"):
            resolve_candidate(pair.client, 20, 2)
        gate = pair.tmp / "gate"
        assert pair.invoke("report", gate_only=True, output_dir=gate, comparison_dir=pair.tmp / "comparisons") == 0
        assert not gate.exists()

    def test_rendering_summary():
        before = evidence({name: [bundle(100, name)] * 3 for name in ("Higher", "Lower", "Same", "Tiny")})
        after = evidence(
            {
                name: [bundle(fps, name)] * 3
                for name, fps in (("Higher", 120), ("Lower", 80), ("Same", 100), ("Tiny", 100.0001), ("New", 130))
            }
        )
        report = compare_evidence(before, after)
        report["baseline"]["run_url"] = "https://bad.invalid/untrusted"
        report["rows"][0]["label"] = "Task | [click](https://bad.invalid)\n<script>bad()</script>"
        unsafe = "leg | </details><script>"
        report["rows"][0]["legs"] = {"baseline": [unsafe], "candidate": [unsafe]}
        report["rows"][0]["notes"].append(unsafe)
        original = copy.deepcopy(report)
        text = render_build_comparison(report)
        primary, details = text.split("<details>", 1)
        assert "Improved 2 · 🔴 Regressed 1 · ⚪ Not comparable 1 · ⚪ Unchanged 1" in primary
        rows = [line for line in primary.splitlines() if line.startswith("|")]
        assert len(rows) == 7 and all(len(line.split("|")) == 7 for line in rows)
        assert "Status | Workload | Baseline FPS | PR FPS | Change %" in primary
        assert "PASS" not in text and "FAIL" not in text
        assert report == original and text.count("<details>") == text.count("</details>") == 1
        assert all(value not in text for value in ("<script>", "[click]", "https://bad.invalid/untrusted"))
        assert all(
            value in details
            for value in (
                "&#91;click&#93;",
                "&#124;",
                "&lt;/details&gt;&lt;script&gt;",
                "3/3 / 3/3",
                "build-comparison.json",
            )
        )
        assert "Baseline: No runtime samples identify this workload." in details
        assert "/actions/runs/1/attempts/1" in details
        zero = compare_evidence(evidence((0,) * 3), evidence((10,) * 3))
        assert_fields(zero["rows"][0], absolute_change=10, change_pct=None)
        assert "baseline median is zero" in zero["rows"][0]["notes"][0]
        primary = render_build_comparison(zero).split("<details>", 1)[0]
        assert "Improved 1" in primary and "N/A (baseline is zero)" in primary
        for report in ({"rows": []}, compare_evidence(None, evidence())):
            report["selection"] = {"reason": "No verified baseline was available."}
            primary = render_build_comparison(report).split("<details>", 1)[0]
            assert "No verified baseline was available" in primary
            if report["rows"]:
                assert "| ⚪ Not comparable | leg | — | 100 | — |" in primary
            else:
                assert "no workload results available" in primary and "Not comparable 0" not in primary
        before = evidence({"leg": [bundle(100, "Before")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        after = evidence({"leg": [bundle(100, "Task")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        primary, details = render_build_comparison(compare_evidence(before, after)).split("<details>", 1)
        labels = [line.split("|")[2].strip() for line in primary.splitlines() if line.startswith("| ⚪")]
        assert len(labels) == len(set(labels)) == 3 and {"leg (1)", "leg (2)"}.issubset(labels)
        assert "newton_mjwarp" not in primary
        assert all(
            value in details
            for value in ("Before · newton_mjwarp", "Task · newton_mjwarp", "Candidate: No runtime samples")
        )

    def test_rendering_provenance():
        a, b = evidence(), evidence(110)
        a.context["measurement"] = {"commit": "d" * 40, "digest": "e" * 64}
        b.context["measurement"] = {"commit": HEAD, "digest": "e" * 64}
        report = compare_evidence(a, b)
        for side, commit, run in (("baseline", PARENT, 12), ("candidate", HEAD, 15)):
            report[side].update(repository=REPO, source_commit=commit, run_id=run, artifact_id=run + 100)
        report["candidate"].update(requested_head_commit="c" * 40, event="pull_request")
        report.update(candidate_kind="merge", report_attempt=1)
        report["source_context"] = {"reference_commit": PARENT, "event_base_commit": HEAD}
        report["selection"] = {"baseline_reused": True, "baseline_origin": report["baseline"]}
        primary, details = render_build_comparison(report).split("<details>", 1)
        assert "| 🟢 Improved | leg | 100 | 110 | +10.00% |" in primary
        assert all(
            value in details
            for value in (
                "A — PR base",
                "B — PR merge result",
                "Baseline measurements: reused.",
                f"Resolved PR base: `{PARENT}`",
                f"The PR event reported `{HEAD}`",
                "Measurement sources",
                "e" * 64,
            )
        )
        for commit in (HEAD, PARENT, "c" * 40, "d" * 40):
            assert f"https://github.com/{REPO}/commit/{commit}" in details and commit not in primary
        for run in (12, 15):
            assert details.count(f"https://github.com/{REPO}/actions/runs/{run}/attempts/1") == 1
            assert f"https://github.com/{REPO}/actions/runs/{run}/artifacts/{run + 100}" in details
        for baseline, reused, expected in (
            (None, False, "unavailable."),
            (None, True, "unavailable."),
            (a, False, "run for this PR."),
        ):
            report = compare_evidence(baseline, b)
            report.update(candidate_kind="merge", selection={"baseline_reused": reused})
            assert f"Baseline measurements: {expected}" in render_build_comparison(report)

    def test_rendering_gate():
        from . import compare

        insufficient = compare.MetricResult("total_fps", "Total FPS", 100, note="insufficient independent runs")
        partial = compare.Report(
            verdict=compare.SKIP, message="No gating metric could be compared", metrics=(insufficient,)
        )
        reports = [
            ("missing", compare.Report(verdict=compare.SKIP, message="Missing credential")),
            ("partial", partial),
            ("failed", compare.errored("invalid JSON")),
        ]
        original = [report.as_dict() for _, report in reports]
        text = render_aggregate(reports)
        assert "Rolling-history CI gate: 🚫 ERROR" in text and "<details>" not in text
        assert "| Combination | Total FPS | Baseline | FPS regression %" in text
        assert all(
            value in text
            for value in (
                "missing — SKIP",
                "Missing credential",
                "Total FPS: insufficient independent runs",
                "failed — ERROR",
                "invalid JSON",
            )
        )
        assert [report.as_dict() for _, report in reports] == original
        reports = [("partial", partial)] + [
            (name, compare.Report(verdict=compare.PASS, message="No performance regression detected"))
            for name in ("g1", "camera", "physx")
        ]
        text = render_aggregate(reports)
        introduction = text.split("| Combination |", 1)[0]
        assert (
            "Rolling-history CI gate: ✅ PASS" in introduction
            and "Workload checks: 3 passed, 1 skipped." in introduction
        )
        assert (
            "No gating metric could be compared" not in introduction
            and "No performance regression detected" not in introduction
        )
        assert "partial — SKIP" in text and "No gating metric could be compared" in text
        unknown = render_aggregate([("unknown", compare.Report(verdict=compare.SKIP))])
        assert "No reason was recorded" in unknown and "credential" not in unknown
        assert "Rolling-history CI gate: no results" in render_aggregate([])

    def test_transport(subtests):
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
                        (lambda: client.download({"id": 7}))
                        if kind == "download"
                        else (lambda: client.run_attempt(10, 1))
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

    @pytest.mark.parametrize(
        "field,value,index",
        [
            ("physics_solvers", ["newton_mjwarp"], None),
            ("physics_solvers", None, None),
            ("physics_solvers", ["newton_vbd"], None),
            ("physics_coupling", "admm", None),
            ("physics_coupling", None, None),
            ("physics_coupling", "admm", 1),
        ],
    )
    def test_solver_identity(field, value, index):
        a, b = evidence(), evidence(fps=110)
        for selected in (a, b):
            for item in selected.samples["leg"]:
                item["bundle"]["run"]["config"].update(
                    physics_solvers=["newton_mjwarp", "newton_vbd"], physics_coupling="proxy"
                )
        for sample in b.samples["leg"] if index is None else [b.samples["leg"][index]]:
            set_field(sample["bundle"], ["run", "config", field], value, remove=value is None)
        report = compare_evidence(a, b)
        assert all(row["change_pct"] is None for row in report["rows"])
        assert all(sum(row[side]["count"] for row in report["rows"]) == 3 for side in ("baseline", "candidate"))
        primary, details = render_build_comparison(report).split("<details>", 1)
        assert "⚪ Not comparable" in primary and "+10.00%" not in primary
        assert "solvers: newton_mjwarp, newton_vbd · coupling: proxy" in details
        if index is not None:
            shared = next(row for row in report["rows"] if row["baseline"]["count"])
            assert len(report["rows"]) == 2 and shared["status"] == "unknown"
            assert "different workload identities" in str(shared["reasons"])

    def test_dynamic_workloads_variation_and_unknowns():
        a = evidence({"old": [bundle(100, "Shared")] * 3, "removed": [bundle(50, "Removed")] * 3})
        b = evidence({"new": [bundle(110, "Shared")] * 3, "added": [bundle(60, "New")] * 3})
        rows = compare_evidence(a, b)["rows"]
        assert len(rows) == 3 and [row["change_pct"] for row in rows if row["status"] == "compared"] == [10]
        assert sum(row["status"] == "missing" for row in rows) == 2
        for renderer in (None, 42, "", " "):
            invalid = bundle()
            invalid["run"]["config"]["rendering_backend"] = renderer
            assert workload_key(invalid) is None
        a = evidence()
        for i, sample in enumerate(a.samples["leg"], 1):
            sample["bundle"]["run"]["seed"] = i
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        assert_fields(row, status="incompatible", absolute_change=None)
        assert row["protocol_differences"][0]["baseline"] == {"sample_values": [1, 2, 3]}
        a, b = evidence(), evidence()
        for selected, names in ((a, ("A100", "H100")), (b, ("H100", "A100"))):
            for sample in selected.samples["leg"]:
                sample["bundle"]["hardware"]["gpu_devices"] = [{"name": name} for name in names]
        assert_fields(compare_evidence(a, b)["rows"][0], status="incompatible", change_pct=None)
        for selected in (a, b):
            for sample in selected.samples["leg"]:
                sample["bundle"].pop("hardware")
        assert compare_evidence(a, b)["rows"][0]["status"] == "unknown"
        b.samples["leg"] = b.samples["leg"][:1]
        assert compare_evidence(a, b)["rows"][0]["status"] == "partial"
        empty = evidence({"leg": []})
        unknown = compare_evidence(empty, empty)["rows"][0]
        assert unknown["workload_key"] is None and "identity unavailable" in unknown["label"]

    @pytest.mark.parametrize(
        "mutation,code",
        [("digest", "corrupt"), ("zip", "corrupt"), ("expired", "expired"), ("missing_time", "ambiguous_attempt")],
    )
    def test_artifact_failures(pair, mutation, code):
        _, artifact = publish_pair(pair)
        attempt = 1
        if mutation == "digest":
            artifact["digest"] = "sha256:" + "0" * 64
        elif mutation == "zip":
            del artifact["digest"]
            pair.client.contents[artifact["id"]] = b"not a zip"
        elif mutation == "expired":
            artifact["expired"] = True
        else:
            attempt = 2
            pair.client.add_run(20, 2)
            pair.client.run_jobs[20, 2][0]["started_at"] = None
        with pytest.raises(EvidenceError) as raised:
            resolve_candidate(pair.client, 20, attempt)
        assert raised.value.code == code
        if mutation == "expired":
            identity = raised.value.identity
            assert_fields(identity, repository=REPO, run_id=20, run_attempt=1, artifact_id=artifact["id"])
            assert_fields(identity, source_commit=HEAD, github_head_commit=HEAD, tested_commit=None)
            assert identity["source_provenance"] == "github_run_metadata"

    def test_corrupt_samples_and_failed_attempt(pair):
        files = artifact_files(pair, "current", 20)
        files["first/sample-1/benchmark_runtime_fixture.json"] = "{invalid"
        files["first/sample-2/benchmark_runtime_fixture.json"]["runtime"]["total_fps"] = []
        files["first/sample-3/benchmark_runtime_fixture.json"]["runtime"]["total_fps"]["mean"] = 10**1000
        files["second/sample-1/benchmark_runtime_fixture.json"]["run"]["status"] = "failed"
        files["missing/status"] = "failed"
        pair.client.run_jobs[20, 1][0]["conclusion"] = "failure"
        pair.client.runs[20, 1]["conclusion"] = "failure"
        pair.client.publish(20, "performance-regression-20-1", files)
        selected = resolve_candidate(pair.client, 20, 1)
        assert len(selected.samples["first"]) == 2 and selected.samples["missing"] == []
        assert selected.statuses["missing"] == "failed" and sum("FPS value" in issue for issue in selected.issues) == 2
        assert "first/sample-1/benchmark_runtime_fixture.json" in str(selected.issues)
        assert selected.samples["second"][0]["bundle"]["run"]["status"] == "failed"

    @pytest.mark.parametrize(
        "change", ["matrix", "launcher", "source_launcher", "measurement", "other_pr", "workflow", "base"]
    )
    def test_reuse_keys(pair, change, monkeypatch):
        baseline = pair.client.publish(10, baseline_name(42, pair.base), artifact_files(pair))
        assert restore(pair)["baseline_reused"]
        if change == "matrix":
            pair.legs.write_text(pair.legs.read_text().replace("512", "1024"))
        elif change in ("launcher", "source_launcher"):
            path = pair.controller / ("run_benchmarks.sh" if change == "launcher" else "source_revision.py")
            path.write_bytes(path.read_bytes() + b"\n# changed controller\n")
        elif change == "measurement":
            measured = {RUNTIME: "c" * 64}
            pair.measurement.update(files=measured, digest=measurement_digest(measured))
        elif change == "other_pr":
            pair.event["pull_request"]["number"] = 43
        elif change == "workflow":
            pair.client.runs[10, 1]["workflow_id"] = 8
        else:
            monkeypatch.setenv("PERF_BASE_COMMIT", pair.old)
            pair.checkout(pair.old)
        selection = restore(pair)
        assert not selection["baseline_reused"] and selection["reference_commit"] == os.environ["PERF_BASE_COMMIT"]
        if change in ("base", "other_pr"):
            assert pair.client.downloaded == [baseline["id"]]
        else:
            rejection = "another workflow" if change == "workflow" else "protocol"
            assert f"Artifact {baseline['id']} was not reused" in str(selection["issues"])
            assert rejection in str(selection["issues"])

    def test_lookup_failures_keep_fallback(pair, monkeypatch):
        monkeypatch.setenv("PERF_BASE_COMMIT", "")
        assert pair.invoke("restore") == 1 and not pair.selection.exists() and not pair.client.queries
        monkeypatch.setenv("PERF_BASE_COMMIT", pair.base)
        baseline, _ = publish_pair(pair)
        pair.client.add_run(9)
        artifacts = pair.client.artifacts

        def unavailable(run):
            if run == 9:
                raise EvidenceError("inaccessible", "HTTP 404")
            return artifacts(run)

        monkeypatch.setattr(pair.client, "artifacts", unavailable)
        selection = restore(pair)
        assert selection["baseline_origin"]["artifact_id"] == baseline["id"] and "Run 9" in str(selection["issues"])
        monkeypatch.setattr(pair.client, "run_attempt", Mock(side_effect=TimeoutError("lookup interrupted")))
        assert pair.invoke("restore") == 1
        saved = json.loads(pair.selection.read_text())
        assert_fields(saved, baseline_reused=False, baseline_origin=None, reference_commit=pair.base)

    @pytest.mark.parametrize(
        "change,code",
        [
            ("parents", "identity_mismatch"),
            ("pin", "identity_mismatch"),
            ("runner", "runner_mismatch"),
            ("fresh", "runner_mismatch"),
        ],
    )
    def test_pinned_identity(pair, change, code):
        baseline = (
            pair.client.publish(20, baseline_name(42, pair.base), artifact_files(pair, run_id=20))
            if change == "runner"
            else None
        )
        _, current = publish_pair(pair, baseline, reused=change != "fresh")
        candidate = resolve_candidate(pair.client, 20, 1)
        if change == "parents":
            candidate.context["source"]["commit_parents"] = [HEAD, pair.head]
            with pytest.raises(EvidenceError) as raised:
                select_pr_baseline(pair.client, candidate)
            assert raised.value.code == code
            return
        pin = json.loads(candidate.files["pr-comparison.json"])
        if change == "pin":
            pin["baseline_origin"]["sha256"] = "0" * 64
        elif change == "runner":
            pin["baseline_reused"] = False
            candidate.context["execution"]["hostname"] = "other"
        candidate.files["pr-comparison.json"] = json.dumps(pin).encode()
        selected, reason = select_pr_baseline(pair.client, candidate)
        assert selected is None and reason["reason_code"] == code

    @pytest.mark.parametrize("detail", [None, "Base checkout failed"])
    def test_missing_binding_metadata(pair, detail, monkeypatch):
        assert pair.invoke("bind", baseline_issue=detail) == 0
        pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert_fields(
            pin,
            pull_request_number=42,
            reference_commit=pair.base,
            requested_head_commit=pair.head,
            tested_commit=pair.merge,
            baseline_reused=False,
            baseline_origin=None,
        )
        assert (detail or "Baseline selection metadata was not produced") in pin["reason"]
        files = artifact_files(pair, "current", 20)
        files["pr-comparison.json"] = pin
        pair.client.publish(20, "performance-regression-20-1", files)
        candidate = resolve_candidate(pair.client, 20, 1)
        selected, selection = select_pr_baseline(pair.client, candidate)
        assert selected is None and selection["reason"] == pin["reason"] and not pair.client.queries
        candidate.files.pop("pr-comparison.json")
        selected, selection = select_pr_baseline(pair.client, candidate)
        assert selected is None and selection["reason_code"] == "missing_source_evidence" and not pair.client.queries
        monkeypatch.delenv("GITHUB_REPOSITORY")
        pair.selection.write_text("{invalid JSON")
        failure_dir = pair.tmp / "missing-repository"
        assert pair.invoke("bind", output_dir=failure_dir) == 1
        assert json.loads((failure_dir / "paired-failure.json").read_text())["reason"] == "'GITHUB_REPOSITORY'"
        assert not (failure_dir / "pr-comparison.json").exists()

    @pytest.mark.parametrize(
        "case,reason",
        [
            ("checkout", "GITHUB_SHA"),
            ("invalid_sha", "GITHUB_SHA"),
            ("head", "second parent"),
            ("nonmerge", "second parent"),
            ("non_pr", "differs from expected commit"),
            ("modified_source", "Checked-out Python source differs"),
            ("measurement_commit", "Measurement source does not identify"),
            ("measurement_missing", "measurement.json"),
        ],
    )
    def test_capture_failures(pair, case, reason, monkeypatch):
        pair.checkout(pair.base if case == "nonmerge" else pair.merge)
        tested = {"checkout": pair.base, "nonmerge": pair.base, "invalid_sha": "invalid", "non_pr": HEAD}
        monkeypatch.setenv("GITHUB_SHA", tested.get(case, pair.merge))
        options = {"measurement_manifest": pair.tmp / "measurement.json"} if case.startswith("measurement") else {}
        if case == "head":
            pair.event["pull_request"]["head"]["sha"] = HEAD
        elif case == "non_pr":
            pair.event.clear()
        elif case == "modified_source":
            (pair.root / RUNTIME).write_text("changed source")
        elif case == "measurement_commit":
            measurement_file(pair, commit=pair.base)
        resolves = case in ("modified_source", "measurement_commit", "measurement_missing")
        assert pair.invoke("resolve") == (0 if resolves else 1)
        outputs = (pair.tmp / "github-output").read_text()
        assert ("base_commit=" in outputs, "reason=" in outputs) == (resolves, not resolves)
        assert pair.invoke("capture", **options) == 1
        failure_path = pair.output / "paired-failure.json"
        failure = json.loads(failure_path.read_text())
        assert reason in failure["reason"] and not (pair.output / "source-manifest.json").exists()
        assert_fields(failure, stage="capture_current", execution={"run_id": 20, "run_attempt": 1})
        sha = os.environ["GITHUB_SHA"]
        assert_fields(failure["source"], commit=pair.git("rev-parse", "HEAD"), tested_commit=sha, intended_commit=sha)
        before = failure_path.read_bytes()
        monkeypatch.setenv("PERF_BASE_COMMIT", "")
        assert pair.invoke("bind") == 1 and failure_path.read_bytes() == before

    @pytest.mark.parametrize("case", ["image", "image_shape", "context_write", "non_pr"])
    def test_capture_without_optional_provenance(pair, case):
        pair.checkout(pair.merge)
        pair.image = {"image": "not JSON", "image_shape": "{}"}.get(case, pair.image)
        if case == "context_write":
            (pair.output / "build-context.json").mkdir(parents=True)
        elif case == "non_pr":
            pair.event.clear()
        assert pair.invoke("capture") == 0
        manifest = json.loads((pair.output / "source-manifest.json").read_text())
        assert manifest["commit"] == pair.merge and RUNTIME in manifest["files"]
        assert not (pair.output / "paired-failure.json").exists()
        context = pair.output / "build-context.json"
        if case.startswith("image"):
            definition = json.loads(context.read_text())["metric_definition"]
            assert definition["total_fps"] is None and "provenance capture failed" in definition["reason"]
        elif case == "non_pr":
            assert not context.exists()
        else:
            assert context.is_dir()
            assert set(pair.output.iterdir()) == {context, pair.output / "source-manifest.json"}

    def test_fresh_partial_baseline_and_clock_independence(pair, monkeypatch):
        files = artifact_files(pair, run_id=20)
        files["second/sample-3/benchmark_runtime_fixture.json"] = b"{invalid JSON"
        baseline = pair.client.publish(20, baseline_name(42, pair.base), files)
        publish_pair(pair, baseline, reused=False, baseline_finished_before="2026-10-01T08:59:00Z")
        # Base and PR sample timestamps overlap; the producing run and runner identify the fresh pair.
        candidate = resolve_candidate(pair.client, 20, 1)
        selected, selection = select_pr_baseline(pair.client, candidate)
        assert selected is not None and not selection["baseline_reused"]
        rows = {row["legs"]["baseline"][0]: row for row in compare_evidence(selected, candidate)["rows"]}
        assert_fields(rows["first"], status="compared", change_pct=0)
        assert_fields(rows["second"], status="partial", change_pct=None)
        restore(pair)
        issue = "Baseline benchmark failed after producing partial evidence."
        assert pair.invoke("bind", artifact_id=baseline["id"], baseline_issue=issue) == 0
        pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert pin["reason"] == issue and issue in pin["issues"]
        assert_fields(pin["baseline_origin"], artifact_id=baseline["id"], run_id=20, source_commit=pair.base)
        monkeypatch.setenv("PERF_BASE_COMMIT", HEAD)
        assert pair.invoke("bind", output_dir=pair.tmp / "wrong-base") == 1
        assert not (pair.tmp / "wrong-base" / "pr-comparison.json").exists()

    def test_workflow_and_baseline_attempt_lifecycle(pair, monkeypatch):
        pair.client.runs[20, 1]["workflow_id"] = 812345
        pair.client.runs[10, 1]["workflow_id"] = 812345
        old = pair.client.publish(10, baseline_name(42, pair.base), artifact_files(pair))
        pair.client.add_run(11)
        obsolete = pair.client.publish(11, baseline_name(42, pair.base), artifact_files(pair, run_id=11))
        selected = restore(pair)
        assert selected["baseline_origin"]["artifact_id"] == old["id"]
        assert any(
            "/812345/runs" in path and parameters == {"event": "pull_request", "branch": "feature"}
            for path, _, parameters in pair.client.queries
        )
        assert f"Artifact {obsolete['id']}" in str(selected["issues"])
        old["expired"] = obsolete["expired"] = True
        for attempt in (2, 3):
            pair.client.add_run(20, attempt, workflow_id=812345)
        files = artifact_files(pair, run_id=20, attempt=2)
        fresh = pair.client.publish(20, baseline_name(42, pair.base, 2), files)
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        assert not restore(pair)["baseline_reused"]
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
        selected = restore(pair)
        assert_fields(selected["baseline_origin"], artifact_id=fresh["id"], run_id=20, run_attempt=2)
        fresh["expired"] = True
        assert not restore(pair)["baseline_reused"]

    @pytest.mark.parametrize("role", ["current", "baseline"])
    def test_actionable_failure_report(pair, role, monkeypatch):
        pair.checkout(pair.merge)
        if role == "current":
            monkeypatch.setenv("GITHUB_SHA", pair.base)
        assert pair.invoke("capture", role=role, output_dir=pair.tmp / "failed") == 1
        failure = json.loads((pair.tmp / "failed" / "paired-failure.json").read_text())
        files = artifact_files(pair, "current", 20) if role == "baseline" else {"first/status": "error"}
        files["paired-failure.json"] = failure
        files["pr-comparison.json"] = {"reference_commit": pair.base, "tested_commit": HEAD}
        pair.client.publish(20, "performance-regression-20-1", files)
        assert pair.invoke("report") == 0
        report = json.loads((pair.output / "build-comparison.json").read_text())
        text = (pair.output / "build-comparison.md").read_text()
        assert failure["stage"] == f"capture_{role}" and report["selection"]["paired_failure"] == failure
        primary = text.split("<details>", 1)[0]
        assert report["selection"]["reason"] == failure["reason"] and failure["reason"] in primary
        if role == "baseline":
            assert report["candidate_kind"] == "merge"
            assert [(row["candidate"]["median"], row["change_pct"]) for row in report["rows"]] == [(100, None)] * 2
        else:
            assert report["selection"]["unavailable_side"] == "candidate" and report["candidate_kind"] is None
            assert "Intended checkout" in text and "PR revision (not verified)" in text
            assert any("FPS formula provenance are unavailable" in note for note in report["notes"])

    @pytest.mark.parametrize("gate_only", [False, True])
    def test_gate_survives_corrupt_provenance(pair, gate_only):
        from . import cli

        files = artifact_files(pair, "current", 20)
        files["build-context.json"] = b"not JSON"
        files["first/comparison.json"] = {"verdict": "FAIL", "label": "first"}
        artifact = pair.client.publish(20, "performance-regression-20-1", files)
        assert pair.invoke("report", comparison_dir=pair.tmp / "comparisons", gate_only=gate_only) == 0
        assert pair.client.downloaded == [artifact["id"]]
        assert cli.main(["aggregate", "--comparison_dir", str(pair.tmp / "comparisons")]) == 1
        if gate_only:
            assert not pair.output.exists()
        else:
            assert (
                json.loads((pair.output / "build-comparison.json").read_text())["selection"]["reason_code"] == "corrupt"
            )

    def test_current_capture_preserves_baseline_failure(pair, monkeypatch):
        pair.git("checkout", "--quiet", "--detach", pair.merge)
        reason = "The tested merge base could not be resolved."
        monkeypatch.setenv("PERF_BASE_COMMIT", "")
        monkeypatch.setenv("PERF_PAIR_ERROR", reason)
        assert pair.invoke("bind") == 1
        failure_path = pair.output / "paired-failure.json"
        original = json.loads(failure_path.read_text())
        assert original["stage"] == "bind" and original["reason"] == reason and original["source"]["commit"] is None
        for resolved in ("", pair.old):
            monkeypatch.setenv("PERF_BASE_COMMIT", resolved)
            assert pair.invoke("capture", role="current") == 0
            source = json.loads((pair.output / "build-context.json").read_text())["source"]
            assert_fields(source, commit=pair.merge, reference_commit=pair.base, commit_parents=[pair.base, pair.head])
            assert json.loads(failure_path.read_text()) == original
            assert pair.invoke("capture", role="baseline") == 1
        (pair.root / RUNTIME).write_text("changed source")
        assert pair.invoke("capture", role="current") == 1
        original["source"].update(commit=pair.merge, commit_parents=[pair.base, pair.head])
        assert json.loads(failure_path.read_text()) == original

    def test_legacy_baseline_needs_shared_measurement(pair):
        files = artifact_files(pair)
        files["source-manifest.json"].pop("measurement")
        context = files["build-context.json"]
        context.pop("measurement")
        context["metric_definition"]["total_fps"] = "source-fps-v2:legacy"
        context["execution"]["benchmark_protocol"].pop("measurement_sha256")
        for path, proof in files.items():
            if path.endswith("source-revision.json"):
                proof.pop("measurement_digest")
        artifact = pair.client.publish(10, baseline_name(42, pair.base), files)
        selected = _read_baseline(pair.client, artifact, 42, pair.base)
        assert selected.issues == []
        restored = restore(pair)
        assert not restored["baseline_reused"] and "protocol" in str(restored["issues"])
