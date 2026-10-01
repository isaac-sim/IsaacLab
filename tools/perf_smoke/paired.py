# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse a PR's verified base measurement and pin its exact comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import baseline as store
from .metric_identity import metric_definition
from .source_revision import RUNTIME_MODULE, prepare_manifest


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _files(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _json(files: dict[str, bytes], name: str) -> dict:
    if name not in files:
        raise store.EvidenceError("missing_source_evidence", f"The artifact has no {name}.")
    return store._object(files[name], name)


def baseline_name(pr_number: int, commit: str, attempt: int = 1) -> str:
    return f"performance-pr-baseline-{pr_number}-{commit}-{attempt}"


def source_issues(evidence: store.Evidence, files: dict[str, bytes], commit: str) -> list[str]:
    """Check each result against its same-process source proof, retaining per-leg failures."""
    try:
        manifest = _json(files, "source-manifest.json")
        if manifest.get("commit") != commit or not manifest.get("files"):
            raise store.EvidenceError("source_mismatch", "Source manifest does not identify the selected commit.")
    except store.EvidenceError as exc:
        return [f"{leg}/source: {exc}" for leg in evidence.samples.keys() | evidence.statuses.keys()]
    issues = []
    for leg, samples in evidence.samples.items():
        for sample in samples:
            path = sample["path"]
            sidecar_path = str(PurePosixPath(path).parent / "source-revision.json")
            try:
                proof = _json(files, sidecar_path)
                runtime = proof.get("runtime_entrypoint") or {}
                modules = proof.get("modules") or []
                if (
                    proof.get("status") != "verified"
                    or proof.get("bytecode_policy") != "fresh_process_cache"
                    or proof.get("commit") != commit
                    or proof.get("benchmark_exit_code") != 0
                    or proof.get("mismatches") != []
                    or runtime.get("source_code_matches") is not True
                    or runtime.get("module") != RUNTIME_MODULE
                    or not any(module.get("name") == RUNTIME_MODULE for module in modules)
                ):
                    raise store.EvidenceError("source_mismatch", "Executing source was not verified for this commit.")
                for module in modules:
                    expected = manifest["files"].get(module.get("relative_path"))
                    if (
                        not expected
                        or module.get("status") != "verified"
                        or module.get("sha256") != expected
                        or module.get("expected_sha256") != expected
                    ):
                        raise store.EvidenceError("source_mismatch", "Loaded source does not match its manifest.")
                matches = [item for item in proof.get("outputs", []) if item.get("path") == PurePosixPath(path).name]
                content = files[path]
                if (
                    len(matches) != 1
                    or matches[0].get("sha256") != hashlib.sha256(content).hexdigest()
                    or matches[0].get("bytes") != len(content)
                ):
                    raise store.EvidenceError("source_mismatch", "Result bytes do not match their source verification.")
            except (store.EvidenceError, KeyError, TypeError, AttributeError) as exc:
                issues.append(f"{leg}/source: {path}: {exc}")
    return issues


def _read_baseline(client: store.GitHubClient, artifact: dict, pr_number: int, base_commit: str) -> store.Evidence:
    contents = store._zip_files(client, artifact)
    context = _json(contents[1], "build-context.json")
    execution, source = context.get("execution", {}), context.get("source", {})
    if not isinstance(execution, dict) or not isinstance(source, dict):
        raise store.EvidenceError("corrupt", "Baseline source/execution context is not an object.")
    run_id, attempt = execution.get("run_id"), execution.get("run_attempt")
    if (
        source.get("commit") != base_commit
        or source.get("event_base_commit") != base_commit
        or source.get("pull_request_number") != pr_number
        or source.get("benchmark_role") != "baseline"
        or not isinstance(run_id, int)
        or not isinstance(attempt, int)
    ):
        raise store.EvidenceError("identity_mismatch", "Baseline does not belong to this PR and exact base commit.")
    run = client.run_attempt(run_id, attempt)
    if run.get("event") != "pull_request":
        raise store.EvidenceError("identity_mismatch", "Baseline was not produced by a PR workflow.")
    evidence = store.read_evidence(
        client,
        run,
        artifact,
        attempt,
        artifact_name=baseline_name(pr_number, base_commit, attempt),
        contents=contents,
    )
    for leg in execution.get("expected_legs", []):
        evidence.samples.setdefault(leg, [])
    evidence.issues.extend(source_issues(evidence, contents[1], base_commit))
    return evidence


def restore_baseline(
    client: store.GitHubClient,
    checkout_root: Path,
    output_dir: Path,
    selection_path: Path,
    event: dict,
    run_id: int,
    run_attempt: int,
) -> dict:
    """Restore a complete verified measurement for this PR/base, or request a fresh one."""
    pr = event["pull_request"]
    pr_number, base_commit = pr["number"], pr["base"]["sha"]
    manifest = prepare_manifest(checkout_root)
    if manifest["commit"] != base_commit:
        raise ValueError("Baseline checkout is not the PR event's exact base commit.")
    selection = {
        "schema_version": 1,
        "comparison_mode": "paired_pr",
        "pull_request_number": pr_number,
        "reference_branch": pr["base"]["ref"],
        "reference_commit": base_commit,
        "requested_head_commit": pr["head"]["sha"],
        "baseline_reused": False,
        "baseline_origin": None,
        "reason": "No reusable verified baseline exists for this PR and base commit; measure it before the PR.",
        "issues": [],
    }
    prefix = f"performance-pr-baseline-{pr_number}-{base_commit}-"
    try:
        current_run = client.run_attempt(run_id, run_attempt)
        workflow_id = current_run.get("workflow_id")
        if not isinstance(workflow_id, int) or workflow_id <= 0:
            raise store.EvidenceError("identity_mismatch", "The producing workflow identity is unavailable.")
        selection["workflow_id"] = workflow_id
        runs = client.paginate(
            f"/repos/{client.repository}/actions/workflows/{workflow_id}/runs",
            "workflow_runs",
            event="pull_request",
            branch=pr["head"]["ref"],
        )
        artifacts = [
            artifact
            for run in runs
            for artifact in client.artifacts(run["id"])
            if artifact.get("name", "").startswith(prefix)
        ]
        for artifact in sorted(artifacts, key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True):
            if artifact.get("expired"):
                continue
            try:
                evidence = _read_baseline(client, artifact, pr_number, base_commit)
                if evidence.identity.get("workflow_id") != workflow_id:
                    raise store.EvidenceError("identity_mismatch", "Baseline belongs to another workflow.")
                saved_manifest = _json(_files(evidence.zip_bytes), "source-manifest.json")
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
                        and all(store._usable(item["bundle"]) for item in evidence.samples[leg])
                        for leg in evidence.samples.keys() | evidence.statuses.keys()
                    )
                )
                if saved_manifest != manifest or not complete:
                    raise store.EvidenceError(
                        "unverified_baseline", "Baseline is incomplete or its source proof is invalid."
                    )
                # An artifact from a later attempt cannot stand in for this execution.
                if evidence.identity["run_id"] == run_id and evidence.identity["run_attempt"] >= run_attempt:
                    continue
                restored_files = _files(evidence.zip_bytes)
                if any(
                    PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in restored_files
                ):
                    raise store.EvidenceError("corrupt", "Baseline artifact contains an invalid path.")
                for path, data in restored_files.items():
                    destination = output_dir.joinpath(*PurePosixPath(path).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                selection.update(
                    baseline_reused=True,
                    baseline_origin=evidence.identity,
                    reason="Reused the verified baseline measurement for this PR's unchanged base commit.",
                )
                break
            except store.EvidenceError as exc:
                selection["issues"].append(f"Artifact {artifact['id']} was not reused: {exc}")
    except store.EvidenceError as exc:
        selection["issues"].append(f"Baseline lookup unavailable; measuring a fresh baseline: {exc}")
    _write(selection_path, selection)
    return selection


def capture_context(root: Path, output_dir: Path, image_ref: str, role: str, event: dict, legs: Path) -> dict:
    """Record the selected checkout, image and runner before its measurement."""
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    parents = [
        line.split()[1]
        for line in subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "-p", "HEAD"],
            text=True,
        )
        .split("\n\n", 1)[0]
        .splitlines()
        if line.startswith("parent ")
    ]
    pr = event.get("pull_request", {})
    if pr:
        intended = pr["base"]["sha"] if role == "baseline" else os.environ["GITHUB_SHA"]
        if commit != intended:
            raise ValueError(f"{role} checkout does not match its intended source revision.")
        if role == "current" and (len(parents) != 2 or parents != [pr["base"]["sha"], pr["head"]["sha"]]):
            raise ValueError("Tested PR merge parents do not match the event's base and head commits.")
    inspected = subprocess.run(["docker", "image", "inspect", image_ref], capture_output=True, text=True)
    image = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else {}
    context = {
        "schema_version": 1,
        "source": {
            "commit": commit,
            "requested_head_commit": pr.get("head", {}).get("sha") or os.environ["GITHUB_SHA"],
            "event": os.environ["GITHUB_EVENT_NAME"],
            "reference_branch": pr.get("base", {}).get("ref") or os.environ["GITHUB_REF_NAME"],
            "reference_commit": pr.get("base", {}).get("sha") if pr else (parents[0] if parents else None),
            "commit_parents": parents,
            "event_base_commit": pr.get("base", {}).get("sha"),
            "pull_request_number": pr.get("number"),
            "benchmark_role": role,
            "image_ref": image_ref,
            "image_digest": next(iter(image.get("RepoDigests", [])), None),
            "image_id": image.get("Id"),
            "provenance": "ci_checkout",
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
        },
        "metric_definition": {
            "total_fps": "aggregate_frames_over_measured_seconds",
            "producer_commit": commit,
            "provenance": "ci_producer_source",
        },
    }
    if pr:
        context["metric_definition"] = {**metric_definition(root), "producer_commit": commit}
    _write(output_dir / "build-context.json", context)
    _write(output_dir / "source-manifest.json", prepare_manifest(root))
    return context


def bind_baseline(
    selection_path: Path,
    artifact_id: str | None,
    output_dir: Path,
    repository: str,
    baseline_issue: str = "",
) -> None:
    """Pin the uploaded baseline before the current PR is measured."""
    selection = json.loads(selection_path.read_text())
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


def select_pr_baseline(client: store.GitHubClient, candidate: store.Evidence) -> store.Selection:
    """Read the exact base measurement pinned by this PR benchmark job, without ancestry search."""
    files = _files(candidate.zip_bytes)
    source = candidate.context.get("source", {})
    for leg in candidate.context.get("execution", {}).get("expected_legs", []):
        candidate.samples.setdefault(leg, [])
    candidate.issues.extend(source_issues(candidate, files, candidate.identity["source_commit"]))
    try:
        selection = _json(files, "pr-comparison.json")
    except store.EvidenceError as exc:
        return store.Selection(None, {"reason": str(exc), "reason_code": exc.code})
    base_commit, pr_number = source.get("event_base_commit"), source.get("pull_request_number")
    if (
        selection.get("reference_commit") != base_commit
        or selection.get("pull_request_number") != pr_number
        or selection.get("tested_commit") != candidate.identity["source_commit"]
        or selection.get("requested_head_commit") != source.get("requested_head_commit")
    ):
        raise store.EvidenceError(
            "identity_mismatch", "The pinned comparison does not identify this tested PR revision."
        )
    origin = selection.get("baseline_origin")
    if not origin:
        return store.Selection(None, selection)
    try:
        artifact = client.get(f"/repos/{client.repository}/actions/artifacts/{origin['artifact_id']}")
        baseline = _read_baseline(client, artifact, pr_number, base_commit)
        for field in ("artifact_id", "run_id", "run_attempt", "source_commit", "sha256"):
            if field in origin and origin[field] != baseline.identity[field]:
                raise store.EvidenceError("identity_mismatch", "Pinned baseline identity or bytes changed.")
        if not selection.get("baseline_reused"):
            left, right = baseline.context["execution"], candidate.context["execution"]
            for field in ("run_id", "run_attempt", "job", "runner_name", "hostname"):
                if not left.get(field) or left.get(field) != right.get(field):
                    raise store.EvidenceError(
                        "runner_mismatch", "Fresh base and PR were not measured on the same runner."
                    )
            if (
                not baseline.measurement_end
                or not candidate.measurement_start
                or (store._time(baseline.measurement_end) > store._time(candidate.measurement_start))
            ):
                raise store.EvidenceError(
                    "measurement_order", "Fresh baseline did not finish before the PR measurement."
                )
        selection.update(baseline_origin=baseline.identity, selected=baseline.identity)
        return store.Selection(baseline, selection)
    except store.EvidenceError as exc:
        selection.update(
            reason=str(exc), reason_code=exc.code, unavailable_evidence=origin, unavailable_side="baseline"
        )
        return store.Selection(None, selection)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--checkout-root", type=Path, required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    restore.add_argument("--selection", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--checkout-root", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--image", required=True)
    capture.add_argument("--role", choices=("baseline", "current"), required=True)
    capture.add_argument("--legs", type=Path, required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("--selection", type=Path, required=True)
    bind.add_argument("--output-dir", type=Path, required=True)
    bind.add_argument("--artifact-id")
    bind.add_argument("--baseline-issue", default="")
    args = parser.parse_args(argv)
    if args.command == "restore":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        selection = restore_baseline(
            store.GitHubClient(os.environ["GITHUB_REPOSITORY"]),
            args.checkout_root,
            args.output_dir,
            args.selection,
            event,
            int(os.environ["GITHUB_RUN_ID"]),
            int(os.environ["GITHUB_RUN_ATTEMPT"]),
        )
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"reused={str(selection['baseline_reused']).lower()}\n")
            name = baseline_name(
                selection["pull_request_number"], selection["reference_commit"], int(os.environ["GITHUB_RUN_ATTEMPT"])
            )
            output.write(f"artifact_name={name}\n")
        print(selection["reason"])
        for issue in selection["issues"]:
            print(issue)
    elif args.command == "capture":
        capture_context(
            args.checkout_root,
            args.output_dir,
            args.image,
            args.role,
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
            args.legs,
        )
    else:
        if not args.selection.exists():
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            pr = event["pull_request"]
            _write(
                args.selection,
                {
                    "schema_version": 1,
                    "comparison_mode": "paired_pr",
                    "pull_request_number": pr["number"],
                    "reference_branch": pr["base"]["ref"],
                    "reference_commit": pr["base"]["sha"],
                    "requested_head_commit": pr["head"]["sha"],
                    "baseline_reused": False,
                    "baseline_origin": None,
                    "issues": [],
                },
            )
            if not args.baseline_issue:
                args.baseline_issue = "Baseline selection metadata was not produced; baseline FPS is unavailable."
        bind_baseline(
            args.selection, args.artifact_id, args.output_dir, os.environ["GITHUB_REPOSITORY"], args.baseline_issue
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
