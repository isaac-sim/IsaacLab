# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse a PR's verified base measurement and pin its exact comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import baseline as store
from .metric_identity import metric_definition
from .source_revision import RUNTIME_MODULE, prepare_manifest


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
    header = subprocess.check_output(["git", "-C", str(root), "cat-file", "-p", commit], text=True)
    parents = [line.split()[1] for line in header.split("\n\n", 1)[0].splitlines() if line.startswith("parent ")]
    return commit, parents


def resolve_base(root: Path, event: dict) -> dict:
    """Resolve A from the immutable tested merge, retaining the event's original base."""
    commit, parents = _checkout_source(root)
    tested = os.environ.get("GITHUB_SHA")
    pr = event["pull_request"]
    head = pr["head"]["sha"]
    if not store.is_commit_sha(tested) or commit != tested:
        raise ValueError("Checked-out PR revision does not match the immutable tested merge GITHUB_SHA.")
    if len(parents) != 2 or parents[1] != head:
        raise ValueError("Tested PR merge does not have the event's head commit as its second parent.")
    return {
        "base_commit": parents[0],
        "tested_commit": commit,
        "requested_head_commit": head,
        "event_base_commit": pr["base"]["sha"],
    }


def _resolved_base(commit: str | None = None) -> str:
    commit = os.environ.get("PERF_BASE_COMMIT") if commit is None else commit
    if not store.is_commit_sha(commit):
        detail = os.environ.get("PERF_PAIR_ERROR")
        raise ValueError(detail or "The tested PR merge's first parent was not resolved; PERF_BASE_COMMIT is missing.")
    return commit


def source_issues(evidence: store.Evidence, files: dict[str, bytes], commit: str) -> list[str]:
    """Check each result against its same-process source proof, retaining per-leg failures."""
    try:
        manifest = store.artifact_json(files, "source-manifest.json")
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
                proof = store.artifact_json(files, sidecar_path)
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
    contents = store.download_artifact(client, artifact)
    context = store.artifact_json(contents[1], "build-context.json")
    execution, source = context.get("execution", {}), context.get("source", {})
    if not isinstance(execution, dict) or not isinstance(source, dict):
        raise store.EvidenceError("corrupt", "Baseline source/execution context is not an object.")
    run_id, attempt = execution.get("run_id"), execution.get("run_attempt")
    if (
        source.get("commit") != base_commit
        or source.get("reference_commit") != base_commit
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


def _initial_selection(event: dict, base_commit: str) -> dict:
    pr = event["pull_request"]
    return {
        "schema_version": 1,
        "comparison_mode": "paired_pr",
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
    client: store.GitHubClient,
    checkout_root: Path,
    output_dir: Path,
    selection_path: Path,
    event: dict,
    run_id: int,
    run_attempt: int,
    *,
    base_commit: str | None = None,
) -> dict:
    """Restore a complete verified measurement for this PR/base, or request a fresh one."""
    pr = event["pull_request"]
    pr_number, base_commit = pr["number"], _resolved_base(base_commit)
    manifest = prepare_manifest(checkout_root)
    if manifest["commit"] != base_commit:
        raise ValueError("Baseline checkout is not the tested PR merge's resolved first parent.")
    selection = _initial_selection(event, base_commit)
    selection["reason"] = "No reusable verified baseline exists for this PR and base commit; measure it before the PR."
    # A terminated lookup leaves a valid request for fresh measurements, never an old reuse decision.
    _write(selection_path, selection)
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
                saved_manifest = store.artifact_json(evidence.files, "source-manifest.json")
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
                        and all(store.has_usable_runtime(item["bundle"]) for item in evidence.samples[leg])
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
                restored_files = evidence.files
                if any(
                    PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in restored_files
                ):
                    raise store.EvidenceError("corrupt", "Baseline artifact contains an invalid path.")
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="baseline-restore-", dir=output_dir.parent) as directory:
                    staged = Path(directory) / "results"
                    staged.mkdir()
                    for path, data in restored_files.items():
                        destination = staged.joinpath(*PurePosixPath(path).parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(data)
                    # Publish only a complete restore; never mix its samples into an existing directory.
                    if output_dir.exists():
                        output_dir.replace(Path(directory) / "previous")
                    staged.replace(output_dir)
                selection.update(
                    baseline_reused=True,
                    baseline_origin=evidence.identity,
                    baseline_metric_definition={
                        "artifact_id": evidence.identity["artifact_id"],
                        "sha256": evidence.identity["sha256"],
                        "source_commit": base_commit,
                        "definition": metric_definition(checkout_root),
                        "provenance": "verified_baseline_checkout",
                    },
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
    commit, parents = _checkout_source(root)
    pr = event.get("pull_request", {})
    reference_commit = _resolved_base() if pr else (parents[0] if parents else None)
    if pr:
        intended = reference_commit if role == "baseline" else os.environ["GITHUB_SHA"]
        if commit != intended:
            raise ValueError(f"{role} checkout does not match its intended source revision.")
        if role == "current" and parents != [reference_commit, pr["head"]["sha"]]:
            raise ValueError("Tested PR merge parents do not match the resolved first parent and event head commit.")
    inspected = subprocess.run(["docker", "image", "inspect", image_ref], capture_output=True, text=True)
    image = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else {}
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
    *,
    reuse_confirmed: bool | None = None,
) -> None:
    """Pin the uploaded baseline before the current PR is measured."""
    selection = json.loads(selection_path.read_text())
    if selection.get("reference_commit") != _resolved_base():
        raise ValueError("Baseline selection does not match the tested PR merge's resolved first parent.")
    if artifact_id or reuse_confirmed is False:
        # The workflow may measure fresh A if restore ended before publishing
        # its reuse output. A failed fresh upload must not resurrect the old pin.
        selection["baseline_reused"] = False
        selection["baseline_origin"] = None
        selection.pop("baseline_metric_definition", None)
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
            # Binding runs after base archival and before current capture/measurement.
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


def select_pr_baseline(client: store.GitHubClient, candidate: store.Evidence) -> tuple[store.Evidence | None, dict]:
    """Read the exact base measurement pinned by this PR benchmark job, without ancestry search."""
    files = candidate.files
    source = candidate.context.get("source", {})
    for leg in candidate.context.get("execution", {}).get("expected_legs", []):
        candidate.samples.setdefault(leg, [])
    candidate.issues.extend(source_issues(candidate, files, candidate.identity["source_commit"]))
    try:
        selection = store.artifact_json(files, "pr-comparison.json")
    except store.EvidenceError as exc:
        return None, {"reason": str(exc), "reason_code": exc.code}
    base_commit, pr_number = source.get("reference_commit"), source.get("pull_request_number")
    if (
        not store.is_commit_sha(base_commit)
        or source.get("commit_parents") != [base_commit, source.get("requested_head_commit")]
        or selection.get("reference_commit") != base_commit
        or selection.get("pull_request_number") != pr_number
        or selection.get("tested_commit") != candidate.identity["source_commit"]
        or selection.get("requested_head_commit") != source.get("requested_head_commit")
    ):
        raise store.EvidenceError(
            "identity_mismatch", "The pinned comparison does not identify this tested PR revision."
        )
    origin = selection.get("baseline_origin")
    if not origin:
        return None, selection
    try:
        artifact = client.get(f"/repos/{client.repository}/actions/artifacts/{origin['artifact_id']}")
        baseline = _read_baseline(client, artifact, pr_number, base_commit)
        for field in ("artifact_id", "run_id", "run_attempt", "source_commit", "sha256"):
            if field in origin and origin[field] != baseline.identity[field]:
                raise store.EvidenceError("identity_mismatch", "Pinned baseline identity or bytes changed.")
        derived = selection.get("baseline_metric_definition")
        if derived is not None and (
            not isinstance(derived, dict)
            or derived.get("provenance") != "verified_baseline_checkout"
            or not isinstance(derived.get("definition"), dict)
            or any(
                derived.get(field) != baseline.identity[field] for field in ("artifact_id", "sha256", "source_commit")
            )
        ):
            raise store.EvidenceError(
                "identity_mismatch", "Derived FPS identity does not identify this baseline artifact."
            )
        if not selection.get("baseline_reused"):
            left, right = baseline.context["execution"], candidate.context["execution"]
            for field in ("run_id", "run_attempt", "job", "runner_name", "hostname"):
                if not left.get(field) or left.get(field) != right.get(field):
                    raise store.EvidenceError(
                        "runner_mismatch", "Fresh base and PR were not measured on the same runner."
                    )
            bound = selection.get("baseline_finished_before")
            completed = store.parse_timestamp(bound if bound is not None else baseline.measurement_end)
            started = store.parse_timestamp(candidate.measurement_start)
            captured = store.parse_timestamp(left.get("measurement_not_before"))
            sample_ends = [
                end
                for samples in baseline.samples.values()
                for sample in samples
                if (end := store.parse_timestamp(sample["bundle"]["run"].get("end_time_utc"))) is not None
            ]
            if (
                not completed
                or not started
                or completed > started
                or (captured is not None and captured > completed)
                or any(end > completed for end in sample_ends)
            ):
                raise store.EvidenceError(
                    "measurement_order", "Fresh baseline did not finish before the PR measurement."
                )
        selection.update(baseline_origin=baseline.identity, selected=baseline.identity)
        return baseline, selection
    except store.EvidenceError as exc:
        selection.update(
            reason=str(exc), reason_code=exc.code, unavailable_evidence=origin, unavailable_side="baseline"
        )
        return None, selection


def _execute(args: argparse.Namespace, event: dict) -> None:
    if args.command == "resolve":
        resolved = resolve_base(args.checkout_root, event)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for field in ("base_commit", "tested_commit", "requested_head_commit"):
                output.write(f"{field}={resolved[field]}\n")
        print(f"Resolved tested PR merge {resolved['tested_commit']} first parent: {resolved['base_commit']}")
    elif args.command == "restore":
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
            event,
            args.legs,
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
            reuse_confirmed=(os.environ["PERF_BASELINE_REUSED"] == "true")
            if "PERF_BASELINE_REUSED" in os.environ
            else None,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--checkout-root", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path)
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
