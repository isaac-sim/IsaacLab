# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paired PR evidence reuse and selection using immutable local source fixtures."""

import contextlib
import copy
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from . import baseline, build_compare, cli, paired, source_revision
from .test_baseline import HEAD, REPO, FixtureClient, bundle, stamp

RUNTIME = "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py"


def encoded(value):
    return json.dumps(value).encode()


class PairedFixtureClient(FixtureClient):
    def __init__(self):
        super().__init__()
        self.pages_requested = []
        self.attempts_requested = []

    def run_attempt(self, run_id, attempt):
        self.attempts_requested.append((run_id, attempt))
        return super().run_attempt(run_id, attempt)

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
            raise baseline.EvidenceError("inaccessible", "Fixture artifact is unavailable")
        return matches[0]


from .test_build_compare import assert_fields, check_comparisons, check_reports, run_report
from .test_compare import check_rolling_history


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
        environment = patch.dict(os.environ, {"PERF_BASE_COMMIT": self.commit})
        stack.enter_context(environment)
        self.manifest = source_revision.prepare_manifest(self.checkout)
        measurement_files = {RUNTIME: hashlib.sha256(b"def run(argv): return 'PR measurement'\n").hexdigest()}
        self.measurement = {
            "schema_version": 1,
            "commit": HEAD,
            "files": measurement_files,
            "digest": source_revision.measurement_digest(measurement_files),
        }
        self.client = PairedFixtureClient()
        self.output = self.root / "baseline output"
        self.selection_path = self.root / "pr-comparison.json"
        self.legs = self.root / "benchmark-legs.tsv"
        self.legs.write_text("first|first|512|600|physics=newton_mjwarp\nsecond|second|512|600|physics=newton_mjwarp\n")
        self.controller = self.root / "controller"
        self.controller.mkdir()
        for name in ("run_benchmarks.sh", "source_revision.py"):
            (self.controller / name).write_bytes(Path(paired.__file__).with_name(name).read_bytes())
        self.protocol = {
            "matrix_sha256": hashlib.sha256(self.legs.read_bytes()).hexdigest(),
            "benchmark_launcher_sha256": hashlib.sha256(
                (self.controller / "run_benchmarks.sh").read_bytes()
            ).hexdigest(),
            "source_launcher_sha256": hashlib.sha256((self.controller / "source_revision.py").read_bytes()).hexdigest(),
        }
        controller = patch.object(paired, "__file__", str(self.controller / "paired.py"))
        stack.enter_context(controller)
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
        event_path.write_bytes(encoded(self.event))
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
                [sys.executable, "-m", "tools.perf_smoke.paired", *args],
                cwd=Path(__file__).resolve().parents[2],
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
                data = encoded(value)
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
        # Keep artifact documents independent before serialization.
        return {name: copy.deepcopy(value) for name, value in files.items()}

    def _add_baseline(self, files=None, *, run_id=10, attempt=1):
        self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return self.client.add_artifact(
            run_id,
            paired.baseline_name(42, self.commit, attempt),
            self._files(run_id, attempt) if files is None else files,
        )

    def _restore(self, *, run_id=20, attempt=1, measurement=None):
        if (run_id, attempt) not in self.client.attempts:
            self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return paired.restore_baseline(
            self.client,
            self.checkout,
            self.selection_path,
            self.event,
            run_id,
            attempt,
            legs=self.legs,
            base_commit=self._git("rev-parse", "HEAD"),
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
            "comparison_mode": "paired_pr",
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
        return baseline.read_evidence(self.client, run, artifact, 1)

    def _origin(self, artifact, *, run_id=20):
        return {
            "repository": REPO,
            "artifact_id": artifact["id"],
            "run_id": run_id,
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
    with paired_fixture(tmp_path, subtests, "merge first parent") as pair:
        base, _, merge, latest = pair._moving_merge()
        assert base != pair.event["pull_request"]["base"]["sha"]
        assert base != latest
        result = pair._cli(["resolve", "--checkout-root", str(pair.checkout)], GITHUB_SHA=merge)
        assert result.returncode == 0, result.stderr + result.stdout
        outputs = dict(line.split("=", 1) for line in (pair.root / "github-output").read_text().splitlines())
        assert outputs == {"base_commit": base}
    with paired_fixture(tmp_path, subtests, "capture optional metadata") as pair:
        base, head, merge, _ = pair._moving_merge()
        original_run = subprocess.run
        original_write = paired._write
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
            patch.object(paired.subprocess, "run", side_effect=inspect_or_run),
            patch.object(paired, "_write", side_effect=write_context),
            patch.object(paired.socket, "gethostname", return_value="fixture-gpu-host"),
        ):
            for role, commit in (("baseline", base), ("current", merge)):
                pair._git("checkout", "--quiet", "--detach", commit)
                output = pair.root / role
                context = paired.capture_context(pair.checkout, output, "same-image", role, pair.event, pair.legs)
                source = context["source"]
                assert source["commit"] == commit
                assert source["reference_commit"] == base
                assert source["event_base_commit"] == pair.commit
                assert source["requested_head_commit"] == head
                assert json.loads((output / "source-manifest.json").read_text())["commit"] == commit
                assert context["execution"]["benchmark_protocol"] == pair.protocol
                saved = json.loads((output / "build-context.json").read_text())
                assert_fields(saved["execution"], hostname="fixture-gpu-host", benchmark_protocol=pair.protocol)
                if role == "current":
                    assert source["commit_parents"] == [base, head]
            for failure in ("image", "image_shape", "context_write"):
                with subtests.test(optional_failure=failure):
                    output = pair.root / failure
                    result = paired.main(pair._capture_args(output))
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
                    assert paired.main(args) == 0
                    manifest = json.loads((output / "source-manifest.json").read_text())
                    context = json.loads((output / "build-context.json").read_text())
                    assert_fields(manifest, commit=commit, measurement=measurement)
                    assert context["source"]["commit"] == commit
                    assert context["measurement"] == {"commit": merge, "digest": measurement["digest"]}
                    assert context["metric_definition"]["total_fps"] == f"measurement-v1:{measurement['digest']}"
                    assert context["execution"]["benchmark_protocol"]["measurement_sha256"] == measurement["digest"]
                (snapshot / "measurement-manifest.json").write_bytes(encoded({**measurement, "commit": base}))
                assert paired.main(args) == 1
                assert (
                    "Measurement source does not identify"
                    in json.loads((output / "paired-failure.json").read_text())["reason"]
                )
                (output / "paired-failure.json").unlink()
                (snapshot / "measurement-manifest.json").unlink()
                assert paired.main(args) == 1
                assert "measurement-manifest.json" in json.loads((output / "paired-failure.json").read_text())["reason"]
    with paired_fixture(tmp_path, subtests, "non-PR capture") as pair:
        with (
            patch.dict(os.environ, {"GITHUB_SHA": pair.commit}),
            patch.object(paired.subprocess, "run", wraps=subprocess.run) as commands,
        ):
            context = paired.capture_context(pair.checkout, pair.output, "same-image", "current", {}, pair.legs)
            assert context == {}
            assert json.loads((pair.output / "source-manifest.json").read_text()) == pair.manifest
            assert not (pair.output / "build-context.json").exists()
            assert not any(call.args[0][0] == "docker" for call in commands.call_args_list)
            with patch.dict(os.environ, {"GITHUB_SHA": HEAD}), pytest.raises(ValueError, match="expected commit"):
                paired.capture_context(pair.checkout, pair.output, "same-image", "current", {}, pair.legs)
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

        with pair._workflow(**env), patch.object(paired.subprocess, "run", side_effect=inspect_or_run):
            for resolved in ("", pair.commit):
                with subtests.test(resolved=resolved), patch.dict(os.environ, {"PERF_BASE_COMMIT": resolved}):
                    context = paired.capture_context(
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
                        paired.capture_context(
                            pair.checkout, pair.output, "same-image", "baseline", pair.event, pair.legs
                        )
            (pair.checkout / RUNTIME).write_text("def run(argv): return 'modified'\n")
            invalid = pair.root / "modified-source"
            with pytest.raises(ValueError, match="Checked-out Python source differs"):
                paired.capture_context(pair.checkout, invalid, "same-image", "current", pair.event, pair.legs)
            assert not (invalid / "source-manifest.json").exists()
            captured = paired.main(pair._capture_args(pair.output))
            assert captured != 0
            expected = copy.deepcopy(original)
            expected["source"].update(commit=merge, commit_parents=[base, head])
            assert json.loads(path.read_text()) == expected


def check_reuse(tmp_path, subtests):
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
            paired.bind_baseline(pair.selection_path, None, pair.output, REPO, "Base image config failed.")
        selection = json.loads((pair.output / "pr-comparison.json").read_text())
        assert selection["baseline_origin"] == restored["baseline_origin"]
        assert selection["reason"] == restored["reason"]
        assert selection["issues"] == restored["issues"]
        candidate = pair._candidate(
            restored["baseline_origin"], reused=True, selection=selection, measurement=measurement
        )
        evidence, _ = paired.select_pr_baseline(pair.client, candidate)
        assert evidence.issues == []
        assert evidence.identity["sha256"] == hashlib.sha256(original).hexdigest()
        assert candidate.issues == []
        report = build_compare.compare_evidence(evidence, candidate)
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
        candidate = pair._candidate(pair._origin(artifact, run_id=10), reused=True, hostname="different-gpu-host")
        candidate.context["source"]["event_base_commit"] = event_base
        evidence, metadata = paired.select_pr_baseline(pair.client, candidate)
        assert_fields(evidence.identity, source_commit=pair.commit, run_id=10)
        assert metadata["baseline_reused"]
    with paired_fixture(tmp_path, subtests, "unresolved reference") as pair:
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": ""}):
            with pytest.raises(ValueError, match="first parent was not resolved"):
                paired.restore_baseline(
                    pair.client, pair.checkout, pair.selection_path, pair.event, 20, 1, legs=pair.legs
                )
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
                raise baseline.EvidenceError("inaccessible", "GitHub evidence request failed (HTTP 404)")
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
                pair.client = PairedFixtureClient()
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
                candidate = pair._candidate(pair._origin(artifact, run_id=10), reused=True)
                evidence, _ = paired.select_pr_baseline(pair.client, candidate)
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
                pair.client = PairedFixtureClient()
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
                candidate = pair._candidate(
                    pair._origin(artifact, run_id=10), reused=True, measurement=pair.measurement
                )
                evidence, _ = paired.select_pr_baseline(pair.client, candidate)
                assert evidence.issues
                report = build_compare.compare_evidence(evidence, candidate)
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
                pair.client = PairedFixtureClient()
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
    with paired_fixture(tmp_path, subtests, "report lifecycle and gate") as pair:
        artifact = pair._add_baseline(pair._files(measurement=pair.measurement))
        candidate = pair._candidate(
            pair._origin(artifact, run_id=10), reused=True, measurement=pair.measurement, fps=80
        )
        files = {**candidate.files, "first/comparison.json": encoded({"verdict": "FAIL", "label": "first"})}
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
            assert_fields(second["candidate"], run_attempt=1, report_attempt=2)
            assert "Measurements: attempt 1; report: attempt 2" in rerun_markdown
            artifact["expired"] = True
            _, result, markdown = pair._run_report(attempt=2)
            assert result["baseline"] is None
            assert result["selection"]["reason_code"] == "expired"
            assert result["selection"]["unavailable_evidence"]["artifact_id"] == artifact["id"]
            assert all(row["candidate"]["median"] == 80 for row in result["rows"])
            assert all(row["change_pct"] is None for row in result["rows"])
            assert "PR base (evidence unavailable)" in markdown
            assert "/actions/runs/10/attempts/1" in markdown
            assert "expired" in markdown
            pair._add_baseline(pair._files(run_id=11, measurement=pair.measurement), run_id=11)
            pair.client.add(20, HEAD, attempt=3, start=stamp(14), event="pull_request", artifact=False)
            pair.client.run_jobs[20, 3][0]["started_at"] = stamp(12)
            _, third, third_markdown = pair._run_report(attempt=3)
            assert third["selection"]["unavailable_evidence"] == result["selection"]["unavailable_evidence"]
            assert third["selection"]["reason_code"] == "expired"
            assert "/actions/runs/10/attempts/1" in third_markdown
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
    with paired_fixture(tmp_path, subtests, "parent mismatch") as pair:
        candidate = pair._candidate({"artifact_id": 123})
        downloaded = list(pair.client.downloaded)
        candidate.context["source"]["commit_parents"] = ["f" * 40, HEAD]
        with pytest.raises(baseline.EvidenceError, match="does not identify this tested PR revision"):
            paired.select_pr_baseline(pair.client, candidate)
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
                pair.client = PairedFixtureClient()
                artifact = pair._add_baseline(run_id=20)
                origin = pair._origin(artifact)
                if case == "pinned_digest":
                    origin["sha256"] = "0" * 64
                candidate = pair._candidate(origin, **arguments)
                evidence, metadata = paired.select_pr_baseline(pair.client, candidate)
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
            paired.bind_baseline(pair.selection_path, str(artifact["id"]), pair.output, REPO, issue)
        pin = json.loads((pair.output / "pr-comparison.json").read_text())
        assert pin["reason"] == issue
        assert issue in pin["issues"]
        assert pin["baseline_origin"]["artifact_id"] == artifact["id"]
        candidate = pair._candidate(origin, selection=pin)
        evidence, _ = paired.select_pr_baseline(pair.client, candidate)
        assert evidence is not None
        report = build_compare.compare_evidence(evidence, candidate)
        by_leg = {row["legs"]["baseline"][0]: row for row in report["rows"]}
        assert_fields(by_leg["first"], status="compared", change_pct=0)
        assert by_leg["second"]["status"] == "partial"
        assert by_leg["second"]["change_pct"] is None
    with paired_fixture(tmp_path, subtests, "bind wrong base") as pair:
        pair._restore()
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": "f" * 40, "GITHUB_SHA": HEAD}):
            with pytest.raises(ValueError, match="selection does not match"):
                paired.bind_baseline(pair.selection_path, None, pair.output, REPO)
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
                    assert paired.main(arguments) == 0
                pin = json.loads((pair.output / "pr-comparison.json").read_text())
                candidate = pair._candidate(pin["baseline_origin"], selection=pin)
                evidence, metadata = paired.select_pr_baseline(pair.client, candidate)
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
    from .test_baseline import check_evidence, check_transport

    check_rolling_history(tmp_path, subtests)
    registry = Path(__file__).resolve().parents[2] / ".github/actions/_lib/test_registry_credential_fallback.py"
    runpy.run_path(str(registry))["check_dependency_cache"](tmp_path / "cache", subtests)
    check_evidence(subtests)
    check_transport(subtests)
    check_comparisons(subtests)
    check_reports(tmp_path, subtests)
    check_capture(tmp_path, subtests)
    check_reuse(tmp_path, subtests)
    check_pair_reports(tmp_path, subtests)
    check_binding(tmp_path, subtests)
