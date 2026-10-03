# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paired PR evidence reuse and selection using immutable local source fixtures."""

import hashlib
import json
import marshal
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from . import baseline, build_compare, paired, source_revision
from . import test_metric_identity as metric_fixtures
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


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.checkout = self.root / "base checkout"
        self.checkout.mkdir()
        self._git("init", "--quiet")
        source = self.checkout / RUNTIME
        source.parent.mkdir(parents=True)
        source.write_text('def run(argv):\n    """Selected base runtime."""\n    return argv\n')
        (self.checkout / "source/isaaclab/isaaclab/__init__.py").write_text("")
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Base"
        )
        self.commit = self._git("rev-parse", "HEAD")
        environment = patch.dict(os.environ, {"PERF_BASE_COMMIT": self.commit})
        environment.start()
        self.addCleanup(environment.stop)
        self.manifest = source_revision.prepare_manifest(self.checkout)
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
        controller.start()
        self.addCleanup(controller.stop)
        self.event = {
            "number": 42,
            "pull_request": {
                "number": 42,
                "base": {"sha": self.commit, "ref": "develop", "repo": {"full_name": REPO}},
                "head": {"sha": HEAD, "ref": "feature", "repo": {"full_name": REPO}},
            },
        }

    def _git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.checkout), *args], text=True).strip()

    def _commit_file(self, name, content):
        (self.checkout / name).write_text(content)
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", name
        )
        return self._git("rev-parse", "HEAD")

    def _moving_merge(self):
        self._git("checkout", "--quiet", "-b", "fixture-pr")
        head = self._commit_file("source/isaaclab/isaaclab/pr.py", "VALUE = 'PR'\n")
        self._git("checkout", "--quiet", "-b", "fixture-target", self.commit)
        base = self._commit_file("source/isaaclab/isaaclab/base.py", "VALUE = 'Advanced base'\n")
        self._git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "merge",
            "--quiet",
            "--no-ff",
            "-m",
            "Tested merge",
            head,
        )
        merge = self._git("rev-parse", "HEAD")
        latest = self._commit_file("source/isaaclab/isaaclab/later.py", "VALUE = 'Later target'\n")
        self._git("checkout", "--quiet", "--detach", merge)
        self.event["pull_request"]["head"]["sha"] = head
        return base, head, merge, latest

    def _cli(self, args, **environment):
        event_path = self.root / "event.json"
        event_path.write_bytes(encoded(self.event))
        env = {
            **os.environ,
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_OUTPUT": str(self.root / "github-output"),
            "GITHUB_RUN_ID": "20",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": REPO,
            **environment,
        }
        return subprocess.run(
            [sys.executable, "-m", "tools.perf_smoke.paired", *args],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_resolve_immutable_merge_first_parent_despite_stale_event_and_moving_target(self):
        base, head, merge, latest = self._moving_merge()
        self.assertNotEqual(base, self.event["pull_request"]["base"]["sha"])
        self.assertNotEqual(base, latest)
        result = self._cli(["resolve", "--checkout-root", str(self.checkout)], GITHUB_SHA=merge)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        outputs = dict(line.split("=", 1) for line in (self.root / "github-output").read_text().splitlines())
        self.assertEqual(outputs, {"base_commit": base, "tested_commit": merge, "requested_head_commit": head})

    def test_resolve_rejects_wrong_checkout_head_or_nonmerge_and_emits_reason(self):
        base, _, merge, _ = self._moving_merge()
        for mismatch in ("checkout", "head", "nonmerge"):
            with self.subTest(mismatch=mismatch):
                self._git("checkout", "--quiet", "--detach", base if mismatch == "nonmerge" else merge)
                event = json.loads(encoded(self.event))
                if mismatch == "head":
                    self.event["pull_request"]["head"]["sha"] = HEAD
                result = self._cli(
                    ["resolve", "--checkout-root", str(self.checkout)],
                    GITHUB_SHA=base if mismatch in ("checkout", "nonmerge") else merge,
                )
                self.event = event
                self.assertNotEqual(result.returncode, 0)
                outputs = (self.root / "github-output").read_text()
                self.assertIn("reason=", outputs)
                self.assertNotIn("base_commit=", outputs)

    def test_capture_uses_resolved_parent_and_preserves_original_event_base_for_both_sides(self):
        base, head, merge, _ = self._moving_merge()
        original_run = subprocess.run

        def inspect_or_run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0, '[{"Id":"same-image","RepoDigests":[]}]', "")
            return original_run(command, **kwargs)

        with (
            patch.dict(
                os.environ,
                {
                    "PERF_BASE_COMMIT": base,
                    "GITHUB_SHA": merge,
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_RUN_ID": "20",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_JOB": "performance-smoke-benchmarks",
                },
            ),
            patch.object(paired.subprocess, "run", side_effect=inspect_or_run),
        ):
            for role, commit in (("baseline", base), ("current", merge)):
                self._git("checkout", "--quiet", "--detach", commit)
                output = self.root / role
                context = paired.capture_context(self.checkout, output, "same-image", role, self.event, self.legs)
                source = context["source"]
                self.assertEqual(source["commit"], commit)
                self.assertEqual(source["reference_commit"], base)
                self.assertEqual(source["event_base_commit"], self.commit)
                self.assertEqual(source["requested_head_commit"], head)
                self.assertEqual(json.loads((output / "source-manifest.json").read_text())["commit"], commit)
                self.assertEqual(context["execution"]["benchmark_protocol"], self.protocol)
                saved = json.loads((output / "build-context.json").read_text())
                self.assertEqual(saved["execution"]["benchmark_protocol"], self.protocol)
                if role == "current":
                    self.assertEqual(source["commit_parents"], [base, head])

    def test_push_and_dispatch_capture_source_formula_changes_and_unknown_producers(self):
        package = self.checkout / metric_fixtures.PACKAGE
        (package / "builders.py").write_text(metric_fixtures.BUILDERS)
        (package / "metrics.py").write_text(metric_fixtures.METRICS)
        original_run = subprocess.run

        def inspect_or_run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0, '[{"Id":"same-image","RepoDigests":[]}]', "")
            return original_run(command, **kwargs)

        definitions = {event_name: {} for event_name in ("push", "workflow_dispatch")}
        for variant, runtime, stepping in (
            ("original", metric_fixtures.RUNTIME, metric_fixtures.STEPPING),
            (
                "changed",
                metric_fixtures.RUNTIME,
                metric_fixtures.STEPPING.replace("time.perf_counter() - start", "(time.perf_counter() - start) / 1000"),
            ),
            (
                "unsupported",
                metric_fixtures.RUNTIME.replace("builders.build_runtime", "other.build_runtime"),
                metric_fixtures.STEPPING,
            ),
        ):
            (package / "stepping.py").write_text(stepping)
            commit = self._commit_file(RUNTIME, runtime)
            for event_name in definitions:
                with (
                    self.subTest(event=event_name, source=variant),
                    patch.dict(
                        os.environ,
                        {
                            "GITHUB_SHA": commit,
                            "GITHUB_EVENT_NAME": event_name,
                            "GITHUB_REF_NAME": "develop",
                            "GITHUB_RUN_ID": "20",
                            "GITHUB_RUN_ATTEMPT": "1",
                            "GITHUB_JOB": "performance-smoke-benchmarks",
                        },
                    ),
                    patch.object(paired.subprocess, "run", side_effect=inspect_or_run),
                ):
                    output = self.root / event_name / variant
                    context = paired.capture_context(self.checkout, output, "same-image", "current", {}, self.legs)
                    definition = context["metric_definition"]
                    definitions[event_name][variant] = definition["total_fps"]
                    self.assertEqual(definition["producer_commit"], commit)
                    saved = json.loads((output / "build-context.json").read_text())
                    self.assertEqual(saved["metric_definition"], definition)
                    if variant == "unsupported":
                        self.assertIsNone(definition["total_fps"])
                        self.assertIn("unknown:", definition["reason"])
                    else:
                        self.assertTrue(definition["total_fps"].startswith("source-fps-v2:"))
        for event_name, observed in definitions.items():
            with self.subTest(event=event_name):
                self.assertNotEqual(observed["original"], observed["changed"])

    def test_capture_failure_preserves_actual_identity_and_does_not_fall_back_to_event_base(self):
        base, head, merge, _ = self._moving_merge()
        reason = "Tested PR merge does not have the event's head commit as its second parent."
        for resolved in ("", self.commit):
            with self.subTest(resolved=resolved):
                output = self.root / (resolved or "unresolved")
                result = self._cli(
                    [
                        "capture",
                        "--checkout-root",
                        str(self.checkout),
                        "--output-dir",
                        str(output),
                        "--image",
                        "unused",
                        "--role",
                        "current",
                        "--legs",
                        str(self.root / "unused.tsv"),
                    ],
                    GITHUB_SHA=merge,
                    PERF_BASE_COMMIT=resolved,
                    PERF_PAIR_ERROR=reason,
                )
                self.assertNotEqual(result.returncode, 0)
                failure_path = output / "paired-failure.json"
                failure = json.loads(failure_path.read_text())
                self.assertEqual(failure["stage"], "capture_current")
                if not resolved:
                    self.assertEqual(failure["reason"], reason)
                self.assertEqual(failure["source"]["commit"], merge)
                self.assertEqual(failure["source"]["intended_commit"], merge)
                self.assertEqual(failure["source"]["tested_commit"], merge)
                self.assertEqual(failure["source"]["commit_parents"], [base, head])
                self.assertEqual(failure["source"]["event_base_commit"], self.commit)
                self.assertEqual(failure["execution"], {"run_id": 20, "run_attempt": 1})
                self.assertFalse((output / "source-manifest.json").exists())
                previous = failure_path.read_bytes()
                self._cli(
                    ["bind", "--selection", str(self.selection_path), "--output-dir", str(output)],
                    GITHUB_SHA=merge,
                    PERF_BASE_COMMIT="",
                )
                self.assertEqual(failure_path.read_bytes(), previous)

    def test_bind_failure_is_enriched_by_later_capture_without_replacing_primary_reason(self):
        base, head, merge, _ = self._moving_merge()
        reason = "CPU could not resolve the tested PR merge."
        env = {"GITHUB_SHA": merge, "PERF_BASE_COMMIT": "", "PERF_PAIR_ERROR": reason}
        bound = self._cli(["bind", "--selection", str(self.selection_path), "--output-dir", str(self.output)], **env)
        self.assertNotEqual(bound.returncode, 0)
        path = self.output / "paired-failure.json"
        original = json.loads(path.read_text())
        self.assertEqual(original["stage"], "bind")
        self.assertEqual(original["reason"], reason)
        self.assertIsNone(original["source"]["commit"])
        captured = self._cli(
            [
                "capture",
                "--checkout-root",
                str(self.checkout),
                "--output-dir",
                str(self.output),
                "--image",
                "unused",
                "--role",
                "current",
                "--legs",
                str(self.root / "unused.tsv"),
            ],
            **env,
        )
        self.assertNotEqual(captured.returncode, 0)
        expected = json.loads(encoded(original))
        expected["source"].update(commit=merge, commit_parents=[base, head])
        self.assertEqual(json.loads(path.read_text()), expected)

    def _files(self, run_id=10, attempt=1):
        files = {
            "source-manifest.json": encoded(self.manifest),
            "build-context.json": encoded(
                {
                    "source": {
                        "commit": self.commit,
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
                        "benchmark_protocol": self.protocol,
                        "measurement_not_before": stamp(8),
                        "job": "performance-smoke-benchmarks",
                        "runner_name": "fixture-gpu-runner",
                        "hostname": "fixture-gpu-host",
                    },
                    "metric_definition": {"total_fps": "source-fps-v2:fixture"},
                }
            ),
        }
        runtime_code = next(
            value
            for value in compile((self.checkout / RUNTIME).read_bytes(), RUNTIME, "exec").co_consts
            if isinstance(value, types.CodeType) and value.co_name == "run"
        )
        modules = [
            {
                "name": "isaaclab.benchmark.entrypoints.runtime" if path == RUNTIME else "isaaclab",
                "path": "/workspace/isaaclab/" + path,
                "relative_path": path,
                "sha256": digest,
                "expected_sha256": digest,
                "status": "verified",
            }
            for path, digest in self.manifest["files"].items()
        ]
        for leg in ("first", "second"):
            files[f"{leg}/status"] = b"ok"
            for sample in range(1, 4):
                path = f"{leg}/sample-{sample}/benchmark_runtime_fixture.json"
                value = bundle(start=stamp(9, sample), end=stamp(9, sample + 1), task=leg)
                value["run"]["seed"] = 42
                value["runtime"].update(
                    iterations_completed=200,
                    steps_per_iteration=512,
                    environment_step_timing={"warmup_steps": 100, "measurement_mode": "host_return"},
                )
                value["hardware"] = {
                    "cpu_name": "Intel(R) Xeon(R) 6975P-C",
                    "cpu_count": 16,
                    "gpu_devices": [{"name": "NVIDIA RTX PRO 4500 Blackwell Server Edition"}],
                }
                data = encoded(value)
                files[path] = data
                files[f"{leg}/sample-{sample}/source-revision.json"] = encoded(
                    {
                        "schema_version": 1,
                        "status": "verified",
                        "bytecode_policy": "fresh_process_cache",
                        "commit": self.commit,
                        "pid": 123,
                        "benchmark_exit_code": 0,
                        "exception": None,
                        "modules": modules,
                        "runtime_entrypoint": {
                            "module": "isaaclab.benchmark.entrypoints.runtime",
                            "function": "run",
                            "code_sha256": hashlib.sha256(marshal.dumps(runtime_code)).hexdigest(),
                            "doc_sha256": hashlib.sha256(b"Selected base runtime.").hexdigest(),
                            "source_code_matches": True,
                        },
                        "outputs": [
                            {
                                "path": "benchmark_runtime_fixture.json",
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "bytes": len(data),
                            }
                        ],
                        "mismatches": [],
                    }
                )
        return files

    def _add_baseline(self, files=None, *, run_id=10, attempt=1):
        self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return self.client.add_artifact(
            run_id,
            paired.baseline_name(42, self.commit, attempt),
            self._files(run_id, attempt) if files is None else files,
        )

    def _restore(self, *, run_id=20, attempt=1):
        if (run_id, attempt) not in self.client.attempts:
            self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return paired.restore_baseline(
            self.client,
            self.checkout,
            self.output,
            self.selection_path,
            self.event,
            run_id,
            attempt,
            legs=self.legs,
            base_commit=self._git("rev-parse", "HEAD"),
        )

    def test_restore_unchanged_base_for_new_pr_head_copies_exact_results(self):
        files = self._files()
        artifact = self._add_baseline(files)
        self.event["pull_request"]["head"]["sha"] = "d" * 40
        selection = self._restore()
        self.assertTrue(selection["baseline_reused"])
        self.assertEqual(selection["requested_head_commit"], "d" * 40)
        self.assertEqual(selection["baseline_origin"]["artifact_id"], artifact["id"])
        self.assertEqual(selection["baseline_origin"]["source_commit"], self.commit)
        self.assertEqual(json.loads(self.selection_path.read_text()), selection)
        for name, data in files.items():
            self.assertEqual((self.output / name).read_bytes(), data)

    def test_restore_requests_fresh_baseline_when_matrix_arguments_change(self):
        self._add_baseline()
        self.legs.write_text(self.legs.read_text().replace("physics=newton_mjwarp", "physics=ovphysx"))
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertIn("protocol", " ".join(selection["issues"]))
        self.assertFalse(self.output.exists())

    def test_restore_requests_fresh_baseline_when_either_controller_launcher_changes(self):
        self._add_baseline()
        for name in ("run_benchmarks.sh", "source_revision.py"):
            with self.subTest(launcher=name):
                path = self.controller / name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n# Changed controller launcher\n")
                    selection = self._restore()
                    self.assertFalse(selection["baseline_reused"])
                    self.assertIn("protocol", " ".join(selection["issues"]))
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(original)

    def test_restore_requests_fresh_baseline_when_protocol_identity_is_missing(self):
        files = self._files()
        context = json.loads(files["build-context.json"])
        del context["execution"]["benchmark_protocol"]
        files["build-context.json"] = encoded(context)
        self._add_baseline(files)
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertIn("protocol", " ".join(selection["issues"]))
        self.assertFalse(self.output.exists())

    def test_reused_legacy_identity_is_migrated_without_changing_measurement_evidence(self):
        files = self._files()
        context = json.loads(files["build-context.json"])
        original_definition = {"total_fps": "source-fps-v1:legacy"}
        context["metric_definition"] = original_definition
        files["build-context.json"] = encoded(context)
        artifact = self._add_baseline(files)
        original_bytes = self.client.contents[artifact["id"]]
        current_definition = {"total_fps": "source-fps-v2:verified", "provenance": "producer_source_ast"}
        with patch.object(paired, "metric_definition", return_value=current_definition):
            restored = self._restore()
        migration = restored["baseline_metric_definition"]
        self.assertTrue(restored["baseline_reused"])
        self.assertEqual(migration["artifact_id"], artifact["id"])
        self.assertEqual(migration["sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(migration["source_commit"], self.commit)
        candidate = self._candidate(
            restored["baseline_origin"], reused=True, selection=restored, definition=current_definition
        )
        evidence, _ = paired.select_pr_baseline(self.client, candidate)
        self.assertEqual(evidence.context["metric_definition"], original_definition)
        self.assertEqual(evidence.files, files)
        self.assertEqual(evidence.identity["sha256"], hashlib.sha256(original_bytes).hexdigest())
        report_dir = self.root / "report"
        with patch.object(build_compare.baseline_mod, "GitHubClient", return_value=self.client):
            self.assertEqual(
                build_compare.main(
                    ["--repository", REPO, "--run_id", "20", "--run_attempt", "1", "--output_dir", str(report_dir)]
                ),
                0,
            )
        report = json.loads((report_dir / "build-comparison.json").read_text())
        self.assertEqual(len(report["rows"]), 2)
        self.assertTrue(all(row["status"] == "compared" and row["change_pct"] == 0 for row in report["rows"]))
        self.assertTrue(all(row["formula"]["baseline"] == "source-fps-v2:verified" for row in report["rows"]))
        self.assertEqual(self.client.contents[artifact["id"]], original_bytes)
        self.assertIn("re-derived from its verified checkout", " ".join(report["notes"]))

    def test_derived_baseline_identity_cannot_be_rebound_to_other_evidence(self):
        artifact = self._add_baseline()
        restored = self._restore()
        for field, value in (("artifact_id", -1), ("sha256", "0" * 64), ("source_commit", HEAD)):
            with self.subTest(field=field):
                pin = json.loads(encoded(restored))
                pin["baseline_metric_definition"][field] = value
                candidate = self._candidate(restored["baseline_origin"], reused=True, selection=pin)
                evidence, metadata = paired.select_pr_baseline(self.client, candidate)
                self.assertIsNone(evidence)
                self.assertEqual(metadata["reason_code"], "identity_mismatch")
                self.assertEqual(metadata["unavailable_evidence"]["artifact_id"], artifact["id"])

    def test_restore_and_selection_use_resolved_reference_not_recorded_event_base(self):
        event_base = "f" * 40
        files = self._files()
        context = json.loads(files["build-context.json"])
        context["source"]["event_base_commit"] = event_base
        files["build-context.json"] = encoded(context)
        artifact = self._add_baseline(files)
        self.event["pull_request"]["base"]["sha"] = event_base
        restored = self._restore()
        self.assertTrue(restored["baseline_reused"])
        self.assertEqual(restored["reference_commit"], self.commit)
        self.assertEqual(restored["event_base_commit"], event_base)
        candidate = self._candidate(self._origin(artifact, run_id=10), reused=True, hostname="different-gpu-host")
        candidate.context["source"]["event_base_commit"] = event_base
        evidence, metadata = paired.select_pr_baseline(self.client, candidate)
        self.assertEqual(evidence.identity["source_commit"], self.commit)
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertTrue(metadata["baseline_reused"])

    def test_restore_requires_resolved_commit_even_when_event_base_matches_checkout(self):
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": ""}):
            with self.assertRaisesRegex(ValueError, "first parent was not resolved"):
                paired.restore_baseline(
                    self.client, self.checkout, self.output, self.selection_path, self.event, 20, 1, legs=self.legs
                )
        self.assertFalse(self.selection_path.exists())
        self.assertEqual(self.client.pages_requested, [])

    def test_restore_uses_current_numeric_workflow_registration_and_excludes_others(self):
        workflow_id = 812345
        self.client.add(20, HEAD, event="pull_request", branch="feature", artifact=False)["workflow_id"] = workflow_id
        expected = self._add_baseline(run_id=10)
        self.client.attempts[10, 1]["workflow_id"] = workflow_id
        obsolete = self._add_baseline(run_id=11)
        self.client.attempts[11, 1]["workflow_id"] = 700000
        selected = self._restore()
        self.assertTrue(selected["baseline_reused"])
        self.assertEqual(selected["baseline_origin"]["artifact_id"], expected["id"])
        self.assertEqual(self.client.attempts_requested[0], (20, 1))
        self.assertIn(
            (
                f"/repos/{REPO}/actions/workflows/{workflow_id}/runs",
                "workflow_runs",
                {"event": "pull_request", "branch": "feature"},
            ),
            self.client.pages_requested,
        )
        self.assertFalse(any("build.yaml" in path for path, _, _ in self.client.pages_requested))
        self.assertTrue(any(f"Artifact {obsolete['id']}" in issue for issue in selected["issues"]))

    def test_changed_base_commit_requests_fresh_measurement(self):
        self._add_baseline()
        source = self.checkout / RUNTIME
        source.write_text(source.read_text().replace("Selected base", "Changed base"))
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Next"
        )
        new_commit = self._git("rev-parse", "HEAD")
        self.event["pull_request"]["base"]["sha"] = new_commit
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertEqual(selection["reference_commit"], new_commit)
        self.assertEqual(self.client.downloaded, [])

    def test_different_pr_does_not_reuse_another_pr_baseline(self):
        self._add_baseline()
        self.event["number"] = self.event["pull_request"]["number"] = 43
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertEqual(self.client.downloaded, [])

    def test_current_measurements_are_never_restored_as_baseline(self):
        files = self._files()
        context = json.loads(files["build-context.json"])
        context["source"]["benchmark_role"] = "current"
        files["build-context.json"] = encoded(context)
        artifact = self._add_baseline(files)
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertEqual(self.client.downloaded, [artifact["id"]])
        self.assertIn("does not belong to this PR and exact base commit", str(selection["issues"]))

    def test_interrupted_lookup_leaves_a_persisted_fresh_selection(self):
        self._add_baseline()
        with patch.object(self.client, "run_attempt", side_effect=TimeoutError("lookup interrupted")):
            with self.assertRaises(TimeoutError):
                self._restore()
        selection = json.loads(self.selection_path.read_text())
        self.assertFalse(selection["baseline_reused"])
        self.assertIsNone(selection["baseline_origin"])
        self.assertEqual(selection["reference_commit"], self.commit)
        self.assertFalse(self.output.exists())

    def test_interrupted_restore_does_not_publish_partial_results(self):
        self._add_baseline()
        write_bytes = Path.write_bytes
        restored = []

        def interrupted(path, content):
            if path.name == "benchmark_runtime_fixture.json":
                restored.append(path)
                if len(restored) == 2:
                    raise OSError("artifact restoration interrupted")
            return write_bytes(path, content)

        with patch.object(Path, "write_bytes", interrupted):
            with self.assertRaisesRegex(OSError, "restoration interrupted"):
                self._restore()
        self.assertEqual(len(restored), 2)
        self.assertFalse(self.output.exists())
        self.assertFalse(json.loads(self.selection_path.read_text())["baseline_reused"])

    def test_expired_baseline_requests_fresh_measurement(self):
        artifact = self._add_baseline()
        artifact["expired"] = True
        self.assertFalse(self._restore()["baseline_reused"])
        self.assertFalse(self.output.exists())

    def test_invalid_source_proof_blocks_reuse_and_identifies_the_affected_sample(self):
        for mutation, sample, reason in (
            ("missing", "second/sample-3", "source-revision.json"),
            ("result", "first/sample-1", "Result bytes do not match their source verification"),
            ("module", "first/sample-1", "Loaded source does not match its manifest"),
        ):
            with self.subTest(mutation=mutation):
                self.client = PairedFixtureClient()
                files = self._files()
                path = f"{sample}/source-revision.json"
                if mutation == "missing":
                    del files[path]
                elif mutation == "result":
                    path = f"{sample}/benchmark_runtime_fixture.json"
                    changed = json.loads(files[path])
                    changed["runtime"]["total_fps"]["mean"] = 900
                    files[path] = encoded(changed)
                else:
                    proof = json.loads(files[path])
                    proof["modules"][0]["sha256"] = "0" * 64
                    files[path] = encoded(proof)
                artifact = self._add_baseline(files)
                candidate = self._candidate(self._origin(artifact, run_id=10), reused=True)
                evidence, _ = paired.select_pr_baseline(self.client, candidate)
                self.assertIn(sample, str(evidence.issues))
                self.assertIn(reason, str(evidence.issues))
                selection = self._restore()
                self.assertFalse(selection["baseline_reused"])
                self.assertIn("Baseline is incomplete or its source proof is invalid.", str(selection["issues"]))
                self.assertFalse(self.output.exists())

    def test_incomplete_or_failed_workload_requests_fresh_measurement(self):
        for mutation in ("missing-sample", "failed-leg"):
            with self.subTest(mutation=mutation):
                self.client = PairedFixtureClient()
                files = self._files()
                if mutation == "missing-sample":
                    del files["second/sample-3/benchmark_runtime_fixture.json"]
                else:
                    files["second/status"] = b"failed"
                self._add_baseline(files)
                self.assertFalse(self._restore()["baseline_reused"])

    def test_same_attempt_does_not_restore_its_own_results(self):
        self._add_baseline(run_id=20)
        self.assertFalse(self._restore()["baseline_reused"])

    def test_entire_missing_workload_cannot_appear_complete(self):
        files = {name: data for name, data in self._files().items() if not name.startswith("second/")}
        self._add_baseline(files)
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertTrue(selection["issues"])
        self.assertFalse(self.output.exists())

    def test_invalid_first_attempt_is_remeasured_then_second_attempt_is_reused(self):
        incomplete = self._files()
        del incomplete["second/sample-3/source-revision.json"]
        first = self._add_baseline(incomplete)
        first_bytes = self.client.contents[first["id"]]
        self.assertFalse(self._restore(run_id=10, attempt=2)["baseline_reused"])
        second = self._add_baseline(run_id=10, attempt=2)
        self.assertNotEqual(first["name"], second["name"])
        selection = self._restore(run_id=10, attempt=3)
        self.assertTrue(selection["baseline_reused"])
        self.assertEqual(selection["baseline_origin"]["artifact_id"], second["id"])
        self.assertEqual(selection["baseline_origin"]["run_attempt"], 2)
        self.assertEqual(self.client.contents[first["id"]], first_bytes)

    def _candidate(
        self, origin, *, reused=False, hostname="fixture-gpu-host", start=stamp(10), selection=None, definition=None
    ):
        files = self._files(run_id=20)
        manifest = json.loads(files["source-manifest.json"])
        manifest["commit"] = HEAD
        files["source-manifest.json"] = encoded(manifest)
        context = json.loads(files["build-context.json"])
        context["source"].update(commit=HEAD, benchmark_role="current", commit_parents=[self.commit, HEAD])
        context["execution"].update(hostname=hostname, measurement_not_before=start)
        if definition is not None:
            context["metric_definition"] = definition
        files["build-context.json"] = encoded(context)
        for name in list(files):
            if name.endswith("source-revision.json"):
                proof = json.loads(files[name])
                proof["commit"] = HEAD
                path = name.rsplit("/", 1)[0] + "/benchmark_runtime_fixture.json"
                data = json.loads(files[path])
                data["run"].update(start_time_utc=start, end_time_utc=stamp(11))
                files[path] = encoded(data)
                proof["outputs"][0].update(sha256=hashlib.sha256(files[path]).hexdigest(), bytes=len(files[path]))
                files[name] = encoded(proof)
        files["pr-comparison.json"] = encoded(
            {
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
        )
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

    def test_fresh_pair_uses_exact_pinned_base_from_same_runner_in_order(self):
        artifact = self._add_baseline(run_id=20)
        evidence, _ = paired.select_pr_baseline(self.client, self._candidate(self._origin(artifact)))
        self.assertEqual(evidence.identity["artifact_id"], artifact["id"])
        self.assertEqual(evidence.identity["source_commit"], self.commit)
        self.assertEqual(self.client.queried, [])
        self.assertEqual(self.client.pages_requested, [])

    def test_selection_rejects_parent_mismatch_before_retrieving_baseline(self):
        candidate = self._candidate({"artifact_id": 123})
        downloaded = list(self.client.downloaded)
        candidate.context["source"]["commit_parents"] = ["f" * 40, HEAD]
        with self.assertRaisesRegex(baseline.EvidenceError, "does not identify this tested PR revision"):
            paired.select_pr_baseline(self.client, candidate)
        self.assertEqual(self.client.downloaded, downloaded)

    def test_fresh_pair_reports_mismatched_runner(self):
        artifact = self._add_baseline(run_id=20)
        candidate = self._candidate(self._origin(artifact), hostname="different-gpu-host")
        evidence, metadata = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNone(evidence)
        self.assertEqual(metadata["reason_code"], "runner_mismatch")

    def test_fresh_pair_reports_base_not_finished_before_current(self):
        artifact = self._add_baseline(run_id=20)
        candidate = self._candidate(self._origin(artifact), start=stamp(9))
        evidence, metadata = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNone(evidence)
        self.assertEqual(metadata["reason_code"], "measurement_order")

    def test_partial_fresh_baseline_preserves_healthy_workload_comparison(self):
        files = self._files(run_id=20)
        files["second/sample-3/benchmark_runtime_fixture.json"] = b"{invalid JSON"
        artifact = self._add_baseline(files, run_id=20)
        origin = self._origin(artifact)
        self.selection_path.write_bytes(
            encoded(
                {
                    "reference_commit": self.commit,
                    "pull_request_number": 42,
                    "requested_head_commit": HEAD,
                    "baseline_reused": False,
                    "baseline_origin": None,
                    "issues": [],
                }
            )
        )
        with (
            patch.dict(os.environ, {"GITHUB_SHA": HEAD, "GITHUB_RUN_ID": "20", "GITHUB_RUN_ATTEMPT": "1"}),
            patch.object(paired, "datetime") as clock,
        ):
            clock.now.return_value.isoformat.return_value = stamp(9, 30)
            paired.bind_baseline(self.selection_path, str(artifact["id"]), self.output, REPO)
        pin = json.loads((self.output / "pr-comparison.json").read_text())
        candidate = self._candidate(origin, selection=pin)
        # GitHub's job completion covers B as well and cannot establish the end of A.
        self.client.run_jobs[20, 1][0]["completed_at"] = stamp(12)
        evidence, _ = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNotNone(evidence)
        report = build_compare.compare_evidence(evidence, candidate)
        by_leg = {row["legs"]["baseline"][0]: row for row in report["rows"]}
        self.assertEqual(by_leg["first"]["status"], "compared")
        self.assertEqual(by_leg["first"]["change_pct"], 0)
        self.assertEqual(by_leg["second"]["status"], "partial")
        self.assertIsNone(by_leg["second"]["change_pct"])

    def test_fresh_completion_bound_cannot_hide_samples_after_pr_start(self):
        artifact = self._add_baseline(run_id=20)
        candidate = self._candidate(
            self._origin(artifact), start=stamp(9), selection={"baseline_finished_before": stamp(8, 59)}
        )
        evidence, metadata = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNone(evidence)
        self.assertEqual(metadata["reason_code"], "measurement_order")

    def test_expired_pinned_baseline_never_falls_back_to_history(self):
        artifact = self._add_baseline(run_id=20)
        artifact["expired"] = True
        evidence, metadata = paired.select_pr_baseline(self.client, self._candidate(self._origin(artifact)))
        self.assertIsNone(evidence)
        self.assertEqual(metadata["reason_code"], "expired")
        self.assertEqual(metadata["unavailable_evidence"]["artifact_id"], artifact["id"])
        self.assertEqual(self.client.queried, [])

    def test_bind_rejects_selection_for_another_resolved_base(self):
        self._restore()
        with patch.dict(os.environ, {"PERF_BASE_COMMIT": "f" * 40, "GITHUB_SHA": HEAD}):
            with self.assertRaisesRegex(ValueError, "selection does not match"):
                paired.bind_baseline(self.selection_path, None, self.output, REPO)
        self.assertFalse((self.output / "pr-comparison.json").exists())

    def test_bind_preserves_failure_stage_when_partial_artifact_exists(self):
        self._restore()
        issue = "The baseline benchmark step failed after producing partial evidence."
        with patch.dict(os.environ, {"GITHUB_SHA": HEAD, "GITHUB_RUN_ID": "20", "GITHUB_RUN_ATTEMPT": "1"}):
            paired.bind_baseline(self.selection_path, "123", self.output, REPO, issue)
        selection = json.loads((self.output / "pr-comparison.json").read_text())
        self.assertEqual(selection["reason"], issue)
        self.assertIn(issue, selection["issues"])
        self.assertEqual(selection["baseline_origin"]["artifact_id"], 123)

    def test_bind_keeps_valid_reused_baseline_despite_unneeded_setup_issue(self):
        self._add_baseline()
        restored = self._restore()
        with patch.dict(os.environ, {"GITHUB_SHA": HEAD}):
            paired.bind_baseline(self.selection_path, None, self.output, REPO, "Base image config failed.")
        selection = json.loads((self.output / "pr-comparison.json").read_text())
        self.assertEqual(selection["baseline_origin"], restored["baseline_origin"])
        self.assertEqual(selection["reason"], restored["reason"])
        self.assertEqual(selection["issues"], restored["issues"])

    def test_unconfirmed_reuse_handoff_cannot_override_fresh_or_missing_upload(self):
        self._add_baseline(run_id=10)
        self._restore()
        fresh = self._add_baseline(run_id=20)
        event_path = self.root / "event.json"
        event_path.write_bytes(encoded(self.event))
        for uploaded in (True, False):
            with self.subTest(fresh_upload_succeeded=uploaded):
                with (
                    patch.dict(
                        os.environ,
                        {
                            "GITHUB_SHA": HEAD,
                            "GITHUB_RUN_ID": "20",
                            "GITHUB_RUN_ATTEMPT": "1",
                            "GITHUB_REPOSITORY": REPO,
                            "GITHUB_EVENT_PATH": str(event_path),
                            "PERF_BASELINE_REUSED": "",
                        },
                    ),
                    patch.object(paired, "datetime") as clock,
                ):
                    clock.now.return_value.isoformat.return_value = stamp(9, 30)
                    arguments = ["bind", "--selection", str(self.selection_path), "--output-dir", str(self.output)]
                    if uploaded:
                        arguments.extend(["--artifact-id", str(fresh["id"])])
                    self.assertEqual(paired.main(arguments), 0)
                pin = json.loads((self.output / "pr-comparison.json").read_text())
                candidate = self._candidate(pin["baseline_origin"], selection=pin)
                evidence, metadata = paired.select_pr_baseline(self.client, candidate)
                if uploaded:
                    self.assertEqual(evidence.identity["artifact_id"], fresh["id"])
                    self.assertEqual(evidence.identity["run_id"], 20)
                else:
                    self.assertIsNone(evidence)
                    self.assertIsNone(metadata["baseline_origin"])
                    self.assertEqual(
                        metadata["reason"], "The base benchmark artifact was not produced; baseline FPS is unavailable."
                    )
                    self.assertEqual(self.client.queried, [])
                self.assertFalse(metadata["baseline_reused"])
                self.assertNotIn("baseline_metric_definition", metadata)

    def test_bind_cli_recovers_missing_selection_with_or_without_failure_detail(self):
        tested_merge = "e" * 40
        for issue in ("The PR base checkout failed; no baseline was measured.", None):
            with self.subTest(failure_detail=issue):
                self.selection_path.unlink(missing_ok=True)
                arguments = ["bind", "--selection", str(self.selection_path), "--output-dir", str(self.output)]
                if issue:
                    arguments.extend(["--baseline-issue", issue])
                result = self._cli(arguments, GITHUB_SHA=tested_merge)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                selection = json.loads((self.output / "pr-comparison.json").read_text())
                if issue:
                    self.assertEqual(selection["reason"], issue)
                    self.assertIn(issue, selection["issues"])
                else:
                    self.assertIn("Baseline selection metadata was not produced", selection["reason"])
                self.assertEqual(selection["pull_request_number"], 42)
                self.assertEqual(selection["reference_commit"], self.commit)
                self.assertEqual(selection["requested_head_commit"], HEAD)
                self.assertEqual(selection["tested_commit"], tested_merge)
                self.assertFalse(selection["baseline_reused"])
                self.assertIsNone(selection["baseline_origin"])


if __name__ == "__main__":
    unittest.main()
