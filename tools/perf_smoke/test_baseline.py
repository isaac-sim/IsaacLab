# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact evidence retrieval tests using local GitHub API and artifact fixtures."""

import hashlib
import io
import json
import unittest
import urllib.error
import urllib.parse
import zipfile
from unittest.mock import Mock, patch

from . import baseline

REPO = "isaac-sim/IsaacLab"
HEAD, PARENT = "a" * 40, "b" * 40


def stamp(hour, minute=0):
    return f"2026-10-01T{hour:02d}:{minute:02d}:00+00:00"


def bundle(start=stamp(10), end=stamp(10, 5), task="task", fps=100, status="completed"):
    return {
        "run": {
            "task": task,
            "num_envs": 512,
            "status": status,
            "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": []},
            "start_time_utc": start,
            "end_time_utc": end,
        },
        "runtime": {"total_fps": {"mean": fps}},
    }


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, value in files.items():
            archive.writestr(path, value if isinstance(value, (str, bytes)) else json.dumps(value))
    return output.getvalue()


class FixtureClient:
    repository = REPO

    def __init__(self):
        self.attempts = {}
        self.run_artifacts = {}
        self.contents = {}
        self.run_jobs = {}
        self.downloaded = []

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
        context=True,
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
            if context:
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
        return self.attempts[run_id, attempt]

    def jobs(self, run_id, attempt):
        return self.run_jobs[run_id, attempt]

    def artifacts(self, run_id):
        return self.run_artifacts[run_id]

    def download(self, artifact):
        self.downloaded.append(artifact["id"])
        if artifact.get("expired"):
            raise baseline.EvidenceError("expired", "Selected artifact has expired")
        return self.contents[artifact["id"]]


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.client = FixtureClient()
        self.client.add(20, HEAD, start=stamp(12), samples={"leg": [bundle(stamp(12, 1), stamp(12, 5))]})

    def candidate(self, attempt=1):
        return baseline.resolve_candidate(self.client, 20, attempt)

    def test_missing_context_does_not_claim_verified_source(self):
        self.client.add(
            10, PARENT, context=False, samples={"leg": [bundle(end=stamp(12), fps=1)]}, conclusion="failure"
        )
        self.assertEqual(self.candidate().measurement_start, stamp(12, 1))
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertEqual(evidence.context, {})
        self.assertIsNone(evidence.identity["tested_commit"])
        self.assertIn("provenance", evidence.issues[0])

    def test_producing_attempt_preserves_incomplete_legs(self):
        self.client.add(10, PARENT)
        self.client.add(
            10,
            PARENT,
            attempt=2,
            start=stamp(11),
            samples={"renamed-leg": [bundle(stamp(11), stamp(11, 5))]},
            statuses={"renamed-leg": "ok", "missing-leg": "failed"},
            conclusion="failure",
        )
        evidence = baseline.resolve_candidate(self.client, 10, 2)
        self.assertEqual(evidence.identity["run_attempt"], 2)
        self.assertEqual(evidence.samples["missing-leg"], [])
        self.assertEqual(evidence.statuses["missing-leg"], "failed")

    def test_missing_candidate_sample_start_uses_capture_or_unknown(self):
        self.client.add(30, HEAD, start=stamp(9), samples={"leg": [bundle(), bundle(start=None)]})
        candidate = baseline.resolve_candidate(self.client, 30, 1)
        self.assertEqual(candidate.measurement_start, stamp(9))
        self.assertIn("capture time", str(candidate.issues))
        self.client.add(31, HEAD, context=False, samples={"leg": [bundle(), bundle(start=None)]})
        self.assertIsNone(baseline.resolve_candidate(self.client, 31, 1).measurement_start)

    def test_failed_sample_without_end_does_not_prove_baseline_finished(self):
        self.client.add(10, PARENT, samples={"leg": [bundle(), bundle(end=None, status="failed")]})
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertIsNone(evidence.measurement_end)
        self.assertIn("runtime has no valid end timestamp", str(evidence.issues))

    def test_corrupt_workload_preserved_alongside_valid_samples(self):
        self.client.add(10, PARENT, artifact=False)
        self.client.run_jobs[10, 1][0]["completed_at"] = stamp(11)
        self.client.add_artifact(
            10,
            "performance-smoke-10-1",
            {
                "leg/sample-1/benchmark_runtime_valid.json": bundle(),
                "leg/status": "ok",
                "broken/sample-1/benchmark_runtime_broken.json": "{not json",
                "broken/status": "failed",
            },
        )
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertEqual(len(evidence.samples["leg"]), 1)
        self.assertEqual(evidence.samples["broken"], [])
        self.assertEqual(evidence.measurement_end, stamp(11))
        self.assertIn("broken/sample-1/benchmark_runtime_broken.json", str(evidence.issues))

    def test_malformed_fps_is_preserved_and_explicitly_unusable(self):
        malformed = bundle()
        malformed["runtime"]["total_fps"] = []
        self.client.add(10, PARENT, samples={"leg": [malformed, bundle(fps=10**1000)]})
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertEqual(len(evidence.samples["leg"]), 2)
        self.assertTrue(all(not baseline.has_usable_runtime(item["bundle"]) for item in evidence.samples["leg"]))
        self.assertIn("FPS value", str(evidence.issues))

    def test_new_benchmark_attempt_uses_new_artifact(self):
        self.client.add(20, HEAD, attempt=2, start=stamp(13), samples={"leg": [bundle(stamp(13), stamp(13, 5))]})
        self.assertEqual(self.candidate(2).identity["run_attempt"], 2)

    def test_missing_job_time_is_explicit_not_old_evidence(self):
        self.client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
        self.client.run_jobs[20, 2][0]["started_at"] = None
        with self.assertRaises(baseline.EvidenceError) as raised:
            self.candidate(2)
        self.assertEqual(raised.exception.code, "ambiguous_attempt")

    def test_corrupt_digest_and_zip_report_explicit_errors(self):
        artifact = self.client.run_artifacts[20][0]
        artifact["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(baseline.EvidenceError, "digest does not match") as raised:
            self.candidate()
        self.assertEqual(raised.exception.code, "corrupt")
        artifact.pop("digest")
        self.client.contents[artifact["id"]] = b"not a zip"
        with self.assertRaisesRegex(baseline.EvidenceError, "ZIP could not be read") as raised:
            self.candidate()
        self.assertEqual(raised.exception.code, "corrupt")

    def test_expired_evidence_retains_api_identity_without_claiming_tested_commit(self):
        self.client.run_artifacts[20][0]["expired"] = True
        with self.assertRaises(baseline.EvidenceError) as raised:
            self.candidate()
        identity = raised.exception.identity
        self.assertEqual((identity["repository"], identity["run_id"], identity["run_attempt"]), (REPO, 20, 1))
        self.assertEqual(identity["artifact_id"], self.client.run_artifacts[20][0]["id"])
        self.assertEqual(identity["source_commit"], HEAD)
        self.assertEqual(identity["github_head_commit"], HEAD)
        self.assertEqual(identity["source_provenance"], "github_run_metadata")
        self.assertIsNone(identity["tested_commit"])
        self.assertEqual(identity["run_url"], f"https://github.com/{REPO}/actions/runs/20/attempts/1")

    def test_failed_samples_are_preserved(self):
        self.client.add(10, PARENT, samples={"leg": [bundle(status="failed")]}, statuses={"leg": "failed"})
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertEqual(evidence.statuses["leg"], "failed")
        self.assertEqual(evidence.samples["leg"][0]["bundle"]["run"]["status"], "failed")
        self.assertFalse(baseline.has_usable_runtime(evidence.samples["leg"][0]["bundle"]))

    def test_missing_or_invalid_renderer_is_unknown_workload_identity(self):
        sample = bundle()
        self.assertIsNotNone(baseline.workload_key(sample))
        sample["run"]["config"].pop("rendering_backend")
        self.assertIsNone(baseline.workload_key(sample))
        for renderer in (None, 42, "", " "):
            with self.subTest(renderer=renderer):
                sample["run"]["config"]["rendering_backend"] = renderer
                self.assertIsNone(baseline.workload_key(sample))


class TestGitHubTransport(unittest.TestCase):
    def test_pagination_at_client_boundary(self):
        client = baseline.GitHubClient(REPO, token="memory-only")
        paths = []

        def get(path):
            paths.append(path)
            page = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["page"][0]
            return {"total_count": 101, "artifacts": [{"id": i} for i in (range(100) if page == "1" else [100])]}

        client.get = get
        self.assertEqual(len(client.artifacts(10)), 101)
        self.assertEqual(len(paths), 2)

    def test_incomplete_history_is_not_treated_as_complete(self):
        client = baseline.GitHubClient(REPO)
        client.get = lambda path: {"total_count": 1001, "artifacts": []}
        with self.assertRaises(baseline.EvidenceError) as raised:
            client.artifacts(10)
        self.assertEqual(raised.exception.code, "incomplete_history")

    def test_redirect_does_not_forward_authorization(self):
        client = baseline.GitHubClient(REPO, token="memory-only")
        redirect = urllib.error.HTTPError(
            "https://api.github.com/artifact",
            302,
            "redirect",
            {
                "Location": "https://storage.example.invalid/artifact?sig=secret",
            },
            None,
        )
        client._opener = Mock()
        client._opener.open.side_effect = redirect
        response = io.BytesIO(b"original zip")
        response.headers = {}
        with patch("urllib.request.urlopen", return_value=response) as opened:
            self.assertEqual(client.download({"id": 7}), b"original zip")
        self.assertIsNone(opened.call_args.args[0].get_header("Authorization"))
        self.assertEqual(client._opener.open.call_args.args[0].get_header("Authorization"), "Bearer memory-only")

    def test_api_and_signed_download_read_timeouts_are_bounded_and_reported_safely(self):
        for download in (False, True):
            with self.subTest(download=download):
                client = baseline.GitHubClient(REPO, token="memory-only")
                client._opener = Mock()
                response = io.BytesIO(b"unfinished response")
                response.headers = {}
                client._opener.open.return_value = response
                if download:
                    client._opener.open.side_effect = urllib.error.HTTPError(
                        "https://api.github.com/artifact",
                        302,
                        "redirect",
                        {"Location": "https://storage.example.invalid/artifact?sig=secret"},
                        None,
                    )
                with (
                    patch("urllib.request.urlopen", return_value=response) as opened,
                    patch.object(response, "read", side_effect=TimeoutError("read stalled: ?sig=secret")),
                    self.assertRaises(baseline.EvidenceError) as raised,
                ):
                    if download:
                        client.download({"id": 7})
                    else:
                        client.run_attempt(10, 1)
                self.assertEqual(raised.exception.code, "timeout")
                self.assertNotIn("secret", str(raised.exception))
                self.assertTrue(response.closed)
                self.assertEqual(client._opener.open.call_args.kwargs["timeout"], 60)
                if download:
                    self.assertEqual(opened.call_args.kwargs["timeout"], 60)
                    self.assertIsNone(opened.call_args.args[0].get_header("Authorization"))

    def test_request_failures_have_safe_reason_codes_and_messages(self):
        for error in (
            urllib.error.URLError(TimeoutError("connect stalled: ?sig=secret")),
            urllib.error.HTTPError("https://storage.example.invalid/?sig=secret", 403, "secret", {}, None),
        ):
            with self.subTest(case=type(error).__name__):
                client = baseline.GitHubClient(REPO)
                client._opener = Mock()
                client._opener.open.side_effect = error
                with self.assertRaises(baseline.EvidenceError) as raised:
                    if isinstance(error, urllib.error.HTTPError):
                        client.download({"id": 7})
                    else:
                        client.run_attempt(10, 1)
                self.assertNotIn("secret", str(raised.exception))
                if isinstance(error, urllib.error.HTTPError):
                    self.assertIn("403", str(raised.exception))
                else:
                    self.assertEqual(raised.exception.code, "timeout")


if __name__ == "__main__":
    unittest.main()
