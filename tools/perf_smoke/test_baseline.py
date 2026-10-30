# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact evidence retrieval tests using local GitHub API and artifact fixtures."""

import hashlib
import http.client
import io
import json
import urllib.error
import urllib.parse
import zipfile
from unittest.mock import Mock, patch

import pytest

from . import baseline

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


class FixtureClient(baseline.GitHubClient):
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
        return self.attempts[run_id, attempt]

    def jobs(self, run_id, attempt):
        return self.run_jobs[run_id, attempt]

    def artifacts(self, run_id):
        return self.run_artifacts[run_id]

    def download(self, artifact):
        self.downloaded.append(artifact["id"])
        with patch.object(self, "_request", return_value=self.contents[artifact["id"]]):
            return super().download(artifact)


def check_evidence(subtests):
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
        evidence = baseline.resolve_candidate(client, 10, 2)
        assert evidence.identity["run_attempt"] == 2
        assert evidence.samples["missing-leg"] == []
        assert evidence.statuses["missing-leg"] == "failed"

    with subtests.test(evidence="corrupt workload beside a valid sample"):
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
            },
        )
        evidence = baseline.resolve_candidate(client, 10, 1)
        assert evidence.identity["run_id"] == 10
        assert evidence.context == {}
        assert evidence.identity["tested_commit"] is None
        assert any("provenance" in issue for issue in evidence.issues)
        assert len(evidence.samples["leg"]) == 1
        assert evidence.samples["broken"] == []
        assert "broken/sample-1/benchmark_runtime_broken.json" in str(evidence.issues)

    with subtests.test(evidence="unusable samples and failed runs"):
        malformed = bundle()
        malformed["runtime"]["total_fps"] = []
        client = FixtureClient()
        client.add(
            10,
            PARENT,
            samples={
                "leg": [malformed, bundle(fps=10**1000)],
                "failed": [bundle(status="failed")],
            },
            statuses={"leg": "ok", "failed": "failed"},
        )
        evidence = baseline.resolve_candidate(client, 10, 1)
        assert len(evidence.samples["leg"]) == 2
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
            with pytest.raises(baseline.EvidenceError, match=message) as raised:
                baseline.resolve_candidate(client, 20, attempt)
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
    assert baseline.workload_key(sample) is None
    for renderer in (None, 42, "", " "):
        with subtests.test(renderer=renderer):
            sample["run"]["config"]["rendering_backend"] = renderer
            assert baseline.workload_key(sample) is None


def check_transport(subtests):
    with subtests.test(transport="pagination"):
        client = baseline.GitHubClient(REPO)
        pages = [{"id": i} for i in range(100)], [{"id": 100}]
        client.get = Mock(side_effect=[{"total_count": 101, "artifacts": page} for page in pages])
        assert len(client.artifacts(10)) == 101
        assert [
            urllib.parse.parse_qs(urllib.parse.urlsplit(call.args[0]).query)["page"]
            for call in client.get.call_args_list
        ] == [["1"], ["2"]]
        client.get = Mock(return_value={"total_count": 1001, "artifacts": []})
        with pytest.raises(baseline.EvidenceError) as raised:
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
            client = baseline.GitHubClient(REPO, token="memory-only" if open_error is None else "")
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
                    with pytest.raises(baseline.EvidenceError) as raised:
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
