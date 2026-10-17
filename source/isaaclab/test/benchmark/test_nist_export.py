# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused test for build_artifact_manifest using temporary run directories."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

_script = Path(__file__).resolve().parents[4] / "scripts/benchmarks/nist_export.py"
_spec = importlib.util.spec_from_file_location("nist_export", _script)
_exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exporter)
build_artifact_manifest = _exporter.build_artifact_manifest

INPUTS = [
    "invocation.json", "execution.json", "process.json", "throughput.json",
    "latency.json", "memory.json", "outcomes.json", "outcomes-native.json",
    "outcomes-recovered.json", "events.jsonl", "outcomes.jsonl", "nvml.jsonl",
    "output.log", "stdout.log",
]


class BuildArtifactManifestTest(unittest.TestCase):
    def test_present_missing_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            contents = {
                "invocation.json": b'{"mode": "throughput"}',
                "throughput.json": b'{"environment_steps": 16}',
                "events.jsonl": b"",
                # Larger than one read chunk, so streaming covers several reads.
                "stdout.log": bytes(range(256)) * 9000,
            }
            for name, data in contents.items():
                (run / name).write_bytes(data)
            history = run / "previous-invocations" / "attempt-1"
            history.mkdir(parents=True)
            (history / "latency.json").write_bytes(b"{}")
            for name in ("benchmark_nist_omniperf.json", "benchmark_result.json", "export.log", "export-status.json"):
                (run / name).write_bytes(b"{}")

            files = build_artifact_manifest(run)["files"]

            self.assertEqual(list(files), INPUTS)
            for name, data in contents.items():
                self.assertEqual(files[name], {
                    "path": name, "status": "present",
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                })
            # latency.json exists only under previous-invocations, so it stays missing.
            for name in set(INPUTS) - set(contents):
                self.assertEqual(files[name], {"path": name, "status": "missing", "sha256": None, "bytes": None})

    def test_unreadable_file_gets_error_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "output.log").mkdir()

            record = build_artifact_manifest(run)["files"]["output.log"]

            self.assertEqual(record["status"], "error")
            self.assertIsNone(record["sha256"])
            self.assertIsNone(record["bytes"])
            self.assertEqual(record["error"]["type"], "IsADirectoryError")


if __name__ == "__main__":
    unittest.main()
