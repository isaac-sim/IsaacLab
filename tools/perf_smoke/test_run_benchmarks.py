# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow cancellation without starting Docker or a simulator."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import suppress
from pathlib import Path

import yaml


class BenchmarkCancellationTests(unittest.TestCase):
    def test_cancelled_workflow_removes_container_and_reaps_launcher(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / ".github/workflows/build.yaml").read_text())
        command = next(
            step["run"]
            for step in workflow["jobs"]["performance-smoke"]["steps"]
            if step.get("name") == "Run runtime benchmarks"
        )
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=stop_signal), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                step_script = workspace / "workflow-step.sh"
                step_script.write_text(command)
                scripts = workspace / "tools/perf_smoke"
                scripts.mkdir(parents=True)
                runner = root / "tools/perf_smoke/run_benchmarks.sh"
                shutil.copy2(runner, scripts)
                (workspace / "benchmark-legs.tsv").write_text("fixture|Fixture-Task|1|600|\n")
                executables = workspace / "bin"
                executables.mkdir()
                docker = executables / "docker"
                docker.write_text(
                    f"#!{sys.executable}\n"
                    + textwrap.dedent(
                        """
                        import json
                        import os
                        import sys
                        import time
                        from pathlib import Path

                        workspace = Path(os.environ["GITHUB_WORKSPACE"])
                        marker = workspace / "running-container"
                        if sys.argv[1] == "run":
                            name = sys.argv[sys.argv.index("--name") + 1]
                            marker.write_text(name)
                            print(json.dumps({"pid": os.getpid()}), flush=True)
                            while marker.exists():
                                time.sleep(0.01)
                        elif sys.argv[1] == "rm" and marker.exists():
                            assert sys.argv[-1] == marker.read_text()
                            marker.unlink()
                        """
                    )
                )
                docker.chmod(0o755)
                # The child stays blocked until cancellation; timeout's expiry is not under test.
                timeout = executables / "timeout"
                timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n')
                timeout.chmod(0o755)
                environment = {
                    **os.environ,
                    "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
                    "GITHUB_WORKSPACE": str(workspace),
                    "GITHUB_RUN_ID": "123",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "CI_IMAGE_TAG": "fixture-image",
                }
                process = subprocess.Popen(
                    ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(step_script)],
                    cwd=workspace,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    started = None
                    for line in process.stdout:
                        if line.startswith("{"):
                            started = json.loads(line)
                            break
                    if started is None:
                        self.fail(process.stderr.read())
                    process.send_signal(stop_signal)
                    # GitHub escalates cancellation after 7.5 s + 2.5 s, then kills the process tree.
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 128 + stop_signal, stdout + stderr)
                    self.assertFalse((workspace / "running-container").exists())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(started["pid"], 0)
                finally:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
