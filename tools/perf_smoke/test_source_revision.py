# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute source-provenance checks against distinct checkout and cached code."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).with_name("source_revision.py")
TASK = Path("source/isaaclab_tasks/isaaclab_tasks/fixture_task.py")
IMPORT_MAP = TASK.with_name("__init__.pyi")
RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"


class SourceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.checkout = self.directory / "checkout"
        self.checkout.mkdir()
        self._git("init", "--quiet")
        self._write(".gitignore", "__pycache__/\n")
        for name in (
            "source/isaaclab/isaaclab/__init__.py",
            "source/isaaclab/isaaclab/benchmark/__init__.py",
            "source/isaaclab/isaaclab/benchmark/entrypoints/__init__.py",
            "source/isaaclab_tasks/isaaclab_tasks/__init__.py",
        ):
            self._write(name, "")
        self._write(
            "source/isaaclab/isaaclab/cli/__init__.py",
            """
            import sys

            def cli():
                from isaaclab.benchmark.entrypoints import runtime
                runtime.run(sys.argv[1:])
            """,
        )
        self._write(
            "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py",
            """
            import json
            import os
            from pathlib import Path

            def run(argv):
                from isaaclab_tasks.fixture_task import observed
                marker = observed()
                output = Path(argv[argv.index("--output_path") + 1])
                output.mkdir(parents=True, exist_ok=True)
                result = {
                    "executed_revision": marker,
                    "worker_pid": os.getpid(),
                }
                (output / "benchmark_runtime_fixture.json").write_text(json.dumps(result))
            """,
        )
        self._write(IMPORT_MAP, "from .fixture_task import observed\n")
        self._task("A")
        self.commit_a = self._commit()
        self.cached = self.directory / "cached-image"
        shutil.copytree(self.checkout / "source", self.cached / "source")
        (self.cached / TASK).write_text("def observed():\n    return 'C'\n")

    def _write(self, relative, contents):
        destination = self.checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(textwrap.dedent(contents).lstrip("\n"))

    def _task(self, marker):
        self._write(TASK, f"def observed():\n    return {marker!r}\n")

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.checkout), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def _commit(self):
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Fixture"
        )
        return self._git("rev-parse", "HEAD")

    def _environment(self, roots=None, **extra):
        environment = dict(os.environ)
        environment.pop("PYTHONPYCACHEPREFIX", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(root / "source" / package)
            for root in (roots or [self.checkout])
            for package in ("isaaclab", "isaaclab_tasks")
        )
        environment.update(extra)
        return environment

    def _invoke(self, operation, *arguments, command=None, roots=None, **extra):
        launcher = [sys.executable, str(LAUNCHER), operation, "--checkout-root", str(self.checkout)]
        return subprocess.run(
            command or launcher + list(map(str, arguments)),
            cwd=self.directory,
            env=self._environment(roots, **extra),
            capture_output=True,
            text=True,
        )

    def _prepare(self, name="manifest", *, success=True):
        path = self.directory / f"{name}.json"
        process = self._invoke("prepare", "--output", path, "--commit", self._git("rev-parse", "HEAD"))
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(path.read_text())["commit"], self._git("rev-parse", "HEAD"))
        return path, process

    def _run(self, manifest, name, *, roots=None, command=None, proof_expected=True, expected_exit=0, **extra):
        output = self.directory / name
        verifier_args = ["--manifest", manifest, "--output-dir", output]
        benchmark_args = ["benchmark", "runtime", "--output_path", output]
        process = self._invoke("run", *verifier_args, "--", *benchmark_args, command=command, roots=roots, **extra)
        if expected_exit is None:
            self.assertNotEqual(process.returncode, 0)
        else:
            self.assertEqual(process.returncode, expected_exit, process.stdout + process.stderr)
        sidecar_path = output / "source-revision.json"
        self.assertEqual(sidecar_path.is_file(), proof_expected, process.stdout + process.stderr)
        sidecar = json.loads(sidecar_path.read_text()) if proof_expected else None
        if sidecar is not None:
            self.assertEqual(sidecar["status"], "verified" if expected_exit == 0 else "failed")
        return process, output, sidecar

    def _assert_verified(self, output, sidecar, marker, commit):
        self.assertEqual(sidecar["commit"], commit)
        self.assertEqual(sidecar["mismatches"], [])
        result_path = output / "benchmark_runtime_fixture.json"
        data = result_path.read_bytes()
        result = json.loads(data)
        self.assertEqual(result["executed_revision"], marker)
        self.assertEqual(result["worker_pid"], sidecar["pid"])
        self.assertEqual(
            sidecar["outputs"],
            [{"path": result_path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
        )
        modules = {item["name"]: item for item in sidecar["modules"]}
        self.assertIn(RUNTIME_MODULE, modules)
        task = modules["isaaclab_tasks.fixture_task"]
        self.assertEqual(task["relative_path"], TASK.as_posix())

    def test_source_verification_workflow(self):
        manifest_a, _ = self._prepare("manifest-a")
        with self.subTest(case="cached package origin is rejected"):
            process, output, sidecar = self._run(
                manifest_a, "stale", expected_exit=None, roots=[self.cached, self.checkout]
            )
            self.assertTrue(sidecar["mismatches"])
            result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
            self.assertEqual(result["executed_revision"], "C")
            self.assertEqual(result["worker_pid"], sidecar["pid"])
        with self.subTest(case="host Git check rejects changed source and import maps"):
            self._git("config", "diff.autoRefreshIndex", "false")
            os.utime(self.checkout / TASK, (1700000000, 1700000000))
            self.assertEqual(self._git("diff", "--name-only", "HEAD", "--", TASK.as_posix()), TASK.as_posix())
            self._prepare("unchanged-source")
            _, output, _ = self._run(manifest_a, "permission-only")
            self._git("config", "core.filemode", "true")
            (self.checkout / TASK).chmod(0o755)
            self.assertIn("mode change", self._git("diff", "--summary", "HEAD", "--", TASK.as_posix()))
            checked = self._invoke("check", "--manifest", manifest_a, "--output-dir", output)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(json.loads((output / "source-revision.json").read_text())["status"], "verified")
            self._prepare("permission-only")
            for path, contents in (
                (TASK, "def observed():\n    return 'B'\n"),
                (IMPORT_MAP, "# changed exports\n"),
            ):
                with self.subTest(path=path, contents=contents):
                    self._git("checkout", "--force", "--detach", self.commit_a)
                    (self.checkout / path).write_text(contents)
                    # The worker checks paths; the host checks for source edits after the attempt.
                    process, output, sidecar = self._run(manifest_a, "wrong-bytes")
                    if path == TASK:
                        (self.checkout / path).chmod(0o755)
                        self.assertEqual(
                            json.loads((output / "benchmark_runtime_fixture.json").read_text())["executed_revision"],
                            "B",
                        )
                    checked = self._invoke("check", "--manifest", manifest_a, "--output-dir", output)
                    self.assertNotEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                    proof = json.loads((output / "source-revision.json").read_text())
                    self.assertEqual(proof["status"], "failed")
                    self.assertTrue(any(path.as_posix() in item["reason"] for item in proof["mismatches"]))
                    if path == TASK:
                        (self.checkout / path).chmod(0o644)
                        self._git("add", "--", path.as_posix())
                    _, process = self._prepare(success=False)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("Traceback (most recent call last):", process.stderr)
                    self.assertIn(path.as_posix(), process.stderr)
        with self.subTest(case="revision B overrides cached packages and stale bytecode"):
            self._git("checkout", "--force", "--detach", self.commit_a)
            task = self.checkout / TASK
            os.utime(task, (1700000000, 1700000000))
            subprocess.run([sys.executable, "-m", "py_compile", str(task)], check=True, env=self._environment())
            original_size = task.stat().st_size
            self._task("B")
            self.assertEqual(task.stat().st_size, original_size)
            os.utime(task, (1700000000, 1700000000))
            commit_b = self._commit()
            # Confirm that an ordinary import loads stale bytecode before testing the fix.
            uncorrected = subprocess.run(
                [sys.executable, "-c", "from isaaclab_tasks.fixture_task import observed; print(observed())"],
                cwd=self.directory,
                env=self._environment(),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(uncorrected.stdout.strip(), "A")
            rejected = self._invoke("prepare", "--output", manifest_a, "--commit", self.commit_a)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("differs from expected commit", rejected.stderr)
            manifest, _ = self._prepare()
            _, output, proof = self._run(manifest, "fresh-bytecode", roots=[self.checkout, self.cached])
            self._assert_verified(output, proof, "B", commit_b)


if __name__ == "__main__":
    unittest.main()
