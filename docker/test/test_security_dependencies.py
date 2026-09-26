# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-selection regressions that also run without the simulation runtime."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSecurityDependencies(unittest.TestCase):
    def test_centralized_requirements_select_security_updates(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        self.assertIn("gitpython>=3.1.59", project["dependencies"])
        self.assertIn("pillow>=12.3.0", project["dependencies"])
        self.assertIn("pyarrow==23.0.1", project["optional-dependencies"]["rerun"])

    def test_wheel_metadata_preserves_dependency_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pyproject.toml"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools/wheel_builder/gen_pyproject.py"),
                    str(REPO_ROOT / "pyproject.toml"),
                    str(output),
                    "3.0.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open("rb") as file:
                project = tomllib.load(file)["project"]
            self.assertIn("gitpython>=3.1.59", project["dependencies"])
            self.assertIn("pillow>=12.3.0", project["dependencies"])
            self.assertIn("pyarrow==23.0.1", project["optional-dependencies"]["all"])

    def test_lock_contains_security_updates(self):
        with (REPO_ROOT / "uv.lock").open("rb") as file:
            packages = {p["name"]: p["version"] for p in tomllib.load(file)["package"]}
        for name, version in {
            "gitpython": "3.1.59",
            "pillow": "12.3.0",
            "pyarrow": "23.0.1",
        }.items():
            with self.subTest(package=name):
                self.assertGreaterEqual(tuple(map(int, packages[name].split("."))), tuple(map(int, version.split("."))))

    def test_images_ship_no_git_lfs(self):
        # Half the pair is the dangerous state: ``git-lfs install --system`` sets
        # ``filter.lfs.required=true``, so a git without the filter fails on an LFS tree.
        for name in ("Dockerfile.base", "Dockerfile.kitless"):
            with self.subTest(dockerfile=name):
                text = (REPO_ROOT / "docker" / name).read_text(encoding="utf-8")
                for line in text.splitlines():
                    if not line.lstrip().startswith("#"):
                        self.assertNotIn("git-lfs", line)

    def test_kitless_runtime_stage_ships_no_git(self):
        text = (REPO_ROOT / "docker/Dockerfile.kitless").read_text(encoding="utf-8")
        runtime = text[text.index("FROM", text.index("AS builder")) :]
        for line in runtime.splitlines():
            if not line.lstrip().startswith("#"):
                self.assertNotEqual(line.strip().rstrip("\\").strip(), "git")

    def test_base_image_purges_its_build_only_git(self):
        # Single-stage: it resolves the ``git+https://`` requirements in place, so git has to
        # be installed, and the purge is what keeps it out of the runtime filesystem.
        text = (REPO_ROOT / "docker/Dockerfile.base").read_text(encoding="utf-8")
        self.assertIn("apt-get purge -y git", text)
        self.assertLess(text.index("apt-get purge -y git"), text.index("USER isaaclab"))

    def test_kitless_builder_keeps_git_for_vcs_dependencies(self):
        # rl-games and robomimic resolve over ``git+https://`` during the builder stage.
        text = (REPO_ROOT / "docker/Dockerfile.kitless").read_text(encoding="utf-8")
        builder = text[text.index("AS builder") : text.index("FROM", text.index("AS builder"))]
        self.assertIn("      git \\", builder)

    def test_contract_workflows_use_hash_locked_dependencies(self):
        requirements = REPO_ROOT / "docker/test/requirements.txt"
        self.assertTrue(requirements.is_file())
        for name in ("build.yaml", "kitless-docker.yml"):
            with self.subTest(workflow=name):
                text = (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertIn("uv pip sync --require-hashes --only-binary :all:", text)
                self.assertIn("docker/test/requirements.txt", text)
                self.assertNotIn("--with pytest", text)
                self.assertIn('"${contract_env}/bin/python" -m pytest', text)

    def test_curobo_avoids_get_pip_bootstrap(self):
        text = (REPO_ROOT / "docker/Dockerfile.curobo").read_text(encoding="utf-8")
        self.assertNotIn("raw.githubusercontent.com/pypa/get-pip", text)
        self.assertNotIn('python3 "${bootstrap_dir}/get-pip.py"', text)
        self.assertNotIn("site-packages/pip*", text)
        self.assertIn("uv sync --frozen --inexact", text)


if __name__ == "__main__":
    unittest.main()
