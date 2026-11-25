# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-selection regressions that also run without the simulation runtime."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("branch_state", ["missing", "image-pin-only", "review-fix"])
def test_nightly_image_update_preserves_review_fixes(tmp_path: Path, branch_state: str):
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"

    def git(*args: str, cwd: Path = checkout) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd, env=environment, text=True).strip()

    git("init", "--bare", str(remote), cwd=tmp_path)
    git("init", "-b", "main", str(checkout), cwd=tmp_path)
    git("config", "user.name", "Test maintainer")
    git("config", "user.email", "test@example.com")
    config = checkout / "config.yaml"
    config.write_text("original image\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Initial image")
    git("remote", "add", "origin", str(remote))
    branch = "ci/image-update"
    previous = None
    if branch_state != "missing":
        git("switch", "-c", branch)
        config.write_text("previous candidate\n", encoding="utf-8")
        if branch_state == "review-fix":
            (checkout / "collider-fix.py").write_text("# Preserve this reviewed fix.\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-m", "Prepare image update")
        git("push", "origin", branch)
        previous = git("rev-parse", "HEAD")
        git("switch", "main")

    config.write_text("latest candidate\n", encoding="utf-8")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/nightly-isaacsim-image.yml").read_text())
    script = next(
        step["run"]
        for step in workflow["jobs"]["update-image-pin"]["steps"]
        if step.get("name") == "Commit and push the update branch"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=checkout,
        env={**environment, "CONFIG_PATH": "config.yaml", "UPDATE_BRANCH": branch},
        capture_output=True,
        text=True,
    )
    actual = git("rev-parse", f"refs/heads/{branch}", cwd=remote)
    if branch_state == "review-fix":
        assert actual == previous, result.stdout + result.stderr
        assert result.returncode != 0
        assert git("show", f"{branch}:collider-fix.py", cwd=remote) == "# Preserve this reviewed fix."
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert git("show", f"{branch}:config.yaml", cwd=remote) == "latest candidate"


class TestSecurityDependencies(unittest.TestCase):
    def test_centralized_requirements_select_security_updates(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        self.assertIn("gitpython>=3.1.60", project["dependencies"])
        self.assertIn("pillow>=12.3.0", project["dependencies"])
        self.assertIn("urllib3>=2.8.0", project["dependencies"])
        self.assertIn("pyarrow==23.0.1", project["optional-dependencies"]["rerun"])
        self.assertIn("PyJWT[crypto]>=2.15.1,<3", project["optional-dependencies"]["isaacsim"])
        self.assertIn("virtualenv>=21.7.13", project["optional-dependencies"]["rlinf"])

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
            self.assertIn("gitpython>=3.1.60", project["dependencies"])
            self.assertIn("pillow>=12.3.0", project["dependencies"])
            self.assertIn("urllib3>=2.8.0", project["dependencies"])
            self.assertIn("pyarrow==23.0.1", project["optional-dependencies"]["all"])
            self.assertIn("PyJWT[crypto]>=2.15.1,<3", project["optional-dependencies"]["isaacsim"])
            self.assertIn("virtualenv>=21.7.13", project["optional-dependencies"]["rlinf"])

    def test_lock_contains_security_updates(self):
        with (REPO_ROOT / "uv.lock").open("rb") as file:
            packages = {p["name"]: p["version"] for p in tomllib.load(file)["package"]}
        for name, version in {
            "aiohttp": "3.14.3",
            "gitpython": "3.1.60",
            "pillow": "12.3.0",
            "pyarrow": "23.0.1",
            "pyjwt": "2.15.1",
            "urllib3": "2.8.0",
            "virtualenv": "21.7.13",
        }.items():
            with self.subTest(package=name):
                self.assertGreaterEqual(tuple(map(int, packages[name].split("."))), tuple(map(int, version.split("."))))

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
