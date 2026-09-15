# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-selection regressions that also run without the simulation runtime."""

import io
import os
import shlex
import subprocess
import sys
import tarfile
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

    def test_both_images_install_verified_git_lfs(self):
        for name in ("Dockerfile.base", "Dockerfile.kitless"):
            with self.subTest(dockerfile=name):
                text = (REPO_ROOT / "docker" / name).read_text(encoding="utf-8")
                self.assertIn("COPY docker/scripts/install_git_lfs.sh /tmp/install_git_lfs.sh", text)
                self.assertIn("RUN /bin/bash /tmp/install_git_lfs.sh", text)
                self.assertLess(text.index("RUN /bin/bash /tmp/install_git_lfs.sh"), text.index("USER isaaclab"))

    def test_git_lfs_changes_invalidate_dependency_cache(self):
        action = REPO_ROOT / ".github/actions/_lib/compute-deps-hash/action.yml"
        self.assertIn("docker/scripts/install_git_lfs.sh", action.read_text(encoding="utf-8"))

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
        self.assertNotIn("get-pip.py", text)
        self.assertNotIn("site-packages/pip*", text)
        self.assertIn("uv sync --frozen --inexact", text)


class TestGitLfsInstaller(unittest.TestCase):
    """Exercise the shell installer with no network, root writes or package-manager changes."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {**os.environ, "PATH": f"{self.bin}:/usr/bin:/bin", "FIXTURE_ROOT": str(self.root)}
        self.env.update(TEST_ARCH="amd64", TEST_INSTALLED="yes", TEST_CHECKSUM="pass")
        with tarfile.open(self.root / "fixture.tar.gz", "w:gz") as archive:
            for name, content, mode in [
                (
                    "git-lfs",
                    b'#!/bin/sh\necho "lfs $*" >> "$FIXTURE_ROOT/events"\necho \'git-lfs/3.8.0 (test)\'\n',
                    0o755,
                ),
                ("README.md", b"Fixture README\n", 0o644),
                ("man/man1/git-lfs.1", b"Fixture manual\n", 0o644),
                ("LICENSE.md", b"Fixture license\n", 0o644),
                ("vendor/example/NOTICE", b"Fixture notice\n", 0o644),
            ]:
                member = tarfile.TarInfo(f"git-lfs-3.8.0/{name}")
                member.size, member.mode = len(content), mode
                archive.addfile(member, io.BytesIO(content))
        self._command("dpkg", 'echo "$TEST_ARCH"')
        self._command("dpkg-query", '[ "$TEST_INSTALLED" = yes ] && printf "install ok installed"')
        self._command(
            "curl",
            """
printf 'download %s\\n' "$*" >> "$FIXTURE_ROOT/events"
while [ "$1" != --output ]; do shift; done
cp "$FIXTURE_ROOT/fixture.tar.gz" "$2"
""",
        )
        # Hash computation belongs to sha256sum. Verify the installer's supplied
        # hashes separately below; simulate both outcomes of that external command.
        self._command(
            "sha256sum",
            """
cat >> "$FIXTURE_ROOT/checksums"
[ "$TEST_CHECKSUM" = pass ]
""",
        )
        self._command("apt-get", 'echo "remove $*" >> "$FIXTURE_ROOT/events"')
        self._command("install", 'echo "install $*" >> "$FIXTURE_ROOT/events"')

    def _command(self, name: str, body: str):
        path = self.bin / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n", encoding="utf-8")
        path.chmod(0o755)

    def _run(self):
        return subprocess.run(
            ["bash", str(REPO_ROOT / "docker/scripts/install_git_lfs.sh")],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installs_each_supported_architecture_and_preserves_notices(self):
        hashes = {
            "amd64": "e455e00f15d9b95661b8d53498ffb0c3367962cf1ec73c31ab7369516cd6ab8d",
            "arm64": "ac9c8efac980bb0505ead384d087e2acb6486fd8498691a2165fa174ec6118c2",
        }
        for architecture, checksum in hashes.items():
            with self.subTest(architecture=architecture):
                self.env["TEST_ARCH"] = architecture
                result = self._run()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(checksum, (self.root / "checksums").read_text())
                events = (self.root / "events").read_text()
                self.assertIn(f"git-lfs-linux-{architecture}-v3.8.0.tar.gz", events)
                self.assertLess(events.index("remove remove -y git-lfs"), events.index("install -m 0755"))
                self.assertIn("/usr/local/share/doc/git-lfs/LICENSE.md", events)
                self.assertIn("/usr/local/share/doc/git-lfs/vendor/example/NOTICE", events)
                self.assertIn("/usr/local/share/man/man1/git-lfs.1", events)
                self.assertIn("lfs install --system --skip-repo", events)

    def test_bad_checksum_cannot_remove_or_install_anything(self):
        self.env["TEST_CHECKSUM"] = "fail"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        events = (self.root / "events").read_text()
        self.assertNotIn("remove ", events)
        self.assertNotIn("install ", events)

    def test_both_downloads_only_allow_https_redirects(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        downloads = [line for line in (self.root / "events").read_text().splitlines() if line.startswith("download ")]
        self.assertEqual(len(downloads), 2)
        for download in downloads:
            with self.subTest(download=download):
                args = shlex.split(download)
                self.assertIn("--fail", args)
                self.assertIn("--location", args)
                for option in ("--proto", "--proto-redir"):
                    self.assertIn(option, args)
                    self.assertEqual(args[args.index(option) + 1], "=https")

    def test_unsupported_architecture_fails_before_downloading(self):
        self.env["TEST_ARCH"] = "s390x"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported Git LFS architecture", result.stderr)
        self.assertFalse((self.root / "events").exists())

    def test_fresh_image_does_not_remove_a_package(self):
        self.env["TEST_INSTALLED"] = "no"
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        events = (self.root / "events").read_text()
        self.assertNotIn("remove ", events)
        self.assertIn("install -m 0755", events)


if __name__ == "__main__":
    unittest.main()
