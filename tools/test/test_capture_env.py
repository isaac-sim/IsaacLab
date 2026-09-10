# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the environment capture bundle."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _bootstrap_paths() -> None:
    """Prepend ``tools/`` so the module under test imports from the working tree."""
    tools_dir = str(Path(__file__).resolve().parents[1])
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)


_bootstrap_paths()

from capture_env import (  # noqa: E402
    collect_isaac_sim,
    collect_repo,
    lock_extras,
    parse_lock,
    render_document,
    resolve_sync_plan,
    scan_distributions,
    select_sync_extras,
    write_bundle,
)

# A lockfile shaped like the real one in the detail that decides what a sync installs: an alias
# extra defined in terms of other extras, rather than in terms of packages.
LOCK = """\
version = 1

[[package]]
name = "demo-dev"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "torch" },
]

[package.optional-dependencies]
all = [
    { name = "demo-dev", extras = ["sb3", "sim"], marker = "extra == 'all'" },
]
docs = [
    { name = "sphinx" },
]
sb3 = [
    { name = "stable-baselines3" },
]
sim = [
    { name = "isaacsim" },
]

[[package]]
name = "torch"
version = "2.11.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }

[[package]]
name = "stable-baselines3"
version = "2.9.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "isaacsim"
version = "6.0.1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "sphinx"
version = "8.2.3"
source = { registry = "https://pypi.org/simple" }
"""


def _installed(**versions: str) -> list[dict]:
    """Return manifest-shaped distribution records for ``name=version`` pairs."""
    return [
        {"name": name.replace("_", "-"), "key": name.replace("_", "-"), "version": version, "installer": "uv"}
        for name, version in versions.items()
    ]


def _manifest(**sections) -> dict:
    """Return a minimal manifest with ``sections`` merged over the empty defaults."""
    base = {
        "captured_at": "2026-01-01T00:00:00Z",
        "capture": {"hostname": "host", "command_under_test": None},
        "gpu": {"devices": [], "driver_version": None},
        "python": {"venv": None, "distributions": []},
        "environment": {"variables": {}, "omitted_count": 0},
        "links": {"symlinks": []},
        "repo": {"root": "/repo", "git": {}},
        "sync": {"lock_available": True, "extras": [], "command": "uv sync --locked"},
    }
    base.update(sections)
    return base


class TestRepositoryRevision:
    """The checked-out revision is read out of ``.git``, without running git."""

    @staticmethod
    def _repo(root: Path, head: str, refs: dict[str, str] | None = None, packed: str | None = None) -> Path:
        """Write a ``.git`` holding ``head``, plus any loose ``refs`` and ``packed-refs`` content."""
        git_dir = root / ".git"
        (git_dir / "refs" / "heads").mkdir(parents=True)
        (git_dir / "HEAD").write_text(head)
        for name, sha in (refs or {}).items():
            (git_dir / "refs" / "heads" / name).write_text(sha + "\n")
        if packed is not None:
            (git_dir / "packed-refs").write_text(packed)
        return root

    def test_a_branch_is_read_from_its_loose_ref(self, tmp_path):
        sha = "a" * 40
        self._repo(tmp_path, "ref: refs/heads/main\n", refs={"main": sha})

        assert collect_repo(tmp_path)[0]["git"] == {"commit": sha, "branch": "main"}

    def test_a_branch_with_no_loose_ref_falls_back_to_packed_refs(self, tmp_path):
        """A clone keeps its branches packed, so the loose file is simply absent."""
        sha = "b" * 40
        packed = "# pack-refs with: peeled fully-peeled sorted\n" + sha + " refs/heads/main\n"
        self._repo(tmp_path, "ref: refs/heads/main\n", packed=packed)

        assert collect_repo(tmp_path)[0]["git"] == {"commit": sha, "branch": "main"}

    def test_a_detached_head_records_the_commit_and_no_branch(self, tmp_path):
        sha = "c" * 40
        self._repo(tmp_path, sha + "\n")

        assert collect_repo(tmp_path)[0]["git"] == {"commit": sha, "branch": None}

    def test_a_worktree_follows_the_redirect_to_the_state_it_shares(self, tmp_path):
        """A worktree keeps its own HEAD beside itself and shares refs with the checkout it came from."""
        sha = "d" * 40
        main = self._repo(tmp_path / "main", "ref: refs/heads/main\n", refs={"main": sha, "side": sha})
        linked = tmp_path / "linked"
        linked.mkdir()
        worktree_dir = main / ".git" / "worktrees" / "linked"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "HEAD").write_text("ref: refs/heads/side\n")
        (worktree_dir / "commondir").write_text("../..\n")
        (linked / ".git").write_text("gitdir: " + str(worktree_dir) + "\n")

        assert collect_repo(linked)[0]["git"] == {"commit": sha, "branch": "side"}

    def test_a_directory_that_is_not_a_checkout_reports_nothing(self, tmp_path):
        """The capture still has to produce a bundle when it is pointed somewhere unexpected."""
        assert collect_repo(tmp_path)[0]["git"] == {"commit": None, "branch": None}

    def test_the_project_files_are_copied_verbatim(self, tmp_path):
        """Written as bytes: a lockfile is copied exactly, line endings included."""
        (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname = 'demo'\n")
        (tmp_path / "uv.lock").write_bytes(b"version = 1\n")

        _, artifacts = collect_repo(tmp_path)

        assert artifacts["files/pyproject.toml"] == "[project]\nname = 'demo'\n"
        assert artifacts["files/uv.lock"] == "version = 1\n"


class TestDistributionScan:
    """Distributions are read from disk, independently of the running interpreter."""

    def test_names_are_normalized_and_installers_recorded(self, tmp_path):
        dist_info = tmp_path / "Isaac_Lab.Tasks-1.0.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: Isaac_Lab.Tasks\nVersion: 1.0\n\nBody.\n")
        (dist_info / "INSTALLER").write_text("pip")

        (scanned,) = scan_distributions(tmp_path)

        assert scanned["name"] == "Isaac_Lab.Tasks"
        assert scanned["key"] == "isaac-lab-tasks"
        assert scanned["version"] == "1.0"
        assert scanned["installer"] == "pip"

    def test_metadata_without_a_name_costs_one_distribution_not_the_inventory(self, tmp_path):
        broken = tmp_path / "broken-1.0.dist-info"
        broken.mkdir(parents=True)
        (broken / "METADATA").write_text("garbage\n")
        good = tmp_path / "torch-2.11.0.dist-info"
        good.mkdir(parents=True)
        (good / "METADATA").write_text("Name: torch\nVersion: 2.11.0\n\n")

        assert [dist["key"] for dist in scan_distributions(tmp_path)] == ["torch"]


class TestSyncPlan:
    """The extras are derived from the lockfile and what is installed, not guessed."""

    def test_an_alias_extra_is_expanded_into_the_extras_it_names(self):
        """``all`` is defined as the project's own extras and has no requirements of its own."""
        graph = parse_lock(LOCK)

        assert graph["root"] == "demo-dev"
        assert lock_extras(graph)["all"] == {"stable-baselines3", "isaacsim"}

    def test_an_extra_covered_by_another_is_dropped(self):
        """``all`` subsumes ``sb3`` and ``sim``, so naming all three would be noise."""
        extras = lock_extras(parse_lock(LOCK))

        assert select_sync_extras(extras, {"torch", "stable-baselines3", "isaacsim"}) == ["all"]

    def test_an_extra_whose_requirements_are_absent_is_not_selected(self):
        extras = lock_extras(parse_lock(LOCK))

        assert select_sync_extras(extras, {"torch", "stable-baselines3"}) == ["sb3"]

    def test_the_command_carries_the_selected_extras(self):
        plan = resolve_sync_plan(LOCK, _installed(torch="2.11.0+cu128", stable_baselines3="2.9.0", isaacsim="6.0.1.0"))

        assert plan["command"] == "uv sync --locked --extra all"

    def test_a_checkout_without_a_lockfile_says_so(self):
        """A bare sync is still offered, but nothing claims the extras were derived."""
        plan = resolve_sync_plan(None, _installed(torch="2.11.0+cu128"))

        assert plan["lock_available"] is False
        assert plan["command"] == "uv sync --locked"


class TestIsaacSimInstall:
    """Isaac Sim arrives three ways, and only the ``VERSION`` string tells the last two apart."""

    WHEEL = [{"key": "isaacsim", "version": "6.0.0.1"}]

    @staticmethod
    def _kit(root: Path, version: str) -> Path:
        """Write a Kit tree at ``root`` holding ``version``, and return it."""
        root.mkdir(parents=True)
        (root / "VERSION").write_text(version + "\n")
        return root

    def _reached_by_env(self, monkeypatch, kit: Path, repo: Path, distributions=()) -> dict:
        """Classify a Kit reached through ``ISAAC_PATH``, which needs no symlink privilege."""
        monkeypatch.setenv("ISAAC_PATH", str(kit))
        return collect_isaac_sim(repo, list(distributions))[0]

    def test_a_wheel_is_told_apart_from_a_kit_tree(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        section = collect_isaac_sim(tmp_path, self.WHEEL)[0]

        assert section["install_method"] == "wheel"
        assert section["wheel_version"] == "6.0.0.1"
        assert section["version"] is None

    def test_a_downloaded_package_records_the_version_to_fetch(self, tmp_path, monkeypatch):
        kit = self._kit(tmp_path / "isaacsim", "4.5.0-rc.36+release.19980.0133eb1e.tc")

        section = self._reached_by_env(monkeypatch, kit, tmp_path / "repo")

        assert section["install_method"] == "binary"
        assert section["version"] == "4.5.0-rc.36+release.19980.0133eb1e.tc"
        assert (section["branch"], section["revision"]) == ("release", "0133eb1e")

    def test_a_local_build_records_the_revision_it_was_built_from(self, tmp_path, monkeypatch):
        """A `.local` suffix marks a Kit compiled on the captured machine, which no bundle rebuilds."""
        kit = self._kit(tmp_path / "isaacsim", "6.1.0-alpha.59+develop.0.4877ef77.local")

        section = self._reached_by_env(monkeypatch, kit, tmp_path / "repo")

        assert section["install_method"] == "source_build"
        assert (section["branch"], section["revision"]) == ("develop", "4877ef77")

    def test_a_kit_inside_a_build_tree_is_a_local_build_whatever_it_is_stamped(self, tmp_path, monkeypatch):
        """A path through `_build` is a compiled Kit even when the version carries no `.local`."""
        kit = self._kit(tmp_path / "IsaacSim" / "_build" / "linux-x86_64" / "release", "5.0.0+main.1234.deadbeef")

        section = self._reached_by_env(monkeypatch, kit, tmp_path / "repo")

        assert section["install_method"] == "source_build"

    def test_the_version_file_is_stored_alongside_the_manifest(self, tmp_path, monkeypatch):
        kit = self._kit(tmp_path / "isaacsim", "6.1.0-alpha.59+develop.0.4877ef77.local")
        monkeypatch.setenv("ISAAC_PATH", str(kit))

        _, artifacts = collect_isaac_sim(tmp_path / "repo", [])

        assert artifacts["isaacsim/VERSION"] == "6.1.0-alpha.59+develop.0.4877ef77.local\n"

    def test_no_isaac_sim_at_all_is_reported_as_such(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        assert collect_isaac_sim(tmp_path, [])[0]["install_method"] == "none"

    def test_an_empty_isaac_path_does_not_pass_for_a_kit_tree(self, tmp_path, monkeypatch):
        """``Path("")`` is the current directory, which must not be mistaken for an installation."""
        monkeypatch.setenv("ISAAC_PATH", "")

        section = collect_isaac_sim(tmp_path, [])[0]

        assert section["install_method"] == "none"
        assert section["path"] is None


class TestDocument:
    """The document is the reproduction, so every step in it has to be actionable."""

    def test_the_sync_step_carries_the_derived_extras(self):
        manifest = _manifest(
            sync={"lock_available": True, "extras": ["all", "test"], "command": "uv sync --locked --extra all"}
        )

        document = render_document(manifest, {})

        assert "uv sync --locked --extra all" in document
        assert "derived from what is installed" in document

    def test_the_bundle_is_unpacked_before_anything_copies_out_of_it(self):
        """The copy step reads files a zip does not provide until it is unpacked."""
        document = render_document(_manifest(), {})

        assert document.index("unzip <this-bundle>.zip") < document.index("cp ../bundle/files/pyproject.toml")

    def test_only_hand_made_symlinks_appear_in_the_recreate_step(self):
        """A link inside the virtual environment is recreated by the sync, so it is not a step."""
        manifest = _manifest(
            links={
                "symlinks": [
                    {"path": "/repo/_isaac_sim", "target": "/build", "exists": True},
                    {
                        "path": "/repo/.venv/bin/python",
                        "target": "/uv/python3.12",
                        "exists": True,
                        "in_virtualenv": True,
                    },
                ]
            },
        )

        document = render_document(manifest, {})

        assert "ln -s /build _isaac_sim" in document
        assert "/uv/python3.12" not in document

    def test_machine_owned_variables_are_recorded_but_not_exported(self):
        """Exporting the captured machine's VIRTUAL_ENV would point yours at a missing path."""
        manifest = _manifest(
            environment={
                "variables": {"VIRTUAL_ENV": "/other/.venv", "CONDA_PREFIX": "/other/conda", "ISAAC_PATH": "/isaac"},
                "omitted_count": 0,
            }
        )

        document = render_document(manifest, {})

        assert "export ISAAC_PATH=/isaac" in document
        assert "export VIRTUAL_ENV" not in document
        assert "`VIRTUAL_ENV`" in document and "`CONDA_PREFIX`" in document

    def test_an_exported_value_survives_a_real_shell(self):
        """``repr`` flips a value holding an apostrophe to double quotes, leaving ``$`` and backticks live."""
        shell = shutil.which("sh")
        if shell is None:
            pytest.skip("no POSIX shell to check the rendered quoting against")
        hostile = "/home/o'brien/$USER/`id`/lib"
        manifest = _manifest(environment={"variables": {"PYTHONPATH": hostile}, "omitted_count": 0})

        document = render_document(manifest, {})

        exported = next(line for line in document.splitlines() if line.startswith("export PYTHONPATH="))
        # The shell itself is the only reference that cannot be fooled by how the value was quoted.
        command = exported + '; printf %s "$PYTHONPATH"'
        done = subprocess.run([shell, "-c", command], capture_output=True, text=True)

        assert done.stdout == hostile

    def test_the_document_states_what_it_cannot_reproduce_and_what_was_left_out(self):
        """The allowlist guarantee is only meaningful if the bundle states its own limits."""
        manifest = _manifest(environment={"variables": {"ISAAC_PATH": "/isaac"}, "omitted_count": 42})

        document = render_document(manifest, {})

        assert "cannot reproduce" in document
        assert "42 variable(s) were present but not collected" in document
        assert "review `env/environment.txt`" in document


class TestBundle:
    """A capture is a zip a support engineer can open without this tool."""

    def test_manifest_and_document_round_trip_through_the_zip(self, tmp_path):
        manifest = _manifest()
        artifacts = {"files/uv.lock": "lock"}
        document = render_document(manifest, artifacts)
        bundle = tmp_path / "bundle.zip"

        write_bundle(bundle, manifest, artifacts, document)

        with zipfile.ZipFile(bundle) as archive:
            assert json.loads(archive.read("manifest.json")) == manifest
            assert archive.read("REPRODUCE.md").decode() == document
            assert archive.read("files/uv.lock").decode() == "lock"
