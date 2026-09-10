# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the environment capture bundle."""

from __future__ import annotations

import json
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
    ISAAC_LAB_ENV_VARS,
    collect_environment,
    collect_repo,
    is_collected_env_var,
    lock_extras,
    parse_lock,
    render_document,
    resolve_sync_plan,
    sanitize_remote_url,
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


def _stub_remotes(monkeypatch, remotes: str) -> None:
    """Make ``git remote -v`` return ``remotes`` and every other git command return nothing."""
    monkeypatch.setattr(
        "capture_env._git",
        lambda root, args, timeout=15: remotes if args[:2] == ["remote", "-v"] else "",
    )


class TestEnvironmentAllowlist:
    """The process environment is captured by an exact, closed list of names."""

    @pytest.mark.parametrize(
        "name",
        ["ISAAC_PATH", "EXP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "CARB_APP_PATH", "WARP_CACHE_PATH"],
    )
    def test_variables_isaac_lab_reads_are_collected(self, name):
        assert is_collected_env_var(name)

    @pytest.mark.parametrize(
        "name",
        ["MY_INTERNAL_HOST", "SSH_AUTH_SOCK", "SLACK_WEBHOOK", "AWS_PROFILE", "NGC_API_KEY", "UV_INDEX_URL"],
    )
    def test_names_outside_the_list_are_never_collected_however_they_are_spelled(self, name):
        """No prefix or pattern matching, so a credential cannot arrive under a known namespace."""
        assert not is_collected_env_var(name)

    def test_the_list_holds_exact_names_only(self):
        """A near-miss on a listed name must not match; only the listed spelling counts."""
        assert is_collected_env_var("ISAACLAB_TEST_DEVICES")
        assert not is_collected_env_var("ISAACLAB_TEST_DEVICES_EXTRA")
        assert not is_collected_env_var("MY_ISAAC_PATH")

    def test_uncollected_variables_are_counted_but_never_named(self, monkeypatch):
        monkeypatch.setattr(
            "os.environ",
            {"ISAAC_PATH": "/isaac", "NGC_API_KEY": "secret", "CUSTOMER_INTERNAL_HOST": "host.corp"},
        )
        section, artifacts = collect_environment()

        assert section["variables"] == {"ISAAC_PATH": "/isaac"}
        assert section["omitted_count"] == 2
        rendered = artifacts["env/environment.txt"]
        for leaked in ("CUSTOMER_INTERNAL_HOST", "host.corp", "secret", "NGC_API_KEY"):
            assert leaked not in rendered
            assert leaked not in json.dumps(section)

    def test_every_listed_name_is_upper_case_and_unqualified(self):
        """Guards against a stray lower-case or prefixed entry slipping into the list."""
        assert all(name == name.upper() and not name.startswith("_") for name in ISAAC_LAB_ENV_VARS)


class TestRemoteSanitization:
    """A remote is recorded only on request, and never with the credential a checkout stored."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://ghp_TOKEN@github.com/org/repo.git", "https://github.com/org/repo.git"),
            ("https://oauth2:glpat_TOKEN@gitlab.example.com/org/repo.git", "https://gitlab.example.com/org/repo.git"),
            ("http://user:password@host.corp/repo.git", "http://host.corp/repo.git"),
            ("ssh://user:password@host.corp/repo.git", "ssh://host.corp/repo.git"),
            ("user:password@host.corp:org/repo.git", "host.corp:org/repo.git"),
        ],
    )
    def test_credentials_are_removed(self, url, expected):
        assert sanitize_remote_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/isaac-sim/IsaacLab.git",
            "git@github.com:isaac-sim/IsaacLab.git",
            "ssh://git@github.com:22/isaac-sim/IsaacLab.git",
            "git://github.com/isaac-sim/IsaacLab.git",
            "/srv/git/IsaacLab.git",
        ],
    )
    def test_a_url_without_a_credential_survives_intact(self, url):
        """The clone step is only actionable if a key-authenticated remote is left usable."""
        assert sanitize_remote_url(url) == url

    def test_remotes_are_omitted_by_default(self, tmp_path, monkeypatch):
        """A fork's URL names a host and an organisation the reproduction does not need."""
        _stub_remotes(monkeypatch, "origin\thttps://github.corp.internal/team/repo.git (fetch)")

        section, artifacts = collect_repo(tmp_path, include_diff=False)

        assert section["git"]["remotes_included"] is False
        assert "remotes" not in section["git"]
        assert "repo/git-remote.txt" not in artifacts
        assert "github.corp.internal" not in json.dumps(section)

    def test_neither_the_manifest_nor_the_stored_listing_carries_the_token(self, tmp_path, monkeypatch):
        _stub_remotes(
            monkeypatch,
            "origin\thttps://ghp_TOKEN@github.com/org/repo.git (fetch)\n"
            "origin\thttps://ghp_TOKEN@github.com/org/repo.git (push)\n",
        )

        section, artifacts = collect_repo(tmp_path, include_diff=False, include_remotes=True)

        assert section["git"]["remotes"] == ["https://github.com/org/repo.git"]
        assert section["git"]["remotes_redacted"] is True
        assert "ghp_TOKEN" not in artifacts["repo/git-remote.txt"]
        assert "ghp_TOKEN" not in json.dumps(section)

    def test_a_remote_with_nothing_to_redact_is_not_reported_as_redacted(self, tmp_path, monkeypatch):
        _stub_remotes(monkeypatch, "origin\tgit@github.com:isaac-sim/IsaacLab.git (fetch)")

        section, artifacts = collect_repo(tmp_path, include_diff=False, include_remotes=True)

        assert section["git"]["remotes_redacted"] is False
        assert artifacts["repo/git-remote.txt"] == "origin\tgit@github.com:isaac-sim/IsaacLab.git (fetch)"


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
        """A tracked link arrives with the clone and a venv link is recreated by the sync."""
        manifest = _manifest(
            links={
                "symlinks": [
                    {"path": "/repo/_isaac_sim", "target": "/build", "exists": True, "tracked": False},
                    {"path": "/repo/.agents/skill", "target": "../s", "exists": True, "tracked": True},
                    {
                        "path": "/repo/.venv/bin/python",
                        "target": "/uv/python3.12",
                        "exists": True,
                        "tracked": False,
                        "in_virtualenv": True,
                    },
                ]
            },
        )

        document = render_document(manifest, {})

        assert "ln -s /build _isaac_sim" in document
        assert "skill" not in document
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

        assert "export ISAAC_PATH='/isaac'" in document
        assert "export VIRTUAL_ENV" not in document
        assert "`VIRTUAL_ENV`" in document and "`CONDA_PREFIX`" in document

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
