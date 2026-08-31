# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the environment capture bundle."""

from __future__ import annotations

import csv
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
    SCHEMA_VERSION,
    Finding,
    analyze,
    check_record_integrity,
    collect_environment,
    collect_isaac_sim,
    collect_repo,
    find_unlocked_distributions,
    is_collected_env_var,
    load_manifest,
    lock_closure,
    lock_extras,
    parse_lock,
    parse_version_pins,
    render_diff,
    render_document,
    resolve_sync_plan,
    sanitize_remote_url,
    scan_distributions,
    select_sync_extras,
    write_bundle,
)


def _write_distribution(
    site_packages: Path,
    name: str,
    version: str,
    files: dict[str, str],
    recorded_extra: list[str] | None = None,
    installer: str = "uv",
) -> Path:
    """Install a fake distribution, optionally recording files that are never written.

    ``recorded_extra`` is what makes a gutted package: paths listed in ``RECORD`` that do not
    exist on disk, which is the state a package is left in when another wheel overwrote its
    files and was then uninstalled.
    """
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nBody.\n")
    (dist_info / "INSTALLER").write_text(installer)

    for relative, content in files.items():
        target = site_packages / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    with (dist_info / "RECORD").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for relative in list(files) + list(recorded_extra or []):
            writer.writerow([relative, "", ""])
        writer.writerow([f"{dist_info.name}/METADATA", "", ""])
    return dist_info


def _manifest(**sections) -> dict:
    """Return a minimal manifest with ``sections`` merged over the empty defaults."""
    base = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": "2026-01-01T00:00:00Z",
        "capture": {"hostname": "host", "command_under_test": None},
        "host": {"os_release": {}},
        "gpu": {"nvidia_smi_available": True, "vulkan_available": True, "devices": [], "device_count": 0},
        "python": {"interpreter": {}, "distributions": [], "duplicates": [], "pth_files": [], "venv": None},
        "environment": {"variables": {}, "omitted_count": 0},
        "links": {"symlinks": []},
        "repo": {"root": "/repo", "git": {}, "pins": {}},
        "isaac_sim": {},
        "sync": {"lock_available": True, "extras": [], "covers": [], "unlocked": [], "command": "uv sync --locked"},
        "findings": [],
    }
    base.update(sections)
    return base


def _with_damage(digest: str | None) -> dict:
    """Return a manifest with one distribution damaged to a fixed extent, digested as ``digest``.

    ``digest`` is ``None`` for a bundle captured before the digest was recorded.
    """
    entry = {
        "distribution": "usd_core-25.5.dist-info",
        "recorded": 78,
        "missing": 12,
        "examples": ["pxr/Usd/__init__.py"],
    }
    if digest is not None:
        entry["digest"] = digest
    return _manifest(
        python={
            "distributions": [],
            "duplicates": [],
            "pth_files": [],
            "venv": None,
            "integrity": {"damaged": [entry]},
        }
    )


def _with_sys_path(*entries: str) -> dict:
    """Return a manifest whose probed interpreter reported ``entries`` as its ``sys.path``."""
    return _manifest(
        python={
            "distributions": [],
            "duplicates": [],
            "pth_files": [],
            "integrity": None,
            "venv": {"interpreter": {"sys_path": list(entries)}},
        }
    )


# A lockfile shaped like the real one in the details that decide what a sync installs: an
# alias extra defined in terms of other extras, and a requirement that reaches a package only
# through an extra declared after intervening `version` and `source` keys.
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
    { name = "isaacsim", extras = ["all", "extscache"] },
]

[[package]]
name = "torch"
version = "2.11.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }
dependencies = [
    { name = "cuda-toolkit", version = "12.8.1", source = { registry = "https://pypi.nvidia.com/" }, \
extra = ["cufft"], marker = "sys_platform == 'linux'" },
]

[[package]]
name = "cuda-toolkit"
version = "12.8.1"
source = { registry = "https://pypi.nvidia.com/" }

[package.optional-dependencies]
cufft = [
    { name = "nvidia-cufft-cu12" },
]

[[package]]
name = "nvidia-cufft-cu12"
version = "11.3.3.83"
source = { registry = "https://pypi.nvidia.com/" }

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

[[package]]
name = "virtualenv"
version = "21.7.0"
source = { registry = "https://pypi.org/simple" }
"""
"""No extra reaches ``virtualenv``: it is in the lockfile only because something outside the
selectable extras resolved it, which is the state `uv sync` deletes without warning."""


def _installed(**versions: str) -> list[dict]:
    """Return manifest-shaped distribution records for ``name=version`` pairs."""
    return [
        {"name": name.replace("_", "-"), "key": name.replace("_", "-"), "version": version, "installer": "uv"}
        for name, version in versions.items()
    ]


class TestEnvironmentAllowlist:
    """The process environment is captured by an exact, closed list of names."""

    @pytest.mark.parametrize(
        "name",
        ["ISAAC_PATH", "EXP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "CARB_APP_PATH", "WARP_CACHE_PATH"],
    )
    def test_variables_isaac_lab_reads_are_collected(self, name):
        assert is_collected_env_var(name)

    @pytest.mark.parametrize("name", ["MY_INTERNAL_HOST", "SSH_AUTH_SOCK", "SLACK_WEBHOOK", "AWS_PROFILE"])
    def test_unrelated_variables_are_not_collected(self, name):
        assert not is_collected_env_var(name)

    @pytest.mark.parametrize(
        "name",
        ["NGC_API_KEY", "AWS_SECRET_ACCESS_KEY", "ISAACLAB_AUTH_TOKEN", "CARB_APP_PASSWORD", "UV_INDEX_URL"],
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


def _stub_remotes(monkeypatch, remotes: str) -> None:
    """Make ``git remote -v`` return ``remotes`` and every other git command return nothing."""
    monkeypatch.setattr(
        "capture_env._git",
        lambda root, args, timeout=15: remotes if args[:2] == ["remote", "-v"] else "",
    )


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


class TestRecordIntegrity:
    """Installed files are checked against the metadata that claims them."""

    def test_an_intact_distribution_reports_no_damage(self, tmp_path):
        _write_distribution(tmp_path, "intact", "1.0", {"intact/__init__.py": "", "intact/core.py": ""})

        assert check_record_integrity(tmp_path)["damaged"] == []

    def test_a_distribution_missing_recorded_files_is_reported(self, tmp_path):
        """The failure mode that leaves a package importable but empty."""
        _write_distribution(
            tmp_path,
            "usd-exchange",
            "2.3.0",
            files={"pxr/Plug/__init__.pyi": ""},
            recorded_extra=["pxr/Plug/__init__.py", "pxr/Plug/_plug.so"],
        )

        damaged = check_record_integrity(tmp_path)["damaged"]

        assert len(damaged) == 1
        assert damaged[0]["missing"] == 2
        assert "pxr/Plug/__init__.py" in damaged[0]["examples"]

    def test_damage_to_a_different_file_is_told_apart_past_the_stored_examples(self, tmp_path):
        """Only a bounded sample of the missing paths is stored, so the digest carries the rest."""

        def damage(site_packages, tail):
            shared = [f"pxr/Plug/m{index:02d}.py" for index in range(11)]
            _write_distribution(
                site_packages,
                "usd-exchange",
                "2.3.0",
                files={"pxr/Plug/__init__.pyi": ""},
                recorded_extra=[*shared, f"pxr/Plug/{tail}"],
            )
            return check_record_integrity(site_packages)["damaged"][0]

        left = damage(tmp_path / "left", "n_alpha.py")
        right = damage(tmp_path / "right", "n_omega.py")

        assert left["missing"] == right["missing"]
        assert left["examples"] == right["examples"], "the sample must be identical for the digest to be the fix"
        assert left["digest"] != right["digest"]

    def test_byte_compiled_caches_are_not_treated_as_missing(self, tmp_path):
        _write_distribution(
            tmp_path,
            "cached",
            "1.0",
            files={"cached/__init__.py": ""},
            recorded_extra=["cached/__pycache__/__init__.cpython-312.pyc"],
        )

        assert check_record_integrity(tmp_path)["damaged"] == []


class TestDistributionScan:
    """Distributions are read from disk, independently of the running interpreter."""

    def test_names_are_normalized_and_installers_recorded(self, tmp_path):
        _write_distribution(tmp_path, "Isaac_Lab.Tasks", "1.0", {"a.py": ""}, installer="pip")

        (scanned,) = scan_distributions(tmp_path)

        assert scanned.name == "Isaac_Lab.Tasks"
        assert scanned.key == "isaac-lab-tasks"
        assert scanned.installer == "pip"


class TestVersionPins:
    """The pinned-version table drives the drift check."""

    def test_only_the_versions_table_is_parsed(self):
        pins = parse_version_pins(
            '[tool.other]\ntorch = "0.0.0"\n\n[tool.isaaclab.versions]\n# comment\ntorch = "2.11.0"\n'
            'newton = "release-1.5"\n\n[tool.ruff]\nline-length = 120\n'
        )

        assert pins == {"torch": "2.11.0", "newton": "release-1.5"}


class TestLockGraph:
    """The lockfile is read as the graph that decides what a sync installs."""

    def test_the_root_project_and_its_forks_are_identified(self):
        graph = parse_lock(LOCK)

        assert graph["root"] == "demo-dev"
        assert graph["packages"]["torch"]["versions"] == {"2.11.0+cu128"}

    def test_an_alias_extra_is_expanded_into_the_extras_it_names(self):
        """``all`` is defined as the project's own extras and has no requirements of its own."""
        extras = lock_extras(parse_lock(LOCK))

        assert extras["all"] == {"stable-baselines3", "isaacsim"}

    def test_a_package_reached_only_through_an_extra_is_in_the_closure(self):
        """Regression: the extra is declared after `version` and `source`, not next to `name`."""
        closure = lock_closure(parse_lock(LOCK), ["all"])

        assert "nvidia-cufft-cu12" in closure

    def test_an_extra_nobody_selected_is_outside_the_closure(self):
        closure = lock_closure(parse_lock(LOCK), ["all"])

        assert "sphinx" not in closure


class TestSyncExtraSelection:
    """The extras are derived from what is installed, not guessed."""

    def test_an_extra_covered_by_another_is_dropped(self):
        """``all`` subsumes ``sb3`` and ``sim``, so naming all three would be noise."""
        extras = lock_extras(parse_lock(LOCK))
        installed = {"torch", "stable-baselines3", "isaacsim", "cuda-toolkit", "nvidia-cufft-cu12"}

        assert select_sync_extras(extras, installed) == ["all"]

    def test_an_extra_whose_requirements_are_absent_is_not_selected(self):
        extras = lock_extras(parse_lock(LOCK))

        assert select_sync_extras(extras, {"torch", "stable-baselines3"}) == ["sb3"]

    def test_the_command_carries_the_selected_extras(self):
        plan = resolve_sync_plan(LOCK, _installed(torch="2.11.0+cu128", stable_baselines3="2.9.0", isaacsim="6.0.1.0"))

        assert plan["command"] == "uv sync --locked --extra all"

    def test_a_checkout_without_a_lockfile_says_so(self):
        plan = resolve_sync_plan(None, _installed(torch="2.11.0+cu128"))

        assert plan["lock_available"] is False
        assert plan["command"] == "uv sync --locked"


class TestUnlockedDistributions:
    """What a sync would delete is found by comparing against the lockfile, not the installer."""

    def _unlocked(self, **versions: str) -> dict[str, str]:
        graph = parse_lock(LOCK)
        distributions = _installed(**versions)
        closure = lock_closure(graph, select_sync_extras(lock_extras(graph), {d["key"] for d in distributions}))
        return {entry["name"]: entry["reason"] for entry in find_unlocked_distributions(distributions, graph, closure)}

    def test_a_package_the_lockfile_never_mentions_is_reported(self):
        assert "absent" in self._unlocked(torch="2.11.0+cu128", pre_commit="4.6.2")["pre-commit"]

    def test_a_package_no_selected_extra_reaches_is_reported(self):
        """It is in the lockfile at exactly this version, and `uv sync` still removes it."""
        reasons = self._unlocked(torch="2.11.0+cu128", virtualenv="21.7.0")

        assert "no selected extra reaches it" in reasons["virtualenv"]

    def test_a_version_the_lockfile_does_not_pin_is_reported(self):
        reasons = self._unlocked(torch="2.11.0+cu128", stable_baselines3="2.8.0", isaacsim="6.0.1.0")

        assert "version differs" in reasons["stable-baselines3"]

    def test_a_local_version_segment_is_not_a_difference(self):
        """2.11.0+cu128 is the locked build, not a package the sync would replace."""
        assert self._unlocked(torch="2.11.0+cu128") == {}

    def test_uv_pip_installs_are_caught_even_though_uv_recorded_them(self):
        """`uv pip install` writes the same INSTALLER as `uv sync`, so only the lockfile tells."""
        graph = parse_lock(LOCK)
        distributions = [dict(dist, installer="uv") for dist in _installed(torch="2.11.0+cu128", pre_commit="4.6.2")]
        closure = lock_closure(graph, [])

        (unlocked,) = [e for e in find_unlocked_distributions(distributions, graph, closure) if e["name"] != "torch"]

        assert unlocked["installer"] == "uv"


class TestIsaacSimInstallMethod:
    """How Isaac Sim was obtained decides how to obtain the same one."""

    def _repo(self, tmp_path, target: Path | None) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        if target is not None:
            (repo / "_isaac_sim").symlink_to(target)
        return repo

    def _isaac_sim(self, tmp_path, name: str, version: str) -> Path:
        install = tmp_path / name
        install.mkdir(parents=True)
        (install / "VERSION").write_text(f"{version}\n")
        return install

    def test_a_wheel_install_has_no_link(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        section, _ = collect_isaac_sim(self._repo(tmp_path, None), _installed(isaacsim="6.0.1.0"))

        assert section["install_method"] == "wheel"
        assert section["wheel_version"] == "6.0.1.0"

    def test_a_downloaded_package_is_told_apart_from_a_build(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)
        install = self._isaac_sim(tmp_path, "isaacsim", "6.0.1+release.19112.f59b3005.gl.linux-x86_64.release")

        section, _ = collect_isaac_sim(self._repo(tmp_path, install), [])

        assert section["install_method"] == "binary"
        assert section["source_build"] is False

    def test_a_local_build_records_the_revision_it_was_built_from(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)
        build = self._isaac_sim(
            tmp_path, "IsaacSim/_build/linux-x86_64/release", "6.1.0-alpha.59+develop.0.4877ef77.local"
        )

        section, _ = collect_isaac_sim(self._repo(tmp_path, build), [])

        assert section["install_method"] == "source_build"
        assert section["source_revision"] == "4877ef77"
        assert section["source_branch"] == "develop"

    def test_nothing_installed_is_reported_as_such(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ISAAC_PATH", raising=False)

        section, _ = collect_isaac_sim(self._repo(tmp_path, None), [])

        assert section["install_method"] == "none"


class TestAnalysis:
    """Checks report what would change how an environment is reproduced."""

    def test_two_usd_providers_are_an_error(self):
        manifest = _manifest(
            python={
                "distributions": [
                    {"key": "usd-core", "name": "usd-core", "version": "25.11"},
                    {"key": "usd-exchange", "name": "usd-exchange", "version": "2.3.0"},
                ],
                "duplicates": [],
                "pth_files": [],
                "venv": None,
                "integrity": None,
            }
        )

        codes = {finding.code: finding for finding in analyze(manifest)}

        assert codes["usd-provider-conflict"].level == "error"

    def test_a_version_matching_its_pin_apart_from_a_local_segment_is_not_drift(self):
        """2.11.0+cu128 is a build of 2.11.0, not a different version."""
        manifest = _manifest(
            python={
                "distributions": [{"key": "torch", "name": "torch", "version": "2.11.0+cu128"}],
                "duplicates": [],
                "pth_files": [],
                "venv": None,
                "integrity": None,
            },
            repo={"root": "/repo", "git": {}, "pins": {"torch": "2.11.0"}},
        )

        assert "pin-drift" not in {finding.code for finding in analyze(manifest)}

    def test_a_package_pinned_but_absent_is_not_drift(self):
        """Most pinned packages are optional extras; absence is not a mismatch."""
        manifest = _manifest(repo={"root": "/repo", "git": {}, "pins": {"torchvision": "0.26.0"}})

        assert "pin-drift" not in {finding.code for finding in analyze(manifest)}

    def test_a_real_version_mismatch_is_drift(self):
        manifest = _manifest(
            python={
                "distributions": [{"key": "ovrtx", "name": "ovrtx", "version": "0.3.0"}],
                "duplicates": [],
                "pth_files": [],
                "venv": None,
                "integrity": None,
            },
            repo={"root": "/repo", "git": {}, "pins": {"ovrtx": "0.4.1.364340"}},
        )

        (drift,) = [finding for finding in analyze(manifest) if finding.code == "pin-drift"]

        assert "installed 0.3.0" in " ".join(drift.detail)

    def test_a_package_the_sync_would_delete_is_a_warning(self):
        manifest = _manifest(
            sync={
                "lock_available": True,
                "extras": ["all"],
                "covers": [],
                "command": "uv sync --locked --extra all",
                "unlocked": [
                    {"name": "pre-commit", "version": "4.6.2", "installer": "uv", "locked": [], "reason": "absent"}
                ],
            }
        )

        codes = {finding.code: finding for finding in analyze(manifest)}

        assert codes["outside-lockfile"].level == "warning"
        assert any("pre-commit==4.6.2" in line for line in codes["outside-lockfile"].detail)

    def test_a_wheel_beside_a_resolving_link_is_worth_noting(self):
        manifest = _manifest(
            isaac_sim={"link_resolves": True, "wheel_version": "6.0.1.0", "resolved_path": "/build", "version": "6.1.0"}
        )

        codes = {finding.code: finding for finding in analyze(manifest)}

        assert codes["isaac-sim-wheel-and-link"].level == "info"

    def test_a_broken_symlink_is_an_error(self):
        manifest = _manifest(
            links={"symlinks": [{"path": "/repo/_isaac_sim", "target": "/gone", "exists": False}]},
        )

        codes = {finding.code: finding for finding in analyze(manifest)}

        assert codes["broken-symlink"].level == "error"

    def test_an_unstartable_venv_interpreter_is_an_error(self):
        manifest = _manifest(
            python={
                "distributions": [],
                "duplicates": [],
                "pth_files": [],
                "venv": {"path": "/repo/.venv", "interpreter": {"available": False, "error": "boom"}},
                "integrity": None,
            }
        )

        codes = {finding.code: finding for finding in analyze(manifest)}

        assert codes["venv-interpreter-unusable"].level == "error"

    def test_findings_are_ordered_by_severity(self):
        manifest = _manifest(
            gpu={"nvidia_smi_available": True, "vulkan_available": False, "devices": [], "device_count": 0},
            links={"symlinks": [{"path": "/repo/link", "target": "/gone", "exists": False}]},
        )

        levels = [finding.level for finding in analyze(manifest)]

        assert levels == sorted(levels, key=["error", "warning", "info"].index)


class TestBundle:
    """A bundle round-trips, and refuses to be read by an incompatible reader."""

    def test_manifest_and_document_round_trip_through_the_zip(self, tmp_path):
        manifest = _manifest()
        document = render_document(manifest, {"files/uv.lock": "lock"})
        bundle = tmp_path / "bundle.zip"

        write_bundle(bundle, manifest, {"files/uv.lock": "lock"}, document)

        with zipfile.ZipFile(bundle) as archive:
            assert archive.read("files/uv.lock").decode() == "lock"
            assert "# Isaac Lab environment capture" in archive.read("REPRODUCE.md").decode()
        assert load_manifest(bundle) == manifest

    def test_a_manifest_from_another_schema_is_refused(self, tmp_path):
        stale = tmp_path / "manifest.json"
        stale.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))

        with pytest.raises(ValueError, match="schema"):
            load_manifest(stale)

    def test_only_hand_made_symlinks_appear_in_the_recreate_step(self):
        """A tracked link arrives with the clone and a venv link is recreated by the sync."""
        manifest = _manifest(
            repo={"root": "/repo", "git": {}, "pins": {}},
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
        assert "export VIRTUAL_ENV=" not in document
        assert "`CONDA_PREFIX`, `VIRTUAL_ENV`" in document

    def test_the_document_states_what_cannot_be_reproduced(self):
        manifest = _manifest(
            isaac_sim={"source_build": True, "link_target": "/build/release"},
            environment={"variables": {}, "omitted_count": 7},
        )

        document = render_document(manifest, {})

        assert "What this bundle cannot reproduce" in document
        assert "/build/release" in document
        assert "7 variable(s) were present but not collected" in document


class TestReproductionSteps:
    """The document prescribes the commands that actually rebuild the captured environment."""

    def test_the_sync_step_carries_the_derived_extras(self):
        """A bare `uv sync` removes everything the extras contribute, so it must not be emitted."""
        manifest = _manifest(
            sync={
                "lock_available": True,
                "extras": ["all", "test"],
                "covers": [],
                "unlocked": [],
                "command": "uv sync --locked --extra all --extra test",
            }
        )

        document = render_document(manifest, {})

        assert "uv sync --locked --extra all --extra test" in document
        assert "\nuv sync --locked\n" not in document

    def test_packages_the_sync_would_delete_are_reinstalled_afterwards(self):
        manifest = _manifest(
            sync={
                "lock_available": True,
                "extras": [],
                "covers": [],
                "command": "uv sync --locked",
                "unlocked": [
                    {
                        "name": "pre-commit",
                        "version": "4.6.2",
                        "installer": "uv",
                        "locked": [],
                        "reason": "absent from uv.lock",
                    }
                ],
            }
        )

        document = render_document(manifest, {})

        assert "uv pip install --no-deps 'pre-commit==4.6.2'" in document

    def test_a_wheel_install_points_at_the_sync_that_restores_it(self):
        manifest = _manifest(
            isaac_sim={"install_method": "wheel", "wheel_version": "6.0.1.0"},
            sync={
                "lock_available": True,
                "extras": ["isaacsim"],
                "covers": ["isaacsim"],
                "unlocked": [],
                "command": "uv sync --locked --extra isaacsim",
            },
        )

        document = render_document(manifest, {})

        assert "Install the same Isaac Sim wheel" in document
        assert "already restores it" in document

    def test_a_wheel_the_extras_miss_is_installed_explicitly(self):
        manifest = _manifest(isaac_sim={"install_method": "wheel", "wheel_version": "6.0.1.0"})

        document = render_document(manifest, {})

        assert "uv pip install 'isaacsim[all,extscache]==6.0.1.0'" in document

    def test_a_downloaded_package_is_linked_rather_than_rebuilt(self):
        manifest = _manifest(
            isaac_sim={
                "install_method": "binary",
                "version": "6.0.1+release.19112",
                "resolved_path": "/home/other/isaacsim",
                "link_target": "/home/other/isaacsim",
            },
            links={"symlinks": [{"path": "/repo/_isaac_sim", "target": "/home/other/isaacsim", "exists": True}]},
        )

        document = render_document(manifest, {})

        assert "ln -s ${ISAACSIM_PATH} _isaac_sim" in document
        assert "6.0.1+release.19112" in document
        # The generic symlink step would emit a bare `ln -s` to a path that exists only on the
        # captured machine, which is exactly the instruction the Isaac Sim step replaces.
        assert "ln -s /home/other/isaacsim _isaac_sim" not in document

    def test_a_local_build_names_the_revision_to_rebuild(self):
        manifest = _manifest(
            isaac_sim={
                "install_method": "source_build",
                "source_build": True,
                "version": "6.1.0-alpha.59+develop.0.4877ef77.local",
                "source_revision": "4877ef77",
                "source_branch": "develop",
                "resolved_path": "/home/other/IsaacSim/_build/linux-x86_64/release",
            }
        )

        document = render_document(manifest, {})

        assert "git checkout 4877ef77" in document
        assert "ln -s /path/to/IsaacSim/_build/linux-x86_64/release _isaac_sim" in document

    def test_the_verification_step_uses_the_bundled_copy_of_the_tool(self):
        """A checkout at the captured commit may predate this script, or not exist at all."""
        document = render_document(_manifest(), {})

        assert "python3 /path/to/bundle/capture_env.py diff" in document

    def test_the_bundle_is_unpacked_before_anything_copies_out_of_it(self):
        """Later steps read `/path/to/bundle/files/...`, which a zip does not provide."""
        document = render_document(_manifest(), {})

        assert document.index("unzip <this-bundle>.zip") < document.index("cp /path/to/bundle/files/uv.lock")


class TestDiff:
    """Comparing two captures reports every recorded difference."""

    def test_a_package_version_change_is_reported(self):
        def with_torch(version):
            return _manifest(
                python={
                    "distributions": [{"key": "torch", "name": "torch", "version": version}],
                    "duplicates": [],
                    "pth_files": [],
                    "venv": None,
                    "integrity": None,
                }
            )

        report = render_diff(with_torch("2.11.0"), with_torch("2.10.0"))

        assert "| torch | 2.11.0 | 2.10.0 |" in report
        assert "1 difference(s) recorded" in report

    def test_obtaining_isaac_sim_a_different_way_is_a_difference(self):
        """Matching versions through a wheel rather than the customer's build is not a match."""
        report = render_diff(
            _manifest(isaac_sim={"install_method": "source_build", "version": "6.1.0"}),
            _manifest(isaac_sim={"install_method": "wheel", "version": "6.1.0"}),
        )

        assert "| isaac_sim_install_method | source_build | wheel |" in report

    def test_a_different_set_of_extras_is_a_difference(self):
        report = render_diff(
            _manifest(sync={"extras": ["all", "test"], "unlocked": [], "covers": [], "command": ""}),
            _manifest(sync={"extras": ["all"], "unlocked": [], "covers": [], "command": ""}),
        )

        assert "| sync_extras | all test | all |" in report

    def test_a_pth_file_adding_a_different_path_is_a_difference(self):
        """Matching package versions still import different code when a `.pth` differs."""

        def with_pth(*paths):
            return _manifest(
                python={
                    "distributions": [{"key": "torch", "name": "torch", "version": "2.11.0"}],
                    "duplicates": [],
                    "pth_files": [{"name": "_isaaclab.pth", "lines": list(paths)}],
                    "venv": None,
                    "integrity": {"damaged": []},
                }
            )

        report = render_diff(with_pth("/home/user/IsaacLab/source"), with_pth("/opt/IsaacLab/source"))

        assert "`_isaaclab.pth`" in report
        assert "/home/user/IsaacLab/source" in report
        assert "/opt/IsaacLab/source" in report
        assert "1 difference(s) recorded" in report

    def test_an_import_path_the_pth_files_do_not_explain_is_a_difference(self):
        """`PYTHONPATH` and a relocated editable install move entries without touching a `.pth`."""
        report = render_diff(
            _with_sys_path("/venv/site-packages", "/home/user/IsaacLab/source"),
            _with_sys_path("/venv/site-packages", "/opt/IsaacLab/source"),
        )

        assert "Only on `host (baseline)`: `/home/user/IsaacLab/source`" in report
        assert "Only on `host (current)`: `/opt/IsaacLab/source`" in report
        assert "2 difference(s) recorded" in report

    def test_the_same_import_path_in_a_different_order_is_a_difference(self):
        """The first entry satisfying an import wins, so order decides which copy is loaded."""
        report = render_diff(
            _with_sys_path("/repo/source", "/venv/site-packages"),
            _with_sys_path("/venv/site-packages", "/repo/source"),
        )

        assert "Shared entries are ordered differently" in report
        assert "1 difference(s) recorded" in report

    def test_a_capture_without_a_probed_interpreter_says_so_rather_than_reading_as_a_match(self):
        report = render_diff(_manifest(), _manifest())

        assert "Not comparable: at least one capture has no interpreter that reported its `sys.path`." in report

    def test_a_package_gutted_on_one_side_is_a_difference(self):
        """A distribution at the right version with its files gone is not an equivalent one."""

        def with_damage(damaged):
            return _manifest(
                python={
                    "distributions": [{"key": "usd-core", "name": "usd-core", "version": "25.5"}],
                    "duplicates": [],
                    "pth_files": [],
                    "venv": None,
                    "integrity": {"damaged": damaged},
                }
            )

        report = render_diff(
            with_damage([{"distribution": "usd_core-25.5.dist-info", "recorded": 78, "missing": 73, "examples": []}]),
            with_damage([]),
        )

        assert "| `usd_core-25.5.dist-info` | 73 of 78 files missing | *intact* |" in report
        assert "1 difference(s) recorded" in report

    def test_equal_damage_to_a_different_set_of_files_is_a_difference(self):
        """Comparing counts alone calls two packages gutted in different places equivalent."""
        report = render_diff(_with_damage("1111aaaa2222bbbb"), _with_damage("3333cccc4444dddd"))

        assert "a different set of files" in report
        assert "1 difference(s) recorded" in report

    def test_a_bundle_predating_digests_is_not_reported_as_damaged_differently(self):
        """An older bundle cannot say which files were missing, so equal counts are all there is."""
        report = render_diff(_with_damage(None), _with_damage("1111aaaa2222bbbb"))

        assert "No differences recorded" in report

    def test_skipping_the_integrity_check_is_reported_rather_than_read_as_a_match(self):
        report = render_diff(_manifest(), _manifest())

        assert "Not comparable: at least one capture ran with `--skip_integrity`." in report

    def test_findings_sharing_a_code_are_compared_one_by_one(self):
        """`missing-package-files` is raised per distribution; the code alone hides which."""

        def damaged_in(name):
            return _manifest(
                findings=[
                    {"level": "error", "code": "missing-package-files", "summary": f"{name} is gutted", "detail": []}
                ]
            )

        report = render_diff(damaged_in("usd-core"), damaged_in("isaacsim-core"))

        assert "usd-core is gutted" in report
        assert "isaacsim-core is gutted" in report
        assert "2 difference(s) recorded" in report

    def test_a_finding_reported_by_both_with_different_detail_is_a_difference(self):
        """The same summary over a different set of missing files is not the same state."""

        def missing(*paths):
            return _manifest(
                findings=[
                    {
                        "level": "error",
                        "code": "missing-package-files",
                        "summary": "usd-core is missing 2 of 78 files",
                        "detail": [f"  missing: {path}" for path in paths],
                    }
                ]
            )

        report = render_diff(missing("pxr/Usd/__init__.py", "pxr/Sdf/__init__.py"), missing("pxr/Gf/__init__.py"))

        assert "Reported by both with different detail" in report
        assert "pxr/Usd/__init__.py" in report
        assert "pxr/Gf/__init__.py" in report
        assert "1 difference(s) recorded" in report

    def test_identical_captures_report_no_differences(self):
        report = render_diff(_manifest(), _manifest())

        assert "No differences recorded" in report

    def test_every_section_names_itself_even_when_it_matches(self):
        """The report doubles as the list of what was checked, so nothing may be dropped."""
        report = render_diff(_manifest(), _manifest())

        for section in (
            "## Host and versions",
            "## Packages",
            "## Environment variables",
            "## Symlinks",
            "## Import path files",
            "## Resolved import path",
            "## Package integrity",
            "## Findings",
        ):
            assert section in report

    def test_columns_are_disambiguated_when_both_captures_share_a_hostname(self):
        report = render_diff(_manifest(), _manifest())

        assert "host (baseline)" in report
        assert "host (current)" in report


def test_findings_carry_an_actionable_repair_for_missing_files():
    """A finding nobody can act on is worse than no finding."""
    manifest = _manifest(
        python={
            "distributions": [],
            "duplicates": [],
            "pth_files": [],
            "venv": None,
            "integrity": {
                "damaged": [
                    {"distribution": "usd_exchange-2.3.0.dist-info", "recorded": 78, "missing": 73, "examples": []}
                ]
            },
        }
    )

    (finding,) = [f for f in analyze(manifest) if f.code == "missing-package-files"]

    assert isinstance(finding, Finding)
    assert any("uv pip install --reinstall-package" in line for line in finding.detail)
