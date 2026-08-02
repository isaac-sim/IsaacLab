# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import isaaclab.utils.assets as assets_utils

pytestmark = pytest.mark.unit


def test_asset_root_environment_override_takes_precedence(monkeypatch):
    """Test the documented Isaac Sim asset-root environment override."""
    monkeypatch.setenv("ISAACSIM_ASSET_ROOT", "/tmp/isaacsim_assets/Assets/Isaac/6.0/")
    monkeypatch.setattr(assets_utils, "_parse_kit_asset_root", lambda: "https://example.com/kit-assets")

    assert assets_utils._resolve_asset_root() == "/tmp/isaacsim_assets/Assets/Isaac/6.0"


def test_asset_root_falls_back_to_kit_file(monkeypatch):
    """Test kitless asset-root resolution when the environment override is absent."""
    monkeypatch.delenv("ISAACSIM_ASSET_ROOT", raising=False)
    monkeypatch.setattr(assets_utils, "_parse_kit_asset_root", lambda: "https://example.com/kit-assets")

    assert assets_utils._resolve_asset_root() == "https://example.com/kit-assets"


def test_asset_root_environment_override_strips_windows_separator(monkeypatch):
    """Test the documented Windows form of the environment override."""
    monkeypatch.setenv("ISAACSIM_ASSET_ROOT", "C:\\assets\\Assets\\Isaac\\6.0\\")
    monkeypatch.setattr(assets_utils, "_parse_kit_asset_root", lambda: "https://example.com/kit-assets")

    assert assets_utils._resolve_asset_root() == "C:\\assets\\Assets\\Isaac\\6.0"


def test_asset_root_ignores_empty_environment_override(monkeypatch):
    """Test an empty override falls back, matching when ``isaacsim.storage.native`` skips it."""
    monkeypatch.setenv("ISAACSIM_ASSET_ROOT", "")
    monkeypatch.setattr(assets_utils, "_parse_kit_asset_root", lambda: "https://example.com/kit-assets")

    assert assets_utils._resolve_asset_root() == "https://example.com/kit-assets"


def test_kit_experience_path_resolves_to_the_shipped_experience():
    """Test the unpatched experience-file path so a broken relative walk fails here."""
    assert Path(assets_utils._KIT_EXPERIENCE_PATH).is_file()
    assert assets_utils._parse_kit_asset_root()


def test_kit_asset_root_prefers_default_setting(tmp_path, monkeypatch):
    """Test the experience-file fallback reads the setting that Isaac Sim resolves."""
    kit_file = tmp_path / "isaaclab.python.kit"
    kit_file.write_text(
        "[settings]\n"
        'persistent.isaac.asset_root.default = "https://example.com/default-assets"\n'
        'persistent.isaac.asset_root.cloud = "https://example.com/cloud-assets"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(assets_utils, "_KIT_EXPERIENCE_PATH", str(kit_file))

    assert assets_utils._parse_kit_asset_root() == "https://example.com/default-assets"


def test_kit_asset_root_falls_back_to_cloud_setting(tmp_path, monkeypatch):
    """Test the experience-file fallback still reads legacy files without a default setting."""
    kit_file = tmp_path / "isaaclab.python.kit"
    kit_file.write_text(
        '[settings]\npersistent.isaac.asset_root.cloud = "https://example.com/cloud-assets"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(assets_utils, "_KIT_EXPERIENCE_PATH", str(kit_file))

    assert assets_utils._parse_kit_asset_root() == "https://example.com/cloud-assets"


def test_exported_asset_root_constants_follow_environment_override(monkeypatch):
    """Test the exported constants, not just the resolver, honor the environment override."""
    try:
        # patch in a nested context so leaving it cannot revert patches owned by other fixtures
        with monkeypatch.context() as patched_env:
            patched_env.setenv("ISAACSIM_ASSET_ROOT", "/tmp/isaacsim_assets/Assets/Isaac/6.0")
            module = importlib.reload(assets_utils)

            assert module.NUCLEUS_ASSET_ROOT_DIR == "/tmp/isaacsim_assets/Assets/Isaac/6.0"
            assert module.ISAAC_NUCLEUS_DIR == "/tmp/isaacsim_assets/Assets/Isaac/6.0/Isaac"
            assert module.ISAACLAB_NUCLEUS_DIR == "/tmp/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab"
    finally:
        # the context restored the caller's environment, so a suite run with a real
        # ISAACSIM_ASSET_ROOT keeps resolving to it
        importlib.reload(assets_utils)


def test_nucleus_connection():
    """Test checking the Nucleus connection."""
    # check nucleus connection
    assert assets_utils.NUCLEUS_ASSET_ROOT_DIR is not None


def test_check_file_path_nucleus():
    """Test checking a file path on the Nucleus server."""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 2


def test_check_file_path_invalid():
    """Test checking an invalid file path."""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_xyz.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 0


def test_find_asset_dependencies_collects_mdl_texture_resources(tmp_path):
    """Test collecting texture resources from quoted MDL strings."""
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        export material Example(*) = OmniPBR(
            diffuse_texture: texture_2d("./textures/Albedo.png", ::tex::gamma_srgb),
            normalmap_texture: texture_2d("../shared/Normal.EXR", ::tex::gamma_linear),
            ORM_texture: texture_2d("https://example.com/materials/orm.<UDIM>.png", ::tex::gamma_linear),
            roughness_texture: texture_2d("omniverse://server/Library/roughness.tx", ::tex::gamma_linear),
            ignored_label: "not_a_texture",
            empty_texture: texture_2d()
        );
        // texture_2d("./textures/commented_line.png")
        /* texture_2d("./textures/commented_block.png") */
        """,
        encoding="utf-8",
    )

    assert assets_utils._find_asset_dependencies(str(mdl_path)) == {
        "./textures/Albedo.png",
        "../shared/Normal.EXR",
        "https://example.com/materials/orm.<UDIM>.png",
        "omniverse://server/Library/roughness.tx",
    }


def test_find_asset_dependencies_collects_mdl_relative_import_modules(tmp_path):
    """Test collecting sibling MDL modules imported by material files."""
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        import .::OmniUe4Function;
        import .::OmniUe4Translucent::*;
        import .::Shared::OmniUe4Base::*;
        import .::Helpers::make_color;
        import ..::Common::Surface::*;
        export using .::Local::Palette import *;
        using ..::Shared::Functions import make_color, make_normal;
        import ::nvidia::core_definitions::*;
        export material Example(*) = OmniPBR();
        // import .::CommentedLine;
        /* import .::CommentedBlock; */
        string ignored = "import .::QuotedString;";
        """,
        encoding="utf-8",
    )

    assert assets_utils._find_asset_dependencies(str(mdl_path)) == {
        "OmniUe4Function.mdl",
        "OmniUe4Translucent.mdl",
        "Shared/OmniUe4Base.mdl",
        "Helpers.mdl",
        "Helpers/make_color.mdl",
        "../Common/Surface.mdl",
        "Local.mdl",
        "Local/Palette.mdl",
        "../Shared.mdl",
        "../Shared/Functions.mdl",
    }


def test_find_asset_dependencies_missing_mdl_does_not_log_traceback(tmp_path, caplog):
    """Test unavailable MDL dependencies do not emit tracebacks in training logs."""
    missing_mdl = tmp_path / "missing.mdl"

    assert assets_utils._find_asset_dependencies(str(missing_mdl)) == set()
    assert "Traceback (most recent call last):" not in caplog.text


def test_retrieve_git_asset_path_uses_local_repo_path(tmp_path):
    """Test retrieving an asset from a local git asset repository."""
    repo_dir = tmp_path / "newton-assets"
    asset_dir = repo_dir / "Robots" / "Disney" / "ExampleBot"
    asset_dir.mkdir(parents=True)
    (asset_dir / "example_bot.usd").write_text("#usda 1.0\n", encoding="utf-8")

    asset_path = Path(assets_utils.retrieve_git_asset_path(str(repo_dir), "Robots/Disney/ExampleBot"))

    assert asset_path == asset_dir
    assert (asset_path / "example_bot.usd").read_text(encoding="utf-8") == "#usda 1.0\n"


def test_retrieve_git_asset_path_clones_default_repo_cache(tmp_path, monkeypatch):
    """Test that git assets are pulled into the default asset cache directory."""
    git_commands = []
    git_path = "https://example.com/example-assets.git"

    def mock_run_git_command(command):
        git_commands.append(command)
        repo_dir = tmp_path / "tmp" / "asset_cache" / "example-assets"
        asset_dir = repo_dir / "Robots" / "Disney" / "ExampleBot"
        asset_dir.mkdir(parents=True)
        (repo_dir / ".git").mkdir()
        (asset_dir / "example_bot.usd").write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(assets_utils, "GIT_ASSET_CACHE_DIR", str(tmp_path / "tmp" / "asset_cache"))
    monkeypatch.setattr(assets_utils, "_run_git_command", mock_run_git_command)

    asset_path = Path(assets_utils.retrieve_git_asset_path(git_path, "Robots/Disney/ExampleBot"))

    assert asset_path == tmp_path / "tmp" / "asset_cache" / "example-assets" / "Robots" / "Disney" / "ExampleBot"
    assert (asset_path / "example_bot.usd").read_text(encoding="utf-8") == "#usda 1.0\n"
    assert git_commands == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            git_path,
            str(tmp_path / "tmp" / "asset_cache" / "example-assets"),
        ]
    ]


def test_retrieve_git_asset_path_uses_cached_asset_without_git(tmp_path, monkeypatch):
    """Test that cached git assets can be used without running git commands."""
    git_path = "https://example.com/example-assets.git"
    cache_dir = tmp_path / "asset_cache"
    asset_dir = cache_dir / "example-assets" / "Robots" / "Disney" / "ExampleBot"
    asset_dir.mkdir(parents=True)
    (asset_dir / "example_bot.usd").write_text("#usda 1.0\n", encoding="utf-8")

    def fail_run_git_command(command):
        raise AssertionError(f"git should not be called for cached asset: {command}")

    monkeypatch.setattr(assets_utils, "_run_git_command", fail_run_git_command)

    asset_path = Path(
        assets_utils.retrieve_git_asset_path(git_path, "Robots/Disney/ExampleBot", cache_dir=str(cache_dir))
    )

    assert asset_path == asset_dir
    assert (asset_path / "example_bot.usd").read_text(encoding="utf-8") == "#usda 1.0\n"


def test_retrieve_git_asset_path_raises_for_missing_asset(tmp_path):
    """Test that git asset retrieval raises when the requested asset is missing."""
    repo_dir = tmp_path / "newton-assets"
    repo_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Unable to find git asset"):
        assets_utils.retrieve_git_asset_path(str(repo_dir), "Robots/Disney/ExampleBot")


_REMOTE_URL = "https://example.com/Assets/Isaac/Robots/example.usd"


@pytest.fixture
def asset_cache(tmp_path, monkeypatch):
    """Isolate the local asset cache: an empty cache directory and no state from other tests."""
    monkeypatch.setattr(assets_utils.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(assets_utils, "_REMOTE_FINGERPRINTS", {})
    monkeypatch.setattr(assets_utils, "_ANNOUNCED_MIRROR_DIRS", set())
    monkeypatch.setattr(assets_utils, "_ANNOUNCED_MIRRORS", set())
    return tmp_path


def _serve(monkeypatch, entries: dict[str, dict | None], payloads: dict[str, bytes] | None = None) -> None:
    """Fake the asset server: ``entries`` maps a URL to its reported metadata (``None`` = absent)."""
    import omni.client

    def fake_stat(url, *args, **kwargs):
        reported = entries.get(url)
        if reported is None:
            return omni.client.Result.ERROR_NOT_FOUND, SimpleNamespace(hash="", version="", size=0, modified_time="")
        return omni.client.Result.OK, SimpleNamespace(**reported)

    def fake_read_file(url, *args, **kwargs):
        if payloads is None or url not in payloads:
            raise AssertionError(f"the server should not have been read for: {url}")
        return omni.client.Result.OK, {}, payloads[url]

    monkeypatch.setattr(omni.client, "stat", fake_stat)
    monkeypatch.setattr(omni.client, "read_file", fake_read_file)


def _cache_asset(cache_dir, url: str, payload: bytes, fingerprint: dict | None) -> Path:
    """Write a locally cached copy of ``url``, with its recorded remote revision."""
    mirrored = Path(assets_utils._mirror_path(url, str(cache_dir)))
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_bytes(payload)
    if fingerprint is not None:
        (mirrored.parent / (mirrored.name + assets_utils._MIRROR_FINGERPRINT_SUFFIX)).write_text(
            json.dumps(fingerprint), encoding="utf-8"
        )
    return mirrored


def test_read_file_uses_the_local_copy_when_it_matches_the_server(asset_cache, monkeypatch):
    """Test an unchanged remote asset is read from disk instead of downloaded again."""
    revision = {"hash": "abc123", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", revision)
    # no payload is served, so any read from the server fails the test
    _serve(monkeypatch, {_REMOTE_URL: revision})

    assert assets_utils.read_file(_REMOTE_URL).read() == b"cached bytes"


def test_read_file_refetches_when_the_server_copy_changed(asset_cache, monkeypatch):
    """Test a changed remote asset is downloaded again rather than served stale."""
    stale = {"hash": "abc123", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    current = {"hash": "def456", "version": "", "size": 11, "modified_time": "2026-07-29 10:00:00"}
    mirrored = _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", stale)
    _serve(monkeypatch, {_REMOTE_URL: current}, payloads={_REMOTE_URL: b"fresh bytes"})

    assert assets_utils.read_file(_REMOTE_URL).read() == b"fresh bytes"
    # the refreshed copy is cached under the new revision, so the next run reads it from disk
    assert mirrored.read_bytes() == b"fresh bytes"
    fingerprint = mirrored.parent / (mirrored.name + assets_utils._MIRROR_FINGERPRINT_SUFFIX)
    assert json.loads(fingerprint.read_text(encoding="utf-8")) == current


def test_local_copy_without_a_recorded_revision_is_refetched(asset_cache, monkeypatch):
    """Test copies left by earlier versions are re-fetched once instead of trusted blindly."""
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", fingerprint=None)
    current = {"hash": "def456", "version": "", "size": 11, "modified_time": "2026-07-29 10:00:00"}
    _serve(monkeypatch, {_REMOTE_URL: current}, payloads={_REMOTE_URL: b"fresh bytes"})

    assert assets_utils.read_file(_REMOTE_URL).read() == b"fresh bytes"


def test_size_and_modification_time_identify_a_revision(asset_cache, monkeypatch):
    """Test providers that report no hash still get a freshness check.

    The local file provider reports only a size and a modification time, and HTTP hosts
    behave the same way, so the check cannot rely on a content hash being available.
    """
    revision = {"hash": "", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", revision)
    _serve(monkeypatch, {_REMOTE_URL: revision})
    assert assets_utils._usable_mirror(_REMOTE_URL)

    assets_utils._REMOTE_FINGERPRINTS.clear()
    _serve(monkeypatch, {_REMOTE_URL: {**revision, "modified_time": "2026-07-29 10:00:00"}})
    assert not assets_utils._usable_mirror(_REMOTE_URL)


def test_unreachable_server_falls_back_to_the_local_copy(asset_cache, monkeypatch, caplog):
    """Test offline runs keep working, with the missing freshness guarantee announced."""
    revision = {"hash": "abc123", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", revision)
    _serve(monkeypatch, {_REMOTE_URL: None})

    with caplog.at_level(logging.WARNING, logger=assets_utils.logger.name):
        assert assets_utils.read_file(_REMOTE_URL).read() == b"cached bytes"

    assert "did not respond" in caplog.text
    assert "may be out of date" in caplog.text


def test_server_without_revision_metadata_is_announced(asset_cache, monkeypatch, caplog):
    """Test a provider reporting nothing to compare cannot silently serve a stale copy."""
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", {"hash": "", "version": "", "size": 0, "modified_time": ""})
    _serve(monkeypatch, {_REMOTE_URL: {"hash": "", "version": "", "size": 0, "modified_time": ""}})

    with caplog.at_level(logging.WARNING, logger=assets_utils.logger.name):
        assert assets_utils.read_file(_REMOTE_URL).read() == b"cached bytes"

    assert "no revision metadata" in caplog.text


def test_two_hosts_serving_the_same_path_do_not_share_a_local_copy(asset_cache, monkeypatch):
    """Test a cloud and an on-prem server exposing the same layout get separate cache entries.

    The on-prem server is unreachable, the case where an unrelated copy would otherwise be
    accepted without a freshness check.
    """
    revision = {"hash": "abc123", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    on_prem_url = _REMOTE_URL.replace("example.com", "nucleus.example-lab.com")
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", revision)
    _serve(monkeypatch, {_REMOTE_URL: revision, on_prem_url: None})

    assert not assets_utils._usable_mirror(on_prem_url)
    assert assets_utils.check_file_path(on_prem_url) == 0


def test_using_local_copies_is_announced_once_per_cache_directory(asset_cache, monkeypatch, caplog):
    """Test the announcement is visible at the default log level without one line per asset."""
    revision = {"hash": "abc123", "version": "", "size": 12, "modified_time": "2026-07-01 10:00:00"}
    other_url = _REMOTE_URL.replace("example.usd", "other.usd")
    _cache_asset(asset_cache, _REMOTE_URL, b"cached bytes", revision)
    _cache_asset(asset_cache, other_url, b"cached bytes", revision)
    _serve(monkeypatch, {_REMOTE_URL: revision, other_url: revision})

    with caplog.at_level(logging.INFO, logger=assets_utils.logger.name):
        assets_utils.read_file(_REMOTE_URL)
        assets_utils.read_file(other_url)

    banners = [record for record in caplog.records if record.levelno == logging.WARNING]
    per_asset = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(banners) == 1
    assert str(asset_cache) in banners[0].getMessage()
    assert len(per_asset) == 2
    assert {_REMOTE_URL, other_url} == {record.args[0] for record in per_asset}


def test_newton_asset_dir_uses_environment_override(tmp_path, monkeypatch):
    """Test that the Newton asset directory is defined from the environment."""
    repo_dir = tmp_path / "newton-assets"
    monkeypatch.setenv("NEWTON_ASSET_DIR", str(repo_dir))

    try:
        module = importlib.reload(assets_utils)
        assert str(repo_dir) == module.NEWTON_ASSET_DIR
    finally:
        monkeypatch.delenv("NEWTON_ASSET_DIR", raising=False)
        importlib.reload(assets_utils)
