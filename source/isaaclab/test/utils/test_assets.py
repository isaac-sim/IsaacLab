# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import importlib
from pathlib import Path

import pytest

import isaaclab.utils.assets as assets_utils


def test_nucleus_connection():
    """Test checking the Nucleus connection."""
    # check nucleus connection
    assert assets_utils.NUCLEUS_ASSET_ROOT_DIR is not None


def test_check_file_path_nucleus():
    """Test checking a file path on the Nucleus server."""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
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
