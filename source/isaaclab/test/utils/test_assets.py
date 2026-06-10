# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
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
