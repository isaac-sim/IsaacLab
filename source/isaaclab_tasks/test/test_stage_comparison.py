# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage snapshots detect structural/pose changes while tolerating generated product names."""

from pathlib import Path

import pytest
from stage_comparison import compare_golden_stage

pytestmark = pytest.mark.kitless


def _usda_with_robot(translate: tuple[float, float, float] = (0.0, 0.0, 0.0), *, extra_prim: bool = False) -> str:
    """Build a self-contained USDA stage with a translated ``/World/Robot`` Xform (no asset refs)."""
    robot = (
        '    def Xform "Robot"\n'
        "    {\n"
        f"        double3 xformOp:translate = ({translate[0]}, {translate[1]}, {translate[2]})\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "    }\n"
    )
    extra = '    def Xform "Extra"\n    {\n    }\n' if extra_prim else ""
    return f'#usda 1.0\ndef Xform "World"\n{{\n{robot}{extra}}}\n'


def _usda_with_hydra_render_product(render_product_name: str, *, extra_prim: bool = False) -> str:
    """Build a USDA stage with one Isaac RTX HydraTextures render product under ``/Render``."""
    extra = '    def Xform "Extra"\n    {\n    }\n' if extra_prim else ""
    return (
        "#usda 1.0\n"
        'def Xform "Render"\n'
        "{\n"
        '    def Xform "OmniverseKit"\n'
        "    {\n"
        '        def Xform "HydraTextures"\n'
        "        {\n"
        f'            def RenderProduct "{render_product_name}"\n'
        "            {\n"
        "            }\n"
        "        }\n"
        "    }\n"
        f"{extra}"
        "}\n"
    )


def test_compare_golden_stage_reports_structure_and_transform_diffs(tmp_path: Path):
    """compare_golden_stage returns no problems when equal and flags added prims and moved transforms."""
    golden = tmp_path / "golden.usda"
    result = tmp_path / "result.usda"

    golden.write_text(_usda_with_robot((1.0, 2.0, 3.0)), encoding="utf-8")
    result.write_text(_usda_with_robot((1.0, 2.0, 3.0)), encoding="utf-8")
    assert compare_golden_stage(str(golden), str(result)) == []

    result.write_text(_usda_with_robot((1.0, 2.0, 3.0 + 1e-7)), encoding="utf-8")
    assert compare_golden_stage(str(golden), str(result)) == []

    result.write_text(_usda_with_robot((1.0, 2.0, 3.0), extra_prim=True), encoding="utf-8")
    problems = compare_golden_stage(str(golden), str(result))
    assert any("added prim" in problem and "/World/Extra" in problem for problem in problems)

    result.write_text(_usda_with_robot((1.0, 2.0, 9.0)), encoding="utf-8")
    problems = compare_golden_stage(str(golden), str(result))
    assert any("transform" in problem and "/World/Robot" in problem for problem in problems)


def test_compare_golden_stage_canonicalizes_replicator_and_uuid_render_products(tmp_path: Path):
    """Legacy ``Replicator`` and UUID ``rp_<hex>`` render-product names compare as equivalent."""
    golden = tmp_path / "golden.usda"
    result = tmp_path / "result.usda"

    golden.write_text(_usda_with_hydra_render_product("Replicator"), encoding="utf-8")
    result.write_text(_usda_with_hydra_render_product("rp_fc992bfc10fd420fa593cd00a09f2ad6"), encoding="utf-8")
    assert compare_golden_stage(str(golden), str(result)) == []

    result.write_text(
        _usda_with_hydra_render_product("rp_fc992bfc10fd420fa593cd00a09f2ad6", extra_prim=True), encoding="utf-8"
    )
    problems = compare_golden_stage(str(golden), str(result))
    assert any("added prim" in problem and "/Render/Extra" in problem for problem in problems)
