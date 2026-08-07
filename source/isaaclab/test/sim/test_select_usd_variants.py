# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for variant selection at spawn time, including the required-set guarantees."""

import pytest

from pxr import Usd, UsdGeom

from isaaclab.sim.utils import select_usd_variants

pytestmark = pytest.mark.unit


def _stage_with_variants(sets: dict[str, tuple[str, ...]]):
    """Build an in-memory stage whose ``/World/Robot`` prim carries ``sets``.

    Args:
        sets: Variant set names mapped to the variants each one offers.

    Returns:
        The stage and the prim path.
    """
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    for set_name, variants in sets.items():
        variant_set = prim.GetVariantSets().AddVariantSet(set_name)
        for variant in variants:
            variant_set.AddVariant(variant)
    return stage, "/World/Robot"


def _selection(stage, prim_path: str, set_name: str) -> str:
    return stage.GetPrimAtPath(prim_path).GetVariantSets().GetVariantSet(set_name).GetVariantSelection()


def test_selecting_an_offered_variant_applies_it():
    """The ordinary path is unchanged."""
    stage, prim_path = _stage_with_variants({"Physics": ("none", "physics", "physx")})

    select_usd_variants(prim_path, {"Physics": "physx"}, stage=stage)

    assert _selection(stage, prim_path, "Physics") == "physx"


def test_missing_optional_variant_set_is_skipped():
    """A cosmetic set the asset lacks stays a warning, so one config can span several assets."""
    stage, prim_path = _stage_with_variants({"Physics": ("none", "physics")})

    select_usd_variants(prim_path, {"colour": "red"}, stage=stage)

    assert "colour" not in stage.GetPrimAtPath(prim_path).GetVariantSets().GetNames()


def test_missing_required_variant_set_raises():
    """An asset with no physics description cannot satisfy a physics selection."""
    stage, prim_path = _stage_with_variants({"colour": ("red", "blue")})

    with pytest.raises(ValueError, match="Variant set 'Physics' does not exist"):
        select_usd_variants(prim_path, {"Physics": "physics"}, stage=stage)


def test_unoffered_variant_of_a_required_set_raises():
    """USD accepts an unknown selection and composes nothing, which loses the physics silently."""
    stage, prim_path = _stage_with_variants({"Physics": ("none", "physics")})

    with pytest.raises(ValueError, match="has no variant 'physx'"):
        select_usd_variants(prim_path, {"Physics": "physx"}, stage=stage)

    assert _selection(stage, prim_path, "Physics") == ""


def test_unoffered_variant_of_an_optional_set_is_left_alone():
    """Optional sets keep their lenient behaviour; only required sets gained a check."""
    stage, prim_path = _stage_with_variants({"colour": ("red", "blue")})

    select_usd_variants(prim_path, {"colour": "green"}, stage=stage)

    assert _selection(stage, prim_path, "colour") == "green"
