# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ovphysx articulation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp

from pxr import Sdf, Usd, UsdPhysics

# The CI isaaclab_ov* pattern unintentionally collects isaaclab_ovphysx tests,
# but the ovphysx wheel is not installed in that environment. Skip gracefully
# so the isaaclab_ov CI pipeline is not blocked by an unrelated dependency.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.assets.articulation.articulation import Articulation  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxManager  # noqa: E402
from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet  # noqa: E402

wp.init()


def _define_tendon_joint(stage: Usd.Stage, path: str, schema_name: str) -> None:
    """Define a revolute joint prim with a tendon schema marker."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    schemas = Sdf.TokenListOp()
    schemas.explicitItems = [schema_name]
    joint.GetPrim().SetMetadata("apiSchemas", schemas)


def _make_articulation_root_stage(tmp_path) -> str:
    """Create a stage with one relevant articulation subtree and unrelated joints elsewhere."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot/root", "Xform")
    stage.DefinePrim("/World/unrelated", "Xform")

    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/spatial_joint",
        "PhysxTendonAttachmentRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_spatial_joint",
        "PhysxTendonAttachmentLeafAPI:inst0",
    )

    stage_path = tmp_path / "scene.usda"
    stage.Export(str(stage_path))
    return str(stage_path)


def _make_articulation_shell() -> Articulation:
    """Create a minimal ovphysx articulation shell for tendon processing tests."""
    articulation = object.__new__(Articulation)
    bindings = MockOvPhysxBindingSet(
        num_instances=1,
        num_joints=2,
        num_bodies=2,
        num_fixed_tendons=1,
        num_spatial_tendons=1,
    )
    object.__setattr__(articulation, "_bindings", bindings.bindings)
    object.__setattr__(articulation, "_articulation_root_path", "/World/envs/env_0/Robot/root")
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(
        articulation,
        "_data",
        SimpleNamespace(
            _num_fixed_tendons=0,
            _num_spatial_tendons=0,
            fixed_tendon_names=[],
            spatial_tendon_names=[],
        ),
    )
    return articulation


def test_process_tendons_scopes_to_articulation_root(tmp_path):
    """Tendon discovery should ignore joints that live outside the current articulation subtree."""
    articulation = _make_articulation_shell()
    stage_path = _make_articulation_root_stage(tmp_path)
    old_stage_path = OvPhysxManager._stage_path
    OvPhysxManager._stage_path = stage_path
    try:
        articulation._process_tendons()
    finally:
        OvPhysxManager._stage_path = old_stage_path

    assert articulation.fixed_tendon_names == ["fixed_joint"]
    assert articulation.spatial_tendon_names == ["spatial_joint"]
    assert articulation._data.fixed_tendon_names == ["fixed_joint"]
    assert articulation._data.spatial_tendon_names == ["spatial_joint"]
