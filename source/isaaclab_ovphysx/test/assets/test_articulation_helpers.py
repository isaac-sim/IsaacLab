# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-only unit tests for articulation helpers.

These tests cover OVPhysX-specific scaffolding (USD tendon-scope resolution,
mock binding-set shape contracts) that has no PhysX equivalent and therefore
does not appear in the PhysX-mirrored ``test_articulation.py``.
"""

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


def test_mock_binding_set_rigid_object_shapes():
    pytest.importorskip("isaaclab_ovphysx.tensor_types").RIGID_BODY_POSE  # gates on wheel
    from isaaclab_ovphysx import tensor_types as TT
    from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet

    bindings = MockOvPhysxBindingSet(
        num_instances=4,
        num_joints=0,
        num_bodies=1,
        asset_kind="rigid_object",
    )
    assert bindings.bindings[TT.RIGID_BODY_POSE].shape == (4, 7)
    assert bindings.bindings[TT.RIGID_BODY_VELOCITY].shape == (4, 6)
    assert bindings.bindings[TT.RIGID_BODY_WRENCH].shape == (4, 9)
    assert bindings.bindings[TT.RIGID_BODY_MASS].shape == (4,)
    assert bindings.bindings[TT.RIGID_BODY_INERTIA].shape == (4, 9)
    # Articulation-only bindings must be absent
    assert TT.DOF_POSITION not in bindings.bindings
    assert TT.LINK_WRENCH not in bindings.bindings
