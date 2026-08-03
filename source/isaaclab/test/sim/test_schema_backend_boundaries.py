# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that core schema writers remain independent of physics backends."""

import ast
from pathlib import Path

import pytest

from pxr import Sdf, Usd, UsdPhysics

_SCHEMAS_SOURCE = Path(__file__).parents[2] / "isaaclab" / "sim" / "schemas" / "schemas.py"
_FROM_FILES_SOURCE = Path(__file__).parents[2] / "isaaclab" / "sim" / "spawners" / "from_files" / "from_files.py"
_BACKEND_PACKAGES = ("isaaclab_newton", "isaaclab_physx", "isaaclab_ovphysx")


def test_file_spawner_excludes_function_local_backend_imports():
    """Keep backend package imports out of core file-spawner functions."""
    tree = ast.parse(_FROM_FILES_SOURCE.read_text(encoding="utf-8"))
    violations = []
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(function):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(_BACKEND_PACKAGES):
                violations.append(f"{function.name}: from {node.module}")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{function.name}: import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(_BACKEND_PACKAGES)
                )

    assert violations == []


def test_fixed_tendon_writer_excludes_backend_schema_tokens():
    """Keep backend schema names out of the core fixed-tendon writer."""
    source = _SCHEMAS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    writer = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "modify_fixed_tendon_properties"
    )
    writer_source = ast.get_source_segment(source, writer)

    assert writer_source is not None
    assert "MjcTendon" not in writer_source
    assert '"mjc:' not in writer_source


def test_legacy_fixed_tendon_cfg_dispatches_to_mujoco_backend():
    """Preserve legacy fixed-tendon behavior through the Newton backend hook."""
    from isaaclab_newton.sim import schemas as newton_schemas  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxFixedTendonPropertiesCfg

    from isaaclab.sim.schemas import modify_fixed_tendon_properties

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    tendon = stage.DefinePrim("/World/Robot/tendon", "MjcTendon")

    modify_fixed_tendon_properties("/World/Robot", PhysxFixedTendonPropertiesCfg(stiffness=2.0, damping=0.25), stage)
    assert tendon.GetAttribute("mjc:stiffness").Get() == pytest.approx(2.0)
    assert tendon.GetAttribute("mjc:damping").Get() == pytest.approx(0.25)


def test_legacy_mujoco_joint_cfg_dispatches_to_newton_backend():
    """Preserve MuJoCo gravity-compensation coupling through cfg-owned dispatch."""
    from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

    from isaaclab.utils.string import string_to_callable

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    body = stage.DefinePrim("/World/Robot/link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/joint")
    joint.GetBody1Rel().SetTargets([Sdf.Path("/World/Robot/link")])
    cfg = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)

    modify_func = string_to_callable(cfg._modify_func)
    modify_func("/World/Robot", cfg, stage)
    assert joint.GetPrim().GetAttribute("mjc:actuatorgravcomp").Get() is True
    assert body.GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
