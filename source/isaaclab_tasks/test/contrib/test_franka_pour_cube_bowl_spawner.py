# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-authoring tests for the Franka pour cube-bowl spawner."""

import subprocess
import sys
import textwrap

from isaaclab.app import AppLauncher

# Launch Omniverse before importing simulator or USD modules.
simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest

from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from isaaclab_tasks.contrib.franka_pour.cube_bowl_mesh import make_cube_bowl_mesh
from isaaclab_tasks.contrib.franka_pour.cube_bowl_spawner_cfg import CubeBowlSpawnerCfg

pytestmark = pytest.mark.isaacsim_ci

_BOWL_DIMS = {
    "inner_width": 0.037,
    "inner_depth": 0.039,
    "cavity_depth": 0.045,
    "wall_thickness": 0.009,
    "bottom_thickness": 0.008,
}


@pytest.fixture
def sim():
    """Create a fresh stage and simulation context for each test."""
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim_utils.update_stage()
    yield sim
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


def _make_cfg(**kwargs) -> CubeBowlSpawnerCfg:
    """Build a bowl config with task-realistic dimensions."""
    return CubeBowlSpawnerCfg(**_BOWL_DIMS, **kwargs)


def _quat_xyzw(prim) -> tuple[float, float, float, float]:
    """Read a prim's authored local orientation in Isaac Lab order."""
    quat = prim.GetAttribute("xformOp:orient").Get()
    imaginary = quat.GetImaginary()
    return (float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quat.GetReal()))


def test_config_import_and_instantiation_do_not_require_physx():
    """The task-local config remains importable when the optional PhysX package is absent."""
    code = textwrap.dedent(
        """
        import builtins
        import sys

        class _PhysxBlocker:
            def find_spec(self, name, path=None, target=None):
                if name == "isaaclab_physx" or name.startswith("isaaclab_physx."):
                    raise ImportError(f"blocked optional PhysX import: {name}")
                return None

        sys.meta_path.insert(0, _PhysxBlocker())
        builtins._isaaclab_tasks_registered = True

        from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
        from isaaclab_tasks.contrib.franka_pour.cube_bowl_spawner_cfg import CubeBowlSpawnerCfg

        cfg = CubeBowlSpawnerCfg(
            inner_width=0.037,
            inner_depth=0.039,
            cavity_depth=0.045,
            wall_thickness=0.009,
            bottom_thickness=0.008,
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
        )
        assert isinstance(cfg.physics_material, RigidBodyMaterialBaseCfg)
        assert cfg.physics_material.static_friction == 0.6
        assert "isaaclab_physx" not in sys.modules
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_dynamic_source_authors_visual_mesh_grasp_proxy_and_material(sim):
    """The source bowl has a visual shell and an invisible rigid grasp proxy."""
    half_extents = (0.028, 0.029, 0.031)
    color = (0.15, 0.55, 0.85)
    cfg = _make_cfg(
        display_color=color,
        grasp_proxy_half_extents=half_extents,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.24),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=0.83,
            dynamic_friction=0.71,
            restitution=0.04,
        ),
    )

    root = cfg.func("/World/Cup", cfg)
    stage = sim.stage
    mesh_prim = stage.GetPrimAtPath("/World/Cup/geometry/mesh")
    proxy_prim = stage.GetPrimAtPath("/World/Cup/geometry/grasp_proxy")

    assert root.IsValid()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)
    assert root.GetAttribute("physics:mass").Get() == pytest.approx(0.24)

    assert mesh_prim.IsA(UsdGeom.Mesh)
    assert not mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    mesh = UsdGeom.Mesh(mesh_prim)
    expected_points, expected_indices = make_cube_bowl_mesh(**_BOWL_DIMS)
    np.testing.assert_allclose(np.asarray(mesh.GetPointsAttr().Get()), expected_points)
    np.testing.assert_array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), expected_indices)
    assert len(mesh.GetPointsAttr().Get()) == 16
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3] * (expected_indices.size // 3)
    assert tuple(mesh.GetDisplayColorAttr().Get()[0]) == pytest.approx(color)

    visual_binding = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBinding()
    assert visual_binding.GetMaterialPath() == "/World/Cup/geometry/visual_material"

    assert proxy_prim.IsA(UsdGeom.Cube)
    assert proxy_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not proxy_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.Imageable(proxy_prim).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert UsdGeom.Cube(proxy_prim).GetSizeAttr().Get() == pytest.approx(1.0)
    assert [tuple(point) for point in UsdGeom.Cube(proxy_prim).GetExtentAttr().Get()] == pytest.approx(
        [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
    )
    assert tuple(proxy_prim.GetAttribute("xformOp:translate").Get()) == pytest.approx((0.0, 0.0, half_extents[2]))
    assert tuple(proxy_prim.GetAttribute("xformOp:scale").Get()) == pytest.approx(
        tuple(2.0 * value for value in half_extents)
    )

    material_path = "/World/Cup/geometry/material"
    material_prim = stage.GetPrimAtPath(material_path)
    assert material_prim.IsA(UsdShade.Material)
    assert material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.83)
    physics_binding = UsdShade.MaterialBindingAPI(proxy_prim).GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == material_path


def test_kinematic_target_without_proxy_authors_pose_and_display_color(sim):
    """The target bowl remains a valid kinematic rigid body without a grasp proxy."""
    translation = (0.42, -0.17, 0.09)
    orientation = (0.0, 0.0, 0.38268343, 0.92387953)
    color = (0.92, 0.31, 0.12)
    cfg = _make_cfg(
        display_color=color,
        grasp_proxy_half_extents=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.64, dynamic_friction=0.52),
    )

    root = cfg.func("/World/TargetCup", cfg, translation=translation, orientation=orientation)
    mesh_prim = sim.stage.GetPrimAtPath("/World/TargetCup/geometry/mesh")

    assert root.IsValid()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get() is True
    assert tuple(root.GetAttribute("xformOp:translate").Get()) == pytest.approx(translation)
    assert _quat_xyzw(root) == pytest.approx(orientation)
    assert not sim.stage.GetPrimAtPath("/World/TargetCup/geometry/grasp_proxy").IsValid()
    assert mesh_prim.IsA(UsdGeom.Mesh)
    assert not mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert tuple(UsdGeom.Mesh(mesh_prim).GetDisplayColorAttr().Get()[0]) == pytest.approx(color)
    physics_binding = UsdShade.MaterialBindingAPI(root).GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == "/World/TargetCup/geometry/material"


def test_rejects_existing_root_prim(sim):
    """Spawning never mutates an existing root prim."""
    sim_utils.create_prim("/World/ExistingCup", "Xform")

    with pytest.raises(ValueError, match="already exists"):
        _make_cfg().func("/World/ExistingCup", _make_cfg())


def test_clone_decorator_spawns_under_matching_parents(sim):
    """A regex path clones the authored bowl hierarchy into every matching parent."""
    sim_utils.create_prim("/World/env_0", "Xform")
    sim_utils.create_prim("/World/env_1", "Xform")
    cfg = _make_cfg(rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True))

    source = cfg.func("/World/env_.*/Cup", cfg)

    assert str(source.GetPath()) == "/World/env_0/Cup"
    for env_index in range(2):
        root = sim.stage.GetPrimAtPath(f"/World/env_{env_index}/Cup")
        mesh = sim.stage.GetPrimAtPath(f"/World/env_{env_index}/Cup/geometry/mesh")
        assert root.HasAPI(UsdPhysics.RigidBodyAPI)
        assert mesh.IsA(UsdGeom.Mesh)
