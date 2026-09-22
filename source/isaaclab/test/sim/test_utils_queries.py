# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import inspect
import textwrap

import pytest

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import queries

pytestmark = pytest.mark.unit


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def _paths(prims) -> list[str]:
    return [prim.GetPath().pathString for prim in prims]


def _define_instanceable_robot(stage, path: str):
    """Author an articulation whose mesh sits inside an instanceable reference, like production assets."""
    stage.DefinePrim("/Prototypes/Link", "Xform")
    stage.DefinePrim("/Prototypes/Link/visuals", "Xform")
    stage.DefinePrim("/Prototypes/Link/visuals/mesh", "Mesh")
    robot = stage.DefinePrim(path, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    link = stage.DefinePrim(f"{path}/link0", "Xform")
    link.GetReferences().AddInternalReference("/Prototypes/Link")
    link.SetInstanceable(True)
    return robot


def test_get_next_free_prim_path(stage):
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube")
    assert sim_utils.get_next_free_prim_path("/World/Wall") == "/World/Wall"
    assert sim_utils.get_next_free_prim_path("/World/Floor") == "/World/Floor_01"
    sim_utils.create_prim("/World/Floor/Box_01", "Cube")
    assert sim_utils.get_next_free_prim_path("/World/Floor/Box") == "/World/Floor/Box_02"
    with pytest.raises(ValueError, match="not a valid prim path"):
        sim_utils.get_next_free_prim_path("World Room")


def test_get_first_matching_ancestor_prim(stage):
    sim_utils.create_prim("/World/Floor/Box", "Cube")
    sim_utils.create_prim("/World/Floor/Box/Sphere", "Sphere")
    is_cube = lambda prim: prim.GetTypeName() == "Cube"  # noqa: E731

    assert (
        sim_utils.get_first_matching_ancestor_prim("/World/Floor/Box/Sphere", is_cube).GetPrimPath()
        == "/World/Floor/Box"
    )
    # the prim itself counts as its first ancestor
    assert sim_utils.get_first_matching_ancestor_prim("/World/Floor/Box", is_cube).GetPrimPath() == "/World/Floor/Box"
    assert sim_utils.get_first_matching_ancestor_prim("/World/Floor/Box", lambda p: p.GetTypeName() == "Cone") is None
    with pytest.raises(ValueError, match="not global"):
        sim_utils.get_first_matching_ancestor_prim("World/Floor", is_cube)


def test_get_matching_child_prims_traverse_instances(stage):
    sim_utils.create_prim("/World/Floor/Box", "Cube")
    sim_utils.create_prim("/World/Wall", "Sphere")
    for env in ("env_1", "env_2", "env_0"):
        _define_instanceable_robot(stage, f"/World/{env}/Robot")
    is_mesh = lambda prim: prim.GetTypeName() == "Mesh"  # noqa: E731

    # all matches, including those inside instance proxies
    assert _paths(sim_utils.get_all_matching_child_prims("/World", lambda p: p.GetTypeName() == "Cube")) == [
        "/World/Floor/Box"
    ]
    assert _paths(sim_utils.get_all_matching_child_prims("/World/env_1/Robot", is_mesh)) == [
        "/World/env_1/Robot/link0/visuals/mesh"
    ]
    assert sim_utils.get_all_matching_child_prims("/World/env_1/Robot", is_mesh, traverse_instance_prims=False) == []
    assert _paths(sim_utils.get_all_matching_child_prims("/World", depth=1)) == [
        "/World",
        "/World/Floor",
        "/World/Wall",
        "/World/env_1",
        "/World/env_2",
        "/World/env_0",
    ]

    # first match follows authoring order
    root = sim_utils.get_first_matching_child_prim("/World", lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI))
    assert root.GetPrimPath() == "/World/env_1/Robot"
    mesh = sim_utils.get_first_matching_child_prim("/World/env_1/Robot", is_mesh)
    assert mesh.GetPrimPath() == "/World/env_1/Robot/link0/visuals/mesh"
    assert sim_utils.get_first_matching_child_prim("/World/Wall", is_mesh) is None

    # match-count validation
    assert len(sim_utils.get_all_matching_child_prims("/World", is_mesh, expected_num_matches=3)) == 3
    with pytest.raises(RuntimeError, match="Expected 2 prims under '/World', found 3"):
        sim_utils.get_all_matching_child_prims("/World", is_mesh, expected_num_matches=2)
    with pytest.raises(ValueError, match="non-negative"):
        sim_utils.get_all_matching_child_prims("/World", expected_num_matches=-1)
    with pytest.raises(ValueError, match="bigger than zero"):
        sim_utils.get_all_matching_child_prims("/World", depth=0)
    with pytest.raises(ValueError, match="not global"):
        sim_utils.get_all_matching_child_prims("World/Room")


def test_path_expression_helpers():
    path_expr = r"/World/envs/env_[^/]+/Robot/link_[0-9]{2}"
    assert sim_utils.split_path_expr(path_expr) == ["", "World", "envs", "env_[^/]+", "Robot", "link_[0-9]{2}"]
    # only the supported segment wildcards are translated to globs
    assert sim_utils.path_expr_to_glob(path_expr) == r"/World/envs/env_*/Robot/link_[0-9]{2}"
    assert sim_utils.path_expr_to_glob(r"/World/Robot/[^/]{2}") == r"/World/Robot/[^/]{2}"

    path_expr = "/World/envs/env_[^/]+/Robot"
    assert sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0")
    assert sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Robot")
    assert not sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Object")
    assert not sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Robot/base")
    assert not sim_utils.matches_path_expr_prefix(
        "/World/envs/env_[^/]+/Robot/cart|pole", "/World/envs/env_0/Robot/cartXX"
    )


def test_find_matching_prims_regex_semantics(stage):
    """The expression is a plain full-path regex: tokens keep their Python semantics across separators."""
    for path in ("/World/Robot/foo", "/World/Robot/foo/bar", "/World/Robot/Arm", "/World/A/foo", "/World/B/foo"):
        sim_utils.create_prim(path)
    stage.DefinePrim("/World/Inactive", "Xform").SetActive(False)
    stage.OverridePrim("/World/Undefined")

    assert _paths(sim_utils.find_matching_prims(r"/World/Robot/[^A]+")) == ["/World/Robot/foo", "/World/Robot/foo/bar"]
    assert sim_utils.find_matching_prim_paths(r"/World/[^/]+/foo") == [
        "/World/Robot/foo",
        "/World/A/foo",
        "/World/B/foo",
    ]
    # anchoring applies to the whole expression, not to each alternative
    assert sim_utils.find_matching_prim_paths(r"/World/Robot/foo/bar|/World/A") == ["/World/Robot/foo/bar", "/World/A"]
    # inactive and undefined prims are exposed rather than silently filtered
    assert sim_utils.find_matching_prim_paths(r"/World/(Inactive|Undefined)") == ["/World/Inactive", "/World/Undefined"]
    assert sim_utils.find_first_matching_prim(r"/World/.*/foo").GetPath() == "/World/Robot/foo"
    assert sim_utils.find_first_matching_prim(r"/World/Missing.*") is None
    with pytest.raises(ValueError, match="not global"):
        sim_utils.find_matching_prims("World/Robot")


def test_find_matching_prims_traverses_instance_proxies(stage):
    _define_instanceable_robot(stage, "/World/Robot")
    assert "/World/Robot/link0/visuals/mesh" in sim_utils.find_matching_prim_paths("/World/Robot(/.*)?")


def test_find_matching_prims_has_no_inferred_traversal_bounds():
    """The query must not narrow or prune USD traversal from the user's regex."""
    sources = (sim_utils.find_matching_prims, queries._iter_matching_prims_in_subtree)
    tree = ast.parse("\n".join(textwrap.dedent(inspect.getsource(function)) for function in sources))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_methods = {func.attr for func in calls if isinstance(func, ast.Attribute)}
    called_functions = {func.id for func in calls if isinstance(func, ast.Name)}

    assert not {"GetPrimAtPath", "PruneChildren"} & called_methods
    assert "_bound_search" not in called_functions


def test_find_global_fixed_joint_prim(stage):
    def define_robot(path: str, fixed_to_world: bool):
        stage.DefinePrim(path, "Xform")
        stage.DefinePrim(f"{path}/base", "Xform")
        # some assets author fixed joints with the generic "Joint" schema
        joint = UsdPhysics.Joint.Define(stage, f"{path}/root_joint")
        joint.GetBody1Rel().SetTargets([f"{path}/base"])
        if not fixed_to_world:
            stage.DefinePrim(f"{path}/link", "Xform")
            joint.GetBody0Rel().SetTargets([f"{path}/link"])
        return joint

    define_robot("/World/Floating", fixed_to_world=False)
    joint = define_robot("/World/Fixed", fixed_to_world=True)

    assert sim_utils.find_global_fixed_joint_prim("/World/Floating") is None
    assert sim_utils.find_global_fixed_joint_prim("/World/Fixed").GetPath() == joint.GetPath()
    # disabled joints only count when requested
    joint.GetJointEnabledAttr().Set(False)
    assert sim_utils.find_global_fixed_joint_prim("/World/Fixed") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Fixed", check_enabled_only=True) is None
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.find_global_fixed_joint_prim("/World/Missing")
