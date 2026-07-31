# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# note: need to enable cameras to be able to make replicator core available
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.queries as query_utils
from isaaclab.cloner import ClonePlan
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()
    sim_utils.update_stage()

    # Yield for the test
    yield

    # Teardown: Clear stage after each test
    sim_utils.clear_stage()


"""
USD Stage Querying.
"""


def test_get_next_free_prim_path():
    """Test get_next_free_prim_path() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Wall", "Sphere", attributes={"radius": 1e3})

    # test
    isaaclab_result = sim_utils.get_next_free_prim_path("/World/Floor")
    assert isaaclab_result == "/World/Floor_01"

    # create another prim
    sim_utils.create_prim("/World/Floor/Box_01", "Cube", position=[75, 75, -150.1], attributes={"size": 300})

    # test again
    isaaclab_result = sim_utils.get_next_free_prim_path("/World/Floor/Box")
    assert isaaclab_result == "/World/Floor/Box_02"


def test_get_first_matching_ancestor_prim():
    """Test get_first_matching_ancestor_prim() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Floor/Box/Sphere", "Sphere", attributes={"radius": 1e3})

    # test with input prim not having the predicate
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box/Sphere", predicate=lambda x: x.GetTypeName() == "Cube"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/Floor/Box"

    # test with input prim having the predicate
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box", predicate=lambda x: x.GetTypeName() == "Cube"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/Floor/Box"

    # test with no predicate match
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box/Sphere", predicate=lambda x: x.GetTypeName() == "Cone"
    )
    assert isaaclab_result is None


def test_matches_path_expr_prefix():
    path_expr = "/World/envs/env_.*/Robot"
    assert sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0")
    assert sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Robot")
    assert not sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Object")
    assert not sim_utils.matches_path_expr_prefix(path_expr, "/World/envs/env_0/Robot/base")


def test_resolve_matching_prims_from_normalized_regex(monkeypatch):
    """A normalized regex controls direct versus descendant predicate matching."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    stage = sim_utils.get_current_stage()
    pelvis_path = "/World/envs/env_0/Robot/pelvis"
    child_path = f"{pelvis_path}/child"
    sim_utils.create_prim(pelvis_path, "Xform")
    sim_utils.create_prim(child_path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(pelvis_path))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(child_path))

    direct_predicate_paths = []

    def direct_has_rigid_body_api(prim):
        direct_predicate_paths.append(prim.GetPath().pathString)
        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

    recursive_predicate_paths = []

    def recursive_has_rigid_body_api(prim):
        recursive_predicate_paths.append(prim.GetPath().pathString)
        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

    path_expr = "/World/envs/env_.*/Robot/pelvis"
    direct = sim_utils.resolve_matching_prims_from_source(f"(?:{path_expr})", direct_has_rigid_body_api)
    recursive = sim_utils.resolve_matching_prims_from_source(f"(?:{path_expr})(?:/.*)?", recursive_has_rigid_body_api)

    assert [prim.GetPath().pathString for prim, _ in direct] == [pelvis_path]
    assert [prim.GetPath().pathString for prim, _ in recursive] == [pelvis_path, child_path]
    assert direct_predicate_paths == [pelvis_path]
    assert recursive_predicate_paths == [pelvis_path, child_path]


def test_resolve_matching_prims_preserves_authored_regex(monkeypatch):
    """Python quantifiers and slash-containing character classes remain unchanged."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    matching_path = "/World/envs/env_0/Robot/link_12"
    nonmatching_path = "/World/envs/env_0/Robot/link_1tail"
    sim_utils.create_prim(matching_path, "Xform")
    sim_utils.create_prim(nonmatching_path, "Xform")

    predicate_paths = []

    def record_match(prim):
        predicate_paths.append(prim.GetPath().pathString)
        return True

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/link_[0-9]*)", record_match)

    assert [prim.GetPath().pathString for prim, _ in matches] == [matching_path]
    assert predicate_paths == [matching_path]


def test_resolve_matching_prims_uses_global_source_domain(monkeypatch):
    """A global regex filters all matches below the global source root."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    first_path = "/World/Table_0/Object"
    second_path = "/World/Table_1/Object"
    sim_utils.create_prim(first_path, "Xform")
    sim_utils.create_prim(second_path, "Xform")

    predicate_paths = []

    def record_match(prim):
        predicate_paths.append(prim.GetPath().pathString)
        return True

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/Table_.*/Object)", predicate=record_match)

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (first_path, first_path),
        (second_path, second_path),
    ]
    assert predicate_paths == [first_path, second_path]


def test_resolve_matching_prims_preserves_global_character_class(monkeypatch):
    """Slashes inside an authored character class are not treated as path separators."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    left_path = "/World/Robot/left"
    right_path = "/World/Robot/right"
    nested_path = f"{left_path}/nested"
    sim_utils.create_prim(left_path, "Xform")
    sim_utils.create_prim(right_path, "Xform")
    sim_utils.create_prim(nested_path, "Xform")

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/Robot/[^/]+)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (left_path, left_path),
        (right_path, right_path),
    ]


def test_resolve_matching_prims_excludes_pseudo_root(monkeypatch):
    """A root-wide regex starts from the first real global prim."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    world_path = "/World"
    robot_path = f"{world_path}/Robot"
    sim_utils.create_prim(robot_path, "Xform")

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/.*)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (world_path, world_path),
        (robot_path, robot_path),
    ]


def test_resolve_matching_prims_normalizes_legacy_wildcard(monkeypatch):
    """A deprecated path keeps its legacy bare-wildcard behavior."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    object_path = "/World/Table_0/Object"
    child_path = f"{object_path}/child"
    sim_utils.create_prim(object_path, "Xform")
    sim_utils.create_prim(child_path, "Xform")

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/Table_*/Object)(?:/.*)?")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (object_path, object_path),
        (child_path, child_path),
    ]


def test_resolve_matching_prims_from_clone_plan_regex(monkeypatch):
    """A destination regex resolves against source-only clone-plan prims."""
    source_path = "/World/prototypes/Robot"
    body_path = f"{source_path}/base"
    sim_utils.create_prim(body_path, "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/base)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (body_path, "/World/envs/env_*/Robot/base")
    ]


def test_resolve_matching_prims_routes_grouped_clone_regex(monkeypatch):
    """Clone routing is based on projected matches rather than regex text."""
    source_path = "/World/prototypes/Robot"
    body_path = f"{source_path}/base"
    sim_utils.create_prim(body_path, "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/(?:Robot)/base)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (body_path, "/World/envs/env_*/Robot/base")
    ]


def test_resolve_matching_prims_uses_clone_plan_source_only(monkeypatch):
    """Authored clone destinations are not scanned after selecting the plan source."""
    source_path = "/World/envs/env_0/Robot"
    body_path = f"{source_path}/base"
    other_body_path = "/World/envs/env_1/Robot/base"
    sim_utils.create_prim(body_path, "Xform")
    sim_utils.create_prim(other_body_path, "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    predicate_paths = []

    def record_match(prim):
        predicate_paths.append(prim.GetPath().pathString)
        return True

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/base)", predicate=record_match)

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (body_path, "/World/envs/env_*/Robot/base")
    ]
    assert predicate_paths == [body_path]


def test_resolve_matching_prims_rejects_partial_clone_plan_regex(monkeypatch):
    """A destination regex cannot silently broaden partial clone-plan coverage."""
    source_path = "/World/prototypes/Robot"
    sim_utils.create_prim(f"{source_path}/base", "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.tensor([[True, False]]),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    with pytest.raises(NotImplementedError, match="partial-env heterogeneous coverage"):
        sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/base)")


def test_resolve_matching_prims_rejects_regex_for_only_later_clone(monkeypatch):
    """A later-environment match remains owned by the first clone source."""
    source_path = "/World/prototypes/Robot"
    sim_utils.create_prim(f"{source_path}/base", "Xform")
    sim_utils.create_prim("/World/envs/env_1/Robot/base", "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    with pytest.raises(NotImplementedError, match="matched only environments"):
        sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_1/Robot/base)")


def test_resolve_matching_prims_uses_first_authored_source_only(monkeypatch):
    """An equivalent environment regex is evaluated only below its first source."""

    class _NoClonePlanContext:
        def get_clone_plan(self):
            return None

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _NoClonePlanContext())

    env_0_path = "/World/envs/env_0/Robot/base"
    env_1_path = "/World/envs/env_1/Robot/base"
    sim_utils.create_prim(env_0_path, "Xform")
    sim_utils.create_prim(env_1_path, "Xform")

    predicate_paths = []

    def record_match(prim):
        predicate_paths.append(prim.GetPath().pathString)
        return True

    matches = sim_utils.resolve_matching_prims_from_source(
        r"(?:/World/envs/env_[0-9]+/Robot/base)", predicate=record_match
    )

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (env_0_path, "/World/envs/env_.*/Robot/base")
    ]
    assert predicate_paths == [env_0_path]


def test_resolve_matching_prims_uses_clone_plan_environment_ids(monkeypatch):
    """Clone-mask columns map through the plan's target environment ids."""
    source_path = "/World/prototypes/Robot"
    body_path = f"{source_path}/base"
    sim_utils.create_prim(body_path, "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.tensor([3, 7]),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_(3|7)/Robot/base)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (body_path, "/World/envs/env_*/Robot/base")
    ]


def test_resolve_matching_prims_prefers_nearest_clone_owner(monkeypatch):
    """A nested clone destination owns its subtree over a parent destination."""
    scene_source = "/World/prototypes/Scene"
    robot_source = "/World/prototypes/Robot"
    parent_body_path = f"{scene_source}/Robot/base"
    child_body_path = f"{robot_source}/base"
    sim_utils.create_prim(parent_body_path, "Xform")
    sim_utils.create_prim(child_body_path, "Xform")
    plan = ClonePlan(
        sources=(scene_source, robot_source),
        destinations=("/World/envs/env_{}", "/World/envs/env_{}/Robot"),
        clone_mask=torch.ones((2, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    matches = sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/base)")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (child_body_path, "/World/envs/env_*/Robot/base")
    ]


def test_resolve_matching_prims_does_not_fall_back_to_parent_clone_owner(monkeypatch):
    """A nested destination owns its subtree even when its source lacks the matching prim."""
    scene_source = "/World/prototypes/Scene"
    robot_source = "/World/prototypes/Robot"
    sim_utils.create_prim(f"{scene_source}/Robot/base", "Xform")
    sim_utils.create_prim(robot_source, "Xform")
    plan = ClonePlan(
        sources=(scene_source, robot_source),
        destinations=("/World/envs/env_{}", "/World/envs/env_{}/Robot"),
        clone_mask=torch.ones((2, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    with pytest.raises(RuntimeError, match="No prim found"):
        sim_utils.resolve_matching_prims_from_source(r"(?:/World/envs/env_.*/Robot/base)")


def test_resolve_matching_prims_uses_first_clone_variant_only(monkeypatch):
    """The predicate is evaluated on the selected source variant only."""
    first_source = "/World/prototypes/RobotA"
    second_source = "/World/prototypes/RobotB"
    first_body_path = f"{first_source}/base"
    second_body_path = f"{second_source}/base"
    sim_utils.create_prim(first_body_path, "Xform")
    sim_utils.create_prim(second_body_path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(sim_utils.get_current_stage().GetPrimAtPath(first_body_path))
    plan = ClonePlan(
        sources=(first_source, second_source),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"),
        clone_mask=torch.tensor([[True, False], [False, True]]),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    predicate_paths = []

    def has_rigid_body_api(prim):
        predicate_paths.append(prim.GetPath().pathString)
        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

    matches = sim_utils.resolve_matching_prims_from_source(
        r"(?:/World/envs/env_.*/Robot/base)", predicate=has_rigid_body_api
    )

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [
        (first_body_path, "/World/envs/env_*/Robot/base")
    ]
    assert predicate_paths == [first_body_path]


def test_resolve_matching_global_prim_under_env_parent(monkeypatch):
    """A clone plan does not hide a global prim below the environment parent."""

    source_path = "/World/prototypes/Robot"
    sim_utils.create_prim(f"{source_path}/base", "Xform")
    plan = ClonePlan(
        sources=(source_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    class _ClonePlanContext:
        def get_clone_plan(self):
            return plan

    monkeypatch.setattr(query_utils.SimulationContext, "instance", lambda: _ClonePlanContext())

    sim_utils.create_prim("/World/envs/env_0/Robot/base", "Xform")
    global_path = "/World/envs/shared/Marker"
    sim_utils.create_prim(global_path, "Xform")

    matches = sim_utils.resolve_matching_prims_from_source(f"(?:{global_path})")

    assert [(prim.GetPath().pathString, destination) for prim, destination in matches] == [(global_path, global_path)]


def test_get_all_matching_child_prims():
    """Test get_all_matching_child_prims() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Wall", "Sphere", attributes={"radius": 1e3})

    # add articulation root prim -- this asset has instanced prims
    # note: isaac sim function does not support instanced prims so we add it here
    #  after the above test for the above test to still pass.
    sim_utils.create_prim(
        "/World/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    )

    # test with predicate
    isaaclab_result = sim_utils.get_all_matching_child_prims("/World", predicate=lambda x: x.GetTypeName() == "Cube")
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Floor/Box"

    # test with predicate and instanced prims
    isaaclab_result = sim_utils.get_all_matching_child_prims(
        "/World/Franka/panda_hand/visuals", predicate=lambda x: x.GetTypeName() == "Mesh"
    )
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Franka/panda_hand/visuals/panda_hand"

    # test expected number of matches
    isaaclab_result = sim_utils.get_all_matching_child_prims(
        "/World", predicate=lambda x: x.GetTypeName() == "Cube", expected_num_matches=1
    )
    assert len(isaaclab_result) == 1
    with pytest.raises(RuntimeError, match="Expected 2 prims under '/World', found 1"):
        sim_utils.get_all_matching_child_prims(
            "/World", predicate=lambda x: x.GetTypeName() == "Cube", expected_num_matches=2
        )
    with pytest.raises(ValueError, match="Expected number of matches must be non-negative"):
        sim_utils.get_all_matching_child_prims("/World", expected_num_matches=-1)

    # test valid path
    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Room")


def test_get_first_matching_child_prim():
    """Test get_first_matching_child_prim() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim(
        "/World/env_1/Franka",
        "Xform",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
    )
    sim_utils.create_prim(
        "/World/env_2/Franka",
        "Xform",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
    )
    sim_utils.create_prim(
        "/World/env_0/Franka",
        "Xform",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
    )

    # test
    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World", predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka"

    # test with instanced prims
    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World/env_1/Franka", predicate=lambda prim: prim.GetTypeName() == "Mesh"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka/panda_link0/visuals/panda_link0"


def test_find_global_fixed_joint_prim():
    """Test find_global_fixed_joint_prim() function."""
    # create scene
    sim_utils.create_prim("/World")
    sim_utils.create_prim("/World/ANYmal", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd")
    sim_utils.create_prim(
        "/World/Franka", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    )
    if "4.5" in ISAAC_NUCLEUS_DIR:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"
    else:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    sim_utils.create_prim("/World/Franka_Isaac", usd_path=franka_usd)

    # test
    assert sim_utils.find_global_fixed_joint_prim("/World/ANYmal") is None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka_Isaac") is not None

    # make fixed joint disabled manually
    joint_prim = sim_utils.find_global_fixed_joint_prim("/World/Franka")
    joint_prim.GetJointEnabledAttr().Set(False)
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka", check_enabled_only=True) is None
