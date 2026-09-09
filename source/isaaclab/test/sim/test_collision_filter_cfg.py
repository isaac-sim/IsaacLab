# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend-neutral declarative collision-filter contract."""

import ast
from pathlib import Path

import pytest

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg
from isaaclab.physics import CollisionGroupCfg, PhysicsCfg, PhysicsManager, PhysxAutoCfg
from isaaclab.physics._collision_filter import CompiledCollisionFilter
from isaaclab.physics.physics_manager_cfg import _resolve_physx_auto_cfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg


def test_collision_policy_has_one_configuration_owner():
    assert "collision_filter" in PhysicsCfg.__dataclass_fields__
    assert "collision_filter" not in InteractiveSceneCfg.__dataclass_fields__
    assert "isolate_environments" in CloneCfg.__dataclass_fields__
    assert "isolate_environments" not in InteractiveSceneCfg.__dataclass_fields__
    assert "filter_collisions" not in InteractiveSceneCfg.__dataclass_fields__
    assert "collision_group" in AssetBaseCfg.__dataclass_fields__
    assert "collision_group" in TerrainImporterCfg.__dataclass_fields__
    assert not hasattr(InteractiveScene, "filter_collisions")
    assert not hasattr(PhysicsManager, "configure_collision_filter")
    assert "include_descendants" not in CollisionGroupCfg.__dataclass_fields__

    scene_path = Path(__file__).parents[2] / "isaaclab" / "scene" / "interactive_scene.py"
    scene_module = ast.parse(scene_path.read_text(encoding="utf-8"))
    scene_class = next(
        node for node in scene_module.body if isinstance(node, ast.ClassDef) and node.name == "InteractiveScene"
    )
    accessed = {node.attr for node in ast.walk(scene_class) if isinstance(node, ast.Attribute)}
    assert accessed.isdisjoint({"collision_group", "filter_collisions", "isolate_environments"})


def test_scene_routes_cloning_policy_only_through_clone_cfg():
    cfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    cfg.from_dict({"clone_cfg": {"replicate_physics": False, "isolate_environments": False}})
    assert cfg.clone_cfg.replicate_physics is False
    assert cfg.clone_cfg.isolate_environments is False

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        InteractiveSceneCfg(num_envs=2, env_spacing=1.0, filter_collisions=False)
    with pytest.raises(KeyError, match="filter_collisions"):
        cfg.from_dict({"filter_collisions": False})


def test_compiled_group_filter_uses_full_paths_and_symmetric_deny_wins():
    groups = {
        "exact": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Exact/collider",)),
        "robot": CollisionGroupCfg(
            prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/.*",),
            filtered_groups=("supports",),
        ),
        "objects": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Object/.*",)),
        "supports": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/.*",)),
        "nut_sdf": CollisionGroupCfg(
            prim_path_exprs=(r"{ENV_REGEX_NS}/Nut/sdf",),
            filtered_groups=("bolt_sdf",),
            invert_filtered_groups=True,
        ),
        "bolt_sdf": CollisionGroupCfg(
            prim_path_exprs=(r"{ENV_REGEX_NS}/Bolt/sdf",),
            filtered_groups=("nut_sdf",),
            invert_filtered_groups=True,
        ),
        "other_convex": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Other/convex",)),
    }
    PhysicsCfg(class_type=object, collision_filter=groups).validate()
    policy = CompiledCollisionFilter(groups, "/World/cells/cell_{}")

    exact = "/World/cells/cell_12/Exact/collider"
    assert policy.memberships(exact) == ("exact",)
    assert policy.memberships(f"/prefix{exact}") == ()
    assert policy.memberships(f"{exact}/child") == ()
    assert not policy.filters(("robot",), ("objects",))
    assert not policy.filters(("objects",), ("supports",))
    assert policy.filters(("robot",), ("supports",))
    assert policy.filters(("supports",), ("robot",))
    assert not policy.filters(("nut_sdf",), ("bolt_sdf",))
    assert policy.filters(("nut_sdf",), ("other_convex",))
    assert policy.filters(("bolt_sdf",), ("other_convex",))
    assert policy.filters(("nut_sdf",), ())
    assert policy.filters(("nut_sdf",), ("bolt_sdf", "other_convex"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prim_path_exprs": r"/World/Robot/.*"}, "prim_path_exprs must be a list or tuple"),
        ({"prim_path_exprs": (1,)}, "prim_path_exprs must contain only strings"),
        (
            {"prim_path_exprs": (r"/World/Robot/.*",), "filtered_groups": "objects"},
            "filtered_groups must be a list or tuple",
        ),
        (
            {"prim_path_exprs": (r"/World/Robot/.*",), "filtered_groups": (1,)},
            "filtered_groups must contain only strings",
        ),
        (
            {"prim_path_exprs": (r"/World/Robot/.*",), "invert_filtered_groups": 1},
            "invert_filtered_groups must be a bool",
        ),
    ],
)
def test_collision_group_rejects_ambiguous_container_and_member_types(kwargs, message):
    with pytest.raises(TypeError, match=message):
        CollisionGroupCfg(**kwargs).validate()


def test_collision_group_accepts_hydra_list_containers():
    CollisionGroupCfg(prim_path_exprs=[r"/World/Robot/.*"], filtered_groups=[]).validate()


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({"robot": CollisionGroupCfg(prim_path_exprs=("(",))}, "Invalid collision-group prim-path regex"),
        (
            {"robot": CollisionGroupCfg(prim_path_exprs=(r"/World/Robot/.*",), filtered_groups=("missing_support",))},
            "unknown groups: missing_support",
        ),
    ],
)
def test_physics_cfg_rejects_invalid_group_policy(groups, message):
    with pytest.raises(ValueError, match=message):
        PhysicsCfg(class_type=object, collision_filter=groups).validate()


@pytest.mark.parametrize("use_isaac_sim", [True, False])
def test_physx_auto_cfg_preserves_or_overrides_backend_neutral_policy(use_isaac_sim):
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    nested_policy = {"nested": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Nested/.*",))}
    outer_policy = {"outer": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Outer/.*",))}
    backend_type = PhysxCfg if use_isaac_sim else OvPhysxCfg
    selected = backend_type(deterministic=True, collision_filter=nested_policy)
    kwargs = {"isaacsim_physx": selected} if use_isaac_sim else {"ovphysx": selected}

    nested = _resolve_physx_auto_cfg(PhysxAutoCfg(**kwargs), use_isaac_sim=use_isaac_sim)
    overridden = _resolve_physx_auto_cfg(
        PhysxAutoCfg(collision_filter=outer_policy, **kwargs),
        use_isaac_sim=use_isaac_sim,
    )

    assert isinstance(nested, backend_type)
    assert nested is not selected
    assert nested.deterministic
    assert nested.collision_filter == nested_policy
    assert overridden.deterministic
    assert overridden.collision_filter == outer_policy
