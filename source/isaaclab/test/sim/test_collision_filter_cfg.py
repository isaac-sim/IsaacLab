# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend-neutral declarative collision-filter contract."""

import ast
import inspect
from pathlib import Path

import pytest

import isaaclab.cloner as cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0
from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg, PhysicsCfg, PhysicsManager, PhysxAutoCfg
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
    assert "replicate_physics" not in InteractiveSceneCfg.__dataclass_fields__
    assert "collision_group" not in AssetBaseCfg.__dataclass_fields__
    assert "collision_group" not in TerrainImporterCfg.__dataclass_fields__
    assert not hasattr(InteractiveScene, "filter_collisions")
    assert not hasattr(cloner, "filter_collisions")
    assert not hasattr(PhysicsManager, "configure_collision_filter")
    assert "include_descendants" not in CollisionGroupCfg.__dataclass_fields__
    assert not (Path(cloner.__path__[0]) / "collision_filter.py").exists()


def test_interactive_scene_does_not_resolve_cloner_policy():
    scene_path = Path(__file__).parents[2] / "isaaclab" / "scene" / "interactive_scene.py"
    scene_module = ast.parse(scene_path.read_text(encoding="utf-8"))
    scene_class = next(
        node for node in scene_module.body if isinstance(node, ast.ClassDef) and node.name == "InteractiveScene"
    )
    scene_init = next(
        node for node in scene_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    accessed_attributes = {node.attr for node in ast.walk(scene_init) if isinstance(node, ast.Attribute)}

    assert accessed_attributes.isdisjoint({"filter_collisions", "isolate_environments", "replicate_physics"})


def test_clone_cfg_is_the_only_public_cloner_policy_input():
    assert set(inspect.signature(clone_plan_from_env_0).parameters) == {
        "source",
        "num_clones",
        "clone_cfg",
        "positions",
        "global_paths",
    }


def test_scene_clone_cfg_supports_structured_updates():
    cfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    cfg.from_dict({"clone_cfg": {"replicate_physics": False, "isolate_environments": False}})

    assert cfg.clone_cfg.replicate_physics is False
    assert cfg.clone_cfg.isolate_environments is False


@pytest.mark.parametrize("removed_field", ["filter_collisions", "replicate_physics"])
def test_removed_scene_cloner_policy_fields_are_rejected(removed_field):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        InteractiveSceneCfg(num_envs=2, env_spacing=1.0, **{removed_field: False})

    cfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    with pytest.raises(KeyError, match=removed_field):
        cfg.from_dict({removed_field: False})


def test_group_selectors_use_whole_path_regex_matching():
    cfg = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",))})
    policy = CompiledCollisionFilter(cfg, "/World/cells/cell_{}")

    assert policy.memberships("/World/cells/cell_12/Robot/collider") == ("robot",)
    assert policy.memberships("/prefix/World/cells/cell_12/Robot/collider") == ()
    assert policy.memberships("/World/cells/cell_12/Robot/collider/child") == ()


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


def test_group_pair_filtering_is_symmetric_and_deny_wins():
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"/World/envs/env_.*/Robot/.*",),
                filtered_groups=("supports",),
            ),
            "objects": CollisionGroupCfg(prim_path_exprs=(r"/World/envs/env_.*/Object/.*",)),
            "supports": CollisionGroupCfg(prim_path_exprs=(r"/World/envs/env_.*/Support/.*",)),
        }
    )
    cfg.validate()
    policy = CompiledCollisionFilter(cfg, "/World/envs/env_{}")

    assert not policy.filters(("robot",), ("objects",))
    assert not policy.filters(("objects",), ("supports",))
    assert policy.filters(("robot",), ("supports",))
    assert policy.filters(("supports",), ("robot",))


def test_per_group_inversion_expresses_collider_specific_allow_lists():
    cfg = CollisionFilterCfg(
        groups={
            "nut_sdf": CollisionGroupCfg(
                prim_path_exprs=(r"/World/envs/env_.*/Nut/sdf",),
                filtered_groups=("bolt_sdf",),
                invert_filtered_groups=True,
            ),
            "bolt_sdf": CollisionGroupCfg(
                prim_path_exprs=(r"/World/envs/env_.*/Bolt/sdf",),
                filtered_groups=("nut_sdf",),
                invert_filtered_groups=True,
            ),
            "other_convex": CollisionGroupCfg(prim_path_exprs=(r"/World/envs/env_.*/Other/convex",)),
        }
    )
    cfg.validate()
    policy = CompiledCollisionFilter(cfg, "/World/envs/env_{}")

    assert not policy.filters(("nut_sdf",), ("bolt_sdf",))
    assert policy.filters(("nut_sdf",), ("other_convex",))
    assert policy.filters(("bolt_sdf",), ("other_convex",))
    assert policy.filters(("nut_sdf",), ())
    assert policy.filters(("nut_sdf",), ("bolt_sdf", "other_convex"))


def test_physics_cfg_rejects_invalid_group_regex():
    cfg = PhysicsCfg(
        class_type=object,
        collision_filter=CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=("(",))}),
    )

    with pytest.raises(ValueError, match="Invalid collision-group prim-path regex"):
        cfg.validate()


def test_unknown_filtered_group_is_rejected():
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"/World/Robot/.*",),
                filtered_groups=("missing_support",),
            )
        }
    )

    with pytest.raises(ValueError, match="unknown groups: missing_support"):
        cfg.validate()


@pytest.mark.parametrize("use_isaac_sim", [True, False])
def test_physx_auto_cfg_preserves_or_overrides_backend_neutral_policy(use_isaac_sim):
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    nested_policy = CollisionFilterCfg(
        groups={"nested": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Nested/.*",))}
    )
    outer_policy = CollisionFilterCfg(
        groups={"outer": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Outer/.*",))}
    )
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
