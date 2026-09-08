# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend-neutral declarative collision-filter contract."""

import pytest

from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg, PhysicsCfg, PhysxAutoCfg
from isaaclab.physics._collision_filter import CompiledCollisionFilter
from isaaclab.physics.physics_manager_cfg import _resolve_physx_auto_cfg
from isaaclab.scene import InteractiveSceneCfg


def test_collision_policy_has_one_configuration_owner():
    assert "collision_filter" in PhysicsCfg.__dataclass_fields__
    assert "collision_filter" not in InteractiveSceneCfg.__dataclass_fields__
    assert "include_descendants" not in CollisionGroupCfg.__dataclass_fields__


def test_group_selectors_use_whole_path_regex_matching():
    cfg = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",))})
    policy = CompiledCollisionFilter(cfg, "/World/cells/cell_{}")

    assert policy.memberships("/World/cells/cell_12/Robot/collider") == ("robot",)
    assert policy.memberships("/prefix/World/cells/cell_12/Robot/collider") == ()
    assert policy.memberships("/World/cells/cell_12/Robot/collider/child") == ()


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


@pytest.mark.parametrize(
    "group, error_type, message",
    [
        (CollisionGroupCfg(prim_path_exprs=()), ValueError, "at least one selector"),
        (CollisionGroupCfg(prim_path_exprs=("",)), ValueError, "empty selector"),
        (CollisionGroupCfg(prim_path_exprs=("(",)), ValueError, "Invalid collision-group prim-path regex"),
        (CollisionGroupCfg(prim_path_exprs=[r"/World/.*"]), TypeError, "must be a tuple"),
        (
            CollisionGroupCfg(prim_path_exprs=(r"/World/.*",), filtered_groups=("",)),
            ValueError,
            "empty group name",
        ),
    ],
)
def test_invalid_group_selectors_are_rejected(group, error_type, message):
    with pytest.raises(error_type, match=message):
        group.validate()


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


def test_physics_cfg_validates_nested_collision_policy():
    cfg = PhysicsCfg(
        class_type=object,
        collision_filter=CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=())}),
    )

    with pytest.raises(ValueError, match="at least one selector"):
        cfg.validate()


@pytest.mark.parametrize("use_isaac_sim", [True, False])
def test_physx_auto_cfg_preserves_backend_neutral_policy(use_isaac_sim):
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    policy = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/.*",))})
    cfg = PhysxAutoCfg(
        deterministic=True,
        collision_filter=policy,
        isaacsim_physx=PhysxCfg(),
        ovphysx=OvPhysxCfg(),
    )

    resolved = _resolve_physx_auto_cfg(cfg, use_isaac_sim=use_isaac_sim)

    assert isinstance(resolved, PhysxCfg if use_isaac_sim else OvPhysxCfg)
    assert resolved is not (cfg.isaacsim_physx if use_isaac_sim else cfg.ovphysx)
    assert resolved.deterministic
    assert resolved.collision_filter == policy


@pytest.mark.parametrize("use_isaac_sim", [True, False])
def test_physx_auto_cfg_preserves_nested_shared_settings_unless_overridden(use_isaac_sim):
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    nested_policy = CollisionFilterCfg(
        groups={"nested": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Nested/.*",))}
    )
    outer_policy = CollisionFilterCfg(
        groups={"outer": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Outer/.*",))}
    )
    selected = (PhysxCfg if use_isaac_sim else OvPhysxCfg)(
        deterministic=True,
        collision_filter=nested_policy,
    )
    kwargs = {"isaacsim_physx": selected} if use_isaac_sim else {"ovphysx": selected}

    nested = _resolve_physx_auto_cfg(PhysxAutoCfg(**kwargs), use_isaac_sim=use_isaac_sim)
    overridden = _resolve_physx_auto_cfg(
        PhysxAutoCfg(collision_filter=outer_policy, **kwargs),
        use_isaac_sim=use_isaac_sim,
    )
    cleared = _resolve_physx_auto_cfg(
        PhysxAutoCfg(collision_filter=CollisionFilterCfg(), **kwargs),
        use_isaac_sim=use_isaac_sim,
    )

    assert nested.deterministic
    assert nested.collision_filter == nested_policy
    assert overridden.deterministic
    assert overridden.collision_filter == outer_policy
    assert cleared.collision_filter == CollisionFilterCfg()
