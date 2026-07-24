# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_experimental.envs.frontend`.

Pure-Python unit tests; no app launch. Covers:

* :class:`Workflow` enum surface.
* :func:`SceneEntityCfg.from_stable` field copy.
* :func:`_require_newton_physics` hard-check.
* :func:`_walk_terms` recursive ManagerTermBaseCfg discovery over instance
  attributes (nested class objects are not descended into).
* :meth:`_mirror_module` / :meth:`_nearest_mdp_module` deterministic mirror
  resolution and :meth:`_cfg_route_modules` cfg-hierarchy resolution.
* :func:`_promote_scene_entity_cfgs` walks ``term.params`` dicts.
* :func:`_swap_mdp` swaps ``func`` *and* ``class_type``; raises with a path
  list when twins are missing.
* :func:`_resolve_warp_twin` rejects stable-origin re-exports.
* :func:`_assert_direct_warp_registration` accepts warp-rooted entry
  points and rejects stable ones.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import isaaclab_experimental.envs.frontend as fe
import pytest
from isaaclab_experimental.envs.frontend import (
    FrontendIncompatibleError,
    WarpFrontend,
    Workflow,
)
from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as WarpSceneEntityCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as StableSceneEntityCfg
from isaaclab.utils.configclass import configclass

# ======================================================================
# Fixtures: fake stable/warp symbols and configclass trees.
#
# These mirror the real cfg shape so :func:`_walk_terms` descends into them
# (it only descends into objects with ``__dataclass_fields__``). Term cfgs
# use the real :class:`EventTermCfg`/:class:`RewardTermCfg`/:class:`ObservationTermCfg`
# so the walker's ``isinstance(ManagerTermBaseCfg)`` discriminator yields them.
# ======================================================================


def _stable_func(env, **params):
    return None


class _StableActionCls:
    pass


_stable_func.__module__ = "isaaclab_tasks.fake_task.mdp"
_StableActionCls.__module__ = "isaaclab_tasks.fake_task.mdp"


def _warp_twin_func(env, out, **params):
    return None


class _WarpActionCls:
    pass


_warp_twin_func.__module__ = "isaaclab_experimental.envs.mdp"
_WarpActionCls.__module__ = "isaaclab_experimental.envs.mdp"


@configclass
class _PolicyObsGroup:
    """Stand-in for a per-task ObservationsCfg sub-group (e.g. PolicyCfg)."""

    o1: ObservationTermCfg | None = None
    o2: ObservationTermCfg | None = None


@configclass
class _ExtraObsGroup:
    """A second obs group (named arbitrarily) to exercise multi-group walks."""

    o3: ObservationTermCfg | None = None


@configclass
class _ObservationsCfg:
    policy: _PolicyObsGroup | None = None
    perception: _ExtraObsGroup | None = None


@configclass
class _RewardsCfg:
    r1: RewardTermCfg | None = None
    r2: RewardTermCfg | None = None


@configclass
class _EventsCfg:
    e1: EventTermCfg | None = None


@configclass
class _CurriculumCfg:
    c1: EventTermCfg | None = None


@configclass
class _CfgFixture:
    observations: _ObservationsCfg | None = None
    rewards: _RewardsCfg | None = None
    events: _EventsCfg | None = None
    curriculum: _CurriculumCfg | None = None


def _term(func=None, params: dict | None = None) -> RewardTermCfg:
    """Cheap RewardTermCfg builder for tests; the cfg class is irrelevant for swap/walk logic."""
    return RewardTermCfg(func=func or _stable_func, weight=1.0, params=params or {})


# ======================================================================
# Enums
# ======================================================================


def test_workflow_values():
    assert Workflow.MANAGER_BASED == "manager_based"
    assert Workflow.DIRECT == "direct"


# ======================================================================
# build()
# ======================================================================


def test_warp_manager_build_constructs_warp_env_with_cfg():
    # Cfg adaptation happens inside ManagerBasedRLEnvWarp.__init__, so build()
    # only selects the env class and forwards the cfg and construct kwargs.
    import isaaclab_experimental.envs as envs

    cfg = object.__new__(ManagerBasedRLEnvCfg)
    expected_env = object()
    calls: list[tuple[Any, Any]] = []

    def fake_env(*, cfg: Any, **kwargs: Any) -> Any:
        calls.append((cfg, kwargs))
        return expected_env

    with patch.object(envs, "ManagerBasedRLEnvWarp", fake_env):
        env = WarpFrontend.build_env(cfg, "Isaac-Test", render_mode="rgb_array")

    assert env is expected_env
    assert calls == [(cfg, {"render_mode": "rgb_array"})]


# ======================================================================
# SceneEntityCfg.from_stable
# ======================================================================


def test_from_stable_copies_minimum_fields():
    stable = StableSceneEntityCfg(name="robot")
    warp = WarpSceneEntityCfg.from_stable(stable)
    assert isinstance(warp, WarpSceneEntityCfg)
    assert warp.name == "robot"
    # Warp-only fields stay None until :meth:`resolve` runs.
    assert warp.joint_mask is None
    assert warp.joint_ids_wp is None
    assert warp.body_ids_wp is None


def test_from_stable_copies_all_selection_fields():
    stable = StableSceneEntityCfg(
        name="robot",
        joint_names=["lf_hip"],
        joint_ids=[0, 1, 2],
        fixed_tendon_names=["tendon_a"],
        fixed_tendon_ids=[5],
        body_names=["base", "lf_foot"],
        body_ids=[0, 4],
        object_collection_names=["objs"],
        object_collection_ids=[7],
        preserve_order=True,
    )
    warp = WarpSceneEntityCfg.from_stable(stable)
    for field in (
        "name",
        "joint_names",
        "joint_ids",
        "fixed_tendon_names",
        "fixed_tendon_ids",
        "body_names",
        "body_ids",
        "object_collection_names",
        "object_collection_ids",
        "preserve_order",
    ):
        assert getattr(warp, field) == getattr(stable, field), f"field {field!r} mismatch"


# ======================================================================
# _require_newton_physics
# ======================================================================


def _cfg_with_physics(physics: Any) -> Any:
    cfg = types.SimpleNamespace()
    cfg.sim = types.SimpleNamespace(physics=physics)
    return cfg


def test_require_newton_passes_for_newton():
    fe.WarpFrontend._require_newton_physics(_cfg_with_physics(NewtonCfg()), "Isaac-Test-v0")  # no raise


def test_require_newton_rejects_physx():
    with pytest.raises(FrontendIncompatibleError) as exc:
        fe.WarpFrontend._require_newton_physics(_cfg_with_physics(PhysxCfg()), "Isaac-Test-v0")
    assert "presets=newton_mjwarp" in str(exc.value)
    assert "PhysxCfg" in str(exc.value)


def test_require_newton_rejects_none():
    with pytest.raises(FrontendIncompatibleError):
        fe.WarpFrontend._require_newton_physics(_cfg_with_physics(None), "Isaac-Test-v0")


# ======================================================================
# _walk_terms
# ======================================================================


def test_walk_terms_yields_each_term_with_its_path():
    cfg = _CfgFixture(
        rewards=_RewardsCfg(r1=_term(), r2=_term()),
        events=_EventsCfg(e1=EventTermCfg(func=_stable_func, mode="reset")),
    )
    # Configclass instances aren't hashable, so collect paths only.
    paths = {".".join(p) for p, _ in fe.WarpFrontend._walk_terms(cfg)}
    assert paths == {"rewards.r1", "rewards.r2", "events.e1"}


def test_walk_terms_descends_into_obs_subgroups():
    cfg = _CfgFixture(
        observations=_ObservationsCfg(
            policy=_PolicyObsGroup(o1=ObservationTermCfg(func=_stable_func)),
            perception=_ExtraObsGroup(o3=ObservationTermCfg(func=_stable_func)),
        ),
    )
    paths = {".".join(p) for p, _ in fe.WarpFrontend._walk_terms(cfg)}
    # Discovery is purely type-driven; no obs group name is hardcoded.
    assert paths == {"observations.policy.o1", "observations.perception.o3"}


def test_walk_terms_stops_at_terms():
    # The walker must not descend into term.params / term.func — yields the term itself.
    nested_se_cfg = StableSceneEntityCfg(name="robot")
    cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(params={"asset_cfg": nested_se_cfg})))
    terms = list(fe.WarpFrontend._walk_terms(cfg))
    assert len(terms) == 1
    _, term = terms[0]
    assert isinstance(term, RewardTermCfg)


def test_walk_terms_skips_non_configclass_attrs():
    # A namespace without __dataclass_fields__ is not descended into.
    cfg = types.SimpleNamespace(some_plain_attr="hello")
    assert list(fe.WarpFrontend._walk_terms(cfg)) == []


def test_walk_terms_skips_none_subtrees():
    cfg = _CfgFixture(rewards=None, events=None)
    assert list(fe.WarpFrontend._walk_terms(cfg)) == []


def test_walk_terms_descends_into_dict_groups():
    # A manager group expressed as a dict of {name: term} (rather than a configclass)
    # must still have its terms discovered — the warp managers accept both forms.
    cfg = _CfgFixture()
    cfg.rewards = {"r1": _term(), "r2": _term()}
    paths = {".".join(p) for p, _ in fe.WarpFrontend._walk_terms(cfg)}
    assert paths == {"rewards.r1", "rewards.r2"}


@configclass
class _GroupWithNestedClass:
    """Mirrors the real ``ObservationsCfg.PolicyCfg`` nested-class pattern."""

    @configclass
    class TemplateCfg:
        t1: ObservationTermCfg | None = None

    o1: ObservationTermCfg | None = None


def test_walk_terms_ignores_nested_class_objects():
    # The walker iterates instance attributes only (mirroring the warp managers'
    # ``cfg.__dict__`` consumption), so a nested class object — reachable as an
    # attribute on the instance — must never contribute paths.
    cfg = _GroupWithNestedClass(o1=ObservationTermCfg(func=_stable_func))
    paths = {".".join(p) for p, _ in fe.WarpFrontend._walk_terms(cfg)}
    assert paths == {"o1"}


# ======================================================================
# Mirror resolution
# ======================================================================


def test_mirror_module_maps_stable_roots():
    assert (
        fe.WarpFrontend._mirror_module("isaaclab_tasks.core.velocity.mdp.rewards")
        == "isaaclab_tasks_experimental.core.velocity.mdp.rewards"
    )
    assert fe.WarpFrontend._mirror_module("isaaclab.envs.mdp.events") == "isaaclab_experimental.envs.mdp.events"
    assert fe.WarpFrontend._mirror_module("isaaclab_tasks") == "isaaclab_tasks_experimental"
    # Unmirrored roots are not resolvable.
    assert fe.WarpFrontend._mirror_module("torch.nn.functional") is None
    assert fe.WarpFrontend._mirror_module("") is None


def test_nearest_mdp_module_truncates_at_mdp_boundary():
    # A symbol module resolves to its own mdp package on the mirrored path.
    assert (
        fe.WarpFrontend._nearest_mdp_module("isaaclab_tasks_experimental.core.locomotion.mdp.rewards")
        == "isaaclab_tasks_experimental.core.locomotion.mdp"
    )
    assert fe.WarpFrontend._nearest_mdp_module("isaaclab_experimental.envs.mdp.events") == (
        "isaaclab_experimental.envs.mdp"
    )


def test_nearest_mdp_module_walks_up_cfg_paths():
    # A cfg module path walks up to the task family's mdp package.
    assert (
        fe.WarpFrontend._nearest_mdp_module("isaaclab_tasks_experimental.core.velocity.config.anymal_d.flat_env_cfg")
        == "isaaclab_tasks_experimental.core.velocity.mdp"
    )
    # Nothing on the path provides an mdp package.
    assert fe.WarpFrontend._nearest_mdp_module("isaaclab_tasks_experimental.nonexistent.task_pkg") is None


def test_cfg_route_modules_resolves_mirrored_hierarchy():
    # A class whose module mirrors into the experimental tree resolves to the
    # task family's mdp package; unmirrored classes contribute nothing.
    fixture_cls = type("FakeVelocityCfg", (), {})
    fixture_cls.__module__ = "isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg"
    modules = fe.WarpFrontend._cfg_route_modules(fixture_cls())
    assert [m.__name__ for m in modules] == ["isaaclab_tasks_experimental.core.velocity.mdp"]

    assert fe.WarpFrontend._cfg_route_modules(_CfgFixture()) == []


# ======================================================================
# _swap_mdp
# ======================================================================


class _FakeMdpModule:
    """Stand-in for a warp mdp module containing twins."""

    __name__ = "test_fake_warp_mdp"


def _fake_mdp_module(name_to_symbol: dict[str, Any]) -> _FakeMdpModule:
    module = _FakeMdpModule()
    for name, symbol in name_to_symbol.items():
        setattr(module, name, symbol)
    return module


def _patch_twin_modules(monkeypatch: pytest.MonkeyPatch, modules: list[Any]) -> None:
    """Pin twin resolution to ``modules`` and keep unit tests hermetic."""
    monkeypatch.setattr(fe.WarpFrontend, "_cfg_route_modules", classmethod(lambda cls, cfg: []))
    monkeypatch.setattr(
        fe.WarpFrontend, "_twin_modules", classmethod(lambda cls, symbol_module, cfg_route_modules: modules)
    )


def test_swap_mdp_swaps_func_and_class_type(monkeypatch: pytest.MonkeyPatch):
    fake = _fake_mdp_module({"_stable_func": _warp_twin_func, "_StableActionCls": _WarpActionCls})
    _patch_twin_modules(monkeypatch, [fake])
    term_reward = _term(func=_stable_func)
    term_action = _term()
    term_action.class_type = _StableActionCls  # set attr to exercise class_type swap
    cfg = _CfgFixture(rewards=_RewardsCfg(r1=term_reward, r2=term_action))
    fe.WarpFrontend._swap_mdp(cfg, "Isaac-Test-v0")
    assert cfg.rewards.r1.func is _warp_twin_func
    assert cfg.rewards.r2.class_type is _WarpActionCls


def test_swap_mdp_missing_twin_raises_with_path_list(monkeypatch: pytest.MonkeyPatch):
    _patch_twin_modules(monkeypatch, [_fake_mdp_module({})])
    cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(func=_stable_func)))
    with pytest.raises(FrontendIncompatibleError) as exc:
        fe.WarpFrontend._swap_mdp(cfg, "Isaac-Test-v0")
    msg = str(exc.value)
    assert "rewards.r1.func" in msg
    assert "_stable_func" in msg
    # The cfg term wasn't mutated for the missing twin.
    assert cfg.rewards.r1.func is _stable_func


def test_swap_mdp_skips_already_warp(monkeypatch: pytest.MonkeyPatch):
    _patch_twin_modules(monkeypatch, [_fake_mdp_module({})])
    cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(func=_warp_twin_func)))
    fe.WarpFrontend._swap_mdp(cfg, "Isaac-Test-v0")  # no raise
    assert cfg.rewards.r1.func is _warp_twin_func


# ======================================================================
# _promote_scene_entity_cfgs
# ======================================================================


def test_promote_scene_entity_cfgs_promotes_in_params():
    cfg = _CfgFixture(
        rewards=_RewardsCfg(
            r1=_term(params={"asset_cfg": StableSceneEntityCfg(name="robot", joint_names=["lf_hip"]), "scale": 1.0})
        )
    )
    fe.WarpFrontend._promote_scene_entity_cfgs(cfg)
    promoted = cfg.rewards.r1.params["asset_cfg"]
    assert isinstance(promoted, WarpSceneEntityCfg)
    assert promoted.name == "robot"
    assert promoted.joint_names == ["lf_hip"]
    # Non-SceneEntityCfg params are untouched.
    assert cfg.rewards.r1.params["scale"] == 1.0


def test_promote_scene_entity_cfgs_skips_already_warp():
    # configclass init deep-copies params, so identity won't hold across
    # construction; what we actually want to assert is "no re-promotion":
    # the asset_cfg remains a WarpSceneEntityCfg (i.e., wasn't passed
    # back through `from_stable`).
    warp = WarpSceneEntityCfg(name="robot", joint_names=["lf_hip"])
    cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(params={"asset_cfg": warp})))
    before = cfg.rewards.r1.params["asset_cfg"]
    fe.WarpFrontend._promote_scene_entity_cfgs(cfg)
    after = cfg.rewards.r1.params["asset_cfg"]
    assert isinstance(after, WarpSceneEntityCfg)
    # The asset_cfg object was not replaced by another from_stable call.
    assert after is before


def test_promote_scene_entity_cfgs_walks_all_term_groups():
    cfg = _CfgFixture(
        rewards=_RewardsCfg(r1=_term(params={"asset_cfg": StableSceneEntityCfg(name="r")})),
        events=_EventsCfg(
            e1=EventTermCfg(func=_stable_func, mode="reset", params={"asset_cfg": StableSceneEntityCfg(name="e")})
        ),
        observations=_ObservationsCfg(
            policy=_PolicyObsGroup(
                o1=ObservationTermCfg(func=_stable_func, params={"asset_cfg": StableSceneEntityCfg(name="o-pol")})
            ),
            perception=_ExtraObsGroup(
                o3=ObservationTermCfg(func=_stable_func, params={"asset_cfg": StableSceneEntityCfg(name="o-per")})
            ),
        ),
        curriculum=_CurriculumCfg(
            c1=EventTermCfg(func=_stable_func, mode="reset", params={"asset_cfg": StableSceneEntityCfg(name="c")})
        ),
    )
    fe.WarpFrontend._promote_scene_entity_cfgs(cfg)
    # Warp-managed groups (rewards, observations incl. sub-groups, events) are promoted.
    assert isinstance(cfg.rewards.r1.params["asset_cfg"], WarpSceneEntityCfg)
    assert isinstance(cfg.observations.policy.o1.params["asset_cfg"], WarpSceneEntityCfg)
    # The perception sub-group is reached even though its attribute name is
    # not hardcoded in the framework.
    assert isinstance(cfg.observations.perception.o3.params["asset_cfg"], WarpSceneEntityCfg)
    # The event manager is warp-first, so its terms are promoted too.
    assert isinstance(cfg.events.e1.params["asset_cfg"], WarpSceneEntityCfg)
    # Curriculum runs on the stable (torch) manager, so its SceneEntityCfg is
    # left untouched — promoting it would hand a warp variant to a torch manager.
    assert isinstance(cfg.curriculum.c1.params["asset_cfg"], StableSceneEntityCfg)
    assert not isinstance(cfg.curriculum.c1.params["asset_cfg"], WarpSceneEntityCfg)


# ======================================================================
# Twin resolution and swap-candidate heuristic
# ======================================================================


def test_resolve_warp_twin_accepts_warp_origin():
    module = types.SimpleNamespace(foo=_warp_twin_func)
    assert fe.WarpFrontend._resolve_warp_twin("foo", [module]) is _warp_twin_func


def test_resolve_warp_twin_rejects_stable_origin():
    module = types.SimpleNamespace(foo=_stable_func)  # same name, stable origin
    assert fe.WarpFrontend._resolve_warp_twin("foo", [module]) is None


def test_resolve_warp_twin_returns_none_when_absent():
    assert fe.WarpFrontend._resolve_warp_twin("missing", [types.SimpleNamespace()]) is None


def test_is_swap_candidate_stable_callable():
    assert fe.WarpFrontend._is_swap_candidate(_stable_func)


def test_is_swap_candidate_rejects_warp_callable():
    assert not fe.WarpFrontend._is_swap_candidate(_warp_twin_func)


def test_is_swap_candidate_rejects_non_callables():
    assert not fe.WarpFrontend._is_swap_candidate(42)
    assert not fe.WarpFrontend._is_swap_candidate("string")


# ======================================================================
# Direct workflow guard
# ======================================================================


_DIRECT_TEST_TASKS = {
    "_Frontend-Test-Warp-Direct-v0": ("isaaclab_tasks_experimental.fake:DirectEnv", None),
    "_Frontend-Test-Stable-Direct-v0": ("isaaclab_tasks.fake:DirectEnv", None),
    "_Frontend-Test-Declared-Direct-v0": ("isaaclab_tasks.fake:DirectEnv", "types:SimpleNamespace"),
    "_Frontend-Test-Broken-Direct-v0": ("isaaclab_tasks.fake:DirectEnv", "definitely_not_a_module:Env"),
    "_Frontend-Test-Missing-Class-Direct-v0": ("isaaclab_tasks.fake:DirectEnv", "types:NoSuchClass"),
    # Follows the <task>_direct_env:<Name>Env convention, so it resolves to the
    # mirrored <task>_warp_env:<Name>WarpEnv with no warp_entry_point.
    "_Frontend-Test-NameMirror-Direct-v0": ("isaaclab_tasks.core.cartpole.cartpole_direct_env:CartpoleEnv", None),
    # Real shared reorient env class: both Allegro and Shadow register it, and it
    # mirrors to the real ReorientDirectWarpEnv (which declares cfg AllegroHandEnvCfg).
    # Used to exercise the cfg-compatibility gate on a genuinely shared base env.
    "_Frontend-Test-Reorient-Direct-v0": ("isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv", None),
}


@pytest.fixture(scope="module", autouse=True)
def _register_direct_stub_tasks():
    # Register stub tasks so ``gym.spec`` resolves them. Entry-point strings
    # are never invoked (the guard only inspects them), so a fake import
    # path is fine. ``warp_entry_point`` targets are imported, so the valid
    # stub points at a real stdlib symbol.
    registered = []
    for task_id, (entry_point, warp_entry_point) in _DIRECT_TEST_TASKS.items():
        kwargs = {"warp_entry_point": warp_entry_point} if warp_entry_point else {}
        with contextlib.suppress(gym.error.Error):
            gym.register(id=task_id, entry_point=entry_point, disable_env_checker=True, kwargs=kwargs)
            registered.append(task_id)
    yield
    # Unregister so later test modules see the real, stub-free registry.
    for task_id in registered:
        gym.registry.pop(task_id, None)


def test_direct_guard_accepts_warp_rooted():
    fe.WarpFrontend._assert_direct_warp_registration("_Frontend-Test-Warp-Direct-v0")  # no raise


def test_direct_guard_rejects_stable_rooted():
    with pytest.raises(FrontendIncompatibleError) as exc:
        fe.WarpFrontend._assert_direct_warp_registration("_Frontend-Test-Stable-Direct-v0")
    assert "isaaclab_experimental" in str(exc.value)


def test_resolve_direct_warp_class_returns_declared_class():
    import types as types_module

    env_class = fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Declared-Direct-v0")
    assert env_class is types_module.SimpleNamespace


def test_resolve_direct_warp_class_returns_none_without_declaration():
    assert fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Stable-Direct-v0") is None


def test_resolve_direct_warp_class_by_name_mirror():
    """A direct task following the ``<task>_direct_env:<Name>Env`` convention resolves
    to ``<task>_warp_env:<Name>WarpEnv`` with no ``warp_entry_point`` declared."""
    env_class = fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-NameMirror-Direct-v0")
    assert env_class is not None
    assert env_class.__name__ == "CartpoleWarpEnv"
    assert env_class.__module__.endswith("cartpole_warp_env")


def test_resolve_direct_warp_class_rejects_broken_module():
    with pytest.raises(FrontendIncompatibleError):
        fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Broken-Direct-v0")


def test_resolve_direct_warp_class_rejects_missing_class():
    with pytest.raises(FrontendIncompatibleError):
        fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Missing-Class-Direct-v0")


def test_resolve_direct_warp_class_rejects_unknown_task():
    with pytest.raises(FrontendIncompatibleError):
        fe.WarpFrontend._resolve_direct_warp_class("Frontend-Test-NotRegistered-v0")


# ======================================================================
# Direct cfg-compatibility guard
# ======================================================================
#
# A stable direct env class can be shared by several tasks (e.g. ReorientDirectEnv
# serves the Allegro and Shadow hands via different cfgs). Name resolution yields a
# single warp twin that may implement only one variant, so the twin declares the cfg
# it supports via its ``cfg`` annotation and the frontend refuses a mismatched cfg
# rather than silently running the wrong MDP.


class _GuardDeclaredCfg:
    pass


class _GuardSubclassCfg(_GuardDeclaredCfg):
    pass


class _GuardUnrelatedCfg:
    pass


class _GuardWarpEnv:
    cfg: _GuardDeclaredCfg  # under `from __future__ import annotations` this stores the
    # identifier string "_GuardDeclaredCfg", exactly as a real warp env's `cfg: <StableCfg>`.


class _GuardWarpEnvNoAnnotation:
    pass


def test_declared_stable_cfg_name_reads_annotation():
    assert fe.WarpFrontend._declared_stable_cfg_name(_GuardWarpEnv) == "_GuardDeclaredCfg"
    assert fe.WarpFrontend._declared_stable_cfg_name(_GuardWarpEnvNoAnnotation) is None


def test_assert_cfg_supported_accepts_exact_cfg():
    fe.WarpFrontend._assert_cfg_supported(_GuardWarpEnv, _GuardDeclaredCfg(), "_task")  # no raise


def test_assert_cfg_supported_accepts_subclass_cfg():
    # A task cfg that IS-A the declared cfg is accepted (covers camera/variant subclasses).
    fe.WarpFrontend._assert_cfg_supported(_GuardWarpEnv, _GuardSubclassCfg(), "_task")  # no raise


def test_assert_cfg_supported_rejects_unrelated_cfg():
    # This is the Shadow-reorient-into-Allegro-twin regression: a cfg the twin does
    # not implement must be a hard error, not a silent wrong-MDP run.
    with pytest.raises(FrontendIncompatibleError) as exc:
        fe.WarpFrontend._assert_cfg_supported(_GuardWarpEnv, _GuardUnrelatedCfg(), "_task")
    msg = str(exc.value)
    assert "_GuardDeclaredCfg" in msg and "_GuardUnrelatedCfg" in msg


def test_assert_cfg_supported_skips_when_no_annotation():
    # No cfg contract declared → nothing to enforce → never raises.
    fe.WarpFrontend._assert_cfg_supported(_GuardWarpEnvNoAnnotation, _GuardUnrelatedCfg(), "_task")


# --- resolve correctness on the REAL shared reorient env class ---------
#
# `_Frontend-Test-Reorient-Direct-v0` uses the real ReorientDirectEnv entry point,
# so resolution mirrors to the real ReorientDirectWarpEnv (declared cfg
# AllegroHandEnvCfg). The gate keys on the cfg *class name*, so lightweight stub
# cfgs named exactly as the real ones exercise the accept/reject paths without
# constructing the heavy real configs.


class AllegroHandEnvCfg:  # noqa: N801 - name mirrors the real cfg the twin implements
    pass


class ShadowHandEnvCfg:  # noqa: N801 - name mirrors the real Shadow cfg the twin does NOT implement
    pass


def test_resolve_returns_twin_when_cfg_matches():
    env_class = fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Reorient-Direct-v0", AllegroHandEnvCfg())
    assert env_class is not None
    assert env_class.__name__ == "ReorientDirectWarpEnv"


def test_resolve_rejects_twin_when_cfg_mismatches():
    # The Shadow-reorient regression: the shared env name-mirrors to the Allegro
    # twin, but the twin does not implement the Shadow cfg → hard error, not a
    # silent wrong-MDP run.
    with pytest.raises(FrontendIncompatibleError) as exc:
        fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Reorient-Direct-v0", ShadowHandEnvCfg())
    msg = str(exc.value)
    assert "AllegroHandEnvCfg" in msg and "ShadowHandEnvCfg" in msg


def test_resolve_name_only_skips_cfg_check():
    # Without a cfg, resolution reports the name mapping alone (no applicability gate).
    env_class = fe.WarpFrontend._resolve_direct_warp_class("_Frontend-Test-Reorient-Direct-v0")
    assert env_class is not None
    assert env_class.__name__ == "ReorientDirectWarpEnv"
