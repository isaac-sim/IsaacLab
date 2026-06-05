# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_experimental.envs.frontend`.

Pure-Python unit tests; no app launch. Covers:

* :class:`Frontend` / :class:`Workflow` enum surface.
* :func:`SceneEntityCfg.from_stable` field copy.
* :func:`_require_newton_physics` hard-check.
* :func:`_walk_terms` recursive ManagerTermBaseCfg discovery.
* :func:`_promote_scene_entity_cfgs` walks ``term.params`` dicts.
* :func:`_swap_mdp` swaps ``func`` *and* ``class_type``; raises with a path
  list when twins are missing.
* :func:`_resolve_warp_twin` rejects stable-origin re-exports.
* :func:`_assert_direct_warp_registration` accepts warp-rooted entry
  points and rejects stable ones.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

import gymnasium as gym
from isaaclab_experimental.envs.frontend import (
    Frontend,
    FrontendIncompatibleError,
    Workflow,
    _assert_direct_warp_registration,
    _is_swap_candidate,
    _promote_scene_entity_cfgs,
    _require_newton_physics,
    _resolve_warp_twin,
    _swap_mdp,
    _walk_terms,
)
from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as WarpSceneEntityCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.managers.manager_term_cfg import EventTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as StableSceneEntityCfg
from isaaclab.utils.configclass import configclass

# ======================================================================
# Enums
# ======================================================================


class TestEnums(unittest.TestCase):
    def test_frontend_values(self):
        self.assertEqual(Frontend.TORCH, "torch")
        self.assertEqual(Frontend.WARP, "warp")

    def test_frontend_coercion(self):
        self.assertIs(Frontend("torch"), Frontend.TORCH)
        self.assertIs(Frontend("warp"), Frontend.WARP)
        with self.assertRaises(ValueError):
            Frontend("kit")

    def test_workflow_values(self):
        self.assertEqual(Workflow.MANAGER_BASED, "manager_based")
        self.assertEqual(Workflow.DIRECT, "direct")


# ======================================================================
# SceneEntityCfg.from_stable
# ======================================================================


class TestFromStable(unittest.TestCase):
    def test_copies_minimum_fields(self):
        stable = StableSceneEntityCfg(name="robot")
        warp = WarpSceneEntityCfg.from_stable(stable)
        self.assertIsInstance(warp, WarpSceneEntityCfg)
        self.assertEqual(warp.name, "robot")
        # Warp-only fields stay None until :meth:`resolve` runs.
        self.assertIsNone(warp.joint_mask)
        self.assertIsNone(warp.joint_ids_wp)
        self.assertIsNone(warp.body_ids_wp)

    def test_copies_all_selection_fields(self):
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
            self.assertEqual(getattr(warp, field), getattr(stable, field), msg=f"field {field!r} mismatch")


# ======================================================================
# _require_newton_physics
# ======================================================================


class TestRequireNewtonPhysics(unittest.TestCase):
    def _cfg_with(self, physics: Any) -> Any:
        cfg = types.SimpleNamespace()
        cfg.sim = types.SimpleNamespace(physics=physics)
        return cfg

    def test_passes_for_newton(self):
        cfg = self._cfg_with(NewtonCfg())
        _require_newton_physics(cfg, "Isaac-Test-v0")  # no raise

    def test_rejects_physx(self):
        cfg = self._cfg_with(PhysxCfg())
        with self.assertRaises(FrontendIncompatibleError) as exc:
            _require_newton_physics(cfg, "Isaac-Test-v0")
        self.assertIn("presets=newton_mjwarp", str(exc.exception))
        self.assertIn("PhysxCfg", str(exc.exception))

    def test_rejects_none(self):
        cfg = self._cfg_with(None)
        with self.assertRaises(FrontendIncompatibleError):
            _require_newton_physics(cfg, "Isaac-Test-v0")


# ======================================================================
# Configclass fixtures for the walker / swap tests.
# ======================================================================
#
# These mirror the real cfg shape so :func:`_walk_terms` descends into them
# (it only descends into objects with ``__dataclass_fields__``). Term cfgs
# use the real :class:`EventTermCfg`/:class:`RewardTermCfg`/:class:`ObservationTermCfg`
# so the walker's ``isinstance(ManagerTermBaseCfg)`` discriminator yields them.


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
# _walk_terms
# ======================================================================


class TestWalkTerms(unittest.TestCase):
    def test_yields_each_term_with_its_path(self):
        cfg = _CfgFixture(
            rewards=_RewardsCfg(r1=_term(), r2=_term()),
            events=_EventsCfg(e1=EventTermCfg(func=_stable_func, mode="reset")),
        )
        # Configclass instances aren't hashable, so collect paths only.
        paths = {".".join(p) for p, _ in _walk_terms(cfg)}
        self.assertEqual(paths, {"rewards.r1", "rewards.r2", "events.e1"})

    def test_descends_into_obs_subgroups(self):
        cfg = _CfgFixture(
            observations=_ObservationsCfg(
                policy=_PolicyObsGroup(o1=ObservationTermCfg(func=_stable_func)),
                perception=_ExtraObsGroup(o3=ObservationTermCfg(func=_stable_func)),
            ),
        )
        paths = {".".join(p) for p, _ in _walk_terms(cfg)}
        # Discovery is purely type-driven; no obs group name is hardcoded.
        self.assertEqual(paths, {"observations.policy.o1", "observations.perception.o3"})

    def test_stops_at_terms(self):
        # The walker must not descend into term.params / term.func — yields the term itself.
        nested_se_cfg = StableSceneEntityCfg(name="robot")
        cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(params={"asset_cfg": nested_se_cfg})))
        terms = list(_walk_terms(cfg))
        self.assertEqual(len(terms), 1)
        _, term = terms[0]
        self.assertIsInstance(term, RewardTermCfg)

    def test_skips_non_configclass_attrs(self):
        # A namespace without __dataclass_fields__ is not descended into.
        cfg = types.SimpleNamespace(some_plain_attr="hello")
        self.assertEqual(list(_walk_terms(cfg)), [])

    def test_skips_none_subtrees(self):
        cfg = _CfgFixture(rewards=None, events=None)
        self.assertEqual(list(_walk_terms(cfg)), [])


# ======================================================================
# _promote_scene_entity_cfgs
# ======================================================================


class TestPromoteSceneEntityCfgs(unittest.TestCase):
    def test_promotes_in_params(self):
        cfg = _CfgFixture(
            rewards=_RewardsCfg(
                r1=_term(params={"asset_cfg": StableSceneEntityCfg(name="robot", joint_names=["lf_hip"]), "scale": 1.0})
            )
        )
        _promote_scene_entity_cfgs(cfg)
        promoted = cfg.rewards.r1.params["asset_cfg"]
        self.assertIsInstance(promoted, WarpSceneEntityCfg)
        self.assertEqual(promoted.name, "robot")
        self.assertEqual(promoted.joint_names, ["lf_hip"])
        # Non-SceneEntityCfg params are untouched.
        self.assertEqual(cfg.rewards.r1.params["scale"], 1.0)

    def test_skips_already_warp(self):
        # configclass init deep-copies params, so identity won't hold across
        # construction; what we actually want to assert is "no re-promotion":
        # the asset_cfg remains a WarpSceneEntityCfg (i.e., wasn't passed
        # back through `from_stable`).
        warp = WarpSceneEntityCfg(name="robot", joint_names=["lf_hip"])
        cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(params={"asset_cfg": warp})))
        before = cfg.rewards.r1.params["asset_cfg"]
        _promote_scene_entity_cfgs(cfg)
        after = cfg.rewards.r1.params["asset_cfg"]
        self.assertIsInstance(after, WarpSceneEntityCfg)
        # The asset_cfg object was not replaced by another from_stable call.
        self.assertIs(after, before)

    def test_walks_all_term_groups(self):
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
        _promote_scene_entity_cfgs(cfg)
        # Warp-managed groups (rewards, observations incl. sub-groups, events) are promoted.
        self.assertIsInstance(cfg.rewards.r1.params["asset_cfg"], WarpSceneEntityCfg)
        self.assertIsInstance(cfg.observations.policy.o1.params["asset_cfg"], WarpSceneEntityCfg)
        # The perception sub-group is reached even though its attribute name is
        # not hardcoded in the framework.
        self.assertIsInstance(cfg.observations.perception.o3.params["asset_cfg"], WarpSceneEntityCfg)
        # The event manager is warp-first, so its terms are promoted too.
        self.assertIsInstance(cfg.events.e1.params["asset_cfg"], WarpSceneEntityCfg)
        # Curriculum runs on the stable (torch) manager, so its SceneEntityCfg is
        # left untouched — promoting it would hand a warp variant to a torch manager.
        self.assertIsInstance(cfg.curriculum.c1.params["asset_cfg"], StableSceneEntityCfg)
        self.assertNotIsInstance(cfg.curriculum.c1.params["asset_cfg"], WarpSceneEntityCfg)


# ======================================================================
# _swap_mdp
# ======================================================================


class _FakeMdpModule:
    """Stand-in for a warp mdp module containing twins."""

    __name__ = "test_fake_warp_mdp"


class TestSwapMdp(unittest.TestCase):
    def _patch_modules(self, name_to_symbol: dict[str, Any]) -> _FakeMdpModule:
        m = _FakeMdpModule()
        for name, sym in name_to_symbol.items():
            setattr(m, name, sym)
        return m

    def _patched_warp_mdp_modules(self, modules: list[Any]):
        import isaaclab_experimental.envs.frontend as fe

        self._orig = fe._warp_mdp_modules
        fe._warp_mdp_modules = lambda task_id: modules  # type: ignore[assignment]

    def setUp(self) -> None:
        self._orig = None

    def tearDown(self) -> None:
        if self._orig is not None:
            import isaaclab_experimental.envs.frontend as fe

            fe._warp_mdp_modules = self._orig

    def test_swaps_func_and_class_type(self):
        fake = self._patch_modules({"_stable_func": _warp_twin_func, "_StableActionCls": _WarpActionCls})
        self._patched_warp_mdp_modules([fake])
        term_reward = _term(func=_stable_func)
        term_action = _term()
        term_action.class_type = _StableActionCls  # set attr to exercise class_type swap
        cfg = _CfgFixture(rewards=_RewardsCfg(r1=term_reward, r2=term_action))
        _swap_mdp(cfg, "Isaac-Test-v0")
        self.assertIs(cfg.rewards.r1.func, _warp_twin_func)
        self.assertIs(cfg.rewards.r2.class_type, _WarpActionCls)

    def test_missing_twin_raises_with_path_list(self):
        fake = self._patch_modules({})
        self._patched_warp_mdp_modules([fake])
        cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(func=_stable_func)))
        with self.assertRaises(FrontendIncompatibleError) as exc:
            _swap_mdp(cfg, "Isaac-Test-v0")
        msg = str(exc.exception)
        self.assertIn("rewards.r1.func", msg)
        self.assertIn("_stable_func", msg)
        # The cfg term wasn't mutated for the missing twin.
        self.assertIs(cfg.rewards.r1.func, _stable_func)

    def test_skips_already_warp(self):
        fake = self._patch_modules({})
        self._patched_warp_mdp_modules([fake])
        cfg = _CfgFixture(rewards=_RewardsCfg(r1=_term(func=_warp_twin_func)))
        _swap_mdp(cfg, "Isaac-Test-v0")  # no raise
        self.assertIs(cfg.rewards.r1.func, _warp_twin_func)


# ======================================================================
# Twin resolution
# ======================================================================


class TestResolveWarpTwin(unittest.TestCase):
    def test_accepts_warp_origin(self):
        m = types.SimpleNamespace()
        m.foo = _warp_twin_func
        result = _resolve_warp_twin("foo", [m])
        self.assertIs(result, _warp_twin_func)

    def test_rejects_stable_origin(self):
        m = types.SimpleNamespace()
        m.foo = _stable_func  # same name, stable origin
        self.assertIsNone(_resolve_warp_twin("foo", [m]))

    def test_returns_none_when_absent(self):
        m = types.SimpleNamespace()
        self.assertIsNone(_resolve_warp_twin("missing", [m]))


# ======================================================================
# Swap candidate heuristic
# ======================================================================


class TestIsSwapCandidate(unittest.TestCase):
    def test_stable_callable_is_candidate(self):
        self.assertTrue(_is_swap_candidate(_stable_func))

    def test_warp_callable_is_not(self):
        self.assertFalse(_is_swap_candidate(_warp_twin_func))

    def test_non_callable_is_not(self):
        self.assertFalse(_is_swap_candidate(42))
        self.assertFalse(_is_swap_candidate("string"))


# ======================================================================
# Direct workflow guard
# ======================================================================


_DIRECT_TEST_TASKS = {
    "_Frontend-Test-Warp-Direct-v0": ("isaaclab_tasks_experimental.fake:DirectEnv", True),
    "_Frontend-Test-Stable-Direct-v0": ("isaaclab_tasks.fake:DirectEnv", False),
}


class TestAssertDirectWarpRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Register stub tasks so ``gym.spec`` resolves them. Entry-point strings
        # are never invoked (the guard only inspects them), so a fake import
        # path is fine.
        cls._registered: list[str] = []
        for task_id, (ep, _) in _DIRECT_TEST_TASKS.items():
            try:
                gym.register(id=task_id, entry_point=ep, disable_env_checker=True)
                cls._registered.append(task_id)
            except gym.error.Error:
                pass

    def test_accepts_warp_rooted(self):
        _assert_direct_warp_registration("_Frontend-Test-Warp-Direct-v0")  # no raise

    def test_rejects_stable_rooted(self):
        with self.assertRaises(FrontendIncompatibleError) as exc:
            _assert_direct_warp_registration("_Frontend-Test-Stable-Direct-v0")
        self.assertIn("isaaclab_experimental", str(exc.exception))

    def test_rejects_unknown_task(self):
        with self.assertRaises(FrontendIncompatibleError):
            _assert_direct_warp_registration("Frontend-Test-NotRegistered-v0")


if __name__ == "__main__":
    unittest.main()
