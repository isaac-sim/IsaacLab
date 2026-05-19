# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_experimental.envs.frontend`.

Pure-Python unit tests; no app launch. Covers:

* :class:`Frontend` / :class:`Workflow` enum surface.
* :func:`SceneEntityCfg.from_stable` field copy.
* :func:`_require_newton_physics` hard-check.
* :func:`_promote_scene_entity_cfgs` walks ``term.params`` dicts.
* :func:`_swap_mdp` swaps ``func`` *and* ``class_type``; raises with a path
  list when twins are missing.
* :func:`_resolve_warp_twin` rejects stable-origin re-exports.
* :func:`_require_direct_is_warp_task` accepts warp-rooted entry points
  and rejects stable ones.
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
    _is_swap_candidate,
    _iter_term_attrs,
    _promote_scene_entity_cfgs,
    _require_direct_is_warp_task,
    _require_newton_physics,
    _resolve_warp_twin,
    _swap_mdp,
    _walk_attrs,
)
from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as WarpSceneEntityCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as StableSceneEntityCfg

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
        self.assertIn("presets=newton", str(exc.exception))
        self.assertIn("PhysxCfg", str(exc.exception))

    def test_rejects_none(self):
        cfg = self._cfg_with(None)
        with self.assertRaises(FrontendIncompatibleError):
            _require_newton_physics(cfg, "Isaac-Test-v0")


# ======================================================================
# _promote_scene_entity_cfgs
# ======================================================================


class _Term:
    """Minimal stand-in for a manager term cfg."""

    def __init__(self, params: dict | None = None, func=None, class_type=None):
        self.params = params if params is not None else {}
        if func is not None:
            self.func = func
        if class_type is not None:
            self.class_type = class_type


class _Group:
    """Minimal stand-in for a manager-cfg group (e.g. RewardsCfg)."""

    def __init__(self, **terms: Any):
        for name, term in terms.items():
            setattr(self, name, term)


class _ObsCfg:
    def __init__(self, policy: _Group):
        self.policy = policy


def _make_cfg(**groups: Any) -> Any:
    cfg = types.SimpleNamespace()
    for name, value in groups.items():
        setattr(cfg, name, value)
    return cfg


class TestPromoteSceneEntityCfgs(unittest.TestCase):
    def test_promotes_in_params(self):
        stable = StableSceneEntityCfg(name="robot", joint_names=["lf_hip"])
        term = _Term(params={"asset_cfg": stable, "scale": 1.0})
        cfg = _make_cfg(rewards=_Group(track=term))
        _promote_scene_entity_cfgs(cfg)
        promoted = term.params["asset_cfg"]
        self.assertIsInstance(promoted, WarpSceneEntityCfg)
        self.assertEqual(promoted.name, "robot")
        self.assertEqual(promoted.joint_names, ["lf_hip"])
        # Non-SceneEntityCfg params are untouched.
        self.assertEqual(term.params["scale"], 1.0)

    def test_skips_already_warp(self):
        warp = WarpSceneEntityCfg(name="robot")
        term = _Term(params={"asset_cfg": warp})
        cfg = _make_cfg(rewards=_Group(t=term))
        _promote_scene_entity_cfgs(cfg)
        self.assertIs(term.params["asset_cfg"], warp)  # unchanged identity

    def test_walks_all_term_paths(self):
        # Drop a stable cfg in every supported group; all should promote.
        groups = {}
        terms: list[_Term] = []
        for group_name in ("rewards", "events", "terminations", "commands", "curriculum"):
            t = _Term(params={"asset_cfg": StableSceneEntityCfg(name=group_name)})
            terms.append(t)
            groups[group_name] = _Group(t=t)
        # observations is nested as observations.policy
        obs_term = _Term(params={"asset_cfg": StableSceneEntityCfg(name="obs")})
        terms.append(obs_term)
        groups["observations"] = _ObsCfg(policy=_Group(t=obs_term))
        # actions group also walked
        act_term = _Term(params={"asset_cfg": StableSceneEntityCfg(name="act")})
        terms.append(act_term)
        groups["actions"] = _Group(t=act_term)

        cfg = _make_cfg(**groups)
        _promote_scene_entity_cfgs(cfg)
        for t in terms:
            self.assertIsInstance(t.params["asset_cfg"], WarpSceneEntityCfg)


# ======================================================================
# _swap_mdp
# ======================================================================


# A fake "stable" mdp module the test fixture stands in for
# ``isaaclab_tasks.<task>.mdp`` — symbols here pretend to live there.
def _stable_func(env, **params):
    return None


class _StableActionCls:
    pass


_stable_func.__module__ = "isaaclab_tasks.fake_task.mdp"
_StableActionCls.__module__ = "isaaclab_tasks.fake_task.mdp"


# And a fake warp twin module the test installs into sys.modules under the
# expected name resolution path. Symbols here pretend to live under
# ``isaaclab_experimental.envs.mdp`` (the fallback).
def _warp_twin_func(env, out, **params):
    return None


class _WarpActionCls:
    pass


_warp_twin_func.__module__ = "isaaclab_experimental.envs.mdp"
_WarpActionCls.__module__ = "isaaclab_experimental.envs.mdp"


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
        # Patch _warp_mdp_modules at the frontend module level so _swap_mdp
        # uses our fake module list instead of doing real importlib lookups.
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
        term_reward = _Term(func=_stable_func)
        term_action = _Term(class_type=_StableActionCls)
        cfg = _make_cfg(rewards=_Group(r=term_reward), actions=_Group(a=term_action))
        _swap_mdp(cfg, "Isaac-Test-v0")
        self.assertIs(term_reward.func, _warp_twin_func)
        self.assertIs(term_action.class_type, _WarpActionCls)

    def test_missing_twin_raises_with_path_list(self):
        # No twins available — every term should be reported as missing.
        fake = self._patch_modules({})
        self._patched_warp_mdp_modules([fake])
        term = _Term(func=_stable_func)
        cfg = _make_cfg(rewards=_Group(track=term))
        with self.assertRaises(FrontendIncompatibleError) as exc:
            _swap_mdp(cfg, "Isaac-Test-v0")
        msg = str(exc.exception)
        self.assertIn("rewards.track.func", msg)
        self.assertIn("_stable_func", msg)
        # The cfg term wasn't mutated (we hard-failed before partial application).
        self.assertIs(term.func, _stable_func)

    def test_skips_already_warp(self):
        fake = self._patch_modules({})
        self._patched_warp_mdp_modules([fake])
        term = _Term(func=_warp_twin_func)  # already warp-origin
        cfg = _make_cfg(rewards=_Group(r=term))
        _swap_mdp(cfg, "Isaac-Test-v0")  # no raise
        self.assertIs(term.func, _warp_twin_func)


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


class TestRequireDirectIsWarpTask(unittest.TestCase):
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
                # Already registered from a previous run — that's fine.
                pass

    def test_accepts_warp_rooted(self):
        _require_direct_is_warp_task("_Frontend-Test-Warp-Direct-v0")  # no raise

    def test_rejects_stable_rooted(self):
        with self.assertRaises(FrontendIncompatibleError) as exc:
            _require_direct_is_warp_task("_Frontend-Test-Stable-Direct-v0")
        self.assertIn("isaaclab_experimental", str(exc.exception))

    def test_rejects_unknown_task(self):
        with self.assertRaises(FrontendIncompatibleError):
            _require_direct_is_warp_task("Frontend-Test-NotRegistered-v0")


# ======================================================================
# Walk helpers
# ======================================================================


class TestWalkHelpers(unittest.TestCase):
    def test_walk_attrs_hit(self):
        root = types.SimpleNamespace(a=types.SimpleNamespace(b=types.SimpleNamespace(c=42)))
        self.assertEqual(_walk_attrs(root, ("a", "b", "c")), 42)

    def test_walk_attrs_miss(self):
        root = types.SimpleNamespace(a=types.SimpleNamespace())
        self.assertIsNone(_walk_attrs(root, ("a", "missing")))
        self.assertIsNone(_walk_attrs(root, ("missing", "b")))

    def test_iter_term_attrs_skips_dunders_and_none(self):
        g = types.SimpleNamespace(t1=_Term(), t2=None, _internal=_Term())
        names = sorted(n for n, _ in _iter_term_attrs(g))
        self.assertEqual(names, ["t1"])


if __name__ == "__main__":
    unittest.main()
