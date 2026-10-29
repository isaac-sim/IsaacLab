# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for composing the tasks of a heterogeneous environment into its configuration."""

from types import SimpleNamespace

import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import HeterogeneousTaskManager, SceneEntityCfg, TerminationTermCfg

pytestmark = pytest.mark.unit


def _holds(env, **params):
    raise NotImplementedError


def _lifted(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")):
    raise NotImplementedError


def _object(name: str, kinematic: bool = False) -> RigidObjectCfg:
    rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=sim_utils.SphereCfg(radius=0.05, rigid_props=rigid_props)
    )


def _task(episode_length_s: float) -> SimpleNamespace:
    """A task with a cube and a termination given as a dictionary, whose term nests a term on the cube."""
    cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/cube", spawn=sim_utils.SphereCfg(radius=0.05))
    inner = TerminationTermCfg(func=_holds, params={"asset_cfg": SceneEntityCfg("cube")})
    terminations = {"done": TerminationTermCfg(func=_holds, params={"term": inner})}
    return SimpleNamespace(
        scene=SimpleNamespace(cube=cube), terminations=terminations, episode_length_s=episode_length_s
    )


def test_compose_tasks_with_dictionary_terminations_and_nested_terms():
    """Dictionary terminations take the tasks' terms, and the entities of nested terms are renamed to the slots."""
    cfg = SimpleNamespace(
        scene=SimpleNamespace(), terminations={}, tasks={"a": _task(5.0), "b": _task(8.0)}, episode_length_s=0.0
    )
    HeterogeneousTaskManager.compose_tasks(cfg)
    assert sorted(cfg.terminations) == ["a_done", "b_done", "time_out"]
    nested = cfg.terminations["a_done"].params["term"].params["term"]
    assert nested.params["asset_cfg"].name == "object_0"
    assert cfg.episode_length_s == 8.0


def test_compose_tasks_renames_default_entities():
    """A term's omitted default scene entity names the task's object, so it is renamed to the object's slot."""
    task = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube")),
        terminations={"done": TerminationTermCfg(func=_lifted)},
        episode_length_s=5.0,
    )
    cfg = SimpleNamespace(scene=SimpleNamespace(), terminations={}, tasks={"a": task}, episode_length_s=0.0)
    HeterogeneousTaskManager.compose_tasks(cfg)
    assert cfg.terminations["a_done"].params["term"].params["asset_cfg"].name == "object_0"
    assert _lifted.__defaults__[0].name == "cube"


def test_placeholders_match_kinematic_slots():
    """A task without the kinematic object of a slot gets a kinematic stand-in there; dynamic slots get dynamic ones."""
    shelf = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube"), shelf=_object("shelf", kinematic=True), ball=_object("ball")),
        terminations={},
        episode_length_s=5.0,
    )
    cube = SimpleNamespace(scene=SimpleNamespace(cube=_object("cube")), terminations={}, episode_length_s=5.0)
    cfg = SimpleNamespace(
        scene=SimpleNamespace(), terminations={}, tasks={"shelf": shelf, "cube": cube}, episode_length_s=0.0
    )
    HeterogeneousTaskManager.compose_tasks(cfg)
    stand_ins = [cfg.scene.object_1.spawn.assets_cfg[1], cfg.scene.object_2.spawn.assets_cfg[1]]
    assert [spawn.rigid_props.kinematic_enabled for spawn in stand_ins] == [True, False]
