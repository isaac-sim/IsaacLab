# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for composing the tasks of a heterogeneous environment into its configuration."""

from dataclasses import MISSING
from types import SimpleNamespace

import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.managers import EventTermCfg, HeterogeneousTaskManager, SceneEntityCfg, TerminationTermCfg

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


def _env_cfg() -> SimpleNamespace:
    """An environment configuration with a reset event, which the tasks are composed into."""
    events = {"reset": EventTermCfg(func=_holds, mode="reset")}
    scene = SimpleNamespace(clone_cfg=CloneCfg())
    return SimpleNamespace(scene=scene, terminations={}, events=events, episode_length_s=0.0)


def test_compose_tasks_with_dictionary_terminations_and_nested_terms():
    """Dictionary terminations take the tasks' terms, and the entities of nested terms are renamed to the slots."""
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": _task(8.0)})
    assert sorted(cfg.terminations) == ["a_done", "b_done", "time_out"]
    nested = cfg.terminations["a_done"].params["term"].params["term"]
    assert nested.params["asset_cfg"].name == "object_0"
    assert cfg.episode_length_s == 8.0
    # The equal cubes of the tasks share one candidate.
    assert len(cfg.scene.object_0.spawn.assets_cfg) == 1
    # The tasks are selected before the environment's own reset events write the default states.
    assert list(cfg.events) == ["select_task", "reset"]


def test_compose_tasks_keeps_the_terms_of_the_environment():
    """A termination that the environment defines itself is not added again for each task."""
    cfg = _env_cfg()
    cfg.terminations["done"] = own = TerminationTermCfg(func=_holds)
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": _task(8.0)})
    assert sorted(cfg.terminations) == ["done", "time_out"]
    assert cfg.terminations["done"] is own


def test_compose_tasks_once_into_a_configuration():
    """Composing the same tasks again, as copying a configuration does, changes nothing; other tasks are rejected."""
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": _task(8.0)})
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": _task(8.0)})
    assert list(cfg.events) == ["select_task", "reset"]
    assert sorted(cfg.terminations) == ["a_done", "b_done", "time_out"]
    with pytest.raises(ValueError, match="composed of other tasks"):
        HeterogeneousTaskManager.compose_tasks(cfg, {"b": _task(8.0)})


def test_compose_tasks_creates_missing_events_and_terminations():
    """A configuration without events or terminations gets the tasks' terms; a task without terminations adds none."""
    scene = SimpleNamespace(clone_cfg=CloneCfg())
    cfg = SimpleNamespace(scene=scene, terminations=MISSING, events=None, episode_length_s=0.0)
    bare = SimpleNamespace(scene=SimpleNamespace(cube=_object("cube")), terminations=None, episode_length_s=3.0)
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": bare})
    assert list(cfg.events) == ["select_task"]
    assert sorted(cfg.terminations) == ["a_done", "time_out"]


def test_compose_tasks_renames_default_entities():
    """A term's omitted default scene entity names the task's object, so it is renamed to the object's slot."""
    task = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube")),
        terminations={"done": TerminationTermCfg(func=_lifted)},
        episode_length_s=5.0,
    )
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": task})
    assert cfg.terminations["a_done"].params["term"].params["asset_cfg"].name == "object_0"
    assert _lifted.__defaults__[0].name == "cube"


def test_compose_tasks_rejects_slots_that_mix_kinematic_objects():
    """A slot whose object is kinematic in one task and dynamic in another is rejected: one body cannot be both."""
    table = SimpleNamespace(
        scene=SimpleNamespace(table=_object("table", kinematic=True)), terminations={}, episode_length_s=5.0
    )
    cube = SimpleNamespace(scene=SimpleNamespace(cube=_object("cube")), terminations={}, episode_length_s=5.0)
    with pytest.raises(ValueError, match="kinematic"):
        HeterogeneousTaskManager.compose_tasks(_env_cfg(), {"table": table, "cube": cube})


def test_placeholders_match_kinematic_slots():
    """A task without the kinematic object of a slot gets a kinematic stand-in there; dynamic slots get dynamic ones."""
    shelf = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube"), shelf=_object("shelf", kinematic=True), ball=_object("ball")),
        terminations={},
        episode_length_s=5.0,
    )
    cube = SimpleNamespace(scene=SimpleNamespace(cube=_object("cube")), terminations={}, episode_length_s=5.0)
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"shelf": shelf, "cube": cube})
    stand_ins = [cfg.scene.object_1.spawn.assets_cfg[1], cfg.scene.object_2.spawn.assets_cfg[1]]
    assert [spawn.rigid_props.kinematic_enabled for spawn in stand_ins] == [True, False]


def test_clone_combinations_pin_the_first_candidate_of_every_slot():
    """Clone planning starts every environment with the first task instead of enumerating the candidates."""
    pins = {"object_0": 0, "object_1": 0}
    task = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube"), ball=_object("ball")), terminations={}, episode_length_s=5.0
    )
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": task})
    assert [(combination.assets, combination.variants) for combination in cfg.scene.clone_cfg.clone_combinations] == [
        (list(pins), pins)
    ]
    # The combinations that the scene declares itself keep their assets.
    cfg = _env_cfg()
    cfg.scene.clone_cfg.clone_combinations = [InclusionSet(assets=["left_arm"]), InclusionSet(assets=["right_arm"])]
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": task})
    assert [(combination.assets, combination.variants) for combination in cfg.scene.clone_cfg.clone_combinations] == [
        (["left_arm"], pins),
        (["right_arm"], pins),
    ]
