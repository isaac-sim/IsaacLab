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
from isaaclab.cloner import CloneCfg, InclusionSet, make_valid_clone_combinations
from isaaclab.managers import (
    EventTermCfg,
    HeterogeneousTaskManager,
    ManagerTermBase,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.managers.heterogeneous_task_manager import task_term

pytestmark = pytest.mark.unit


def _holds(env, **params):
    raise NotImplementedError


def _lifted(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")):
    raise NotImplementedError


class _Stateful(ManagerTermBase):
    """A class-based term that records the environments it is reset for."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self.reset_ids = None

    def reset(self, env_ids=None):
        self.reset_ids = env_ids

    def __call__(self, env):
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
    # The task's own term keeps neither the populated default nor the renamed entity.
    assert task.terminations["done"].params == {}


def test_compose_tasks_renames_entities_in_lists_and_of_string_functions():
    """Scene entities listed in a parameter are renamed, as is the default of a function given as a string."""
    terminations = {
        "listed": TerminationTermCfg(func=_holds, params={"objects": [SceneEntityCfg("cube"), SceneEntityCfg("ball")]}),
        "named": TerminationTermCfg(func=f"{__name__}:_lifted"),
    }
    task = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube"), ball=_object("ball")),
        terminations=terminations,
        episode_length_s=5.0,
    )
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": task})
    listed = cfg.terminations["a_listed"].params["term"].params["objects"]
    assert [entity.name for entity in listed] == ["object_0", "object_1"]
    assert cfg.terminations["a_named"].params["term"].params["asset_cfg"].name == "object_0"


def test_task_terms_reset_their_nested_class_terms():
    """Resetting a task's term resets its class-based nested term, which the termination manager does not see."""
    env = SimpleNamespace(num_envs=2, device="cpu")
    nested = TerminationTermCfg(func=_Stateful)
    # The termination manager instantiates class-based terms, nested ones included, when the simulation plays.
    nested.func = _Stateful(nested, env)
    term = task_term(TerminationTermCfg(func=task_term, params={"task": 0, "term": nested}), env)
    term.reset([1])
    # The term configuration copies its parameters, so the nested term is read back from the term.
    assert term.cfg.params["term"].func.reset_ids == [1]


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


def test_task_inclusion_starts_with_one_world_and_retains_shared_choices():
    """Distinct candidates stay available, but initial layouts do not multiply task slots."""
    first = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("cube"), ball=_object("ball")), terminations={}, episode_length_s=5.0
    )
    second = SimpleNamespace(
        scene=SimpleNamespace(cube=_object("other_cube"), ball=_object("other_ball")),
        terminations={},
        episode_length_s=5.0,
    )
    second.scene.cube.spawn.radius = 0.08
    second.scene.ball.spawn.radius = 0.09
    cfg = _env_cfg()
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": first, "b": second})
    counts = [len(cfg.scene.object_0.spawn.assets_cfg), len(cfg.scene.object_1.spawn.assets_cfg)]
    assert counts == [2, 2]
    worlds, weights = make_valid_clone_combinations(
        ["object_0", "object_1"], counts, cfg.scene.clone_cfg.clone_combinations
    )
    assert worlds == ((0, 2),)
    assert weights.tolist() == [1.0]
    # Independent robot variants remain legal without multiplying the task object candidates.
    worlds, weights = make_valid_clone_combinations(
        ["robot", "object_0", "object_1"], [2, *counts], cfg.scene.clone_cfg.clone_combinations
    )
    assert worlds == ((0, 2, 4), (1, 2, 4))
    assert weights.tolist() == [0.5, 0.5]

    # A caller's explicit world definitions are authoritative rather than rewritten by composition.
    cfg = _env_cfg()
    selection = InclusionSet(assets=["robot", "object_0", "object_1"], layouts=[(1, 0, 0)])
    cfg.scene.clone_cfg.clone_combinations = [selection]
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": first, "b": second})
    assert cfg.scene.clone_cfg.clone_combinations == [selection]
    worlds, _ = make_valid_clone_combinations(
        ["robot", "object_0", "object_1"], [2, *counts], cfg.scene.clone_cfg.clone_combinations
    )
    assert worlds == ((1, 2, 4),)
