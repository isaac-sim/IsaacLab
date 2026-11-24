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
from isaaclab.cloner import CloneCfg, InclusionSet, make_clone_plan, make_valid_clone_combinations
from isaaclab.managers import (
    EventTermCfg,
    HeterogeneousTaskCfg,
    HeterogeneousTaskManager,
    ManagerTermBase,
    SceneEntityCfg,
    TerminationTermCfg,
    heterogeneous_task_manager,
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


def _task(episode_length_s: float) -> HeterogeneousTaskCfg:
    """A declarative task with a nested termination on the shared scene's cube."""
    inner = TerminationTermCfg(func=_holds, params={"asset_cfg": SceneEntityCfg("cube")})
    return HeterogeneousTaskCfg(
        inclusion=InclusionSet(assets=["cube"]),
        episode_length_s=episode_length_s,
        terminations={"done": TerminationTermCfg(func=_holds, params={"term": inner})},
    )


def _env_cfg() -> SimpleNamespace:
    """An environment configuration with a reset event, which the tasks are composed into."""
    events = {"reset": EventTermCfg(func=_holds, mode="reset")}
    scene = SimpleNamespace(clone_cfg=CloneCfg(), cube=_object("cube"), ball=_object("ball"))
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
    scene = SimpleNamespace(clone_cfg=CloneCfg(), cube=_object("cube"), ball=_object("ball"))
    cfg = SimpleNamespace(scene=scene, terminations=MISSING, events=None, episode_length_s=0.0)
    bare = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["cube"]), terminations={}, episode_length_s=3.0)
    HeterogeneousTaskManager.compose_tasks(cfg, {"a": _task(5.0), "b": bare})
    assert list(cfg.events) == ["select_task"]
    assert sorted(cfg.terminations) == ["a_done", "time_out"]


def test_compose_tasks_renames_default_entities():
    """A term's omitted default scene entity names the task's object, so it is renamed to the object's slot."""
    task = HeterogeneousTaskCfg(
        inclusion=InclusionSet(assets=["cube"]),
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
    task = HeterogeneousTaskCfg(
        inclusion=InclusionSet(assets=["cube", "ball"]),
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
    cfg = _env_cfg()
    cfg.scene.table = _object("table", kinematic=True)
    table = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["table"]), episode_length_s=5.0)
    with pytest.raises(ValueError, match="kinematic"):
        HeterogeneousTaskManager.compose_tasks(cfg, {"table": table, "cube": _task(5.0)})


def test_placeholders_match_kinematic_slots():
    cfg = _env_cfg()
    cfg.scene.shelf = _object("shelf", kinematic=True)
    shelf = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["cube", "shelf", "ball"]), episode_length_s=5.0)
    HeterogeneousTaskManager.compose_tasks(cfg, {"shelf": shelf, "cube": _task(5.0)})
    stand_ins = [cfg.scene.object_1.spawn.assets_cfg[1], cfg.scene.object_2.spawn.assets_cfg[1]]
    assert [spawn.rigid_props.kinematic_enabled for spawn in stand_ins] == [True, False]


def test_task_inclusion_starts_with_one_world_and_retains_shared_choices():
    """A flat catalog and inclusion records own task composition; runtime slots are a derived projection."""
    assert not any(
        isinstance(value, type) and value is not InclusionSet and issubclass(value, InclusionSet)
        for value in vars(heterogeneous_task_manager).values()
    ), "Task composition uses declarative inclusions rather than an InclusionSet subclass"
    cfg = _env_cfg()
    cube = cfg.scene.cube
    cube.spawn.semantic_tags = [("class", "gear")]
    cfg.scene.other_cube = _object("other_cube")
    cfg.scene.other_cube.spawn.radius = 0.08
    cfg.scene.other_ball = _object("other_ball")
    cfg.scene.other_ball.spawn.radius = 0.09
    cfg.scene.third_ball = _object("third_ball")
    cfg.scene.third_ball.spawn.radius = 0.11
    first = HeterogeneousTaskCfg(
        inclusion=InclusionSet(assets=["cube", "ball"]),
        episode_length_s=5.0,
        init_states={"cube": RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.2, 0.3))},
    )
    second = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["other_cube", "other_ball"]), episode_length_s=5.0)
    third = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["cube", "third_ball"]), episode_length_s=5.0)
    tasks = {"a": first, "b": second, "c": third}
    HeterogeneousTaskManager.compose_tasks(cfg, tasks)
    assert cfg.events["select_task"].params["tasks"] is tasks
    assert not hasattr(first, "scene"), "Task records must not embed environment or scene configurations"
    assert cfg.scene.cube is None, "Source assets have one owner in the reset term's catalog"
    assert list(cfg.events["select_task"].params["objects"]) == [
        "cube",
        "ball",
        "other_cube",
        "other_ball",
        "third_ball",
    ]
    assert cfg.scene.object_0.init_state.pos == (0.1, 0.2, 0.3)
    assert cube.init_state.pos == (0.0, 0.0, 0.0)
    candidates = [cfg.scene.object_0.spawn.assets_cfg, cfg.scene.object_1.spawn.assets_cfg]
    assert [len(group) for group in candidates] == [2, 3]
    assert candidates[0][0].semantic_tags == [("class", "gear"), ("task", "a"), ("task", "c"), ("initial_task", "a")]
    assert cube.spawn.semantic_tags == [("class", "gear")]
    groups, weights = make_valid_clone_combinations(
        ["object_0", "object_1"], candidates, cfg.scene.clone_cfg.clone_combinations
    )
    assert groups == (((0,), (2,)),)
    assert weights.tolist() == [1.0]
    plan = make_clone_plan([*candidates[0], *candidates[1]], groups, 128, weights=weights)
    assert plan.topology.world_prototype_starts.tolist() == [0, 0, 2]
    assert plan.topology.world_prototypes.tolist() == [0, 2]
    assert plan.topology.world_prototype_layout.tolist() == [0] * 128
    robots = [sim_utils.SphereCfg(radius=0.2), sim_utils.SphereCfg(radius=0.3)]
    groups, weights = make_valid_clone_combinations(
        ["robot", "object_0", "object_1"], [robots, *candidates], cfg.scene.clone_cfg.clone_combinations
    )
    assert groups == (((0, 1), (2,), (4,)),)
    plan = make_clone_plan([*robots, *candidates[0], *candidates[1]], groups, 4, weights=weights)
    assert plan.topology.world_prototypes.tolist() == [0, 2, 4, 1, 2, 4]
    assert plan.topology.world_prototype_layout.tolist() == [0, 0, 1, 1]
    HeterogeneousTaskManager.compose_tasks(cfg, tasks)
    assert cfg.scene.object_0.spawn.assets_cfg[0].semantic_tags == candidates[0][0].semantic_tags
    assert len(cfg.scene.object_0.spawn.assets_cfg) == 2
    # Caller-owned rules can describe all legal families without embedding full task environments.
    selection = InclusionSet(assets=["object_0", "object_1"], match_tags=["task"])
    cfg.scene.clone_cfg.clone_combinations = [selection]
    HeterogeneousTaskManager.compose_tasks(cfg, tasks)
    assert cfg.scene.clone_cfg.clone_combinations[0] is selection
    groups, _ = make_valid_clone_combinations(
        ["object_0", "object_1"],
        [cfg.scene.object_0.spawn.assets_cfg, cfg.scene.object_1.spawn.assets_cfg],
        cfg.scene.clone_cfg.clone_combinations,
    )
    assert groups == (((0,), (2,)), ((0,), (4,)), ((1,), (3,)))


@pytest.mark.parametrize("invalid", ["unknown", "not-rigid", "incompatible", "inactive-pose"])
def test_flat_task_rejects_invalid_asset_bindings(invalid):
    cfg = _env_cfg()
    task = HeterogeneousTaskCfg(inclusion=InclusionSet(assets=["cube", "ball"]), episode_length_s=5.0)
    if invalid == "unknown":
        task.inclusion.assets.append("missing")
        message = "Unknown task asset"
    elif invalid == "not-rigid":
        cfg.scene.cube = object()
        message = "must be a RigidObjectCfg"
    elif invalid == "incompatible":
        cfg.scene.cube.spawn.semantic_tags = [("size", "small")]
        cfg.scene.ball.spawn.semantic_tags = [("size", "large")]
        task.inclusion.match_tags = ["size"]
        message = "at least one compatible world"
    else:
        task.init_states = {"missing": RigidObjectCfg.InitialStateCfg()}
        message = "inactive task assets"
    with pytest.raises(ValueError, match=message):
        HeterogeneousTaskManager.compose_tasks(cfg, {"task": task})
