# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task manager for environments that each run one of several tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner import InclusionSet
from isaaclab.utils import dict_to_md5_hash, string_to_callable

from .manager_base import ManagerTermBase, _populate_term_defaults
from .manager_term_cfg import EventTermCfg, ManagerTermBaseCfg, TerminationTermCfg
from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class HeterogeneousTaskManager:
    """Manager for environments that each run one of several tasks.

    The tasks are complete environment configurations, of which their rigid objects, termination terms and
    episode lengths are used; :meth:`compose_tasks` adds them to an environment configuration, with the reset
    event that creates the manager as ``env.task_manager``. The other managers and scene entities are shared by
    all tasks. Each environment runs the task set with :meth:`set_task` from its next reset on: the reset selects
    the task's objects in the slots, where the task changed, and makes their task poses the default root poses,
    which the environment's reset events write.

    The objects are selected as reset-time mesh variants, which need the Newton backend with the XPBD solver, or
    with the MJWarp solver and ``use_mujoco_contacts=False``.

    .. code-block:: python

        @configclass
        class MultiTaskEnvCfg(ManagerBasedRLEnvCfg):
            def __post_init__(self):
                HeterogeneousTaskManager.compose_tasks(self, {"lift": LiftEnvCfg(), "stack": StackEnvCfg()})


        env = ManagerBasedRLEnv(MultiTaskEnvCfg())
        # the first two environments run the stack task from their next reset on
        env.task_manager.set_task([0, 1], [1, 1])
    """

    def __init__(self, cfg: dict[str, ManagerBasedRLEnvCfg], env: ManagerBasedRLEnv):
        """Initialize the task manager.

        Args:
            cfg: The task configurations, keyed by task name.
            env: The environment instance.
        """
        self._env = env
        self._tasks = cfg
        self.task_names = list(cfg)
        """Task names, in task index order."""
        slots = _slot_objects(cfg)
        self._slots = [env.scene[f"object_{slot}"] for slot in range(len(slots))]
        poses = [[(*obj.init_state.pos, *obj.init_state.rot) for obj in objects] for objects in slots]
        self._poses = torch.tensor(poses, device=env.device)
        variants = [_variants(objects)[1] for objects in slots]
        self._variants = torch.tensor(variants, dtype=torch.long, device=env.device)
        self.max_episode_length = torch.tensor(
            [math.ceil(task.episode_length_s / env.step_dt) for task in cfg.values()], device=env.device
        )
        """Episode length of each task in environment steps."""
        self.task_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        """Index of the task each environment runs."""
        self._next_task_ids = self.task_ids.clone()
        # Task whose objects the slots of each environment hold; none before the first reset.
        self._slot_task_ids = torch.full_like(self.task_ids, -1)

    @staticmethod
    def compose_tasks(cfg: ManagerBasedRLEnvCfg, tasks: dict[str, ManagerBasedRLEnvCfg]) -> None:
        """Add the objects and terminations of ``tasks`` to ``cfg``, and the reset event that selects the tasks.

        The k-th rigid objects of the tasks are the candidates of the scene entity ``object_<k>``, spawned as the
        reset-time mesh variants of a :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg`; objects with
        equal spawn configurations share a candidate. Each task's terminations, other than time-outs and terms that
        ``cfg`` defines itself, are added as ``<task>_<term>`` with their scene entities, the omitted defaults
        included, renamed to the slots, and hold only in the environments running the task. A term that ``cfg``
        defines itself can evaluate the tasks' terms of its name together (see :meth:`task_terms`). The ``time_out``
        term ends episodes at the length of their task. The reset event ``select_task`` comes before the other
        events of ``cfg``, which then write the default states of the selected tasks. Every environment starts
        with the objects of the first task: the clone combinations of the scene pin the first candidate of every
        slot, so that clone planning does not enumerate the combinations of all candidates.

        Call it once, before the environment is created, for example in the ``__post_init__`` of its
        configuration. Where the scene, events or terminations of ``cfg`` or of the tasks are presets, call it after
        they are resolved instead.

        Args:
            cfg: The environment configuration, modified in place.
            tasks: The task configurations, keyed by task name.

        Raises:
            ValueError: If ``cfg`` is composed of other tasks already.
        """
        if cfg.events is None:
            cfg.events = {}
        if cfg.terminations is None or isinstance(cfg.terminations, type(MISSING)):
            cfg.terminations = {}
        events = _terms(cfg.events)
        # Copying a configuration composes its tasks again; other tasks would leave slots and terms of the first.
        previous = events.pop("select_task", None)
        if previous is not None and list(previous.params.get("tasks", ())) != list(tasks):
            raise ValueError("The configuration is composed of other tasks already: compose its tasks once.")
        slots = _slot_objects(tasks)
        for slot, objects in enumerate(slots):
            spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=_variants(objects)[0], static=False, random_choice=False)
            obj = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/object_{slot}", spawn=spawn, init_state=objects[0].init_state
            )
            setattr(cfg.scene, f"object_{slot}", obj)
        # Every environment starts with the objects of the first task, the first candidates of the slots.
        pins = {f"object_{slot}": 0 for slot in range(len(slots))}
        combinations = cfg.scene.clone_cfg.clone_combinations
        if not combinations:
            combinations.append(InclusionSet(assets=list(pins)))
        for combination in combinations:
            combination.variants.update(pins)
        terms = _terms(cfg.terminations)
        for index, (task_name, task) in enumerate(tasks.items()):
            for name, term in _terms(task.terminations).items():
                if term is None or term.time_out or name in terms:
                    continue
                term = TerminationTermCfg(func=task_term, params={"task": index, "term": _slot_term(task, term)})
                terms[f"{task_name}_{name}"] = term
        terms["time_out"] = TerminationTermCfg(func=task_time_out, time_out=True)
        cfg.episode_length_s = max(task.episode_length_s for task in tasks.values())
        # The tasks are selected before the other reset events write the default states.
        select = EventTermCfg(func=select_task, mode="reset", params={"tasks": tasks})
        others = dict(events)
        events.clear()
        events.update({"select_task": select, **others})

    @staticmethod
    def placeholder(slot: int, kinematic: bool = False) -> RigidObjectCfg:
        """Return a hidden stand-in for an object that a task does not have in a slot, parked away from the robot.

        Tasks with fewer rigid objects than others have it in their last slots, kinematic where the slot's objects
        are; a task can leave a slot before its last object empty by declaring it there.

        Args:
            slot: Index of the slot, which places the stand-ins of different slots apart.
            kinematic: Whether the stand-in is kinematic, as the other candidates of its slot must match.
        """
        spawn = sim_utils.SphereCfg(
            radius=0.01,
            visible=False,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic),
            mass_props=sim_utils.MassCfg(mass=0.01),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )
        init_state = RigidObjectCfg.InitialStateCfg(pos=(0.1 * slot, -50.0, 0.5))
        return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/empty_{slot}", spawn=spawn, init_state=init_state)

    def candidate_ids(self, name: str) -> torch.Tensor:
        """Return the candidate of the slot ``name`` that each environment uses.

        Args:
            name: Scene entity name of the slot, ``object_<k>``.

        Returns:
            Index into the slot's :attr:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg.assets_cfg` per
            environment, shape (num_envs,).
        """
        return self._variants[int(name.removeprefix("object_")), self.task_ids]

    def task_terms(self, name: str) -> dict[int, TerminationTermCfg]:
        """Return the termination term ``name`` of the tasks that have one, with their scene entities renamed to the
        slots.

        Args:
            name: Term name in the tasks' termination configurations.

        Returns:
            The terms keyed by task index.
        """
        terms = {}
        for index, task in enumerate(self._tasks.values()):
            term = _terms(task.terminations).get(name)
            if term is not None:
                terms[index] = _slot_term(task, term)
        return terms

    def set_task(self, env_ids: Sequence[int] | torch.Tensor, task_ids: Sequence[int] | torch.Tensor) -> None:
        """Set the tasks of environments, which take effect when the environments are next reset.

        Args:
            env_ids: Environment indices.
            task_ids: Task index of each environment.
        """
        device = self._env.device
        self._next_task_ids[torch.as_tensor(env_ids, device=device)] = torch.as_tensor(task_ids, device=device)

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Switch the slots of environments to the objects of their tasks, where the task changed, and to their poses.

        Args:
            env_ids: Environment indices.
        """
        self.task_ids[env_ids] = self._next_task_ids[env_ids]
        tasks = self.task_ids[env_ids]
        changed = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        changed = changed[self._slot_task_ids[changed] != self.task_ids[changed]]
        if len(changed):
            for slot, obj in enumerate(self._slots):
                # A slot whose tasks all have the same object has no variant to select.
                if obj.num_mesh_variants > 1:
                    variant_ids = self._variants[slot, self.task_ids[changed]].int()
                    obj.write_mesh_variant_to_sim(variant_ids=variant_ids, env_ids=changed)
            self._slot_task_ids[changed] = self.task_ids[changed]
        for slot, obj in enumerate(self._slots):
            obj.data.default_root_pose.torch[env_ids] = self._poses[slot, tasks]


class select_task(ManagerTermBase):
    """Reset event that creates the task manager, ``env.task_manager``, and switches the environments to their tasks.

    :meth:`HeterogeneousTaskManager.compose_tasks` adds it to the environment configuration.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        env.task_manager = HeterogeneousTaskManager(cfg.params["tasks"], env)

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, tasks: dict[str, ManagerBasedRLEnvCfg]
    ) -> None:
        env.task_manager.reset(env_ids)


class task_term(ManagerTermBase):
    """Termination term of one task, which holds only in the environments running the task."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        term = self.cfg.params["term"]
        if isinstance(term.func, ManagerTermBase):
            term.func.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, task: int, term: TerminationTermCfg) -> torch.Tensor:
        return term.func(env, **term.params) & (env.task_manager.task_ids == task)


def task_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episodes that reached the episode length of their task."""
    manager = env.task_manager
    return env.episode_length_buf >= manager.max_episode_length[manager.task_ids]


def _task_objects(task: ManagerBasedRLEnvCfg) -> list[tuple[str, RigidObjectCfg]]:
    """Return the named rigid objects of a task's scene in declaration order."""
    return [(name, value) for name, value in task.scene.__dict__.items() if isinstance(value, RigidObjectCfg)]


def _slot_term(task: ManagerBasedRLEnvCfg, term: TerminationTermCfg) -> TerminationTermCfg:
    """Return a copy of a task's term with its scene entities renamed to the slots of the task's objects."""
    term = term.copy()
    _rename_entities(term, {name: f"object_{slot}" for slot, (name, _) in enumerate(_task_objects(task))})
    return term


def _slot_objects(tasks: dict[str, ManagerBasedRLEnvCfg]) -> list[list[RigidObjectCfg]]:
    """Return the object of every task in every slot: its k-th rigid object, or a placeholder if it has fewer."""
    objects = [[obj for _, obj in _task_objects(task)] for task in tasks.values()]
    slots = []
    for slot in range(max(map(len, objects))):
        kinematic = any(_kinematic(task[slot]) for task in objects if slot < len(task))
        placeholder = HeterogeneousTaskManager.placeholder(slot, kinematic)
        slots.append([task[slot] if slot < len(task) else placeholder for task in objects])
    return slots


def _kinematic(obj: RigidObjectCfg) -> bool:
    """Return whether an object's spawn configuration makes it kinematic."""
    rigid_props = getattr(obj.spawn, "rigid_props", None)
    return rigid_props is not None and bool(rigid_props.kinematic_enabled)


def _variants(objects: list[RigidObjectCfg]) -> tuple[list[sim_utils.SpawnerCfg], list[int]]:
    """Return the distinct spawn configurations of a slot's objects and the index of each object's among them."""
    keys = [dict_to_md5_hash(obj.spawn) for obj in objects]
    variants = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    return [objects[keys.index(key)].spawn for key in variants], [variants[key] for key in keys]


def _terms(cfg: object | dict) -> dict[str, Any]:
    """Return the terms of a manager configuration, a configuration class or a dictionary, as a writable mapping."""
    return cfg if isinstance(cfg, dict) else vars(cfg)


def _rename_entities(value: Any, names: dict[str, str]) -> None:
    """Rename the scene entities referenced in a term or a term parameter, in place, also in nested terms.

    A term first takes the defaults of the parameters it omits, which the managers would add after the renaming.
    """
    if isinstance(value, SceneEntityCfg):
        value.name = names.get(value.name, value.name)
    elif isinstance(value, ManagerTermBaseCfg):
        _populate_term_defaults(
            string_to_callable(value.func) if isinstance(value.func, str) else value.func, value.params
        )
        _rename_entities(value.params, names)
    elif isinstance(value, dict):
        for item in value.values():
            _rename_entities(item, names)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _rename_entities(item, names)
