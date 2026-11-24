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
from isaaclab.cloner import InclusionSet, make_valid_clone_combinations
from isaaclab.utils import configclass, dict_to_md5_hash, string_to_callable

from .manager_base import ManagerTermBase, _populate_term_defaults
from .manager_term_cfg import EventTermCfg, ManagerTermBaseCfg, TerminationTermCfg
from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


@configclass
class HeterogeneousTaskCfg:
    """One task over named assets in a shared flat scene, without a separate environment configuration."""

    inclusion: InclusionSet = MISSING
    """Named scene assets active in this task, in slot order."""

    episode_length_s: float = MISSING
    """Episode duration for this task [s]."""

    instruction: str = ""
    """Language instruction for this task."""

    init_states: dict[str, RigidObjectCfg.InitialStateCfg] = {}
    """Per-asset initial states; omitted entries use the flat scene asset's initial state."""

    terminations: dict[str, TerminationTermCfg] = {}
    """Task-specific termination terms referencing the original scene asset names."""


class HeterogeneousTaskManager(ManagerTermBase):
    """Run declarative tasks over a shared flat scene, switching on reset.

    Declare rigid objects once in the scene. Each :class:`HeterogeneousTaskCfg` selects named assets
    with an :class:`~isaaclab.cloner.InclusionSet`, supplies their task poses, instruction, episode
    duration and termination terms. :meth:`compose_tasks` compiles that catalog into fixed runtime
    slots, retaining the source catalog once in the reset term's configuration. No environment
    instance or environment configuration is needed per task.

    Reset-time mesh switching requires Newton with XPBD or MJWarp with external contacts. Only the
    Newton Warp camera renderer follows switched variants. Every environment starts with the first
    task; :meth:`set_task` schedules a different task for its next reset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Initialize the task manager, the reset event ``select_task`` of the environment.

        Args:
            cfg: The event configuration, whose ``tasks`` parameter holds the task configurations by task name.
            env: The environment instance, which gets the manager as ``env.task_manager``.
        """
        super().__init__(cfg, env)
        self._tasks: dict[str, HeterogeneousTaskCfg] = cfg.params["tasks"]
        self.task_names = list(self._tasks)
        """Task names, in task index order."""
        slots = _slot_objects(self._tasks, cfg.params["objects"])
        self._slots = [env.scene[f"object_{slot}"] for slot in range(len(slots))]
        poses = [[(*obj.init_state.pos, *obj.init_state.rot) for obj in objects] for objects in slots]
        self._poses = torch.tensor(poses, device=env.device)
        variants = [_variants(objects)[1] for objects in slots]
        self._variants = torch.tensor(variants, dtype=torch.long, device=env.device)
        self.max_episode_length = torch.tensor(
            [math.ceil(task.episode_length_s / env.step_dt) for task in self._tasks.values()], device=env.device
        )
        """Episode length of each task in environment steps."""
        self.task_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        """Index of the task each environment runs."""
        self._next_task_ids = self.task_ids.clone()
        # Track the loaded task; the initial-task tag selects the first task's variants.
        self._slot_task_ids = self.task_ids.clone()
        env.task_manager = self

    @staticmethod
    def compose_tasks(cfg: ManagerBasedRLEnvCfg, tasks: dict[str, HeterogeneousTaskCfg]) -> None:
        """Compile task inclusions over a flat scene into reset-time variant slots.

        Source assets are moved from the scene into the reset term's ``objects`` catalog. Task
        poses stay in their task records; equal spawn configurations share a slot candidate.
        Existing shared scene entities and manager terms remain independent. Termination entity
        references are resolved to runtime slots on copies, leaving task definitions unchanged.

        Candidate copies carry their task memberships as semantic tags. The default inclusion
        matches ``initial_task`` to retain one initial world. Caller-supplied clone inclusions stay
        authoritative and must agree with first-task initialization. Compose before creating the
        environment, after resolving presets. Repeating composition for the same task names is safe.

        Args:
            cfg: Shared environment configuration, whose scene declares the source rigid objects.
            tasks: Declarative task records keyed by task name.

        Raises:
            ValueError: Task names change after composition, an asset is unknown, semantic matching
                fails, or a runtime slot mixes kinematic and dynamic assets.
        """
        if cfg.events is None:
            cfg.events = {}
        if cfg.terminations is None or isinstance(cfg.terminations, type(MISSING)):
            cfg.terminations = {}
        events = _terms(cfg.events)
        # Copying a configuration composes its tasks again; other tasks would leave slots and terms of the first.
        previous = events.pop("select_task", None)
        if previous is not None and list(previous.params["tasks"]) != list(tasks):
            raise ValueError("The configuration is composed of other tasks already: compose its tasks once.")
        objects = {}
        for name in dict.fromkeys(name for task in tasks.values() for name in task.inclusion.assets):
            if vars(cfg.scene).get(name) is not None:
                objects[name] = getattr(cfg.scene, name)
            elif previous is not None and name in previous.params["objects"]:
                objects[name] = previous.params["objects"][name]
            else:
                raise ValueError(f"Unknown task asset {name!r}: declare it in the shared scene.")
            if not isinstance(objects[name], RigidObjectCfg):
                raise ValueError(f"Task asset {name!r} must be a RigidObjectCfg.")
        for task in tasks.values():
            if unknown := task.init_states.keys() - set(task.inclusion.assets):
                raise ValueError(f"Initial states refer to inactive task assets: {sorted(unknown)}.")
            if task.inclusion.match_tags:
                names = list(dict.fromkeys(task.inclusion.assets))
                make_valid_clone_combinations(names, [[objects[name].spawn] for name in names], [task.inclusion])
        slots = _slot_objects(tasks, objects)
        for name in objects:
            setattr(cfg.scene, name, None)
        for slot, slot_objects in enumerate(slots):
            candidates, task_variants = _variants(slot_objects)
            candidates = [candidate.copy() for candidate in candidates]
            for task_name, variant in zip(tasks, task_variants, strict=True):
                candidate = candidates[variant]
                candidate.semantic_tags = [*(candidate.semantic_tags or ()), ("task", task_name)]
            initial = candidates[task_variants[0]]
            initial.semantic_tags.append(("initial_task", next(iter(tasks))))
            spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=candidates, static=False, random_choice=False)
            obj = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/object_{slot}", spawn=spawn, init_state=slot_objects[0].init_state
            )
            setattr(cfg.scene, f"object_{slot}", obj)
        # Default layouts start with the first task; caller-supplied inclusion definitions stay authoritative.
        if not cfg.scene.clone_cfg.clone_combinations:
            cfg.scene.clone_cfg.clone_combinations = [
                InclusionSet(assets=[f"object_{slot}" for slot in range(len(slots))], match_tags=["initial_task"])
            ]
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
        select = EventTermCfg(func=HeterogeneousTaskManager, mode="reset")
        select.params = {"tasks": tasks, "objects": objects}
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
        """Return the termination term ``name`` of each task that has one, with its scene entities renamed to the slots.

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
        task_ids = torch.as_tensor(task_ids, device=device, dtype=torch.long)
        self._next_task_ids[torch.as_tensor(env_ids, device=device)] = task_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        tasks: dict[str, HeterogeneousTaskCfg],
        objects: dict[str, RigidObjectCfg],
    ) -> None:
        """Switch the slots of environments to the objects of their tasks, where the task changed, and to their poses.

        The environment applies it as the reset event ``select_task``, before its other reset events.

        Args:
            env: The environment instance.
            env_ids: Environment indices.
            tasks: Declarative task records retained by the reset term.
            objects: Shared source asset catalog, consumed during initialization.
        """
        self.task_ids[env_ids] = self._next_task_ids[env_ids]
        tasks = self.task_ids[env_ids]
        changed = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        changed = changed[self._slot_task_ids[changed] != self.task_ids[changed]]
        if len(changed):
            for slot, obj in enumerate(self._slots):
                # A slot whose tasks all have the same object has no variant to select.
                if obj.num_mesh_variants > 1:
                    variant_ids = self._variants[slot, self.task_ids[changed]]
                    obj.write_mesh_variant_to_sim(variant_ids=variant_ids, env_ids=changed)
            self._slot_task_ids[changed] = self.task_ids[changed]
        for slot, obj in enumerate(self._slots):
            obj.data.default_root_pose.torch[env_ids] = self._poses[slot, tasks]


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


def _slot_term(task: HeterogeneousTaskCfg, term: TerminationTermCfg) -> TerminationTermCfg:
    """Return a copy of a task's term with its scene entities renamed to the slots of the task's objects."""
    term = term.copy()
    _rename_entities(term, {name: f"object_{slot}" for slot, name in enumerate(task.inclusion.assets)})
    return term


def _slot_objects(
    tasks: dict[str, HeterogeneousTaskCfg], catalog: dict[str, RigidObjectCfg]
) -> list[list[RigidObjectCfg]]:
    """Return the object of every task in every slot: its k-th rigid object, or a placeholder if it has fewer.

    The objects of a slot must agree on being kinematic, as the physics gives all candidates the body of the first.
    """
    objects = [
        [
            catalog[name].replace(init_state=task.init_states.get(name, catalog[name].init_state))
            for name in task.inclusion.assets
        ]
        for task in tasks.values()
    ]
    slots = []
    for slot in range(max(map(len, objects))):
        kinematic = {_kinematic(task[slot]) for task in objects if slot < len(task)}
        if len(kinematic) > 1:
            raise ValueError(
                f"The objects of slot {slot} mix kinematic and dynamic bodies, but the candidates of a slot share the"
                " body of its first candidate: declare kinematic objects in the same slots in every task."
            )
        placeholder = HeterogeneousTaskManager.placeholder(slot, kinematic.pop())
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


def _terms(cfg: object | dict | None) -> dict[str, Any]:
    """Return the terms of a manager configuration, a configuration class, a dictionary or ``None``, as a mapping.

    The mapping is the configuration's own, writable, unless the configuration is ``None`` or missing.
    """
    if cfg is None or isinstance(cfg, type(MISSING)):
        return {}
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
