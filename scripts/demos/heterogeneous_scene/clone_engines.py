# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The two clone engines that turn harvested tasks into a running heterogeneous scene.

A :class:`CloneEngine` consumes the harvested ``(tasks, prototypes)`` (see
:mod:`registry_harvest`). The base class owns everything that is *the same
regardless of cloning API* -- the run loop, the locomotion/manipulation alternation,
and the per-prototype pose/joint writes -- and leaves just two things to subclasses:

* :meth:`CloneEngine.build` -- spawn + clone the scene and create physics views.
* :meth:`CloneEngine.resolve_envs` -- where a prototype lives: its physics-view indices
  and matching world origins for a set of active envs.

The split exists so :class:`InteractiveSceneEngine` and :class:`ManualCloneEngine` read
top-to-bottom as the **two recipes** for building a heterogeneous cloner:

* :class:`InteractiveSceneEngine` -- hand a declarative
  :class:`~isaaclab.scene.InteractiveSceneCfg` (prototypes + a heterogeneous
  :class:`~isaaclab.cloner.CloneCfg`) to :class:`~isaaclab.scene.InteractiveScene`,
  which lays out envs, clones, wires the selector, and builds physics views for you.
* :class:`ManualCloneEngine` -- do it by hand: ``grid_transforms`` for env origins,
  ``usd_replicate`` to lay out env containers, then a
  :class:`~isaaclab.cloner.ReplicateSession` (``make_clone_plan`` + ``replicate``) that
  spawns each prototype once in ``env_0`` and clones it into the envs that use it.

Both author into the same ``/World/envs/env_*`` containers and publish a single clone
plan, so a scene uses exactly one engine. Import after ``AppLauncher`` has started.
"""

from __future__ import annotations

import re

import torch
import warp as wp
from registry_harvest import Prototype, TaskGroup, is_locomotion, noise_scale

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet, ReplicateSession, make_valid_clone_combinations, sequential
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, SelectorCfg, SelectorTermCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

# Env-namespace prefix the harvested prototype prim paths are bound to. It differs per engine
# (InteractiveScene resolves the ``{ENV_REGEX_NS}`` macro itself; the manual path needs the
# expanded regex), so each engine exposes its own via ``PRIM_PREFIX``.
PRIM_PREFIX_INTERACTIVE_SCENE = "{ENV_REGEX_NS}"
PRIM_PREFIX_MANUAL = "/World/envs/env_.*"


def _is_clone_group(cfg) -> bool:
    """Whether ``make_clone_plan`` treats this cfg as a clone group.

    Mirrors the plan's own filter: an env-scoped cfg that carries a spawner. Assets
    without a spawn (e.g. a :class:`~isaaclab.assets.SurfaceGripperCfg`, whose prims
    live inside their owner's USD) are not replicated as their own group, so they must
    not become columns of the ``valid_set`` handed to :func:`make_valid_clone_combinations`.
    """
    prim_path = getattr(cfg, "prim_path", None)
    return bool(prim_path) and getattr(cfg, "spawn", None) is not None and "/World/envs/" in prim_path


def _global_assets() -> dict[str, AssetBaseCfg]:
    """The two global shared assets (a single ground + dome-light prim, never cloned)."""
    return {
        "ground": AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=GroundPlaneCfg(),
        ),
        "light": AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        ),
    }


class CloneEngine:
    """Build a heterogeneous scene from ``(tasks, prototypes)`` and drive it.

    Subclasses implement only the cloning API -- :meth:`build`, :meth:`resolve_envs`,
    and a few one-line accessors. The run loop, the group alternation, and the
    pose/joint writes live here and are identical for both engines.
    """

    SWITCH_INTERVAL = 60  # steps between locomotion <-> manipulation group switches
    PRIM_PREFIX = ""  # set by each subclass; the caller harvests with it

    def __init__(
        self, sim, simulation_app, tasks, prototypes, num_envs, env_spacing, device, clone_strategy=sequential
    ):
        self.sim = sim
        self.simulation_app = simulation_app
        self.tasks: list[TaskGroup] = tasks
        self.prototypes: list[Prototype] = prototypes
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.device = device
        self.clone_strategy = clone_strategy
        self.origins: torch.Tensor | None = None  # [num_envs, 3] world env origins (set in build)
        self._by_name = {p.name: p for p in prototypes}

    # ----------------------------------------------------------------
    # API-specific hooks -- the only thing the two engines differ in
    # ----------------------------------------------------------------

    def build(self) -> None:
        """Spawn + clone the scene, create physics views, and set :attr:`origins`."""
        raise NotImplementedError

    def resolve_envs(self, proto: Prototype, active: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(view_indices, world_origins)`` for ``proto`` over the ``active`` envs.

        ``view_indices`` index into the prototype's own physics view; ``world_origins`` is
        the matching ``[k, 3]`` slice of env origins. Empty tensors if it is in none of
        ``active``.
        """
        raise NotImplementedError

    def iter_articulations(self):
        """Yield ``(Prototype, articulation)`` for every articulation in the scene."""
        raise NotImplementedError

    def _instance(self, proto: Prototype):
        """Return the live Articulation/RigidObject backing ``proto``."""
        raise NotImplementedError

    def flush(self) -> None:
        """Push buffered writes to the simulation."""
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Refresh asset data after a physics step."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Shared driving -- identical across engines
    # ----------------------------------------------------------------

    def run(self) -> None:
        """Build once, then alternate the locomotion / manipulation groups forever."""
        self.build()
        envs_by_category = self._envs_by_category()
        categories = list(envs_by_category)
        print("[INFO] env categories: " + ", ".join(f"{c}={len(v)} envs" for c, v in envs_by_category.items()))

        dt = self.sim.get_physics_dt()
        self.reset(list(range(self.num_envs)))
        self.flush()
        step, turn, active = 0, 0, envs_by_category[categories[0]]
        while self.simulation_app.is_running():
            if step % self.SWITCH_INTERVAL == 0:
                category = categories[turn % len(categories)]
                turn += 1
                active = envs_by_category[category]
                self.reset(active)
                print(f"[step {step:>5d}] reset group '{category}': {len(active)} envs -> {active}")
            self.apply_actions(active)
            self.flush()
            self.sim.step()
            step += 1
            self.update(dt)

    def reset(self, active: list[int]) -> None:
        """Write each task's intended init pose into its (active) envs.

        Walks tasks then their prototypes, asks the engine where each prototype lives via
        :meth:`resolve_envs`, then writes that task's pose -- so a shared prototype is
        re-posed once per task that uses it.
        """
        n_tasks = len(self.tasks)
        active_set = set(active)
        for task_idx, task in enumerate(self.tasks):
            task_envs = [e for e in active_set if e % n_tasks == task_idx]
            if not task_envs:
                continue
            for name, init_cfg in task.init_cfgs.items():
                proto = self._by_name[name]
                if not proto.resettable:
                    continue
                view_idx, world_pos = self.resolve_envs(proto, task_envs)
                if view_idx.numel() == 0:
                    continue
                self._write_init_state(proto, view_idx, world_pos, init_cfg)

    def apply_actions(self, active: list[int]) -> None:
        """Hold default joint targets everywhere, add random offsets in the active envs."""
        for proto, articulation in self.iter_articulations():
            default = wp.to_torch(articulation.data.default_joint_pos)
            articulation.set_joint_position_target_index(target=default)
            view_idx, _ = self.resolve_envs(proto, active)
            if view_idx.numel() == 0:
                continue
            noise = noise_scale(proto.name) * torch.randn(view_idx.shape[0], default.shape[1], device=default.device)
            articulation.set_joint_position_target_index(target=default[view_idx] + noise, env_ids=view_idx)

    def _write_init_state(self, proto: Prototype, view_idx: torch.Tensor, world_pos: torch.Tensor, init_cfg) -> None:
        """Write ``init_cfg``'s root pose (+ default joints for articulations) at ``view_idx``."""
        obj = self._instance(proto)
        init = getattr(init_cfg, "init_state", None)
        pos = getattr(init, "pos", None) or (0.0, 0.0, 0.0)
        rot = getattr(init, "rot", None) or (1.0, 0.0, 0.0, 0.0)
        dev, dtype = world_pos.device, world_pos.dtype
        pose = torch.zeros((view_idx.shape[0], 7), device=dev, dtype=dtype)
        pose[:, :3] = torch.tensor(pos, device=dev, dtype=dtype) + world_pos
        pose[:, 3:7] = torch.tensor(rot, device=dev, dtype=dtype)
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_idx)
        obj.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((view_idx.shape[0], 6), device=dev, dtype=dtype), env_ids=view_idx
        )
        if proto.kind == "ArticulationCfg":
            jpos = wp.to_torch(obj.data.default_joint_pos)[view_idx].clone()
            jvel = wp.to_torch(obj.data.default_joint_vel)[view_idx].clone()
            obj.write_joint_position_to_sim_index(position=jpos, env_ids=view_idx)
            obj.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=view_idx)

    def _envs_by_category(self) -> dict[str, list[int]]:
        """Group env ids into locomotion / manipulation (sequential: env i -> task i % n)."""
        category_of_task = ["locomotion" if is_locomotion(t) else "manipulation" for t in self.tasks]
        out: dict[str, list[int]] = {}
        for e in range(self.num_envs):
            out.setdefault(category_of_task[e % len(self.tasks)], []).append(e)
        return out


# ======================================================================
# Recipe A -- declarative cfg handed to InteractiveScene (explicit / high level)
# ======================================================================


class InteractiveSceneEngine(CloneEngine):
    """High-level cloning: assemble a scene cfg and let ``InteractiveScene`` do the rest."""

    PRIM_PREFIX = PRIM_PREFIX_INTERACTIVE_SCENE

    @staticmethod
    def _selector_term(scene_assets: dict[str, object], names: list[str]) -> tuple[str, ...]:
        """Selector-term func: keep the listed prototype names that ended up in the scene."""
        return tuple(name for name in names if name in scene_assets)

    def build(self) -> None:
        # --- 1. assemble one declarative scene cfg --------------------------
        # One clone combination + selector term per task; a shared prototype name
        # appears in every task that uses it, so the cloner clones it into their union.
        clone_combinations = [InclusionSet(assets=list(t.prototype_names), weight=1) for t in self.tasks]
        selector_terms = {
            # selector attr names must be valid identifiers -> swap the gym id's dashes
            re.sub(r"\W+", "_", t.task_id).strip("_"): SelectorTermCfg(
                func=self._selector_term, params={"names": list(t.prototype_names)}
            )
            for t in self.tasks
        }
        selector_cls = configclass(type("RegistrySelectorCfg", (SelectorCfg,), selector_terms))

        namespace = _global_assets()
        namespace["clone_cfg"] = CloneCfg(clone_strategy=self.clone_strategy, clone_combinations=clone_combinations)
        namespace["selector_cfg"] = selector_cls()
        for proto in self.prototypes:
            namespace[proto.name] = proto.cfg
        scene_cls = configclass(type("RegistrySceneCfg", (InteractiveSceneCfg,), namespace))

        # --- 2. hand it over: InteractiveScene lays out envs, clones, wires the ---
        # --- selector and builds physics views; sim.reset() starts physics.    ---
        cfg = scene_cls(num_envs=self.num_envs, env_spacing=self.env_spacing, replicate_physics=False)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.origins = self.scene.env_origins
        self._print_selector_info()

    # The selector knows, per prototype, exactly which envs hold it (from the clone
    # plan), so locating a prototype's active envs is one selector query.
    def resolve_envs(self, proto, active):
        glob, view_idx = self.scene.selector.filter_reset_ids(
            proto.name, torch.tensor(active, device=self.scene.device, dtype=torch.long)
        )
        return view_idx, self.origins[glob]

    def iter_articulations(self):
        for proto in self.prototypes:
            if proto.name in self.scene.articulations:
                yield proto, self.scene.articulations[proto.name]

    def _instance(self, proto):
        if proto.name in self.scene.articulations:
            return self.scene.articulations[proto.name]
        return self.scene.rigid_objects[proto.name]

    def flush(self):
        self.scene.write_data_to_sim()

    def update(self, dt):
        self.scene.update(dt)

    def _print_selector_info(self) -> None:
        """Show the clean per-task env partition and the (overlapping) selector groups."""
        selector = self.scene.selector
        n_tasks = len(self.tasks)
        print("\n" + "=" * 78)
        print(f"  RUNTIME SELECTOR  --  {selector}")
        print("=" * 78)
        print("  Clean task -> owned envs  (sequential: env i -> task i % n_tasks):")
        for task_idx, task in enumerate(self.tasks):
            owned = [e for e in range(self.num_envs) if e % n_tasks == task_idx]
            print(f"    [{task_idx:2d}] {task.task_id:34s} owns {len(owned):2d} envs -> {owned}")
        # A selector group's env ids are the union of envs holding any of its assets, so
        # tasks that share a prototype (e.g. all Franka tasks) overlap here -- expected.
        print("\n  Selector groups (union of envs per asset; shared prototypes overlap):")
        for name in selector.selector_names:
            view = selector[name]
            ids = view.env_ids
            ids = list(range(self.num_envs)) if isinstance(ids, slice) else ids.tolist()
            print(f"  - {name:30s} count={view.count:3d}  env ids={ids}")
        print("=" * 78 + "\n")


# ======================================================================
# Recipe B -- grid_transforms + usd_replicate + ReplicateSession (implicit / low level)
# ======================================================================


class ManualCloneEngine(CloneEngine):
    """Low-level cloning: lay out envs and clone every prototype by hand."""

    PRIM_PREFIX = PRIM_PREFIX_MANUAL

    def build(self) -> None:
        stage = self.sim.stage

        # --- 1. shared globals: one ground + light prim, spawned directly ---
        for cfg in _global_assets().values():
            cfg.spawn.func(cfg.prim_path, cfg.spawn)

        # --- 2. lay out env containers: clone an empty env_0 Xform to env_1..N ---
        stage.DefinePrim("/World/envs/env_0", "Xform")
        all_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.origins, _ = cloner.grid_transforms(self.num_envs, self.env_spacing, device=self.device)
        with cloner.disabled_fabric_change_notifies(stage, restore=False):
            cloner.usd_replicate(
                stage, ["/World/envs/env_0"], ["/World/envs/env_{}"], all_indices, positions=self.origins
            )

        # --- 3. heterogeneous valid set: one clone combination per task ---
        # valid_set has one column per clone group, and make_clone_plan only groups env-scoped
        # cfgs with a spawner -- so restrict columns (and each task's inclusion set) to clonable
        # prototypes to stay in lockstep with the plan. Non-clonable assets (e.g. surface
        # grippers) ride along inside their owner's env and are constructed below.
        clonable = [p for p in self.prototypes if _is_clone_group(p.cfg)]
        clonable_names = {p.name for p in clonable}
        valid_set = make_valid_clone_combinations(
            [p.name for p in clonable],
            [1] * len(clonable),  # multi-asset spawners were collapsed to one variant
            [InclusionSet(assets=[n for n in t.prototype_names if n in clonable_names], weight=1) for t in self.tasks],
            self.device,
        )

        # --- 4. spawn each prototype in env_0 and let the clone plan replicate it ---
        with ReplicateSession(
            [p.cfg for p in self.prototypes],
            num_clones=self.num_envs,
            env_spacing=self.env_spacing,
            device=self.device,
            stage=stage,
            clone_strategy=self.clone_strategy,
            valid_set=valid_set,
        ):
            # make_clone_plan (in __enter__) has pointed each cfg's spawn_path at env_0.
            for proto in self.prototypes:
                cfg = proto.cfg
                if cfg.class_type is not None:
                    # Articulation / RigidObject: constructor spawns + registers replication.
                    proto.instance = cfg.class_type(cfg)
                elif cfg.spawn is not None:
                    # Static prop (no physics view): spawn in env_0 + queue its USD copy.
                    init = getattr(cfg, "init_state", None)
                    cfg.spawn.func(
                        cfg.spawn.spawn_path,
                        cfg.spawn,
                        translation=getattr(init, "pos", None),
                        orientation=getattr(init, "rot", None),
                    )
                    cloner.queue_usd_replication(cfg)

        self.sim.reset()  # initialise physics views on every spawned prototype
        self._live = [p.instance for p in self.prototypes if p.instance is not None]

    # No selector here: a prototype's actual envs are the analytic sequential set
    # (env i -> task i % n_tasks), recorded on the Prototype by assign_env_ids.
    def resolve_envs(self, proto, active):
        active_set = set(active)
        envs = [e for e in proto.env_ids if e in active_set]
        if not envs:
            empty = torch.empty(0, dtype=torch.long, device=self.origins.device)
            return empty, self.origins[empty]
        view_idx = torch.tensor([proto.env_ids.index(e) for e in envs], device=self.origins.device, dtype=torch.long)
        return view_idx, self.origins[envs]

    def iter_articulations(self):
        for proto in self.prototypes:
            if proto.kind == "ArticulationCfg" and proto.instance is not None:
                yield proto, proto.instance

    def _instance(self, proto):
        return proto.instance

    def flush(self):
        for instance in self._live:
            instance.write_data_to_sim()

    def update(self, dt):
        for instance in self._live:
            instance.update(dt)
