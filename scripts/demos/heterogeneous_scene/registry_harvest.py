# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry harvesting for the heterogeneous-scene demo.

This is the front half of the pipeline: discover gym tasks, pull each task's
env-scoped assets off its cfg, and **de-duplicate them into prototypes** -- one
prototype per distinct model, shared by every task that uses it. The back half (how a
prototype is actually spawned and cloned) lives in :mod:`clone_engines`.

Vocabulary:

* **task** -- one gym task, harvested into a :class:`TaskGroup` (the prototype names
  it uses + each asset's per-task init cfg).
* **prototype** -- a :class:`Prototype`: one asset spawned a single time and cloned
  into every env whose task uses it. Same-model assets across tasks collapse to one.

Import this from a demo *after* :class:`~isaaclab.app.AppLauncher` has started the
simulator (its isaaclab imports require the running app).
"""

from __future__ import annotations

import re
from dataclasses import MISSING

import gymnasium as gym

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import interleaved, sequential

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

try:
    from isaaclab_physx.assets import DeformableObjectCfg
except Exception:  # pragma: no cover - physx may be unavailable
    DeformableObjectCfg = ()  # type: ignore[assignment]

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Task sources, keyed by the module prefix of their ``env_cfg_entry_point``. Harvesting
# more task families is a one-liner: add a prefix here and import the package that
# registers it.
CORE_TASK_PREFIX = "isaaclab_tasks.core"
CONTRIB_TASK_PREFIX = "isaaclab_tasks.contrib"
_GROUND_SPAWN_NAMES = {"GroundPlaneCfg"}
_TERRAIN_CFG_NAMES = {"TerrainImporterCfg"}

# Random joint-perturbation magnitude per robot family; legged platforms wiggle
# less than arms so they stay upright while remaining visually expressive.
ARM_NOISE = 0.4
LEG_NOISE = 0.06
DEFAULT_NOISE = 0.15
LEGGED_HINTS = ("anymal", "unitree", "spot", "a1", "go1", "go2", "cassie", "digit", "h1", "g1", "humanoid", "ant")
ARM_HINTS = ("panda", "franka", "ur5", "ur10", "kinova", "sawyer", "flexiv", "allegro", "shadow", "robot", "arm")

# Selectable clone strategies (prototype-combination -> env assignment). Only the
# deterministic round-robin strategies are exposed; the cloner's :func:`~isaaclab.cloner.random`
# is omitted on purpose. This demo relies on the analytic ``env i -> task i % n`` map for
# per-task reset, grouping, and the report, and on every task getting envs -- ``random`` breaks
# both (non-deterministic split, uneven coverage). ``interleaved`` is a readability alias.
STRATEGIES = {"sequential": sequential, "interleaved": interleaved}

# Per-model display tweaks, matched against the spawn's USD basename. Robots that are
# oversized or sit high above the floor (cartpole / cart-double-pendulum at z=2, ant)
# are scaled down and dropped closer to the ground for a balanced scene.
SCALE_OVERRIDES = {"cart": 0.4, "ant": 0.4}
Z_OFFSET_OVERRIDES = {"cart": -1.5}

# Per-task vertical nudges, keyed by a gym-id substring and applied to *every* asset in a
# matching task (on top of ground re-basing).
TASK_Z_OFFSETS = {"Place-Mug-Agibot-Left-Arm": -1.0}


# ------------------------------------------------------------------
# Prototype identity: signature (de-dup) + display label + unique name
# ------------------------------------------------------------------


def spawn_signature(cfg: AssetBaseCfg) -> str:
    """String identifying the physical model a spawn produces (USD set + scale)."""
    spawn = getattr(cfg, "spawn", None)
    if spawn is None:
        return f"{type(cfg).__name__}:no_spawn"
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd is not None:
        body = usd
    elif sub:
        body = "+".join(sorted(getattr(s, "usd_path", None) or type(s).__name__ for s in sub))
    else:
        body = type(spawn).__name__
    return f"{type(cfg).__name__}|{body}|scale={getattr(spawn, 'scale', None)}"


def model_label(cfg: AssetBaseCfg) -> str:
    """Short human-readable model name for reports (USD file, shape class, or variant set)."""
    spawn = getattr(cfg, "spawn", None)
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd:
        return usd.split("/")[-1]
    if sub:
        labels = [(getattr(s, "usd_path", None) or "").split("/")[-1] or type(s).__name__ for s in sub]
        return "{" + ", ".join(labels) + "}"
    return type(spawn).__name__ if spawn else "-"


def prototype_key(cfg: AssetBaseCfg, occurrence: int) -> str:
    """Identity key deciding which assets collapse into the same cloned prototype.

    Resettable assets (articulation / rigid object) share by *model* across tasks
    (re-posed per env at reset); static props also key on pose, since they cannot move
    after spawn. ``occurrence`` keeps same-model duplicates within one task (e.g. two
    identical cubes) distinct so they remain separate prototypes.
    """
    key = f"{spawn_signature(cfg)}#{occurrence}"
    resettable = isinstance(cfg, (ArticulationCfg, RigidObjectCfg)) or bool(
        DeformableObjectCfg and isinstance(cfg, DeformableObjectCfg)
    )
    if not resettable:
        init = getattr(cfg, "init_state", None)
        pos = tuple(round(float(x), 3) for x in (getattr(init, "pos", None) or (0.0, 0.0, 0.0)))
        rot = tuple(round(float(x), 3) for x in (getattr(init, "rot", None) or (1.0, 0.0, 0.0, 0.0)))
        key += f"|pos={pos}|rot={rot}"
    return key


def unique_prim_name(cfg: AssetBaseCfg, taken: set[str]) -> str:
    """A unique scene-field name (and prim leaf) derived from the asset's model."""
    spawn = getattr(cfg, "spawn", None)
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd:
        stem = usd.split("/")[-1].rsplit(".", 1)[0]
    elif sub:
        first_usd = getattr(sub[0], "usd_path", None)
        stem = (first_usd.split("/")[-1].rsplit(".", 1)[0] + "_multi") if first_usd else "multi_asset"
    elif spawn is not None:
        stem = type(spawn).__name__.replace("Cfg", "")
    else:
        stem = type(cfg).__name__
    base = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower() or "asset"
    name, i = base, 1
    while name in taken:
        i += 1
        name = f"{base}_{i}"
    taken.add(name)
    return name


# ------------------------------------------------------------------
# Task driving semantics (used by the engines' run loop)
# ------------------------------------------------------------------


def noise_scale(asset_name: str) -> float:
    """Per-asset joint-perturbation magnitude inferred from the asset name."""
    lname = asset_name.lower()
    if any(h in lname for h in LEGGED_HINTS):
        return LEG_NOISE
    if any(h in lname for h in ARM_HINTS):
        return ARM_NOISE
    return DEFAULT_NOISE


def is_locomotion(task: TaskGroup) -> bool:
    """Classify a task as locomotion (legged/control) vs manipulation for grouped driving."""
    return any(h in (task.task_id + " " + " ".join(task.prototype_names)).lower() for h in LEGGED_HINTS)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


class Prototype:
    """One asset spawned a single time and cloned into the envs that use it.

    Same-model assets across tasks collapse to one :class:`Prototype`; :attr:`task_ids`
    lists every task that shares it and :attr:`env_ids` the envs it is cloned into.
    """

    def __init__(self, key: str, name: str, cfg: AssetBaseCfg):
        self.key = key
        self.name = name  # scene-field name == prim leaf, e.g. "panda_instanceable"
        self.cfg = cfg  # prim_path already bound under the env namespace
        self.kind = type(cfg).__name__  # "ArticulationCfg" / "RigidObjectCfg" / "AssetBaseCfg"
        self.label = model_label(cfg)  # human-readable model name for reports
        self.task_ids: list[str] = []  # gym ids of tasks that use this prototype
        self.env_ids: list[int] = []  # global envs it is cloned into (set by assign_env_ids)
        self.instance = None  # live Articulation/RigidObject after spawn (manual engine only)

    @property
    def shared(self) -> bool:
        return len(self.task_ids) > 1

    @property
    def resettable(self) -> bool:
        """True if it has a writable root pose (articulation / rigid object)."""
        return self.kind in ("ArticulationCfg", "RigidObjectCfg")


class TaskGroup:
    """One harvested gym task: the prototype names it uses + its per-task init cfgs.

    A shared prototype is spawned once, but each task keeps its own re-based cfg in
    :attr:`init_cfgs` so the engine can reset that prototype to this task's pose.
    """

    def __init__(self, task_id: str, ground_z: float):
        self.task_id = task_id
        self.ground_z = ground_z
        self.prototype_names: list[str] = []
        self.init_cfgs: dict[str, AssetBaseCfg] = {}  # prototype name -> this task's cfg


# ------------------------------------------------------------------
# Discovery + harvest
# ------------------------------------------------------------------


def discover_task_ids(
    include: str | None,
    exclude: str | None,
    *,
    manager_based: bool,
    module_prefixes: tuple[str, ...] = (CORE_TASK_PREFIX,),
) -> list[str]:
    """Collect task ids of one workflow, minus Play/Camera/deprecated noise.

    Args:
        manager_based: Keep manager-based tasks (``isaaclab.envs:ManagerBasedRLEnv``)
            when True, else keep Direct-workflow tasks (custom env class).
        module_prefixes: env-cfg module prefixes to harvest from; defaults to ``core``.
            Pass e.g. ``(CORE_TASK_PREFIX, CONTRIB_TASK_PREFIX)`` to also pull contrib
            tasks (``import isaaclab_tasks`` registers both core and contrib).
    """
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    ids: list[str] = []
    for task_id, spec in gym.registry.items():
        ep = spec.kwargs.get("env_cfg_entry_point")
        module = (ep.split(":")[0] if isinstance(ep, str) else getattr(ep, "__module__", None)) if ep else None
        if module is None or not module.startswith(module_prefixes):
            continue
        if ("ManagerBased" in str(spec.entry_point)) != manager_based:
            continue
        if "Play" in task_id or "Camera" in task_id or spec.kwargs.get("deprecated"):
            continue
        if (inc and not inc.search(task_id)) or (exc and exc.search(task_id)):
            continue
        ids.append(task_id)
    return sorted(ids)


def asset_fields(cfg) -> dict:
    """The cfg fields that may hold env-scoped assets, unified across workflows.

    Manager-based tasks declare assets inside ``cfg.scene``; Direct-workflow tasks
    declare them as top-level ``cfg`` fields. Merging both namespaces lets a single
    :func:`harvest_tasks` pass handle either workflow -- or a mix of both.
    """
    scene = getattr(cfg, "scene", None)
    return {**vars(cfg), **(vars(scene) if scene is not None else {})}


def _harvest_task(task_id: str, fields: dict, randomize_variants: bool) -> tuple[float, list[AssetBaseCfg]] | None:
    """Pull one task's env-scoped asset cfgs out of ``fields``, re-based + variant-collapsed.

    Returns ``(ground_z, asset_cfgs)`` or ``None`` (with a logged reason) for tasks that
    contain a deformable object or expose no env-scoped assets.
    """
    if any(
        (DeformableObjectCfg and isinstance(v, DeformableObjectCfg)) or "Deformable" in type(v).__name__
        for v in fields.values()
    ):
        print(f"[skip] {task_id}: deformable (cloth/soft) task filtered")
        return None

    ground_z = 0.0
    env_scoped: list[AssetBaseCfg] = []
    dropped_grippers: list[str] = []
    for value in fields.values():
        if value is MISSING or value is None:
            continue
        # SurfaceGripper requires a CPU physics backend; drop just the gripper (not the
        # whole task) so the scene still runs on GPU. Its owner robot + the rest of the
        # task's assets spawn and clone normally.
        if "SurfaceGripper" in type(value).__name__:
            dropped_grippers.append(type(value).__name__)
            continue
        spawn = getattr(value, "spawn", None)
        # Ground / terrain: record its height for re-basing, then drop it.
        if (spawn is not None and type(spawn).__name__ in _GROUND_SPAWN_NAMES) or type(
            value
        ).__name__ in _TERRAIN_CFG_NAMES:
            gpos = getattr(getattr(value, "init_state", None), "pos", None)
            ground_z = float(gpos[2]) if gpos is not None else 0.0
            continue
        if not isinstance(value, AssetBaseCfg):  # sensors / knobs / sim / scene ... are not AssetBaseCfg
            continue
        prim = getattr(value, "prim_path", "") or ""
        # Env-scoped assets use the {ENV_REGEX_NS} macro OR the expanded literal form.
        if "{ENV_REGEX_NS}" not in prim and "/World/envs/env_" not in prim:
            continue  # global asset (ground/light/sky) -> the shared scene supplies its own
        env_scoped.append(value)

    if dropped_grippers:
        print(f"[note] {task_id}: dropped {len(dropped_grippers)} SurfaceGripper asset(s) (CPU-only backend)")

    if not env_scoped:
        print(f"[skip] {task_id}: no env-scoped assets")
        return None

    delta = -ground_z  # lift the whole group so its ground lands at world z=0
    # Whole-task vertical nudge (applies to every asset in this task), if one is configured.
    task_z = next((z for key, z in TASK_Z_OFFSETS.items() if key in task_id), 0.0)
    out: list[AssetBaseCfg] = []
    for value in env_scoped:
        model = model_label(value).lower()  # match per-model tweaks on the USD/shape name
        # Re-base z onto the shared floor, plus any per-model drop toward the ground.
        z_off = next((z for h, z in Z_OFFSET_OVERRIDES.items() if h in model), 0.0)
        init = getattr(value, "init_state", None)
        pos = getattr(init, "pos", None)
        if init is not None and pos is not None:
            new_z = float(pos[2]) + delta + z_off + task_z
            value = value.replace(init_state=init.replace(pos=(float(pos[0]), float(pos[1]), new_z)))
        # Collapse a multi-variant spawner to one variant (keeping wrapper + props).
        spawn = getattr(value, "spawn", None)
        sub = getattr(spawn, "assets_cfg", None)
        if sub and len(sub) > 1 and not randomize_variants:
            value = value.replace(spawn=spawn.replace(assets_cfg=list(sub[:1])))
            spawn = value.spawn
        # Shrink oversized models via spawn scale.
        factor = next((s for h, s in SCALE_OVERRIDES.items() if h in model), None)
        if factor is not None and spawn is not None and hasattr(spawn, "scale"):
            b = spawn.scale or (1.0, 1.0, 1.0)
            value = value.replace(spawn=spawn.replace(scale=(b[0] * factor, b[1] * factor, b[2] * factor)))
        # Some tasks declare a static prop with rigid_props (e.g. a GR1 table), so PhysX
        # simulates it as a free body that drifts on contact. We never re-pose static props,
        # so force kinematic -- keeps the collider but stays put (cf. packing_table).
        resettable = isinstance(value, (ArticulationCfg, RigidObjectCfg)) or bool(
            DeformableObjectCfg and isinstance(value, DeformableObjectCfg)
        )
        rigid_props = getattr(spawn, "rigid_props", None)
        if not resettable and rigid_props is not None and not rigid_props.kinematic_enabled:
            value = value.replace(spawn=spawn.replace(rigid_props=rigid_props.replace(kinematic_enabled=True)))
        out.append(value)
    return ground_z, out


def harvest_tasks(
    task_ids: list[str],
    *,
    fields_of,
    prim_prefix: str,
    device: str,
    max_tasks: int | None = None,
    randomize_variants: bool = False,
) -> tuple[list[TaskGroup], list[Prototype]]:
    """Harvest tasks into ``(tasks, prototypes)`` with same-model assets de-duplicated.

    Args:
        fields_of: ``cfg -> dict`` returning the cfg fields to scan for assets
            (use :func:`asset_fields`).
        prim_prefix: Env-namespace prefix for cloned prim paths, e.g. ``"{ENV_REGEX_NS}"``
            for InteractiveScene or ``"/World/envs/env_.*"`` for the manual clone path.
    """
    tasks: list[TaskGroup] = []
    prototypes: list[Prototype] = []  # registration order == stable scene-field order
    by_key: dict[str, Prototype] = {}  # prototype_key -> shared Prototype
    taken_names: set[str] = set()
    seen_task_sigs: set[frozenset] = set()

    for task_id in task_ids:
        try:
            cfg = parse_env_cfg(task_id, device=device, num_envs=1)
        except Exception as exc:  # noqa: BLE001 - one bad task must not abort discovery
            print(f"[skip] {task_id}: cfg load failed ({type(exc).__name__}: {exc})")
            continue
        fields = fields_of(cfg)
        harvested = _harvest_task(task_id, fields, randomize_variants)
        if harvested is None:
            continue
        ground_z, asset_cfgs = harvested

        # Number same-model duplicates within a task (#0, #1, ...) so two identical
        # cubes stay distinct while still sharing the k-th slot across tasks.
        per_model_count: dict[str, int] = {}
        keyed_assets: list[tuple[str, AssetBaseCfg]] = []
        for asset_cfg in asset_cfgs:
            sig = spawn_signature(asset_cfg)
            occ = per_model_count.get(sig, 0)
            per_model_count[sig] = occ + 1
            keyed_assets.append((prototype_key(asset_cfg, occ), asset_cfg))

        task_sig = frozenset(key for key, _ in keyed_assets)
        if task_sig in seen_task_sigs:
            print(f"[dup ] {task_id}: asset set identical to an earlier task, skipped")
            continue
        seen_task_sigs.add(task_sig)

        task = TaskGroup(task_id, ground_z)
        for key, asset_cfg in keyed_assets:
            proto = by_key.get(key)
            if proto is None:  # first task to use this model -> register a new prototype
                # scale + ground drop were already applied in _harvest_task, so spawn and
                # reset share one adjusted pose; here we only bind the cloned prim path.
                name = unique_prim_name(asset_cfg, taken_names)
                proto = Prototype(key, name, asset_cfg.replace(prim_path=f"{prim_prefix}/{name}"))
                by_key[key] = proto
                prototypes.append(proto)
            if task_id not in proto.task_ids:
                proto.task_ids.append(task_id)
            if proto.name not in task.prototype_names:
                task.prototype_names.append(proto.name)
            task.init_cfgs[proto.name] = asset_cfg  # this task's intended pose for reset

        tasks.append(task)
        if max_tasks is not None and len(tasks) >= max_tasks:
            print(f"[INFO] reached --max_tasks={max_tasks}, stopping discovery")
            break
    return tasks, prototypes


def assign_env_ids(tasks: list[TaskGroup], prototypes: list[Prototype], num_envs: int) -> None:
    """Fill ``Prototype.env_ids`` from the deterministic sequential map (env i -> task i % n)."""
    n_tasks = len(tasks)
    task_to_idx = {t.task_id: i for i, t in enumerate(tasks)}
    for proto in prototypes:
        owners = {task_to_idx[tid] for tid in proto.task_ids}
        proto.env_ids = [e for e in range(num_envs) if e % n_tasks in owners]


def print_preprocess_report(tasks: list[TaskGroup], prototypes: list[Prototype], num_envs: int, *, title: str) -> None:
    """Print the harvested layout: counts, env->task map, and per-prototype spawn/clone.

    :func:`assign_env_ids` must have run first so ``Prototype.env_ids`` is set.
    """
    n_tasks = len(tasks)
    n_shared = sum(1 for p in prototypes if p.shared)
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  tasks                 : {n_tasks}")
    print(f"  total envs            : {num_envs}   (sequential: env i -> task i % {n_tasks})")
    print("  global shared assets  : 2   (ground + dome light, one prim each, never cloned)")
    print(
        f"  unique prototypes     : {len(prototypes)}   ({n_shared} shared across >1 task, "
        f"{len(prototypes) - n_shared} exclusive; each spawned once, then cloned)"
    )

    print("\n  env -> task layout:")
    cells = [f"e{i:<2d}={tasks[i % n_tasks].task_id}" for i in range(num_envs)]
    for r in range(0, len(cells), 3):
        print("    " + "  ".join(f"{c:<38s}" for c in cells[r : r + 3]).rstrip())

    print("\n  prototypes (★ = shared across tasks):")
    for proto in prototypes:
        src, clones = proto.env_ids[0], proto.env_ids[1:]
        print(f"\n  {'★' if proto.shared else ' '} {proto.name:26s} [{proto.kind:15s}] model={proto.label}")
        print(f"      used by {len(proto.task_ids)} task(s): {proto.task_ids}")
        print(f"      spawn @ env {src:<3d} -> clone into envs {clones if clones else '(none)'}")
    print("=" * 78 + "\n")
