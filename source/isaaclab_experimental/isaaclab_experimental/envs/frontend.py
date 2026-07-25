# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp runtime frontend for IsaacLab tasks (``--frontend warp``).

:class:`WarpFrontend` encapsulates everything the warp runtime needs from a
stable task definition:

* Manager-based tasks: adapt the stable cfg in place (Newton physics check,
  :class:`SceneEntityCfg` promotion, MDP twin swap) and construct
  :class:`ManagerBasedRLEnvWarp`. All warp MDP twins (functions *and* classes)
  must exist for the chosen task — a missing twin is a hard failure, not a
  silent drop.
* Direct tasks: resolve the warp env class by name from the mirrored
  experimental package (or an explicit ``warp_entry_point`` override) and
  construct it with the stable cfg. The resolved twin is verified against the
  task's cfg, so a stable env class shared by several tasks (e.g. the Allegro
  and Shadow reorient hands) never silently runs one variant's warp env for
  another.

Twin resolution is purely mechanical: the experimental packages mirror the
stable tree (``isaaclab.* ↔ isaaclab_experimental.*``,
``isaaclab_tasks.* ↔ isaaclab_tasks_experimental.*``), and a stable symbol's
twin is looked up by name on the nearest ``.mdp`` package of the mirrored
path. There is no registration; anything unresolved is collected and reported
in one hard failure. The torch runtime never touches this module — the shared
RL CLI dispatches to plain ``gym.make`` before importing it, so this
(optional) package stays warp-only.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
from types import ModuleType
from typing import Any, ClassVar

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _StableSceneEntityCfg

logger = logging.getLogger(__name__)


__all__ = [
    "FrontendIncompatibleError",
    "WarpFrontend",
    "Workflow",
]


class Workflow(StrEnum):
    """Manager-based vs direct env workflow."""

    MANAGER_BASED = "manager_based"
    DIRECT = "direct"


class FrontendIncompatibleError(RuntimeError):
    """Raised when the warp frontend can't run the requested task."""


class WarpFrontend:
    """Builds environments on the warp runtime from stable task definitions.

    The class is a stateless namespace: twin resolution is a pure function of
    the stable module paths and the mirrored package tree (:attr:`MIRROR_ROOTS`);
    there is no registry and no import-order dependence.
    """

    # Stable package roots mirrored by their warp implementation packages. Twin
    # resolution is purely mechanical: mirror the root, then look up the nearest
    # enclosing ``.mdp`` package on the mirrored path.
    MIRROR_ROOTS: ClassVar[dict[str, str]] = {
        "isaaclab_tasks": "isaaclab_tasks_experimental",
        "isaaclab": "isaaclab_experimental",
    }

    # Prefixes that identify warp-origin symbols and registrations.
    WARP_ROOT_PREFIXES: ClassVar[tuple[str, ...]] = tuple(MIRROR_ROOTS.values())

    # Top-level cfg groups whose managers run warp-first. Only terms under these
    # are adapted (SceneEntityCfg promotion + MDP twin swap). The event manager
    # is warp-first too — it invokes term funcs with a Warp env-mask, so a stable
    # event func (which expects torch ``env_ids``) breaks at runtime; its funcs
    # must be swapped to warp twins. The curriculum, recorder and command
    # managers run on the stable (torch) implementation, so their terms are left
    # untouched. A stable term left on a warp manager would break, so a missing
    # twin in these groups is a hard error; a stable term on a stable manager is
    # correct, so those groups are skipped.
    WARP_MANAGED_GROUPS: ClassVar[frozenset[str]] = frozenset(
        {"observations", "rewards", "terminations", "actions", "events"}
    )

    # ------------------------------------------------------------------
    # Env construction
    # ------------------------------------------------------------------

    @classmethod
    def build_env(cls, env_cfg: Any, task_id: str, **construct_kwargs: Any) -> gym.Env:
        """Construct the warp env for ``task_id`` from a stable env cfg.

        Args:
            env_cfg: Stable env cfg. Manager-based cfgs are mutated in place.
            task_id: Gym registration id, e.g. ``"Isaac-Cartpole"``.
            **construct_kwargs: Forwarded to the env constructor (``render_mode``, …).

        Raises:
            FrontendIncompatibleError: If the warp runtime can't run the task
                (wrong physics, missing MDP twins, direct task without a warp
                implementation).
        """
        if cls._detect_workflow(env_cfg) is Workflow.DIRECT:
            return cls._build_direct_env(env_cfg, task_id, **construct_kwargs)
        return cls._build_manager_env(env_cfg, **construct_kwargs)

    @classmethod
    def _build_manager_env(cls, env_cfg: Any, **construct_kwargs: Any) -> gym.Env:
        """Construct :class:`ManagerBasedRLEnvWarp`; the env adapts the cfg itself.

        Imported lazily so that ``--frontend=torch`` callers don't pay the
        ``isaaclab_experimental.envs`` import cost. The env runs
        :meth:`adapt_cfg` in ``__init__``, so stable cfgs and warp-native cfgs
        take the same path.
        """
        from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

        return ManagerBasedRLEnvWarp(cfg=env_cfg, **construct_kwargs)

    @classmethod
    def _build_direct_env(cls, env_cfg: Any, task_id: str, **construct_kwargs: Any) -> gym.Env:
        """Construct a direct warp env.

        Direct workflows aren't cfg-adapted: a hand-written warp env class
        implements the task, constructed with the *stable* cfg. The class is
        resolved by name from the mirrored experimental package (or an explicit
        ``warp_entry_point`` override); see :meth:`_resolve_direct_warp_class`.
        If name resolution finds nothing, the task itself must be a warp-native
        registration.

        Several stable tasks can share one direct env class (e.g.
        ``ReorientDirectEnv`` serves both the Allegro and Shadow hands via
        different cfgs), so :meth:`_resolve_direct_warp_class` returns a
        name-mirrored twin only when it implements ``env_cfg`` — otherwise it
        raises rather than let the wrong env run silently.
        """
        env_class = cls._resolve_direct_warp_class(task_id, env_cfg)
        if env_class is None:
            # No warp twin by name: the task itself must be a warp-native registration.
            cls._assert_direct_warp_registration(task_id)
            return gym.make(task_id, cfg=env_cfg, **construct_kwargs)
        # Name-resolved warp class that implements this cfg: swap only the env class.
        cls._require_newton_physics(env_cfg, type(env_cfg).__name__)
        return env_class(cfg=env_cfg, **construct_kwargs)

    # ------------------------------------------------------------------
    # Cfg adaptation (manager-based)
    # ------------------------------------------------------------------

    @classmethod
    def adapt_cfg(cls, cfg: Any) -> None:
        """Mutate a stable manager-based ``cfg`` in place for the warp managers.

        Idempotent: re-running on an already-adapted cfg is a no-op (the steps
        below skip warp-origin symbols and already-promoted entities).

        Three steps, each independently testable:

        1. :meth:`_require_newton_physics` — hard check that ``cfg.sim.physics``
           is :class:`~isaaclab_newton.physics.NewtonCfg`. The user is
           responsible for selecting the Newton variant of the task's
           :class:`PresetCfg` via ``presets=newton_mjwarp``; we don't auto-inject.
        2. :meth:`_promote_scene_entity_cfgs` — replace stable
           :class:`~isaaclab.managers.SceneEntityCfg` instances under each
           term's ``params`` with the warp variant (which adds warp-cached
           ``joint_mask``, ``joint_ids_wp``, ``body_ids_wp`` fields).
        3. :meth:`_swap_mdp` — for every MDP term found anywhere in the cfg tree
           (discovered by :meth:`_walk_terms` via :class:`ManagerTermBaseCfg`
           subclassing, not by hard-coded attribute names), replace any stable
           ``func`` *or* ``class_type`` with its same-named warp twin. A missing
           twin raises :class:`FrontendIncompatibleError` — partial coverage is
           unsafe under the warp managers' kernel-only signature.
        """
        label = type(cfg).__name__
        cls._require_newton_physics(cfg, label)
        cls._promote_scene_entity_cfgs(cfg)
        cls._swap_mdp(cfg, label)
        cls._swap_noise_cfgs(cfg, label)

    @staticmethod
    def _require_newton_physics(cfg: Any, label: str) -> None:
        """Block unless ``cfg.sim.physics`` is :class:`NewtonCfg`.

        The warp managers' assets read state through :class:`NewtonManager`;
        a :class:`PhysxCfg` (or unresolved :class:`PresetCfg`) is a hard
        incompatibility. The fix is to pass ``presets=newton_mjwarp`` on the CLI
        so Hydra resolves the task's :class:`PresetCfg` wrapper to the Newton
        field before construction.
        """
        from isaaclab_newton.physics import NewtonCfg

        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        if isinstance(physics, NewtonCfg):
            return
        raise FrontendIncompatibleError(
            f"warp env {label!r}: expected cfg.sim.physics to be NewtonCfg,"
            f" got {type(physics).__name__!r}. Pass `presets=newton_mjwarp` on the CLI so"
            f" Hydra resolves the task's PresetCfg wrapper to the Newton variant."
        )

    @classmethod
    def _promote_scene_entity_cfgs(cls, cfg: Any) -> None:
        """Replace stable :class:`SceneEntityCfg` instances with the warp variant.

        Iterates every term cfg in the tree (via :meth:`_walk_terms`) and
        rebuilds any stable :class:`SceneEntityCfg` value under ``term.params``
        through :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable`.
        The warp variant subclasses the stable one, so type checks elsewhere
        stay valid; the new fields (``joint_mask`` / ``joint_ids_wp`` /
        ``body_ids_wp``) are filled at :meth:`resolve` time by the warp scene.
        """
        from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as _WarpSceneEntityCfg

        promoted: list[str] = []
        for path, term in cls._walk_terms(cfg):
            if not path or path[0] not in cls.WARP_MANAGED_GROUPS:
                continue  # term runs on a stable manager; keep the stable entity
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            for key, value in list(params.items()):
                # Skip non-entities and already-promoted values (idempotence).
                if isinstance(value, _WarpSceneEntityCfg) or not isinstance(value, _StableSceneEntityCfg):
                    continue
                params[key] = _WarpSceneEntityCfg.from_stable(value)
                promoted.append(f"{'.'.join(path)}.params[{key!r}] ({value.name!r})")
        if promoted:
            logger.info(
                "frontend.warp: promoted %d SceneEntityCfg instance(s) to warp variant:\n  %s",
                len(promoted),
                "\n  ".join(promoted),
            )

    @classmethod
    def _swap_mdp(cls, cfg: Any, label: str) -> None:
        """Replace ``term.func`` and ``term.class_type`` with their warp twins.

        Iterates every term cfg in the tree (via :meth:`_walk_terms`) and on
        each term swaps whichever of ``func`` / ``class_type`` is a
        stable-origin callable. Twin lookup is name-based, in the order given by
        :meth:`_twin_modules`: the warp mirror of the cfg's own task MDP
        namespace first (task twins win, even for symbols defined in core
        packages), then the mirror of the symbol's defining package (covers
        terms borrowed from another task family, e.g. the Ant task reusing
        Humanoid MDP terms), then the shared fallback. Any missing twin raises
        :class:`FrontendIncompatibleError` listing every affected term —
        partial swaps would leave torch-style callables in the cfg and the warp
        managers would call them with the wrong signature.

        The warp-side declarations (``out_dim``, ``axes``, ``observation_type``)
        that the warp managers need at init are *not* supplied by this swap;
        they travel with the warp twin function itself via its own
        ``@generic_io_descriptor_warp(out_dim=…)`` decorator. This method only
        substitutes the callable; the manager reads the new func's annotations
        when it parses the term cfg.
        """
        # Highest-priority twin source: warp mirrors of the cfg's own task namespace.
        cfg_route_modules = cls._cfg_route_modules(cfg)
        module_cache: dict[str, list[ModuleType]] = {}  # defining module -> modules to search
        searched: set[str] = set()  # module names, for the error message

        swapped = 0
        missing: list[tuple[str, str, str]] = []  # (location, attr, symbol)
        for path, term in cls._walk_terms(cfg):
            if not path or path[0] not in cls.WARP_MANAGED_GROUPS:
                continue  # term runs on a stable manager; leave it stable
            location = ".".join(path)
            for attr in ("func", "class_type"):  # a term implements via either attr
                stable = getattr(term, attr, None)
                if stable is None or not cls._is_swap_candidate(stable):
                    continue
                origin = getattr(stable, "__module__", "") or ""
                if origin not in module_cache:
                    module_cache[origin] = cls._twin_modules(origin, cfg_route_modules)
                    searched.update(m.__name__ for m in module_cache[origin])
                twin = cls._resolve_warp_twin(stable.__name__, module_cache[origin])
                if twin is None:
                    missing.append((location, attr, stable.__name__))  # collect every miss; report once below
                    continue
                setattr(term, attr, twin)
                swapped += 1

        if missing:
            lines = "\n  ".join(f"{loc}.{attr}: no warp twin for {sym!r}" for loc, attr, sym in missing)
            raise FrontendIncompatibleError(
                f"warp env {label!r}: missing warp MDP twins (searched {sorted(searched)}):\n  {lines}"
            )

        logger.info("frontend.warp: swapped %d MDP symbol(s) to warp twins", swapped)

    # ------------------------------------------------------------------
    # Workflow detection and direct-task resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_workflow(cfg: Any) -> Workflow:
        """Classify the env cfg into manager-based or direct (used to pick the build path).

        Note:
            The env cfg roots (ManagerBasedRLEnvCfg, DirectRLEnvCfg,
            DirectMARLEnvCfg) do not share a common base class. When a new cfg
            root is added, extend the isinstance tuples below.
        """
        if isinstance(cfg, ManagerBasedRLEnvCfg):
            return Workflow.MANAGER_BASED
        if isinstance(cfg, (DirectRLEnvCfg, DirectMARLEnvCfg)):
            return Workflow.DIRECT
        raise FrontendIncompatibleError(
            f"Unrecognised env cfg type {type(cfg).__name__!r};"
            f" expected ManagerBasedRLEnvCfg / DirectRLEnvCfg / DirectMARLEnvCfg subclass."
        )

    @classmethod
    def _resolve_direct_warp_class(cls, task_id: str, env_cfg: Any = None) -> type | None:
        """Return the warp env class that implements a stable direct task, or ``None``.

        Resolution order:

        1. An explicit ``warp_entry_point`` kwarg on the stable registration
           (``"module.path:ClassName"``), if present. This is a deliberate
           per-registration override for env classes that cannot follow the
           naming convention below, and is trusted as declared.
        2. Name-based mirror: the stable env entry point
           ``isaaclab_tasks.<...>.<task>_direct_env:<Name>Env`` resolves to
           ``isaaclab_tasks_experimental.<...>.<task>_warp_env:<Name>WarpEnv``
           on the mirrored experimental package. A direct warp env is discovered
           by following the convention — no per-task registration needed.

        The class is constructed with the *stable* cfg, so the task needs no
        parallel warp registration or duplicated configuration. Because one
        stable env class can be shared by several tasks (e.g. ``ReorientDirectEnv``
        serves the Allegro and Shadow hands via different cfgs), a name-mirrored
        twin is returned only when it implements ``env_cfg``, verified via
        :meth:`_assert_cfg_supported`; a mismatch is a hard error. Pass
        ``env_cfg=None`` to perform the name mapping alone (used by callers that
        only probe the convention, not applicability).
        """
        try:
            spec = gym.spec(task_id)
        except gym.error.NameNotFound as exc:
            raise FrontendIncompatibleError(
                f"--frontend=warp: task {task_id!r} is not registered with gymnasium."
            ) from exc
        target = (spec.kwargs or {}).get("warp_entry_point")
        if isinstance(target, str):
            module_name, _, class_name = target.partition(":")
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise FrontendIncompatibleError(
                    f"--frontend=warp: task {task_id!r} declares warp_entry_point {target!r},"
                    f" but it failed to import: {exc}"
                ) from exc
            env_class = getattr(module, class_name, None)
            if env_class is None:
                raise FrontendIncompatibleError(
                    f"--frontend=warp: task {task_id!r} declares warp_entry_point {target!r},"
                    f" but {class_name!r} is not defined in {module_name!r}."
                )
            return env_class
        env_class = cls._mirror_direct_warp_class(spec.entry_point)
        if env_class is not None and env_cfg is not None:
            # Name resolution keys on the (possibly shared) stable env class, so
            # confirm the mirrored twin actually implements this task's cfg
            # before offering it as the resolution.
            cls._assert_cfg_supported(env_class, env_cfg, task_id)
        return env_class

    @classmethod
    def _mirror_direct_warp_class(cls, entry_point: Any) -> type | None:
        """Resolve a stable direct env entry point to its warp twin by convention.

        ``isaaclab_tasks.<...>.<task>_direct_env:<Name>Env`` mirrors to
        ``isaaclab_tasks_experimental.<...>.<task>_warp_env:<Name>WarpEnv``.
        Returns ``None`` when the entry point does not follow the convention or
        the twin is absent (the caller then reports the task as unsupported).
        """
        if not isinstance(entry_point, str) or ":" not in entry_point:
            return None
        stable_module, _, stable_class = entry_point.partition(":")
        warp_module = cls._mirror_module(stable_module)
        if warp_module is None or not warp_module.endswith("_direct_env") or not stable_class.endswith("Env"):
            return None
        warp_module = f"{warp_module[: -len('_direct_env')]}_warp_env"
        warp_class = f"{stable_class[: -len('Env')]}WarpEnv"
        try:
            module = importlib.import_module(warp_module)
        except ImportError:
            return None
        env_class = getattr(module, warp_class, None)
        origin = getattr(env_class, "__module__", "") or ""
        if env_class is None or not origin.startswith(cls.WARP_ROOT_PREFIXES):
            return None
        return env_class

    @classmethod
    def _assert_direct_warp_registration(cls, task_id: str) -> None:
        """For direct workflows without a ``warp_entry_point``, the task must be warp-registered."""
        spec = gym.spec(task_id)
        ep = spec.entry_point
        module = ep if isinstance(ep, str) else (getattr(ep, "__module__", "") or "")
        if not module.startswith(cls.WARP_ROOT_PREFIXES):
            raise FrontendIncompatibleError(
                f"--frontend=warp on direct task {task_id!r}: entry_point {ep!r}"
                f" is not under {list(cls.WARP_ROOT_PREFIXES)}. Direct tasks must either"
                f" declare a `warp_entry_point` in their stable registration or be"
                f" registered as a warp env class."
            )

    @staticmethod
    def _declared_stable_cfg_name(env_class: type) -> str | None:
        """Return the stable cfg class name a direct warp env declares it implements.

        The contract is the env's own ``cfg`` class annotation (e.g.
        ``cfg: AllegroHandEnvCfg``). It is read straight from ``__annotations__``
        rather than via :func:`typing.get_type_hints`, which eagerly resolves
        *every* annotation on the class — including unrelated ``TYPE_CHECKING``
        forward references the runtime cannot import — and would raise. Returns
        ``None`` when the annotation is absent or is not a bare class name, in
        which case no compatibility check is enforced.
        """
        annotation = getattr(env_class, "__annotations__", {}).get("cfg")
        if isinstance(annotation, str) and annotation.isidentifier():
            return annotation
        return None

    @classmethod
    def _assert_cfg_supported(cls, env_class: type, env_cfg: Any, task_id: str) -> None:
        """Guard that a name-resolved direct warp env implements the task's cfg.

        A direct warp env is constructed with the *stable* cfg, so it must
        actually implement that cfg. When one stable env class is shared by
        several tasks (e.g. ``ReorientDirectEnv`` serves the Allegro and Shadow
        hands via different cfgs), name resolution yields a single warp twin
        that may implement only one of them. The twin advertises the cfg it
        supports via its ``cfg`` annotation; this passes when the task's cfg is
        that class or a subclass of it, and raises otherwise so the caller never
        trains a different MDP under a mismatched env.
        """
        declared = cls._declared_stable_cfg_name(env_class)
        if declared is None:
            return  # env declares no cfg contract — keep prior behavior
        cfg_type_names = {base.__name__ for base in type(env_cfg).__mro__}
        if declared not in cfg_type_names:
            raise FrontendIncompatibleError(
                f"--frontend=warp: task {task_id!r} resolves by name to warp env "
                f"{env_class.__module__}.{env_class.__qualname__}, which implements config "
                f"{declared!r}, but the task's config is {type(env_cfg).__name__!r}. This warp "
                f"env does not implement that config, so the task has no warp implementation. "
                f"(This happens when several stable tasks share one direct env class but only "
                f"one has a warp twin; declare a `warp_entry_point` on the stable registration "
                f"if a dedicated twin exists.)"
            )

    # ------------------------------------------------------------------
    # Twin resolution
    # ------------------------------------------------------------------

    @classmethod
    def _mirror_module(cls, module: str) -> str | None:
        """Mechanical mirror of a stable module path, or ``None`` if not mirrored."""
        if not isinstance(module, str) or not module:
            return None
        root, _, rest = module.partition(".")  # only the top-level package root is swapped
        mirror_root = cls.MIRROR_ROOTS.get(root)
        if mirror_root is None:
            return None
        return f"{mirror_root}.{rest}" if rest else mirror_root

    @staticmethod
    def _nearest_mdp_module(mirrored: str) -> str | None:
        """Deepest existing ``<prefix>.mdp`` package on a mirrored module path.

        If the path already contains an ``.mdp`` segment, the search starts at
        that boundary (``...velocity.mdp.rewards`` resolves to
        ``...velocity.mdp``); otherwise the path is walked upward until a
        mirrored ``.mdp`` package exists. Existence is probed with
        :func:`importlib.util.find_spec`; a prefix that does not exist simply
        ends that probe, so resolution is a pure function of the installed
        package tree.
        """
        parts = mirrored.split(".")
        if "mdp" in parts:
            parts = parts[: parts.index("mdp")]  # start the walk at the .mdp boundary
        for depth in range(len(parts), 0, -1):  # deepest prefix first
            candidate = ".".join([*parts[:depth], "mdp"])
            try:
                if importlib.util.find_spec(candidate) is not None:
                    return candidate
            except ModuleNotFoundError:
                continue
        return None

    @classmethod
    def _swap_noise_cfgs(cls, cfg: Any, label: str) -> None:
        """Replace stable observation-noise cfgs with their warp-native twins.

        The warp observation manager applies noise through the experimental noise
        cfg family (in-place Warp kernels); a stable noise cfg would otherwise be
        silently ignored, training against a different MDP. Twins resolve by class
        name; data fields are copied while ``func`` keeps the twin's warp default.
        Noise cfgs without a warp twin (e.g. class-based ``NoiseModelCfg``) are a
        hard error — silently changing the MDP is never acceptable.
        """
        import dataclasses

        from isaaclab.utils import noise as stable_noise

        from isaaclab_experimental.utils import noise as warp_noise

        unsupported: list[str] = []
        swapped = 0
        for path, term in cls._walk_terms(cfg):
            if not path or path[0] != "observations":
                continue
            noise_cfg = getattr(term, "noise", None)
            if noise_cfg is None:
                continue
            if type(noise_cfg).__module__.startswith(cls.WARP_ROOT_PREFIXES):
                continue  # already warp-native
            twin_cls = getattr(warp_noise, type(noise_cfg).__name__, None)
            if (
                twin_cls is None
                or not twin_cls.__module__.startswith(cls.WARP_ROOT_PREFIXES)
                or not isinstance(noise_cfg, stable_noise.NoiseCfg)
            ):
                unsupported.append(f"{'.'.join(path)}: {type(noise_cfg).__name__}")
                continue
            if noise_cfg.func != type(noise_cfg)().func:
                # A user-customized noise func has no warp twin; replacing it with the
                # twin's default would silently change the MDP.
                unsupported.append(f"{'.'.join(path)}: {type(noise_cfg).__name__} with custom func {noise_cfg.func!r}")
                continue
            fields = {f.name: getattr(noise_cfg, f.name) for f in dataclasses.fields(noise_cfg) if f.name != "func"}
            if any(isinstance(value, torch.Tensor) for value in fields.values()):
                # The warp noise kernels take scalar parameters; tensor-valued ranges
                # (per-element noise) have no warp twin yet.
                unsupported.append(f"{'.'.join(path)}: {type(noise_cfg).__name__} with tensor-valued parameters")
                continue
            term.noise = twin_cls(**fields)
            swapped += 1
        if unsupported:
            raise FrontendIncompatibleError(
                f"warp env {label!r}: observation noise cfgs without warp twins:\n  " + "\n  ".join(unsupported)
            )
        if swapped:
            logger.info("frontend.warp: swapped %d observation noise cfg(s) to warp twins", swapped)

    @staticmethod
    def _import_twin_module(target: str) -> ModuleType:
        """Import a resolved twin module; a module that exists but breaks is a hard error."""
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            raise FrontendIncompatibleError(f"warp twin module {target!r} exists but failed to import: {exc}") from exc

    @classmethod
    def _cfg_route_modules(cls, cfg: Any) -> list[ModuleType]:
        """Warp MDP mirrors of the cfg's class hierarchy, subclass first.

        A stable cfg consumes terms through its task's ``mdp`` namespace, which
        re-exports symbols that may be *defined* in core or shared packages. The
        warp mirror of that namespace is therefore the primary place to resolve
        twins; it is found by mirroring the modules of ``type(cfg).__mro__`` —
        the subclass module first (robot-specific cfgs), then base-cfg modules
        (task-family bases).
        """
        modules: list[ModuleType] = []
        for klass in type(cfg).__mro__:  # subclass first: robot cfg wins over task-family base
            mirrored = cls._mirror_module(getattr(klass, "__module__", "") or "")
            if mirrored is None:
                continue
            target = cls._nearest_mdp_module(mirrored)
            if target is None:
                continue
            module = cls._import_twin_module(target)
            if module not in modules:
                modules.append(module)
        return modules

    @classmethod
    def _twin_modules(cls, symbol_module: str, cfg_route_modules: list[ModuleType]) -> list[ModuleType]:
        """Warp modules to consult for a stable symbol's twin, in preference order.

        1. The warp mirrors of the cfg's own task MDP namespace
           (:meth:`_cfg_route_modules`) — task-specific twins win, including
           twins for symbols defined in core/shared packages.
        2. The mirror of the symbol's defining package — covers terms borrowed
           from another task family and, for core-defined symbols, the shared
           :mod:`isaaclab_experimental.envs.mdp` twins (the mirror of
           :mod:`isaaclab.envs.mdp`).
        """
        modules = list(cfg_route_modules)
        # Fallback: the mirror of the package that defines the symbol.
        mirrored = cls._mirror_module(symbol_module)
        if mirrored is not None:
            target = cls._nearest_mdp_module(mirrored)
            if target is not None:
                module = cls._import_twin_module(target)
                if module not in modules:
                    modules.append(module)
        return modules

    @classmethod
    def _resolve_warp_twin(cls, name: str, modules: Iterable[ModuleType]) -> Any | None:
        """Return the same-named symbol from ``modules`` that originates under the warp packages."""
        for module in modules:
            candidate = getattr(module, name, None)
            if candidate is None:
                continue
            origin = getattr(candidate, "__module__", "") or ""
            if origin.startswith(cls.WARP_ROOT_PREFIXES):  # re-exported stable symbols don't count
                return candidate
        return None

    @classmethod
    def _is_swap_candidate(cls, value: Any) -> bool:
        """Heuristic: callable or class whose origin is *not already* warp."""
        if not callable(value):
            return False
        origin = getattr(value, "__module__", "") or ""
        if origin.startswith(cls.WARP_ROOT_PREFIXES):
            return False  # already a warp twin (idempotent)
        return True

    @staticmethod
    def _walk_terms(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
        """Yield ``(path, term)`` for every MDP term cfg in the cfg tree.

        A "term" is a :class:`ManagerTermBaseCfg` (observation/reward/
        termination/event/curriculum) *or* an :class:`ActionTermCfg` — the
        latter is a separate base that is **not** a ``ManagerTermBaseCfg``
        subclass, yet carries a swappable ``class_type``, so it must be matched
        explicitly.

        Behavior at each node:

        * Match (a term cfg instance): yield ``(path, node)`` and stop — do not
          descend into ``term.params`` / ``term.func`` / ``term.class_type``.
        * Configclass: don't yield; recurse into every non-underscore *instance*
          attribute (``vars(node)``), extending the path. ``observations``,
          ``rewards``, ``events``, ``actions``, sub-groups like
          ``observations.policy``, and anything nested deeper are reached
          transparently.
        * Mapping (a group expressed as a ``{name: term}`` dict rather than a
          configclass): don't yield; recurse into each value. The warp managers
          accept both dict and configclass groups, so the walker must reach the
          terms either way or they would never be swapped.
        * Anything else (plain Python data, callables, non-configclass objects):
          stop. No yield, no recursion.

        Iterating the instance ``__dict__`` mirrors how the warp managers
        consume group cfgs (they iterate ``cfg.__dict__.items()``), so the
        walker sees exactly the terms the managers will see — including terms
        assigned in ``__post_init__`` — while never descending into methods or
        nested class objects, which live on the class rather than the instance.

        Driven entirely by type — no attribute names are hardcoded — so future
        cfg layouts (extra observation groups, new nesting, etc.) are picked up
        automatically as long as their terms subclass one of the term base cfgs.
        """
        from isaaclab.managers.manager_term_cfg import ActionTermCfg, ManagerTermBaseCfg

        if isinstance(node, (ManagerTermBaseCfg, ActionTermCfg)):
            yield path, node  # a term: yield and stop; never descend into params/func
            return
        if isinstance(node, Mapping):
            # A manager group can be a dict of {name: term} instead of a
            # configclass; the warp managers accept both, so descend into dict
            # values too or those terms would never be promoted/swapped.
            for key, value in node.items():
                if value is not None:
                    yield from WarpFrontend._walk_terms(value, path + (str(key),))
            return
        if not hasattr(node, "__dataclass_fields__"):
            return  # plain data / callables: not a cfg subtree
        for name, value in vars(node).items():  # instance attrs — the same view the managers iterate
            if name.startswith("_") or value is None:
                continue
            yield from WarpFrontend._walk_terms(value, path + (name,))
