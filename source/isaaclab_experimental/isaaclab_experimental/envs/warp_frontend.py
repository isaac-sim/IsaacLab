# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt a stable manager-based RL env cfg to run on the warp runtime.

Module-level imports are deliberately limited to lightweight stdlib modules.
Warp-library imports fire only inside :meth:`WarpFrontend.build`, which must
be called *after* :class:`SimulationApp` is alive — otherwise the warp lib
loading will race with Kit's USD/pxr extension initialisation.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MissingItem:
    """A symbol the adapter looked for but couldn't resolve to a warp twin."""

    group: str
    """Where it was looked up — e.g. ``rewards``, ``observations.policy``, ``actions``."""
    term_name: str
    """Field name on the group, e.g. ``track_lin_vel_xy_exp``."""
    expected_name: str
    """The symbol name searched for in the warp modules (the stable ``__name__``)."""
    kind: str
    """Either ``"func"`` (mdp function) or ``"class_type"`` (action runtime class)."""
    searched: tuple[str, ...]
    """Module dotted-paths searched, in order."""
    action: str
    """Outcome: ``"dropped"`` (term set to None) or ``"raised"`` (caller saw an error)."""


@dataclass
class WarpAdaptReport:
    """Result of :meth:`WarpFrontend.adapt`. Empty ``missing`` means clean run."""

    missing: list[MissingItem] = field(default_factory=list)

    def __bool__(self) -> bool:  # truthy when there is something to report
        return bool(self.missing)

    def format(self) -> str:
        if not self.missing:
            return "WarpFrontend: no missing items."
        lines = [f"WarpFrontend: {len(self.missing)} missing warp twin(s):"]
        for m in self.missing:
            lines.append(
                f"  - {m.group}.{m.term_name}: stable {m.kind}={m.expected_name!r} → "
                f"no twin in {list(m.searched)} ({m.action})"
            )
        return "\n".join(lines)


class _NameSwap:
    """Replace ``term.<attr>`` with the same-named symbol in ``modules``.

    Drop the term entirely if no twin exists and ``drop_if_missing`` is set;
    otherwise raise :class:`LookupError`. Either way, append a record to
    ``report.missing`` so the caller can summarise what was lost.

    Used for both ``term.func`` (mdp swap) and ``action.class_type`` swaps —
    the algorithm is identical, only the attribute name and policy differ.
    """

    def __init__(
        self,
        modules: Iterable[ModuleType],
        attr: str,
        drop_if_missing: bool,
        report: WarpAdaptReport,
        group_label: str,
        warp_module_prefixes: tuple[str, ...] = ("isaaclab_experimental", "isaaclab_tasks_experimental"),
    ):
        self._modules = tuple(modules)
        self._attr = attr
        self._drop = drop_if_missing
        self._report = report
        self._group = group_label
        # The warp mdp __init__ re-exports stable mdp terms via ``from isaaclab.envs.mdp import *``
        # before adding its own overrides. ``getattr(warp_mdp, name)`` therefore can return the
        # *stable* implementation untouched. We accept a candidate as a real warp twin only if
        # its ``__module__`` lives under one of these prefixes.
        self._warp_module_prefixes = warp_module_prefixes

    def apply_to(self, group: Any) -> None:
        if group is None:
            return
        for name in (n for n in dir(group) if not n.startswith("_")):
            term = getattr(group, name, None)
            if term is None or not hasattr(term, self._attr):
                continue
            stable = getattr(term, self._attr)
            if not callable(stable):
                continue
            twin = self._resolve_twin(stable.__name__)
            if twin is not None:
                setattr(term, self._attr, twin)
                continue

            # No twin found.
            missing = MissingItem(
                group=self._group,
                term_name=name,
                expected_name=stable.__name__,
                kind=self._attr,
                searched=tuple(m.__name__ for m in self._modules),
                action="dropped" if self._drop else "raised",
            )
            self._report.missing.append(missing)
            if self._drop:
                setattr(group, name, None)
            else:
                raise LookupError(
                    f"WarpFrontend: cannot adapt {self._group}.{name} — no warp twin for"
                    f" stable {self._attr}={stable.__name__!r}. Searched {missing.searched}."
                )

    def _resolve_twin(self, name: str) -> Any | None:
        """Return the warp implementation of ``name`` from the search modules.

        Skips candidates that are merely re-exports of stable code: the warp mdp
        package does ``from isaaclab.envs.mdp import *`` and a ``getattr`` lookup
        will happily return the stable symbol untouched. We confirm a candidate
        is actually a warp implementation by checking its ``__module__``.
        """
        for module in self._modules:
            candidate = getattr(module, name, None)
            if candidate is None:
                continue
            origin = getattr(candidate, "__module__", "") or ""
            if origin.startswith(self._warp_module_prefixes):
                return candidate
        return None


class WarpFrontend:
    """Adapt a stable env cfg in place; build a warp-runtime env.

    Call :meth:`build` only after ``SimulationApp`` is alive — warp lib loads
    lazily inside ``build``. All knobs are constructor args; subclass only if
    you need to change the algorithm itself, not its parameters.

    Example::

        with launch_simulation(env_cfg, args_cli):
            env = WarpFrontend().build(env_cfg, task_id=args_cli.task)
    """

    DEFAULT_TERM_GROUPS: tuple[tuple[str, ...], ...] = (
        ("observations", "policy"),
        ("events",),
        ("rewards",),
        ("terminations",),
        ("curriculum",),
        ("actions",),  # included for SceneEntityCfg upgrade in action params
    )

    def __init__(
        self,
        stable_root: str = "isaaclab_tasks",
        warp_root: str = "isaaclab_tasks_experimental",
        fallback_mdp: str = "isaaclab_experimental.envs.mdp",
        warp_actions: str = "isaaclab_experimental.envs.mdp.actions.joint_actions",
        drop_sensors: Iterable[str] = ("height_scanner",),
        term_groups: Iterable[tuple[str, ...]] = DEFAULT_TERM_GROUPS,
        strict: bool = False,
    ):
        self._stable_root = stable_root
        self._warp_root = warp_root
        self._fallback_mdp = fallback_mdp
        self._warp_actions = warp_actions
        self._drop_sensors = tuple(drop_sensors)
        self._term_groups = tuple(term_groups)
        self._strict = strict

    # -- public ---------------------------------------------------------------

    def adapt(self, cfg: Any, task_id: str) -> WarpAdaptReport:
        """Mutate ``cfg`` in place; return a report of what couldn't be adapted.

        With ``strict=True``, any missing twin (term func or action class) raises
        :class:`LookupError` immediately so the caller sees the failure in the
        traceback instead of a silently-dropped term.
        """
        report = WarpAdaptReport()

        # 1. PresetCfg → newton (also dodges the lazy PhysxCfg materialisation).
        # Note: by the time we run, Hydra's `resolve_presets` has already
        # collapsed any PresetCfg wrapper to a concrete preset. The auto
        # ``presets=newton`` injected from train.py for ``--manager=warp``
        # ensures that concrete preset is the newton one. If the preset
        # somehow still has a ``newton`` attribute (custom subclass), pick it.
        physics = getattr(cfg.sim, "physics", None)
        if physics is not None and hasattr(physics, "newton"):
            cfg.sim.physics = physics.newton

        # 2. Drop sensors warp can't run yet.
        scene = getattr(cfg, "scene", None)
        for sensor in self._drop_sensors:
            if scene is not None and getattr(scene, sensor, None) is not None:
                setattr(scene, sensor, None)

        # 3. Upgrade SceneEntityCfg instances inside term.params dicts to the
        #    warp-extended variant (adds joint_mask / *_ids_wp wp.array fields
        #    that warp mdp kernels read).
        self._upgrade_scene_entity_cfgs(cfg)

        # 4. Swap term.func across all groups.
        warp_mdp_modules = self._mdp_modules_for(task_id)
        # Always log which warp mdp modules were resolved so it's easy to debug
        # missing-twin reports.
        logger.info(
            "WarpFrontend: warp mdp modules for %r → %s",
            task_id,
            [m.__name__ for m in warp_mdp_modules],
        )
        for path in self._term_groups:
            if path == ("actions",):
                continue  # actions don't have ``func`` — handled below
            group = self._walk(cfg, path)
            label = ".".join(path)
            swap = _NameSwap(
                warp_mdp_modules, "func", drop_if_missing=not self._strict, report=report, group_label=label
            )
            swap.apply_to(group)

        # 5. Swap action class_type. An action with no warp class_type can't
        #    run on the warp manager, so this is always strict (raise).
        action_swap = _NameSwap(
            [importlib.import_module(self._warp_actions)],
            attr="class_type",
            drop_if_missing=False,
            report=report,
            group_label="actions",
        )
        action_swap.apply_to(getattr(cfg, "actions", None))

        # 6. Surface the report.
        if report.missing:
            level = logging.ERROR if self._strict else logging.WARNING
            logger.log(level, report.format())

        return report

    def build(self, cfg: Any, task_id: str):
        """Adapt ``cfg`` and return a :class:`ManagerBasedRLEnvWarp` instance.

        Always logs the missing-twin report (if any). With ``strict=True``,
        any missing twin raises before the env is constructed.
        """
        # Lazy: this is the first warp-lib load. Caller must already be inside
        # the SimulationApp context, i.e. inside ``launch_simulation``.
        from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

        self.adapt(cfg, task_id)
        return ManagerBasedRLEnvWarp(cfg=cfg)

    # -- helpers --------------------------------------------------------------

    def _upgrade_scene_entity_cfgs(self, cfg: Any) -> None:
        """Upgrade every stable :class:`SceneEntityCfg` to the warp variant.

        Walks ``term.params`` dicts inside each term group and re-classes any
        stable :class:`SceneEntityCfg` instance into the warp subclass so warp
        mdp kernels can read ``joint_mask`` / ``joint_ids_wp`` / ``body_ids_wp``
        after :meth:`resolve`.
        """
        from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _StableSE

        from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as _WarpSE

        def upgrade(obj: Any) -> None:
            if isinstance(obj, _WarpSE) or not isinstance(obj, _StableSE):
                return
            # In-place class promotion + initialise warp-only fields. The class
            # hierarchy guarantees compatible memory layout (warp inherits stable).
            obj.__class__ = _WarpSE
            obj.joint_mask = None
            obj.joint_ids_wp = None
            obj.body_ids_wp = None

        for path in self._term_groups:
            group = self._walk(cfg, path)
            if group is None:
                continue
            for name in (n for n in dir(group) if not n.startswith("_")):
                term = getattr(group, name, None)
                if term is None:
                    continue
                params = getattr(term, "params", None)
                if isinstance(params, dict):
                    for v in params.values():
                        upgrade(v)

    @staticmethod
    def _walk(root: Any, path: tuple[str, ...]) -> Any:
        node = root
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                return None
        return node

    def _mdp_modules_for(self, task_id: str) -> list[ModuleType]:
        """Locate warp mdp modules that mirror the stable task's mdp.

        Convention: replace ``isaaclab_tasks`` with ``isaaclab_tasks_experimental``
        in the stable cfg's module path; walk up looking for an ``mdp``
        sub-module on the warp side; always append the package-wide warp mdp
        as a fallback for cross-task base terms.
        """
        modules: list[ModuleType] = []
        try:
            import gymnasium as gym

            entry = gym.spec(task_id).kwargs.get("env_cfg_entry_point")
        except Exception:
            entry = None
        if isinstance(entry, str) and entry.startswith(self._stable_root):
            warp_pkg = entry.rsplit(".", 1)[0].replace(self._stable_root, self._warp_root, 1)
            parts = warp_pkg.split(".")
            for depth in range(len(parts), 0, -1):
                try:
                    modules.append(importlib.import_module(".".join(parts[:depth] + ["mdp"])))
                    break
                except ImportError:
                    continue
        try:
            modules.append(importlib.import_module(self._fallback_mdp))
        except ImportError:
            logger.warning("WarpFrontend: fallback mdp module %r not importable", self._fallback_mdp)
        return modules
