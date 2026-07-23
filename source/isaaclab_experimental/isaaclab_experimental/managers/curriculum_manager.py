# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first curriculum manager with mask-native term dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING

import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab.managers import CurriculumTermCfg as StableCurriculumTermCfg
from isaaclab.managers import ManagerTermBase as StableManagerTermBase
from isaaclab.utils import string_to_callable

from .manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CurriculumManager(ManagerBase):
    """Manage curriculum terms that write state into persistent Warp scalar outputs."""

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the curriculum manager.

        Args:
            cfg: Configuration object or dictionary of curriculum terms.
            env: Environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[StableCurriculumTermCfg] = []
        self._term_modes: list[str] = []

        super().__init__(cfg, env)

        num_terms = len(self._term_names)
        self._term_states_wp = wp.zeros(num_terms, dtype=wp.float32, device=self.device)
        self._term_state_views_wp: list[wp.array] = []
        if num_terms > 0:
            stride = self._term_states_wp.strides[0]
            for term_idx, term_cfg in enumerate(self._term_cfgs):
                out_view = wp.array(
                    ptr=self._term_states_wp.ptr + term_idx * stride,
                    dtype=wp.float32,
                    shape=(1,),
                    strides=(stride,),
                    device=self.device,
                )
                self._term_state_views_wp.append(out_view)
                term_cfg.out = out_view

        self._term_states = wp.to_torch(self._term_states_wp)
        self._reset_extras = {
            f"Curriculum/{term_name}": self._term_states[term_idx]
            for term_idx, term_name in enumerate(self._term_names)
        }

    def __str__(self) -> str:
        """Return a string representation of the curriculum manager."""
        msg = f"<CurriculumManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Curriculum Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        return msg + table.get_string() + "\n"

    @property
    def active_terms(self) -> list[str]:
        """Names of active curriculum terms."""
        return self._term_names

    @property
    def reset_extras(self) -> dict[str, torch.Tensor]:
        """Persistent scalar views containing the latest curriculum states."""
        return self._reset_extras

    def reset(
        self,
        env_mask: wp.array(dtype=wp.bool),
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reset selected class terms and return persistent curriculum logging outputs.

        Args:
            env_mask: Boolean Warp mask selecting environments to reset.
            env_ids: Precomputed compact environment IDs matching :paramref:`env_mask`.
                When omitted, legacy terms materialize IDs once at this boundary.

        Returns:
            Persistent scalar curriculum states keyed by their logging paths.
        """
        env_mask = self._resolve_reset_mask(None, env_mask)
        # Dispatch by term kind: Warp class terms consume the mask directly; stable
        # class terms receive compact IDs, materialized at most once per call.
        compact_env_ids = env_ids
        for term_cfg, mode in zip(self._term_cfgs, self._term_modes):
            if isinstance(term_cfg.func, ManagerTermBase):
                term_cfg.func.reset(env_mask=env_mask)
            elif isinstance(term_cfg.func, StableManagerTermBase):
                if compact_env_ids is None:
                    compact_env_ids = self._compact_legacy_env_ids(env_mask)
                term_cfg.func.reset(env_ids=compact_env_ids)
        return self._reset_extras

    @property
    def requires_host_ids(self) -> bool:
        """Whether any active legacy term genuinely consumes compact environment IDs."""
        return "legacy" in self._term_modes

    @property
    def requires_host_boundary(self) -> bool:
        """Whether any active legacy term must run only when a reset occurs."""
        return any(mode != "mask" for mode in self._term_modes)

    def compute(
        self,
        env_mask: wp.array(dtype=wp.bool),
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Update curriculum terms for selected environments.

        Args:
            env_mask: Boolean Warp mask selecting environments to update.
            env_ids: Precomputed compact environment IDs matching :paramref:`env_mask`.
                When omitted, legacy terms materialize IDs once at this boundary.
        """
        env_mask = self._resolve_reset_mask(None, env_mask)
        # Logging states are recomputed every call; mask-native terms write their
        # scalar slot on-device through the pointer-stable ``term_cfg.out`` view.
        self._term_states_wp.zero_()
        compact_env_ids = env_ids
        # Term modes: "mask" = Warp-native ``(env, env_mask, out)``;
        # "legacy" = stable ``(env, env_ids)`` with compact IDs materialized at most once.
        for term_idx, (term_cfg, mode) in enumerate(zip(self._term_cfgs, self._term_modes)):
            if mode == "mask":
                term_cfg.func(self._env, env_mask, term_cfg.out, **term_cfg.params)
                continue
            if compact_env_ids is None:
                compact_env_ids = self._compact_legacy_env_ids(env_mask)
            state = term_cfg.func(self._env, compact_env_ids, **term_cfg.params)
            if state is not None:
                if isinstance(state, torch.Tensor) and state.numel() != 1:
                    raise TypeError(
                        f"Curriculum term '{self._term_names[term_idx]}' must return a scalar state;"
                        f" received tensor shape {tuple(state.shape)}."
                    )
                if not isinstance(state, (Real, torch.Tensor)):
                    raise TypeError(
                        f"Curriculum term '{self._term_names[term_idx]}' returned {type(state).__name__}."
                        " Warp CurriculumManager supports scalar logging states only."
                    )
                self._term_states[term_idx] = state

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Return curriculum states for debug inspection.

        Args:
            env_idx: Unused environment index retained for the manager interface.

        Returns:
            Curriculum term names and their scalar states.
        """
        del env_idx
        states = self._term_states.detach().cpu().tolist()
        return [(name, [states[index]]) for index, name in enumerate(self._term_names)]

    def _prepare_terms(self):
        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, StableCurriculumTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CurriculumTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            if isinstance(term_cfg.func, str):
                term_cfg.func = string_to_callable(term_cfg.func)
            if self._is_mask_term(term_cfg.func):
                mode = "mask"
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=3)
            else:
                mode = "legacy"
                self._resolve_legacy_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            self._term_modes.append(mode)

    @staticmethod
    def _is_mask_term(func) -> bool:
        """Return whether a term follows ``(env, env_mask, out, ...)``."""
        func_static = func.__call__ if inspect.isclass(func) else func
        parameters = list(inspect.signature(func_static).parameters)
        if inspect.isclass(func):
            parameters = parameters[1:]
        return parameters[:3] == ["env", "env_mask", "out"]

    def _resolve_legacy_term_cfg(self, term_name: str, term_cfg: StableCurriculumTermCfg) -> None:
        """Validate and initialize a stable ``(env, env_ids, ...)`` curriculum term."""
        if not callable(term_cfg.func):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")
        is_class = inspect.isclass(term_cfg.func)
        func_static = term_cfg.func.__call__ if is_class else term_cfg.func
        min_argc = 3 if is_class else 2
        signature = inspect.signature(func_static)
        parameters = list(signature.parameters.values())
        for parameter in parameters[min_argc:]:
            if (
                parameter.default is not inspect.Parameter.empty
                and parameter.name not in term_cfg.params
                and hasattr(parameter.default, "__dataclass_fields__")
            ):
                term_cfg.params[parameter.name] = parameter.default.copy()
        required = {
            parameter.name for parameter in parameters[min_argc:] if parameter.default is inspect.Parameter.empty
        }
        accepted = {parameter.name for parameter in parameters[min_argc:]}
        provided = set(term_cfg.params)
        if not required.issubset(provided) or not provided.issubset(accepted):
            raise ValueError(
                f"The legacy curriculum term '{term_name}' expects parameters {sorted(accepted)},"
                f" but received {sorted(provided)}."
            )
        graph_cache = getattr(self._env, "_warp_graph_cache", None)
        if graph_cache is not None:
            graph_cache.register_capturability(type(self).__name__, False)
        if self._env.sim.is_playing():
            self._process_term_cfg_at_play(term_name, term_cfg)

    @staticmethod
    def _compact_legacy_env_ids(env_mask: wp.array(dtype=wp.bool)) -> torch.Tensor:
        """Materialize compact Torch IDs at the legacy curriculum boundary."""
        return wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
