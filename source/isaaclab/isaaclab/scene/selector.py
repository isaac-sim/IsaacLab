# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime env/view selector index for heterogeneous scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from .env_view_index import EnvToViewMap, EnvViewIndex, build_layout, is_contiguous_slice
from .selector_cfg import SelectorTermCfg

if TYPE_CHECKING:
    from .selector_cfg import SelectorCfg


class Selector(EnvViewIndex):
    """Runtime selector index mapping names to env and asset-view indices."""

    def __init__(self, cfg: SelectorCfg, *, num_envs: int, device: str = "cpu"):
        super().__init__(cfg, num_envs=num_envs, device=device)
        self._selector_terms = {name: term for name, term in cfg.__dict__.items() if isinstance(term, SelectorTermCfg)}
        self._selector_assets: dict[str, tuple[str, ...]] = {}
        self._selector_env_ids: dict[str, torch.Tensor] = {}

    @property
    def selector_names(self) -> tuple[str, ...]:
        """Names of configured selector terms."""
        return tuple(self._selector_terms.keys())

    @property
    def selector_assets(self) -> Mapping[str, tuple[str, ...]]:
        """Read-only mapping from selector name to selected asset names."""
        return MappingProxyType(self._selector_assets)

    def resolve_terms(self, asset_cfgs: Mapping[str, object]) -> None:
        """Resolve configured selector terms against raw scene asset cfgs."""
        for name, term in self._selector_terms.items():
            result = term.func(asset_cfgs, **term.params)
            self._selector_assets[name] = self._normalize_asset_names(name, result)

    def apply_asset_env_ids(self, asset_env_ids: Mapping[str, torch.Tensor]) -> None:
        """Register sampled env ids for assets and configured selectors."""
        for asset_name, env_ids in asset_env_ids.items():
            self.register(asset_name, env_ids.to(device=self._device, dtype=torch.long))
            self.register_asset(asset_name, asset_name)

        for selector_name, asset_names in self._selector_assets.items():
            env_ids = self._union_asset_env_ids(asset_names)
            self._selector_env_ids[selector_name] = env_ids
            self.register(selector_name, env_ids)

    def get(self, selector: str | Sequence[str] | None, asset: str | None = None) -> EnvToViewMap:
        """Return env/view ids for a selector term.

        Args:
            selector: Selector term name. A sequence is accepted for legacy
                group-style callers and resolves to the union of those names.
            asset: Optional asset name whose local view ids should be produced.

        Returns:
            An :class:`EnvToViewMap` for the selected env rows and asset view.
        """
        if selector is None:
            return self._make_view(None, asset)
        if isinstance(selector, str):
            if selector in self._selector_env_ids:
                return self._make_selector_view(selector, asset)
            return super().get([selector], asset=asset)
        return super().get(list(selector), asset=asset)

    def _make_selector_view(self, selector: str, asset: str | None) -> EnvToViewMap:
        env_ids_tensor = self._selector_env_ids[selector]
        if asset is None:
            asset_names = self._selector_assets.get(selector, ())
            if len(asset_names) != 1:
                raise ValueError(
                    f"Selector '{selector}' spans {len(asset_names)} assets. Pass an asset name to resolve view ids."
                )
            asset = asset_names[0]
        else:
            asset_env_ids = self._get_asset_env_ids(asset)
            env_ids_tensor = env_ids_tensor[torch.isin(env_ids_tensor, asset_env_ids)]

        env_ids = self._compact_env_ids(env_ids_tensor)
        view_ids = self._view_ids_for_asset(asset, env_ids_tensor)
        return EnvToViewMap(
            env_ids=env_ids, view_ids=view_ids, layout=build_layout(env_ids_tensor.tolist(), self._device)
        )

    def _union_asset_env_ids(self, asset_names: tuple[str, ...]) -> torch.Tensor:
        tensors = [self._get_asset_env_ids(asset_name) for asset_name in asset_names]
        if not tensors:
            return torch.tensor([], dtype=torch.long, device=self._device)
        return torch.cat(tensors).unique().sort().values if len(tensors) > 1 else tensors[0]

    def _view_ids_for_asset(self, asset_name: str, env_ids: torch.Tensor) -> slice | torch.Tensor:
        if env_ids.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=self._device)
        asset_env_ids = self._get_asset_env_ids(asset_name)
        if asset_env_ids.numel() == 0:
            raise ValueError(f"Selector references envs where asset '{asset_name}' is absent: {env_ids.tolist()}.")
        positions = torch.searchsorted(asset_env_ids, env_ids)
        valid = (positions < asset_env_ids.numel()) & (
            asset_env_ids[positions.clamp(max=asset_env_ids.numel() - 1)] == env_ids
        )
        if not bool(valid.all()):
            missing = env_ids[~valid].tolist()
            raise ValueError(f"Selector references envs where asset '{asset_name}' is absent: {missing}.")
        if is_contiguous_slice(positions.tolist()):
            return slice(int(positions[0].item()), int(positions[-1].item()) + 1)
        return positions

    def _compact_env_ids(self, env_ids: torch.Tensor) -> slice | torch.Tensor:
        if env_ids.numel() == self.num_envs and bool(
            torch.equal(env_ids, torch.arange(self.num_envs, device=self._device))
        ):
            return slice(None)
        if env_ids.numel() > 0 and is_contiguous_slice(env_ids.tolist()):
            return slice(int(env_ids[0].item()), int(env_ids[-1].item()) + 1)
        return env_ids

    def _normalize_asset_names(self, selector_name: str, result: object) -> tuple[str, ...]:
        if isinstance(result, str):
            return (result,)
        if isinstance(result, Sequence):
            names = tuple(result)
            if all(isinstance(name, str) for name in names):
                return names
        raise TypeError(
            f"Selector term '{selector_name}' must return an asset name or a sequence of asset names, got {result!r}."
        )

    def __repr__(self) -> str:
        names = ", ".join(self._selector_env_ids)
        return f"Selector(num_envs={self._num_envs}, selectors=[{names or 'none'}])"
