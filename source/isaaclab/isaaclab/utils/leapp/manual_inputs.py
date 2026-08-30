# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for manually exposing computed tensors as LEAPP inputs."""

from __future__ import annotations

from typing import Any

import torch


class _ManualTensorView:
    """Small adapter matching the ``data.<property>.torch`` access pattern."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @property
    def torch(self) -> torch.Tensor:
        return self._tensor


def _cache_manual_data_proxy(data_proxy: Any, real_data: Any, property_name: str, tensor: torch.Tensor) -> None:
    """Cache a manual input tensor on a LEAPP data proxy."""
    manual_cache = object.__getattribute__(data_proxy, "_manual_cache")
    manual_cache[(id(real_data), property_name)] = _ManualTensorView(tensor)


def leapp_real_env(env: Any) -> Any:
    """Return the wrapped real environment when LEAPP passes an export proxy."""
    if type(env).__name__ == "_EnvProxy":
        return object.__getattribute__(env, "_real_env")
    return env


def leapp_input_tensor(
    env: Any,
    name: str,
    tensor: torch.Tensor,
    *,
    kind: Any | None = None,
    element_names: list[str] | list[list[str]] | None = None,
    connection: str | dict[str, str] | None = None,
    cache: tuple[str, str] | None = None,
) -> torch.Tensor:
    """Expose a tensor as a LEAPP input during export and return it unchanged otherwise.

    Args:
        env: Environment passed into the observation term.
        name: LEAPP input tensor name.
        tensor: Tensor to expose.
        kind: Optional LEAPP input kind used by deployment tooling.
        element_names: Optional element names for tensor dimensions.
        connection: Optional Isaac Lab connection string or metadata dict.
        cache: Optional ``(entity_name, property_name)`` tuple. When provided,
            the annotated tensor is cached as that LEAPP scene-data property read.

    Returns:
        The annotated tensor during LEAPP export, otherwise ``tensor`` unchanged.
    """
    if type(env).__name__ != "_EnvProxy":
        return tensor

    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    extra = None
    if isinstance(connection, str):
        extra = {"isaaclab_connection": connection}
    elif connection is not None:
        extra = connection

    if kind is None and element_names is None and extra is None:
        annotated = annotate.input_tensors(env.unwrapped.spec.id, {name: tensor})
    else:
        semantics = TensorSemantics(
            name=name,
            ref=tensor,
            kind=kind,
            element_names=element_names,
            extra=extra,
        )
        annotated = annotate.input_tensors(env.unwrapped.spec.id, semantics)

    if cache is not None:
        entity_name, property_name = cache
        real_env = leapp_real_env(env)
        real_data = real_env.scene[entity_name].data
        scene_proxy = object.__getattribute__(env, "_scene_proxy")
        proxy_cache = object.__getattribute__(scene_proxy, "_cache")
        tensor_view = _ManualTensorView(annotated)
        proxy_cache[(id(real_data), property_name)] = tensor_view

        proxied_entities = object.__getattribute__(scene_proxy, "_proxied")
        entity_proxy = proxied_entities.get(entity_name)
        if entity_proxy is not None:
            data_proxy = object.__getattribute__(entity_proxy, "_data_proxy")
            _cache_manual_data_proxy(data_proxy, real_data, property_name, annotated)

        action_manager = getattr(real_env, "action_manager", None)
        if action_manager is not None:
            for action_term in action_manager._terms.values():
                asset_proxy = getattr(action_term, "_asset", None)
                if type(asset_proxy).__name__ != "_ArticulationWriteProxy":
                    continue
                if object.__getattribute__(asset_proxy, "_entity_name") != entity_name:
                    continue
                data_proxy = object.__getattribute__(asset_proxy, "_data_proxy")
                _cache_manual_data_proxy(data_proxy, real_data, property_name, annotated)

    return annotated
