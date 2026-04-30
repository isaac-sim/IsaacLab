# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextlib import suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.utils.warp.proxy_array import ProxyArray

if TYPE_CHECKING:
    from .leapp_semantics import LeappTensorSemantics


class TracedProxyArray(ProxyArray):
    _traced_array: torch.Tensor

    def __init__(
        self,
        proxy_array: ProxyArray,
        *,
        input_name: str,
        semantics_meta: LeappTensorSemantics,
        real_data: Any,
        entity_name: str,
        property_name: str,
        task_name: str,
    ) -> None:
        from .leapp_semantics import resolve_leapp_element_names

        super().__init__(proxy_array.warp)
        astorch = super().torch
        sem = TensorSemantics(
            name=input_name,
            ref=astorch,
            kind=semantics_meta.kind,
            element_names=resolve_leapp_element_names(semantics_meta, real_data),
            extra=build_state_connection(entity_name, property_name),
        )
        annotated = annotate.input_tensors(task_name, sem)
        object.__setattr__(self, "_traced_array", annotated)

    @property
    def torch(self) -> torch.Tensor:
        return self._traced_array

    @property
    def warp(self) -> Any:
        raise AttributeError("warp arrays are not supported for leapp export")


def select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
    """Select element names using optional runtime indices."""
    if names is None:
        return None
    if indices is None or indices == slice(None):
        return list(names)
    if isinstance(indices, slice):
        return list(names[indices])
    with suppress(AttributeError):
        indices = indices.tolist()
    if isinstance(indices, (list, tuple)):
        return [names[int(index)] for index in indices]
    if isinstance(indices, int):
        return [names[indices]]
    return None


def ensure_env_spec_id(env, fallback_task_name: str = "policy") -> str:
    """Return ``env.unwrapped.spec.id``, creating a fallback spec when needed."""
    spec = getattr(env.unwrapped, "spec", None)
    if spec is None:
        env.unwrapped.spec = SimpleNamespace(id=fallback_task_name)
        return fallback_task_name

    task_name = getattr(spec, "id", None)
    if task_name is None:
        spec.id = fallback_task_name
        return fallback_task_name

    return task_name


# ══════════════════════════════════════════════════════════════════
# Connection Builders
# ══════════════════════════════════════════════════════════════════


def build_state_connection(entity_name: str, property_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a state property."""
    return {"isaaclab_connection": f"state:{entity_name}:{property_name}"}


def build_command_connection(command_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a command term."""
    return {"isaaclab_connection": f"command:{command_name}"}


def build_write_connection(entity_name: str, method_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for an articulation write target."""
    return {"isaaclab_connection": f"write:{entity_name}:{method_name}"}
