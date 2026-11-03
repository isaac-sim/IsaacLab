# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from leapp import TensorSemantics, annotate

from ..warp.proxy_array import ProxyArray
from .leapp_semantics import LeappTensorSemantics, resolve_leapp_axes, resolve_leapp_element_names


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
        super().__init__(proxy_array.warp)
        astorch = super().torch
        sem = TensorSemantics(
            name=input_name,
            ref=astorch,
            kind=semantics_meta.kind,
            axes=resolve_leapp_axes(
                astorch, semantics_meta.kind, resolve_leapp_element_names(semantics_meta, real_data)
            ),
            is_setpoint=semantics_meta.is_setpoint,
            reference=semantics_meta.reference,
            expressed_in=semantics_meta.expressed_in,
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
