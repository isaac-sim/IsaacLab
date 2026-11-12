# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

import leapp
import torch
from leapp import Kind, TensorSemantics, annotate

from ..warp.proxy_array import ProxyArray
from .leapp_semantics import LeappTensorSemantics, resolve_semantics_axes


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
        kind = semantics_meta.kind
        # LEAPP 0.8 velocity helpers still require the old vx/wx component labels.
        constructor = (
            getattr(leapp, kind.name.lower())
            if isinstance(kind, Kind)
            and kind not in (Kind.FRAME_TWIST, Kind.FRAME_LINEAR_VELOCITY, Kind.FRAME_ANGULAR_VELOCITY)
            else partial(TensorSemantics, kind=kind)
        )
        kwargs = {
            "axes": resolve_semantics_axes(semantics_meta, real_data, astorch),
            "is_setpoint": semantics_meta.is_setpoint,
        }
        for field in ("reference", "expressed_in"):
            value = getattr(semantics_meta, field)
            if value is not None:
                kwargs[field] = value
        sem = constructor(input_name, astorch, **kwargs)
        sem.extra = build_state_connection(entity_name, property_name)
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
