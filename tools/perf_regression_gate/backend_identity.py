# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical backend identity helpers for the performance regression gate"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DEFAULT_PHYSICS_BACKEND = "physx"
_EMPTY_VALUES = {"", "none", "null"}
_DEFAULT_VALUES = _EMPTY_VALUES | {"default"}
_PHYSICS_PRESET_TO_BACKEND = {
    "physx": "physx",
    "newton": "newton",
    "newton_mjwarp": "newton",
}
_RENDER_PRESET_TOKENS = frozenset(
    {
        "newton_renderer",
        "ovrtx_renderer",
        "warp_renderer",
        "rtx_renderer",
    }
)
_KNOWN_PHYSICS_BACKENDS = ("physx", "newton")


@dataclass(frozen=True)
class BackendIdentity:
    physics_backend: str
    render_backend: str | None = None

    @property
    def backend_key(self) -> str:
        return make_backend_key(self.physics_backend, self.render_backend)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "physics_backend": self.physics_backend,
            "render_backend": self.render_backend,
            "backend_key": self.backend_key,
        }


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def normalize_physics_backend(value: Any, *, default: str | None = None) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return default
    lowered = cleaned.lower()
    if lowered in _DEFAULT_VALUES:
        return default
    return _PHYSICS_PRESET_TO_BACKEND.get(lowered, lowered)


def normalize_render_backend(value: Any) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    if lowered in _DEFAULT_VALUES:
        return None
    return lowered


def make_backend_key(physics_backend: str, render_backend: str | None = None) -> str:
    physics = normalize_physics_backend(physics_backend)
    if not physics:
        raise ValueError("physics_backend is required to build backend_key")
    render = normalize_render_backend(render_backend)
    return f"{physics}_{render}" if render else physics


def identity_from_parts(physics_backend: Any, render_backend: Any = None) -> BackendIdentity | None:
    physics = normalize_physics_backend(physics_backend)
    if not physics:
        return None
    return BackendIdentity(physics, normalize_render_backend(render_backend))


def split_backend_key(backend_key: Any) -> BackendIdentity | None:
    key = _clean(backend_key)
    if not key:
        return None
    for physics in _KNOWN_PHYSICS_BACKENDS:
        if key == physics:
            return BackendIdentity(physics, None)
        prefix = f"{physics}_"
        if key.startswith(prefix):
            return BackendIdentity(physics, normalize_render_backend(key[len(prefix) :]))
    if "_" in key:
        physics, render = key.split("_", 1)
        return identity_from_parts(physics, render)
    return BackendIdentity(key, None)


def preset_tokens(value: Any) -> frozenset[str]:
    cleaned = _clean(value)
    if not cleaned:
        return frozenset()
    tokens: set[str] = set()
    for chunk in cleaned.replace(";", ",").split(","):
        token = chunk.strip().lower()
        if token:
            tokens.add(token)
    return frozenset(tokens)


def identity_from_presets(presets: Any) -> BackendIdentity | None:
    tokens = preset_tokens(presets)
    if not tokens:
        return None
    physics = "newton" if "newton_mjwarp" in tokens else _DEFAULT_PHYSICS_BACKEND
    render = next((token for token in sorted(tokens) if token in _RENDER_PRESET_TOKENS), None)
    return BackendIdentity(physics, render)


def backend_identity_from_launch_config(config: dict[str, Any]) -> BackendIdentity | None:
    identity = identity_from_parts(config.get("physics_backend"), config.get("render_backend"))
    if identity is not None:
        return identity
    return split_backend_key(config.get("backend_key") or config.get("backend"))


def backend_identity_from_benchmark_info(info: dict[str, Any]) -> BackendIdentity | None:
    direct_key = info.get("backend_key") or info.get("backend")
    if direct_key:
        return split_backend_key(direct_key)

    identity = identity_from_parts(
        info.get("physics_backend") or info.get("physics"),
        info.get("render_backend") or info.get("render"),
    )
    if identity is not None:
        return identity

    return identity_from_presets(info.get("presets") or info.get("preset"))
