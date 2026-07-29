# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and validate ``odin.yaml``.

Named ``config_file`` rather than ``config`` because ``tools/odin/config/`` is a
sibling directory; the two would shadow each other in some import orders.

Every validation error raises :class:`OdinConfigError` naming the key path, so
the operator knows exactly which field to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ImageSpec",
    "OdinConfig",
    "OdinConfigError",
    "ResourcesSpec",
    "RetrySpec",
    "load_odin_config",
]

_PRIORITIES = frozenset({"HIGH", "NORMAL", "LOW"})


class OdinConfigError(ValueError):
    """Raised when ``odin.yaml`` fails validation."""


@dataclass(frozen=True)
class ImageSpec:
    """Registry coordinates for the benchmark image.

    Args:
        registry: Registry host and namespace, e.g. ``nvcr.io/nvidian``.
        repository: Repository name within the registry.
        pull_credential: OSMO credential name used to pull, or ``None`` for a
            public image.
    """

    registry: str
    repository: str
    pull_credential: str | None

    @property
    def registry_host(self) -> str:
        """Return just the registry host, e.g. ``nvcr.io`` from ``nvcr.io/nvidian``.

        OSMO's task ``credentials:`` block maps a credential name to the registry
        host it authenticates, not to the full namespaced path.
        """
        return self.registry.split("/", 1)[0]


@dataclass(frozen=True)
class ResourcesSpec:
    """Per-task resource request sent to OSMO.

    Args:
        cpu: CPU cores.
        gpu: GPU count.
        memory: Memory request in Kubernetes quantity form, e.g. ``64Gi``.
        storage: Ephemeral storage request, e.g. ``64Gi``.
        platform: OSMO platform selector, e.g. ``ovx-l40s``.
    """

    cpu: int
    gpu: int
    memory: str
    storage: str
    platform: str


@dataclass(frozen=True)
class RetrySpec:
    """OSMO ``exitActions`` exit-code ranges.

    Args:
        reschedule_codes: Codes OSMO reschedules onto another node.
        restart_codes: Codes OSMO restarts in place.
    """

    reschedule_codes: str
    restart_codes: str


@dataclass(frozen=True)
class OdinConfig:
    """Validated contents of ``odin.yaml``.

    Args:
        osmo_profile: OSMO profile name, exported as ``OSMO_PROFILE``.
        pool: Target OSMO pool.
        priority: Scheduling priority, one of ``HIGH``, ``NORMAL``, ``LOW``.
        image: Registry coordinates for the benchmark image.
        resources: Per-task resource request.
        exec_timeout_s: Workflow execution timeout [s]. Each task is wrapped in
            its own group, so the OSMO clock is per-group and this acts as a
            ceiling that catches wedged tasks rather than a per-task budget.
        queue_timeout_s: Workflow queue timeout [s].
        chunk_size: Maximum tasks per submitted OSMO workflow.
        results_uri: Bucket URI prefix under which results are published.
        retry: OSMO exit-action code ranges.
    """

    osmo_profile: str
    pool: str
    priority: str
    image: ImageSpec
    resources: ResourcesSpec
    exec_timeout_s: int
    queue_timeout_s: int
    chunk_size: int
    results_uri: str
    retry: RetrySpec


def _require(mapping: Any, key: str, ctx: str) -> Any:
    path = f"{ctx}.{key}" if ctx else key
    if not isinstance(mapping, dict) or key not in mapping:
        raise OdinConfigError(f"missing required field: {path}")
    return mapping[key]


def _require_str(mapping: Any, key: str, ctx: str) -> str:
    value = _require(mapping, key, ctx)
    path = f"{ctx}.{key}" if ctx else key
    if not isinstance(value, str) or not value:
        raise OdinConfigError(f"{path} must be a non-empty string")
    return value


def _require_int(mapping: Any, key: str, ctx: str) -> int:
    value = _require(mapping, key, ctx)
    path = f"{ctx}.{key}" if ctx else key
    # bool subclasses int, so "gpu: true" would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise OdinConfigError(f"{path} must be an integer")
    return value


def load_odin_config(path: Path) -> OdinConfig:
    """Load and validate ``odin.yaml``.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        OdinConfigError: If the file is unreadable, is not a mapping, or any
            field is missing or malformed.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise OdinConfigError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise OdinConfigError(f"could not parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise OdinConfigError(f"{path} must contain a top-level mapping")

    priority = _require_str(raw, "priority", "")
    if priority not in _PRIORITIES:
        raise OdinConfigError(f"priority must be one of {sorted(_PRIORITIES)}, got {priority!r}")

    image_raw = _require(raw, "image", "")
    resources_raw = _require(raw, "resources", "")
    retry_raw = raw.get("retry") or {}

    chunk_size = _require_int(raw, "chunk_size", "")
    if chunk_size < 1:
        raise OdinConfigError(f"chunk_size must be >= 1, got {chunk_size}")

    return OdinConfig(
        osmo_profile=_require_str(raw, "osmo_profile", ""),
        pool=_require_str(raw, "pool", ""),
        priority=priority,
        image=ImageSpec(
            registry=_require_str(image_raw, "registry", "image"),
            repository=_require_str(image_raw, "repository", "image"),
            pull_credential=image_raw.get("pull_credential"),
        ),
        resources=ResourcesSpec(
            cpu=_require_int(resources_raw, "cpu", "resources"),
            gpu=_require_int(resources_raw, "gpu", "resources"),
            memory=_require_str(resources_raw, "memory", "resources"),
            storage=_require_str(resources_raw, "storage", "resources"),
            platform=_require_str(resources_raw, "platform", "resources"),
        ),
        exec_timeout_s=_require_int(raw, "exec_timeout_s", ""),
        queue_timeout_s=_require_int(raw, "queue_timeout_s", ""),
        chunk_size=chunk_size,
        results_uri=_require_str(raw, "results_uri", ""),
        retry=RetrySpec(
            reschedule_codes=str(retry_raw.get("reschedule_codes") or ""),
            restart_codes=str(retry_raw.get("restart_codes") or ""),
        ),
    )
