# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_NUMERIC_SUFFIX = re.compile(r"(\d+)(?=\.pt$)")


@dataclass
class CheckpointEntry:
    """Observed state of one direct-child checkpoint file."""

    path: Path
    size: int
    mtime_ns: int
    stable_scans: int = 1
    ready: bool = False
    iteration: int | None = None
    error: str | None = None
    status: str = "waiting"

    @property
    def filename_iteration(self) -> int:
        match = _NUMERIC_SUFFIX.search(self.path.name)
        return int(match.group(1)) if match else -1

    @property
    def rank(self) -> tuple[int, int, int, str]:
        return (
            self.iteration if self.iteration is not None else -1,
            self.filename_iteration,
            self.mtime_ns,
            self.path.name,
        )


class CheckpointCatalog:
    """Watch completed direct ``*.pt`` children of an RSL-RL run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._entries: dict[Path, CheckpointEntry] = {}

    @property
    def entries(self) -> tuple[CheckpointEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.rank, reverse=True))

    def scan(self) -> tuple[CheckpointEntry, ...]:
        current: set[Path] = set()
        try:
            children = tuple(self.run_dir.iterdir())
        except OSError as exc:
            raise RuntimeError(f"Unable to scan policy debug folder {self.run_dir}: {exc}") from exc

        for path in children:
            if not path.is_file() or path.suffix != ".pt":
                continue
            current.add(path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entry = self._entries.get(path)
            signature = (stat.st_size, stat.st_mtime_ns)
            if entry is None:
                self._entries[path] = CheckpointEntry(path, *signature)
            elif signature == (entry.size, entry.mtime_ns):
                entry.stable_scans += 1
                if entry.stable_scans >= 2:
                    entry.ready = True
                    if entry.status == "waiting":
                        entry.status = "ready"
            else:
                entry.size, entry.mtime_ns = signature
                entry.stable_scans = 1
                entry.ready = False
                entry.error = None
                entry.status = "waiting"
                entry.iteration = None

        for deleted in set(self._entries) - current:
            del self._entries[deleted]
        return self.entries

    def rescan_from_scratch(self) -> tuple[CheckpointEntry, ...]:
        self._entries.clear()
        return self.scan()


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Trusted local PyTorch checkpoint and its validation metadata."""

    path: Path
    payload: dict[str, Any]
    iteration: int | None
    metadata: dict[str, Any]
    parameter_shapes: dict[str, tuple[int, ...]]


class CheckpointLoader:
    """Load trusted local RSL-RL checkpoints on CPU before activation."""

    def load(self, entry: CheckpointEntry) -> LoadedCheckpoint:
        import torch

        try:
            payload = torch.load(entry.path, map_location="cpu", weights_only=False)
        except Exception as exc:
            entry.error = f"Could not load checkpoint: {exc}"
            entry.status = "error"
            raise ValueError(entry.error) from exc
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint root must be a dictionary produced by RSL-RL")
        state_dicts = self._state_dicts(payload)
        if not state_dicts:
            message = (
                "Checkpoint has no supported RSL-RL model state "
                "(expected actor/critic, student/teacher, or model_state_dict)"
            )
            entry.error = message
            entry.status = "error"
            raise ValueError(message)
        raw_iteration = payload.get("iter", payload.get("iteration"))
        iteration = int(raw_iteration) if raw_iteration is not None else None
        metadata = payload.get("policy_debug", payload.get("metadata", payload.get("infos", {})))
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        grouped = len(state_dicts) > 1
        shapes = {}
        for group_name, group in state_dicts.items():
            for name, value in group.items():
                if hasattr(value, "shape"):
                    key = f"{group_name}.{name}" if grouped else name
                    shapes[key] = tuple(value.shape)
        entry.iteration = iteration
        entry.error = None
        entry.status = "loaded"
        return LoadedCheckpoint(entry.path, payload, iteration, metadata, shapes)

    @staticmethod
    def _state_dicts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for group_name in ("actor", "critic", "student", "teacher"):
            value = payload.get(f"{group_name}_state_dict")
            if isinstance(value, dict):
                groups[group_name] = value
        if groups:
            return groups
        value = payload.get("model_state_dict")
        return {"model": value} if isinstance(value, dict) else {}
