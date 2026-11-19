# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a fixed buffer of accepted states using ordinary task callbacks."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
import torch

from .reset_state_buffer_cfg import ResetStateBufferCfg


class ResetStateBuffer:
    """Collect accepted task states independently of their sampling or application.

    The state representation is a nested dictionary of tensors with a leading row dimension,
    compatible with :meth:`isaaclab.scene.InteractiveScene.get_state`. Physical units follow
    the state fields supplied by the task. Capture scene states relative to environment origins
    when the same buffer will be shared across environments.

    Generation and validation run only during :meth:`collect`. They may use simulation to
    establish grasps or settle objects. At runtime, :meth:`get_state` only indexes stored tensors;
    the task restores them through its existing scene and controller APIs.

    Args:
        cfg: Collection size, attempt limit, and seed.
    """

    State: TypeAlias = dict[str, "torch.Tensor | ResetStateBuffer.State"]

    def __init__(self, cfg: ResetStateBufferCfg) -> None:
        if any(type(value) is not int or value < 1 for value in (cfg.size, cfg.max_batches)):
            raise ValueError("size and max_batches must be positive integers.")
        if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.cfg = cfg
        self._state: ResetStateBuffer.State = {}
        self._group_names: tuple[str, ...] = ()
        self._group_ids = torch.empty(0, dtype=torch.long)
        self.collection_report: dict[str, dict[str, int]] = {}
        """Attempted, accepted, and rejected counts for the most recent successful collection."""

    @property
    def num_states(self) -> int:
        """Number of collected states."""
        return self._group_ids.numel()

    @property
    def group_names(self) -> tuple[str, ...]:
        """Generator names, in collection order."""
        return self._group_names

    @property
    def group_ids(self) -> torch.Tensor:
        """Copy of each stored row's generator index, shape [num_states]."""
        return self._group_ids.clone()

    @property
    def state(self) -> ResetStateBuffer.State:
        """Copy of the collected tensor tree, preserving the captured fields' units.

        Use :meth:`get_state` to copy only selected rows during reset.
        """
        return _map_state(self._state, lambda value: value.clone())

    def collect(
        self,
        env_ids: torch.Tensor,
        generate: Callable[[torch.Tensor], torch.Tensor | None]
        | Mapping[str, Callable[[torch.Tensor], torch.Tensor | None]],
        capture: Callable[[torch.Tensor], ResetStateBuffer.State],
        checks: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        """Generate, check, and capture a complete buffer before training starts.

        Each generator receives all candidate environment indices on every attempt. Its optional
        boolean return mask reports failed construction, such as an unsuccessful IK solve. Named
        checks return additional acceptance masks. Only candidates passing every condition are
        captured. Named generators receive equal quotas, differing by at most one row.

        The callbacks own simulation synchronization and any settling; collection does not step
        the simulation or advance the environment's episode bookkeeping itself.
        Failed collection preserves the previous catalog, group metadata, and report. An exhausted
        generation budget includes the failed attempt's report in its exception.

        Args:
            env_ids: Candidate environment indices, shape [N], on the simulation device.
            generate: A generator callback or a non-empty mapping of generator names to callbacks.
                The optional returned mask has shape [N].
            capture: Callback returning a tensor tree with one leading row per supplied index.
            checks: Named validity callbacks returning boolean tensors of shape [N]. The name
                ``generator`` is reserved for the generator's own validity mask.

        Raises:
            RuntimeError: A generator cannot fill its quota within :attr:`ResetStateBufferCfg.max_batches`.
            ValueError: There are no candidate environments, or the buffer cannot cover every group.
        """
        generators = dict(generate) if isinstance(generate, Mapping) else {"default": generate}
        if env_ids.numel() == 0 or not generators or self.cfg.size < len(generators):
            raise ValueError("Collection needs candidate environments and at least one buffer row per generator.")
        checks = checks or {}
        if "generator" in checks:
            raise ValueError("The check name 'generator' is reserved for generator validity.")
        state: ResetStateBuffer.State = {}
        group_ids = torch.empty(self.cfg.size, dtype=torch.long, device=env_ids.device)
        collected = 0
        collection_report: dict[str, dict[str, int]] = {}
        devices = [env_ids.device.index] if env_ids.is_cuda else []
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.cfg.seed)
                if env_ids.is_cuda:
                    with torch.cuda.device(env_ids.device):
                        torch.cuda.manual_seed(self.cfg.seed)
                np.random.seed(self.cfg.seed)
                random.seed(self.cfg.seed)
                for group, (name, generator) in enumerate(generators.items()):
                    quota = self.cfg.size // len(generators) + int(group < self.cfg.size % len(generators))
                    accepted = 0
                    report = collection_report[name] = {"attempted": 0, "accepted": 0}
                    for _ in range(self.cfg.max_batches):
                        report["attempted"] += env_ids.numel()
                        reported = generator(env_ids)
                        masks = {"generator": reported} if reported is not None else {}
                        masks.update({key: check(env_ids) for key, check in checks.items()})
                        valid = torch.ones_like(env_ids, dtype=torch.bool)
                        for key, mask in masks.items():
                            if mask.shape != env_ids.shape or mask.dtype != torch.bool:
                                raise ValueError(
                                    f"{name}/{key} must return a boolean mask shaped {tuple(env_ids.shape)}."
                                )
                            rejected = int((~mask).sum())
                            report[f"rejected/{key}"] = report.get(f"rejected/{key}", 0) + rejected
                            valid &= mask
                        selected = env_ids[valid][: quota - accepted]
                        if selected.numel():
                            captured = capture(selected)
                            if not _validate_state(captured, selected.numel(), state if state else None):
                                raise ValueError("capture must return a non-empty tree of batched tensors.")
                            if not state:
                                state = _map_state(
                                    captured, lambda value: value.new_empty((self.cfg.size, *value.shape[1:]))
                                )
                            _copy_state(state, captured, collected)
                            group_ids[collected : collected + selected.numel()] = group
                            collected += selected.numel()
                            accepted += selected.numel()
                            report["accepted"] = accepted
                        if accepted == quota:
                            break
                    if accepted < quota:
                        raise RuntimeError(
                            f"Reset group '{name}' collected {accepted}/{quota} states after "
                            f"{self.cfg.max_batches} batches. Collection report: {collection_report}"
                        )
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self._state, self._group_ids, self._group_names, self.collection_report = (
            state,
            group_ids,
            tuple(generators),
            collection_report,
        )

    def get_state(self, row_ids: torch.Tensor) -> ResetStateBuffer.State:
        """Return the selected rows of the captured state tree.

        Args:
            row_ids: Buffer row indices, shape [N], on the buffer device.

        Returns:
            Copied tensor tree with a leading dimension of N and the captured fields' units.
        """
        return _map_state(self._state, lambda value: value[row_ids])


def _validate_state(
    state: ResetStateBuffer.State,
    rows: int,
    reference: ResetStateBuffer.State | None = None,
    path: str = "",
) -> int:
    if not isinstance(state, dict) or reference is not None and state.keys() != reference.keys():
        raise ValueError(f"Captured state structure changed at '{path}'.")
    leaves = 0
    for name, value in state.items():
        key = f"{path}/{name}"
        expected = reference[name] if reference is not None else None
        if isinstance(value, dict):
            if expected is not None and not isinstance(expected, dict):
                raise ValueError(f"Captured state structure changed at '{key}'.")
            leaves += _validate_state(value, rows, expected, key)
        else:
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != rows:
                raise ValueError(f"Captured state '{key}' needs a leading dimension of {rows}.")
            if expected is not None and (
                not isinstance(expected, torch.Tensor)
                or (value.shape[1:], value.dtype, value.device) != (expected.shape[1:], expected.dtype, expected.device)
            ):
                raise ValueError(f"Captured tensor shape, dtype, or device changed at '{key}'.")
            leaves += 1
    return leaves


def _map_state(
    state: ResetStateBuffer.State, function: Callable[[torch.Tensor], torch.Tensor]
) -> ResetStateBuffer.State:
    return {
        key: _map_state(value, function) if isinstance(value, dict) else function(value) for key, value in state.items()
    }


def _copy_state(destination: ResetStateBuffer.State, source: ResetStateBuffer.State, start: int) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            _copy_state(destination[key], value, start)
        else:
            destination[key][start : start + value.shape[0]].copy_(value.detach())
