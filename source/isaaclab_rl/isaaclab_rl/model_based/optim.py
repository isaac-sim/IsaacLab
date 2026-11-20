# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Shared learner utilities."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict


def build_optimizers(
    modules: Sequence[nn.Module],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    device: str = "cpu",
    optimizer: str = "muon_adam",
) -> list[torch.optim.Optimizer]:
    """Build fused capturable Adam, optionally using Muon for intermediate Linear weights.

    ``muon_adam`` retains the pinned source optimizer; ``adam`` supports the paper
    recipe with a tensor-valued learning rate. Muon's rate is fixed after capture.
    """
    if optimizer not in {"adam", "muon_adam"}:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    hidden = [
        linear.weight
        for module in modules
        for seq in module.modules()
        if optimizer == "muon_adam" and isinstance(seq, nn.Sequential)
        for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
    adam = torch.optim.Adam(rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True)
    if not hidden:
        return [adam]
    return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def _optimize_step(
    optimizers: Sequence[torch.optim.Optimizer],
    loss: torch.Tensor,
    params: Iterable[torch.Tensor] | None = None,
    max_norm: float | None = None,
    clip_value: float = 0.0,
) -> torch.Tensor | None:
    """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Average before clipping so every rank applies the combined-batch update.
        gradients = [
            parameter.grad
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        flat = torch.cat([gradient.flatten() for gradient in gradients])
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
            offset += gradient.numel()
    grad_norm = None
    if params is not None and max_norm is not None:
        # Don't set error_if_nonfinite=True: it forces a per-step CPU<->GPU sync.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        if clip_value > 0:
            torch.nn.utils.clip_grad_value_(params, clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


def graph_state(modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]) -> list[torch.Tensor]:
    """What a captured update reads that Python can rebind: parameters, buffers,
    optimizer state and learning-rate tensors."""
    state: list[torch.Tensor] = []
    for module in modules:
        state += [*module.parameters(), *module.buffers()]
    for optimizer in optimizers:
        state += [t for s in optimizer.state.values() for t in s.values()]
        state += [group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])]
    return state


def _identity(state: list[torch.Tensor]) -> list[int]:
    return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
    """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
    a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
    capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
    its inputs updated in place. Python values it reads are fixed at capture, and
    each replay overwrites the outputs it returns. All CapturedUpdates on a device
    record on one stream into one memory pool and must replay in the order they
    were recorded. A replay raises if a tensor in ``state()`` moved; a learning rate
    that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

    def __init__(
        self,
        fn: Callable[[TensorDict], dict[str, torch.Tensor]],
        state: Callable[[], list[torch.Tensor]],
        warmup: int = 3,
    ):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        """Release graph-held NCCL communicators before process-group shutdown."""
        if self._graph is not None:
            self._graph.reset()
            self._graph = None

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        state = self._state()
        if not state[0].is_cuda:
            return self._fn(inputs)
        with torch.cuda.device(state[0].device):
            stream, pool = _capture_stream_and_pool(state[0].device)
            if self._inputs is None:
                self._inputs = inputs.clone()
            else:
                self._inputs.update_(inputs)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._inputs)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._inputs)
                self._graph, self._identity = graph, _identity(self._state())
            elif _identity(state) != self._identity:
                raise RuntimeError("a captured tensor moved")
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs
