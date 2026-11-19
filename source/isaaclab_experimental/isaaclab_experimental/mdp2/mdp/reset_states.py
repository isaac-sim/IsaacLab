# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset from a bank of start states: the MDP2 term over :mod:`isaaclab.envs.reset`.

The term composes three parts that are chosen independently:

- the states, a :class:`~isaaclab.envs.reset.ResetStateBuffer` that a fill builds once at startup;
- the choice, a :class:`~isaaclab.envs.reset.ResetStateSampler` that credits each ended episode to the state it
  started from and draws the next one, from the use's random stream;
- the restore, which gathers the drawn states into the event's ``outputs``.

The outputs are the state: any writable fields, such as ``robot.joint_pos`` and ``object.root_link_pose_w``. A fill is
a callable ``fill(ctx, shared) -> {output: tensor}`` that returns, per output, ``(num_worlds, bank_size, ...)`` states
for a bank per world, or ``(bank_size, ...)`` for one bank that every world shares.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from isaaclab.envs.reset import ResetStateBuffer, ResetStateSampler, ResetStateSamplerCfg
from isaaclab.utils.warp import ProxyArray

from ..builder import MDPBuilder
from ..config import TaskCfg
from ..eager import TermContext
from ..runtime import evaluate, reset, startup
from ..terms import Term


@dataclass(frozen=True)
class FillFromStates:
    """Fill the bank with given states, such as states saved from an earlier fill or demonstrations.

    ``states`` maps each output to ``(num_worlds, bank_size, ...)`` states, or ``(bank_size, ...)`` for a shared bank,
    in the frame the event restores: poses relative to the environment origins when the use binds ``env_origins``.
    """

    states: Mapping[str, torch.Tensor]

    def __call__(self, ctx: TermContext, shared: bool) -> dict[str, torch.Tensor]:
        return {name: value.to(ctx.rng.device) for name, value in self.states.items()}


@dataclass(frozen=True)
class FillFromCandidates:
    """Fill the bank by rejection sampling the reset events of a ``candidates`` task.

    The candidates' reset events draw a state in every world, and the states that none of its terminations reject are
    kept. If the task has a ``descriptor`` observation group, ``oversample_factor`` times as many states are harvested
    and the ones most spread over the descriptor are kept.
    """

    candidates: TaskCfg
    bank_size: int
    oversample_factor: float = 2.0
    max_rolls: int = 2000

    def __call__(self, ctx: TermContext, shared: bool) -> dict[str, torch.Tensor]:
        num_worlds, device = ctx.num_worlds, ctx.rng.device
        builder = MDPBuilder(ctx.scene_data.schema)
        builder.add_task(self.candidates)
        mdp = builder.finalize(device=str(device), backend="warp")
        state, transition = mdp.state(ctx.scene_data, seed=int(ctx.rng[0])), mdp.transition()
        startup(mdp, state)

        descriptor = transition.observations.get("descriptor")
        size = self.bank_size if descriptor is None else round(self.bank_size * self.oversample_factor)
        if self.bank_size < 1 or size < self.bank_size:
            raise ValueError("The bank must harvest at least as many states as it keeps, and keep at least one.")
        fields = dict(ctx.outputs)
        if descriptor is not None:
            fields["descriptor"] = descriptor.torch
        # Root poses are kept relative to the environment origins when the use binds them.
        origins = ctx.inputs.get("env_origins")
        poses = {name for name in ctx.outputs if origins is not None and _warp_view(ctx, name).dtype == wp.transformf}

        rows = (num_worlds, size) if not shared else (size,)
        harvest = {name: value.new_zeros(*rows, *value.shape[1:]) for name, value in fields.items()}
        filled = torch.zeros(num_worlds if not shared else 1, dtype=torch.long, device=device)
        worlds = torch.arange(num_worlds, device=device)
        mask = ProxyArray(wp.ones(num_worlds, dtype=wp.bool, device=str(device)))
        commit = functools.partial(ctx.scene_data.commit, mask.warp, mdp.reset_writes)
        for _ in range(self.max_rolls):
            reset(mdp, state, mask, commit)
            commit()
            evaluate(mdp, state, transition)
            accepted = ~transition.terminated.torch
            values = {name: _minus_origin(value, origins) if name in poses else value for name, value in fields.items()}
            if shared:
                take = worlds[accepted][: size - int(filled)]
                for name, value in values.items():
                    harvest[name][int(filled) : int(filled) + len(take)] = value[take]
                filled += len(take)
            else:
                take = accepted & (filled < size)
                slot = filled.clamp(max=size - 1)
                for name, value in values.items():
                    harvest[name][worlds, slot] = torch.where(_rows(take, value), value, harvest[name][worlds, slot])
                filled += take
            if (filled == size).all():
                break
        else:
            raise RuntimeError(
                f"The reset bank harvested fewer than {size} valid start states after {self.max_rolls} rolls of the"
                f" candidates ({filled.min().item()} in the worst {'bank' if shared else 'world'})."
            )

        if descriptor is not None:
            # Keep the most spread states, each descriptor axis rescaled to unit span so that all axes count.
            points = harvest.pop("descriptor")
            points = points if not shared else points[None]
            span = (points.amax(dim=(0, 1)) - points.amin(dim=(0, 1))).clamp(min=1e-6)
            kept = farthest_points(points / span, self.bank_size)
            if shared:
                harvest = {name: value[kept[0]] for name, value in harvest.items()}
            else:
                harvest = {name: value[worlds[:, None], kept] for name, value in harvest.items()}
        return harvest


def farthest_points(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return, for each batch of ``points`` of shape ``(batch, n, dim)``, the indices of ``count`` points spread as far
    apart as possible."""
    batch = torch.arange(len(points), device=points.device)
    chosen = torch.zeros(len(points), count, dtype=torch.long, device=points.device)
    nearest = torch.full(points.shape[:2], float("inf"), device=points.device)
    farthest = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for k in range(count):
        chosen[:, k] = farthest
        nearest = torch.minimum(nearest, (points - points[batch, farthest][:, None]).norm(dim=-1))
        farthest = nearest.argmax(dim=-1)
    return chosen


def _init(
    ctx: TermContext,
    *,
    fill: Callable[[TermContext, bool], dict[str, torch.Tensor]],
    sampler: ResetStateSamplerCfg | None = None,
    shared: bool = False,
) -> None:
    """Fill the bank once, then build the sampler over it and bind the outputs it restores."""
    outputs = {name: _warp_view(ctx, name) for name in ctx.outputs}
    if shared and "env_origins" not in ctx.inputs and any(value.dtype == wp.transformf for value in outputs.values()):
        raise ValueError("A shared bank of poses needs the env_origins input, so that poses transfer across worlds.")
    states = fill(ctx, shared)
    if states.keys() != outputs.keys():
        raise ValueError(f"The fill returned states for {sorted(states)}, but the event writes {sorted(outputs)}.")

    num_worlds = ctx.num_worlds
    leading = 1 if shared else 2
    bank_sizes = {value.shape[leading - 1] for value in states.values()}
    if len(bank_sizes) != 1 or (not shared and any(value.shape[0] != num_worlds for value in states.values())):
        raise ValueError("Every output needs the same number of states, per world unless the bank is shared.")
    bank_size = bank_sizes.pop()
    rows = {
        name: wp.from_torch(value.reshape(-1, *value.shape[leading:]).contiguous(), dtype=outputs[name].dtype)
        for name, value in states.items()
    }
    num_groups = 1 if shared else num_worlds
    sampler_cfg = sampler if sampler is not None else ResetStateSamplerCfg(history_length=1, scores=[])
    device = outputs[next(iter(outputs))].device
    groups = None if shared else wp.array(range(num_worlds), dtype=wp.int32, device=device)
    origins = ctx.inputs.get("env_origins")

    ctx.cache.update(
        bank=ResetStateBuffer(rows),
        sampler=ResetStateSampler(
            sampler_cfg, num_worlds, bank_size * num_groups, device, group_ids=groups, num_groups=num_groups
        ),
        outputs=outputs,
        poses=[value for value in outputs.values() if value.dtype == wp.transformf] if origins is not None else [],
        origins=None if origins is None else wp.from_torch(origins.contiguous(), dtype=wp.vec3f),
    )
    # Load every kernel and the sampler's sort and scan scratch before a graph captures the hook: a reset of no worlds.
    _reset_from_bank(
        wp.zeros(num_worlds, dtype=wp.bool, device=device), ctx.cache, ctx.rng, ctx.inputs.get("succeeded")
    )


def _reset_from_bank(mask: Any, cache: dict, rng: Any, succeeded: Any = None, **_: Any) -> None:
    """Credit the ended episodes, draw the next states, and write them to the masked worlds' outputs."""
    mask, rng = _as_warp(mask), _as_warp(rng)
    sampler: ResetStateSampler = cache["sampler"]
    success = None if succeeded is None else _as_warp(succeeded)
    sampler.reset(mask, success=success, rng=rng)
    cache["bank"].get_state(sampler.indices, out=cache["outputs"], mask=mask)
    for pose in cache["poses"]:
        wp.launch(_add_origin, mask.shape, [mask, cache["origins"], pose], device=pose.device)


reset_from_bank = Term(torch=_reset_from_bank, warp=_reset_from_bank, random=True, init=_init)
"""Restart the reset worlds from start states drawn from a bank, the MDP2 counterpart of MDP1's lift
``conditional_reset``.

Params: ``fill``, a callable that returns the states, such as :class:`FillFromCandidates`; ``sampler``, a
:class:`~isaaclab.envs.reset.ResetStateSamplerCfg` that weighs the states by their recent success, or None to draw
uniformly; and ``shared``, whether every world draws from one bank rather than its own. Share a bank only between
worlds whose states fit each other: a state banked with one object spawned does not fit a world with another.

Inputs: ``succeeded``, the ended episode's success, credited to the state it started from when bound; and
``env_origins``, which stores the poses relative to the environment origins so that a shared bank transfers across
worlds.

Place it after the other reset events: it overwrites the outputs they wrote. The bank and the sampler's history persist
across episodes.
"""


@wp.kernel
def _add_origin(mask: wp.array(dtype=wp.bool), origins: wp.array(dtype=wp.vec3f), pose: wp.array(dtype=wp.transformf)):
    world = wp.tid()
    if mask[world]:
        pose[world] = wp.transformf(
            wp.transform_get_translation(pose[world]) + origins[world], wp.transform_get_rotation(pose[world])
        )


def _warp_view(ctx: TermContext, name: str) -> wp.array:
    """Return a Warp view of an output, keeping the element type of scene fields such as ``wp.transformf``."""
    selector = ctx.selectors[name]
    if getattr(selector, "scene", False):
        return ctx.scene_data.data(selector.entity, selector.field).warp
    return wp.from_torch(ctx.outputs[name])


def _minus_origin(pose: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
    """Return root poses of shape ``(num_worlds, 7)`` relative to the environment origins."""
    relative = pose.clone()
    relative[:, :3] -= origins
    return relative


def _rows(take: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return take.view(-1, *[1] * (value.ndim - 1))


def _as_warp(value: Any) -> wp.array:
    return wp.from_torch(value) if isinstance(value, torch.Tensor) else value
