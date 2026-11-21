# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset from a bank of start states: the MDP2 event over :mod:`isaaclab.envs.reset`.

A :class:`ResetBank` composes three parts that are chosen independently:

- the states, a :class:`~isaaclab.envs.reset.ResetStateBuffer` that a fill builds once at startup, and that recording
  extends with states visited during training;
- the choice, a :class:`~isaaclab.envs.reset.ResetStateSampler` that credits each ended episode to the state it
  started from and draws the next one from the use's random stream, unless a slot is requested for replay;
- the restore, which gathers the drawn states into the event's ``outputs``.

The outputs are the state: any writable fields, such as ``robot.joint_pos`` and ``object.root_link_pose_w``, and
commands, whose restored value then stands over the resample. A fill is a callable ``fill(ctx, shared) -> {output:
tensor}`` that returns, per output, ``(num_worlds, size, ...)`` states for a bank per world, or ``(size, ...)`` for one
bank that every world shares. :func:`reset_bank` returns the bank of a use, to inspect, replay, or save it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import warp as wp

from isaaclab.envs.reset import BetaSamplingScoreCfg, ResetStateBuffer, ResetStateSampler, ResetStateSamplerCfg
from isaaclab.utils.warp import ProxyArray

from ..builder import MDPBuilder
from ..compiled import MDPState
from ..config import TaskCfg
from ..eager import TermContext
from ..runtime import evaluate, reset, startup
from ..terms import Implementation, Term


@dataclass(frozen=True)
class RecordStates:
    """Record states that the worlds visit during training into slots of their own bank.

    Each step, a world that did not terminate records its outputs with ``probability``, and only where the
    ``record_when`` input, if bound, is above 0.5. Records go to ``slots`` extra slots per world, the oldest replaced
    first; they start as copies of the filled states. A replaced slot forgets its outcomes and starts at the success
    rate that the sampler's first score targets, so that new states are tried.
    """

    slots: int
    probability: float


@dataclass(frozen=True)
class FillFromStates:
    """Fill the bank with given states, such as demonstrations.

    ``states`` maps each output to ``(num_worlds, size, ...)`` states, or ``(size, ...)`` for a shared bank, in the
    frame the event restores: poses relative to the environment origins when the use binds ``env_origins``.
    """

    states: Mapping[str, torch.Tensor]

    def __call__(self, ctx: TermContext, shared: bool) -> dict[str, torch.Tensor]:
        return {name: value.to(ctx.rng.device) for name, value in self.states.items()}


@dataclass(frozen=True)
class FillFromFile:
    """Fill the bank with states that :meth:`ResetBank.save` wrote, such as a separate collection run."""

    path: str | Path

    def __call__(self, ctx: TermContext, shared: bool) -> dict[str, torch.Tensor]:
        saved = torch.load(self.path, map_location=ctx.rng.device)
        if saved["shared"] != shared or not shared and saved["num_worlds"] != ctx.num_worlds:
            kind = "a shared bank" if saved["shared"] else f"banks of {saved['num_worlds']} worlds"
            wanted = "a shared bank" if shared else f"banks of {ctx.num_worlds} worlds"
            raise ValueError(f"{self.path} holds {kind}, but the event needs {wanted}.")
        return saved["states"]


class Candidates:
    """A candidates task bound to a scene, rolled to propose start states.

    Each :meth:`roll` resets every world through the task's reset events, commits them to the scene, and evaluates
    its terminations, which reject states. The scene then holds the proposals.
    """

    def __init__(self, task: TaskCfg, scene_data: Any, device: str, seed: int = 0) -> None:
        builder = MDPBuilder(scene_data.schema)
        builder.add_task(task)
        self.mdp = builder.finalize(device=device, backend="warp")
        self.state, self.transition = self.mdp.state(scene_data, seed=seed), self.mdp.transition()
        startup(self.mdp, self.state)
        self._mask = ProxyArray(wp.ones(self.mdp.num_worlds, dtype=wp.bool, device=device))
        self._commit = functools.partial(scene_data.commit, self._mask.warp, self.mdp.reset_writes)

    def roll(self) -> torch.Tensor:
        """Propose a state in every world and return which ones no termination rejects, ``(num_worlds,)``."""
        reset(self.mdp, self.state, self._mask, self._commit)
        self._commit()
        evaluate(self.mdp, self.state, self.transition)
        return ~self.transition.terminated.torch

    def field(self, selector: Any) -> torch.Tensor:
        """Return the proposed values of a field of the MDP, such as a command that the candidates sample."""
        if selector not in self.state.arrays:
            raise ValueError(f"The candidates task must define '{selector}', which the event restores.")
        return self.state.arrays[selector].torch


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
        candidates = Candidates(self.candidates, ctx.scene_data, str(device), seed=int(ctx.rng[0]))
        descriptor = candidates.transition.observations.get("descriptor")
        size = self.bank_size if descriptor is None else round(self.bank_size * self.oversample_factor)
        if self.bank_size < 1 or size < self.bank_size:
            raise ValueError("The bank must harvest at least as many states as it keeps, and keep at least one.")
        # Scene outputs are read from the scene the candidates write; commands from the candidates' own state.
        fields = {
            name: value if ctx.selectors[name].scene else candidates.field(ctx.selectors[name])
            for name, value in ctx.outputs.items()
        }
        if descriptor is not None:
            fields["descriptor"] = descriptor.torch
        # Root poses are kept relative to the environment origins when the use binds them.
        origins = ctx.inputs.get("env_origins")
        poses = {name for name in ctx.outputs if origins is not None and _warp_view(ctx, name).dtype == wp.transformf}

        rows = (num_worlds, size) if not shared else (size,)
        harvest = {name: value.new_zeros(*rows, *value.shape[1:]) for name, value in fields.items()}
        filled = torch.zeros(num_worlds if not shared else 1, dtype=torch.long, device=device)
        worlds = torch.arange(num_worlds, device=device)
        for _ in range(self.max_rolls):
            accepted = candidates.roll()
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


class ResetBank:
    """The start states of a reset event, the sampler that chooses among them, and the worlds' current choices.

    Slots are numbered within a world's bank, or within the shared bank. Get it with :func:`reset_bank`. Reading
    allocates and synchronizes; :meth:`replay` and the event's hooks do not, and replay inside the step's CUDA graph.
    """

    def __init__(
        self,
        states: Mapping[str, torch.Tensor],
        outputs: Mapping[str, wp.array],
        sampler: ResetStateSamplerCfg,
        *,
        shared: bool,
        origins: wp.array | None,
        nominal_probability: float,
        record: RecordStates | None,
    ) -> None:
        first = next(iter(outputs.values()))
        num_worlds, device = first.shape[0], first.device
        filled = next(iter(states.values())).shape[0 if shared else 1]
        recorded = 0 if record is None else record.slots
        self.num_worlds, self.shared, self.size = num_worlds, shared, filled + recorded
        self._filled, self._record, self._nominal = filled, record, float(nominal_probability)

        # Recorded slots start as copies of the filled ones, so every slot can be drawn from the start.
        cycle = torch.arange(self.size, device=states[next(iter(states))].device) % filled
        rows = {}
        for name, value in states.items():
            value = value[cycle] if shared else value[:, cycle]
            rows[name] = wp.from_torch(
                value.reshape(-1, *value.shape[1 if shared else 2 :]).contiguous(), dtype=outputs[name].dtype
            )
        self.buffer = ResetStateBuffer(rows)
        num_groups = 1 if shared else num_worlds
        groups = None if shared else wp.array(range(num_worlds), dtype=wp.int32, device=device)
        self.sampler = ResetStateSampler(
            sampler, num_worlds, self.size * num_groups, device, group_ids=groups, num_groups=num_groups
        )
        scores = [score for score in sampler.scores if isinstance(score, BetaSamplingScoreCfg)]
        self._prior = scores[0].target if scores else 0.0

        self._outputs, self._origins = dict(outputs), origins
        self._poses = (
            [value for value in outputs.values() if value.dtype == wp.transformf] if origins is not None else []
        )
        zeros = functools.partial(wp.zeros, num_worlds, device=device)
        self._requests = wp.full(num_worlds, -1, dtype=wp.int32, device=device)
        self._restored, self._credit, self._valid = zeros(dtype=wp.bool), zeros(dtype=wp.bool), zeros(dtype=wp.bool)
        self._recording, self._record_rows = zeros(dtype=wp.bool), zeros(dtype=wp.int32)
        self._cursor = zeros(dtype=wp.int32)
        # Recording stores poses relative to the origins, from copies of the outputs.
        self._record_values = {
            name: wp.empty_like(value) if any(value is pose for pose in self._poses) else value
            for name, value in self._outputs.items()
        }

    @property
    def filled(self) -> int:
        """Slots that the fill wrote, first in each bank; the others record."""
        return self._filled

    @property
    def states(self) -> dict[str, torch.Tensor]:
        """Copies of the stored states, ``(num_worlds, size, ...)`` per output, or ``(size, ...)`` if shared. Poses
        are relative to the environment origins when the use binds them."""
        rows = {name: value.torch for name, value in self.buffer.states.items()}
        return (
            rows
            if self.shared
            else {name: value.view(self.num_worlds, self.size, *value.shape[1:]) for name, value in rows.items()}
        )

    @property
    def success_rate(self) -> torch.Tensor:
        """Recent success rate of each slot, ``(num_worlds, size)``, or ``(size,)`` if shared."""
        return self._per_slot(self.sampler.success_rate.torch)

    @property
    def episode_counts(self) -> torch.Tensor:
        """Episodes that started from each slot since it was filled, ``(num_worlds, size)`` or ``(size,)``."""
        return self._per_slot(self.sampler.episode_counts.torch)

    @property
    def probabilities(self) -> torch.Tensor:
        """Probability that the next draw picks each slot, ``(num_worlds, size)`` or ``(size,)``."""
        return self._per_slot(self.sampler.probabilities.torch)

    @property
    def slots(self) -> torch.Tensor:
        """The slot each world's episode started from, ``(num_worlds,)``; -1 for a nominal start or before the first
        reset."""
        indices = self.sampler.indices.torch.long()
        slots = indices if self.shared else indices - torch.arange(self.num_worlds, device=indices.device) * self.size
        return torch.where(wp.to_torch(self._restored) & (indices >= 0), slots, -1)

    def replay(self, slots: torch.Tensor | None) -> None:
        """Restart the worlds from the given slots at their next resets, ``(num_worlds,)``, -1 to draw as usual.

        The requests hold until replaced; None clears them. A slot outside the bank leaves the world's start unchanged
        and sets the sampler's ``invalid_requests``. Replayed episodes are credited like drawn ones.
        """
        requests = wp.to_torch(self._requests)
        if slots is None:
            requests.fill_(-1)
            return
        slots = slots.to(requests.device, torch.int32)
        if not self.shared:
            in_bank = (slots >= 0) & (slots < self.size)
            rows = torch.arange(self.num_worlds, device=slots.device, dtype=torch.int32) * self.size + slots
            slots = torch.where(in_bank, rows, torch.where(slots < 0, -1, self.size * self.num_worlds))
        requests.copy_(slots)

    def discard_outcomes(self) -> None:
        """Credit none of the worlds' current episodes, such as before resetting them by hand mid-episode."""
        self._credit.zero_()

    def save(self, path: str | Path) -> None:
        """Save the states, to fill a later run's bank with :class:`FillFromFile`."""
        torch.save({"states": self.states, "shared": self.shared, "num_worlds": self.num_worlds}, path)

    def restore(self, mask: wp.array, rng: wp.array, succeeded: wp.array | None) -> None:
        """Credit the ended episodes, choose the next starts, and write the restored ones to the outputs."""
        wp.launch(
            _choose_starts,
            mask.shape,
            [mask, rng, self._requests, self._nominal, self._credit, self._valid, self._restored],
            device=mask.device,
        )
        self.sampler.reset(mask, success=succeeded, valid=self._valid, state_ids=self._requests, rng=rng)
        self.buffer.get_state(self.sampler.indices, out=self._outputs, mask=self._restored)
        for pose in self._poses:
            wp.launch(_shift_poses, mask.shape, [self._restored, self._origins, pose, pose, 1.0], device=mask.device)

    def record(self, rng: wp.array, terminated: wp.array, record_when: wp.array | None) -> None:
        """Record the states of the worlds selected this step into their oldest recorded slots."""
        wp.launch(
            _choose_records,
            self.num_worlds,
            [
                rng,
                terminated,
                record_when,
                self._record.probability,
                self._filled,
                self._record.slots,
                self.size,
                self.sampler.indices.warp,
                self._cursor,
                self._credit,
                self._recording,
                self._record_rows,
            ],
            device=rng.device,
        )
        for name, pose in self._outputs.items():
            if self._record_values[name] is not pose:
                wp.launch(
                    _shift_poses,
                    self.num_worlds,
                    [self._recording, self._origins, pose, self._record_values[name], -1.0],
                    device=rng.device,
                )
        self.sampler.forget(self._record_rows, self._recording, success_rate=self._prior)
        self.buffer.set_state(self._record_rows, self._record_values, mask=self._recording)

    def _per_slot(self, values: torch.Tensor) -> torch.Tensor:
        return values.clone() if self.shared else values.view(self.num_worlds, self.size).clone()


def reset_bank(state: MDPState, use: str = "events.reset") -> ResetBank:
    """Return the bank of a :data:`reset_from_bank` use, by its name in the schedule."""
    cache = state.caches.get(use, {})
    if "bank" not in cache:
        raise KeyError(f"'{use}' is not a reset_from_bank use. Uses with a cache: {sorted(state.caches)}.")
    return cache["bank"]


def _init(
    ctx: TermContext,
    *,
    fill: Callable[[TermContext, bool], dict[str, torch.Tensor]],
    sampler: ResetStateSamplerCfg | None = None,
    shared: bool = False,
    nominal_probability: float = 0.0,
    record: RecordStates | None = None,
) -> None:
    """Fill the bank once, then build its sampler and bind the outputs it restores."""
    outputs = {name: _warp_view(ctx, name) for name in ctx.outputs}
    has_poses = any(value.dtype == wp.transformf for value in outputs.values())
    if shared and "env_origins" not in ctx.inputs and has_poses:
        raise ValueError("A shared bank of poses needs the env_origins input, so that poses transfer across worlds.")
    if record is not None and (shared or "terminated" not in ctx.inputs or record.slots < 1):
        raise ValueError(
            "Recording needs a bank per world, at least one slot, and the terminated input (mdp2.terminated), so"
            " that states which end the episode are not recorded."
        )
    if not 0.0 <= nominal_probability <= 1.0:
        raise ValueError(f"nominal_probability must be in [0, 1], got {nominal_probability}.")
    states = fill(ctx, shared)
    if states.keys() != outputs.keys():
        raise ValueError(f"The fill returned states for {sorted(states)}, but the event writes {sorted(outputs)}.")
    leading = 1 if shared else 2
    if (
        len({value.shape[leading - 1] for value in states.values()}) != 1
        or not shared
        and any(value.shape[0] != ctx.num_worlds for value in states.values())
    ):
        raise ValueError("Every output needs the same number of states, per world unless the bank is shared.")

    origins = ctx.inputs.get("env_origins")
    bank = ResetBank(
        states,
        outputs,
        sampler if sampler is not None else ResetStateSamplerCfg(history_length=1, scores=[]),
        shared=shared,
        origins=None if origins is None else wp.from_torch(origins.contiguous(), dtype=wp.vec3f),
        nominal_probability=nominal_probability,
        record=record,
    )
    ctx.cache["bank"] = bank
    # Load every kernel and the sampler's sort and scan scratch before a graph captures the hooks: a step and a reset
    # of no worlds.
    nothing = wp.zeros(ctx.num_worlds, dtype=wp.bool, device=bank.sampler.device)
    rng, succeeded = _as_warp(ctx.rng), ctx.inputs.get("succeeded")
    bank.restore(nothing, rng, None if succeeded is None else _as_warp(succeeded))
    if record is not None:
        when = ctx.inputs.get("record_when")
        bank.record(rng, wp.ones_like(nothing), None if when is None else _as_warp(when))


def _restore(mask: Any, cache: dict, rng: Any, succeeded: Any = None, **_: Any) -> None:
    cache["bank"].restore(_as_warp(mask), _as_warp(rng), None if succeeded is None else _as_warp(succeeded))


def _record(cache: dict, rng: Any, terminated: Any = None, record_when: Any = None, **_: Any) -> None:
    bank: ResetBank = cache["bank"]
    if bank._record is not None:
        bank.record(_as_warp(rng), _as_warp(terminated), None if record_when is None else _as_warp(record_when))


_hooks = Implementation(compute=_restore, update=_record)
reset_from_bank = Term(torch=_hooks, warp=_hooks, random=True, init=_init)
"""Restart the reset worlds from start states drawn from a bank, the MDP2 counterpart of MDP1's lift
``conditional_reset``.

Params:

- ``fill``: a callable that returns the states, such as :class:`FillFromCandidates` or :class:`FillFromFile`;
- ``sampler``: a :class:`~isaaclab.envs.reset.ResetStateSamplerCfg` that weighs the states by their recent success,
  or None to draw uniformly;
- ``shared``: whether every world draws from one bank rather than its own. Share a bank only between worlds whose
  states fit each other: a state banked with one object spawned does not fit a world with another;
- ``nominal_probability``: the fraction of resets that keep what the earlier reset events wrote. Those episodes are
  not credited to any slot;
- ``record``: :class:`RecordStates`, to bank states visited during training.

Inputs: ``succeeded``, the ended episode's success, credited to the slot it started from when bound;
``env_origins``, which stores the poses relative to the environment origins; ``terminated`` (``mdp2.terminated``),
required to record; and ``record_when``, a quantity above 0.5 where a state may be recorded.

Place it after the other reset events: it overwrites the outputs they wrote. The bank and the sampler's history persist
across episodes; :func:`reset_bank` returns them.
"""


@wp.kernel
def _choose_starts(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.int32),
    requests: wp.array(dtype=wp.int32),
    nominal_probability: float,
    credit: wp.array(dtype=wp.bool),
    valid: wp.array(dtype=wp.bool),
    restored: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    if not mask[world]:
        return
    # The ended episode counts if it started from a slot that was not recorded over since.
    valid[world] = credit[world]
    restore = requests[world] >= 0 or wp.randf(wp.rand_init(rng[world], 1)) >= nominal_probability
    restored[world] = restore
    credit[world] = restore


@wp.kernel
def _choose_records(
    rng: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.bool),
    record_when: wp.array(dtype=wp.float32),
    probability: float,
    first: int,
    slots: int,
    size: int,
    indices: wp.array(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),
    credit: wp.array(dtype=wp.bool),
    recording: wp.array(dtype=wp.bool),
    rows: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    record = not terminated[world] and wp.randf(wp.rand_init(rng[world], 2)) < probability
    if record_when:
        record = record and record_when[world] > 0.5
    recording[world] = record
    if not record:
        return
    row = world * size + first + cursor[world]
    cursor[world] = (cursor[world] + 1) % slots
    rows[world] = row
    if indices[world] == row:
        credit[world] = False


@wp.kernel
def _shift_poses(
    mask: wp.array(dtype=wp.bool),
    origins: wp.array(dtype=wp.vec3f),
    pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.transformf),
    sign: float,
):
    world = wp.tid()
    if mask[world]:
        out[world] = wp.transformf(
            wp.transform_get_translation(pose[world]) + sign * origins[world], wp.transform_get_rotation(pose[world])
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
