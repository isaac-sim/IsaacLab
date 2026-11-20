# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset to stored reset states: the MDP2 event over :mod:`isaaclab.envs.reset`.

A :class:`ResetStates` composes three parts that are chosen independently:

- the states, a :class:`~isaaclab.envs.reset.ResetStateBuffer` that a source builds once at startup, and that
  recording extends with states visited during training;
- the choice, a :class:`~isaaclab.envs.reset.ResetStateSampler` that credits each ended episode to the state it
  started from and draws the next one from the use's random stream, unless a slot is pinned;
- the restore, which gathers the drawn states into the event's ``outputs``.

The outputs are the state: any writable fields, such as ``robot.joint_pos`` and ``object.root_link_pose_w``, and
commands, whose restored value then stands over the resample. A source is a callable ``source(ctx, groups) ->
{output: tensor}`` that returns, per output, ``(num_groups, num_states, ...)`` states for each group of worlds, given
the group of each world as ``groups``, ``(num_worlds,)``, or ``(num_states, ...)`` for one set that every world shares,
given None. Reset states per world are a group per world. :func:`get_reset_states` returns the reset states of a use,
to inspect, pin, or save them.
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
class RecordVisited:
    """Record states that the worlds visit during training as extra reset states of their own.

    Each step, a world that did not terminate records its outputs with ``probability``, and only where the
    ``record_if`` input, if bound, is above 0.5. Records go to ``num_states`` extra slots per world, the oldest
    replaced first; they start as copies of the source's states. A replaced slot forgets its outcomes and starts at the
    success rate that the sampler's first score targets, so that new states are tried.
    """

    num_states: int
    probability: float


@dataclass(frozen=True)
class FromTensors:
    """Reset states given as tensors, such as demonstrations.

    ``states`` maps each output to ``(num_groups, num_states, ...)`` states, ``(num_worlds, num_states, ...)`` for
    states per world, or ``(num_states, ...)`` if shared, in the frame the event restores: poses relative to the
    environment origins when the use binds ``env_origins``.
    """

    states: Mapping[str, torch.Tensor]

    def __call__(self, ctx: TermContext, groups: torch.Tensor | None) -> dict[str, torch.Tensor]:
        return {name: value.to(ctx.rng.device) for name, value in self.states.items()}


@dataclass(frozen=True)
class FromFile:
    """Reset states that :meth:`ResetStates.save` wrote, such as a separate generation run."""

    path: str | Path

    def __call__(self, ctx: TermContext, groups: torch.Tensor | None) -> dict[str, torch.Tensor]:
        saved = torch.load(self.path, map_location=ctx.rng.device)
        held = saved["groups"]
        if (held is None) != (groups is None) or groups is not None and not torch.equal(held, groups):
            raise ValueError(f"{self.path} holds {_describe(held)}, but the event needs {_describe(groups)}.")
        return saved["states"]


def _describe(groups: torch.Tensor | None) -> str:
    if groups is None:
        return "shared reset states"
    num_groups = int(groups.max()) + 1
    if num_groups == len(groups):
        return f"reset states of {len(groups)} worlds"
    return f"reset states of {num_groups} groups over {len(groups)} worlds"


def prototype_groups(ctx: TermContext) -> torch.Tensor:
    """Group the worlds by the prototype combination that the scene cloned into them, such as the object each world
    spawned, as MDP1's lift reset does: worlds of a group share their reset states and their success statistics."""
    from isaaclab.sim import SimulationContext

    layout = SimulationContext.instance().get_clone_plan().topology.world_prototype_layout
    return torch.unique(torch.as_tensor(layout, device=ctx.rng.device), return_inverse=True)[1]


class StateGenerator:
    """A reset state generator task bound to a scene, rolled to propose reset states.

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
        """Return the proposed values of a field of the MDP, such as a command that the generator samples."""
        if selector not in self.state.arrays:
            raise ValueError(f"The generator task must define '{selector}', which the event restores.")
        return self.state.arrays[selector].torch


@dataclass(frozen=True)
class FromTask:
    """Generate reset states by rejection sampling the reset events of a generator ``task``.

    The task's reset events draw a state in every world, and the states that none of its terminations reject are
    kept. If the task has a ``descriptor`` observation group, ``oversample_factor`` times as many states are harvested
    and the ones most spread over the descriptor are kept.
    """

    task: TaskCfg
    num_states: int
    oversample_factor: float = 2.0
    max_rolls: int = 2000

    def __call__(self, ctx: TermContext, groups: torch.Tensor | None) -> dict[str, torch.Tensor]:
        num_worlds, device = ctx.num_worlds, ctx.rng.device
        group = groups if groups is not None else torch.zeros(num_worlds, dtype=torch.long, device=device)
        num_groups = int(group.max()) + 1
        generator = StateGenerator(self.task, ctx.scene_data, str(device), seed=int(ctx.rng[0]))
        descriptor = generator.transition.observations.get("descriptor")
        size = self.num_states if descriptor is None else round(self.num_states * self.oversample_factor)
        if self.num_states < 1 or size < self.num_states:
            raise ValueError("FromTask must harvest at least as many states as it keeps, and keep at least one.")
        # Scene outputs are read from the scene the generator writes; commands from the generator's own state.
        fields = {
            name: value if ctx.selectors[name].scene else generator.field(ctx.selectors[name])
            for name, value in ctx.outputs.items()
        }
        if descriptor is not None:
            fields["descriptor"] = descriptor.torch
        # Root poses are kept relative to the environment origins when the use binds them.
        origins = ctx.inputs.get("env_origins")
        poses = {name for name in ctx.outputs if origins is not None and _warp_view(ctx, name).dtype == wp.transformf}

        harvest = {name: value.new_zeros(num_groups, size, *value.shape[1:]) for name, value in fields.items()}
        filled = torch.zeros(num_groups, dtype=torch.long, device=device)
        worlds = torch.arange(num_worlds, device=device)
        for _ in range(self.max_rolls):
            accepted = generator.roll()
            values = {name: _minus_origin(value, origins) if name in poses else value for name, value in fields.items()}
            # Each group takes its accepted worlds in order, into its next free slots.
            take = worlds[accepted]
            order = torch.argsort(group[take], stable=True)
            take = take[order]
            members = group[take]
            starts = torch.searchsorted(members, members)  # index of the first accepted world of each one's group
            slot = filled[members] + torch.arange(len(take), device=device) - starts
            keep = slot < size
            take, members, slot = take[keep], members[keep], slot[keep]
            for name, value in values.items():
                harvest[name][members, slot] = value[take]
            filled += torch.bincount(members, minlength=num_groups)
            if (filled == size).all():
                break
        else:
            raise RuntimeError(
                f"FromTask harvested fewer than {size} valid reset states after {self.max_rolls} rolls of the"
                f" generator ({filled.min().item()} in the worst group)."
            )

        if descriptor is not None:
            # Keep the most spread states, each descriptor axis rescaled to unit span so that all axes count.
            points = harvest.pop("descriptor")
            span = (points.amax(dim=(0, 1)) - points.amin(dim=(0, 1))).clamp(min=1e-6)
            kept = farthest_points(points / span, self.num_states)
            rows = torch.arange(num_groups, device=device)
            harvest = {name: value[rows[:, None], kept] for name, value in harvest.items()}
        return harvest if groups is not None else {name: value[0] for name, value in harvest.items()}


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


class ResetStates:
    """The reset states of a reset event, the sampler that chooses among them, and the worlds' current choices.

    Slots are numbered within a group's reset states, a world's for states per world, or within the shared ones. Get
    it with :func:`get_reset_states`. Reading allocates and synchronizes; :meth:`pin` and the event's hooks do not, and
    pinned slots restore inside the step's CUDA graph.
    """

    def __init__(
        self,
        states: Mapping[str, torch.Tensor],
        outputs: Mapping[str, wp.array],
        sampler: ResetStateSamplerCfg,
        *,
        groups: torch.Tensor | None,
        origins: wp.array | None,
        default_reset_probability: float,
        record: RecordVisited | None,
    ) -> None:
        first = next(iter(outputs.values()))
        num_worlds, device = first.shape[0], first.device
        shared = groups is None
        filled = next(iter(states.values())).shape[0 if shared else 1]
        recorded = 0 if record is None else record.num_states
        self.num_worlds, self.shared, self.size = num_worlds, shared, filled + recorded
        self.groups = groups  # The group of each world, None if shared
        self.num_groups = 1 if shared else int(groups.max()) + 1
        self._group = torch.zeros(num_worlds, dtype=torch.long, device=wp.device_to_torch(device))
        if not shared:
            self._group.copy_(groups)
        self._filled, self._record, self._default = filled, record, float(default_reset_probability)

        # Recorded slots start as copies of the source's states, so every slot can be drawn from the start.
        cycle = torch.arange(self.size, device=states[next(iter(states))].device) % filled
        rows = {}
        for name, value in states.items():
            value = value[cycle] if shared else value[:, cycle]
            rows[name] = wp.from_torch(
                value.reshape(-1, *value.shape[1 if shared else 2 :]).contiguous(), dtype=outputs[name].dtype
            )
        self.buffer = ResetStateBuffer(rows)
        self.sampler = ResetStateSampler(
            sampler,
            num_worlds,
            self.size * self.num_groups,
            device,
            group_ids=None if shared else wp.from_torch(self._group.int()),
            num_groups=self.num_groups,
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
        self._writes = zeros(dtype=wp.bool)  # the worlds that this reset restores
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
        """Copies of the stored states, ``(num_groups, size, ...)`` per output, ``(num_worlds, size, ...)`` for states
        per world, or ``(size, ...)`` if shared. Poses are relative to the environment origins when the use binds
        them."""
        rows = {name: value.torch for name, value in self.buffer.states.items()}
        return (
            rows
            if self.shared
            else {name: value.view(self.num_groups, self.size, *value.shape[1:]) for name, value in rows.items()}
        )

    @property
    def success_rate(self) -> torch.Tensor:
        """Recent success rate of each slot, ``(num_groups, size)``, or ``(size,)`` if shared."""
        return self._per_slot(self.sampler.success_rate.torch)

    @property
    def episode_counts(self) -> torch.Tensor:
        """Episodes that started from each slot since it was stored, ``(num_groups, size)`` or ``(size,)``."""
        return self._per_slot(self.sampler.episode_counts.torch)

    @property
    def probabilities(self) -> torch.Tensor:
        """Probability that the next draw picks each slot, ``(num_groups, size)`` or ``(size,)``.

        With groups, a slot's probability is that of a draw by a world of its group."""
        return self._per_slot(self.sampler.probabilities.torch)

    @property
    def started_from(self) -> torch.Tensor:
        """The slot each world's episode started from, ``(num_worlds,)``; -1 for a default reset or before the first
        reset."""
        indices = self.sampler.indices.torch.long()
        slots = indices - self._group * self.size
        return torch.where(wp.to_torch(self._restored) & (indices >= 0), slots, -1)

    def pin(self, slots: torch.Tensor) -> None:
        """Reset the worlds to the given slots at their next resets, ``(num_worlds,)``, -1 to draw as usual.

        The pins hold until replaced or :meth:`unpin`. A slot out of range leaves the world's reset unchanged and sets
        the sampler's ``invalid_requests``. Pinned episodes are credited like drawn ones.
        """
        requests = wp.to_torch(self._requests)
        slots = slots.to(requests.device, torch.int32)
        if not self.shared:
            in_range = (slots >= 0) & (slots < self.size)
            rows = self._group.int() * self.size + slots
            slots = torch.where(in_range, rows, torch.where(slots < 0, -1, self.size * self.num_groups))
        requests.copy_(slots)

    def unpin(self) -> None:
        """Draw every world's reset states as usual again."""
        wp.to_torch(self._requests).fill_(-1)

    def skip_current_episodes(self) -> None:
        """Credit none of the worlds' current episodes, such as before resetting them by hand mid-episode."""
        self._credit.zero_()

    def save(self, path: str | Path) -> None:
        """Save the states, for a later run to load with :class:`FromFile`."""
        torch.save({"states": self.states, "groups": self.groups}, path)

    def restore(self, mask: wp.array, rng: wp.array, succeeded: wp.array | None) -> None:
        """Credit the ended episodes, choose the next starts, and write the restored ones to the outputs."""
        wp.launch(
            _choose_starts,
            mask.shape,
            [mask, rng, self._requests, self._default, self._credit, self._valid, self._restored, self._writes],
            device=mask.device,
        )
        self.sampler.reset(mask, success=succeeded, valid=self._valid, state_ids=self._requests, rng=rng)
        self.buffer.get_state(self.sampler.indices, out=self._outputs, mask=self._writes)
        for pose in self._poses:
            wp.launch(_shift_poses, mask.shape, [self._writes, self._origins, pose, pose, 1.0], device=mask.device)

    def record(self, rng: wp.array, terminated: wp.array, record_if: wp.array | None) -> None:
        """Record the states of the worlds selected this step into their oldest recorded slots."""
        wp.launch(
            _choose_records,
            self.num_worlds,
            [
                rng,
                terminated,
                record_if,
                self._record.probability,
                self._filled,
                self._record.num_states,
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
        return values.clone() if self.shared else values.view(self.num_groups, self.size).clone()


def get_reset_states(state: MDPState, use: str = "events.reset") -> ResetStates:
    """Return the reset states of a :data:`reset_to_states` use, by its name in the schedule."""
    cache = state.caches.get(use, {})
    if "reset_states" not in cache:
        raise KeyError(f"'{use}' is not a reset_to_states use. Uses with a cache: {sorted(state.caches)}.")
    return cache["reset_states"]


def _init(
    ctx: TermContext,
    *,
    source: Callable[[TermContext, torch.Tensor | None], dict[str, torch.Tensor]],
    sampler: ResetStateSamplerCfg | None = None,
    shared: bool = False,
    groups: Callable[[TermContext], torch.Tensor] | None = None,
    default_reset_probability: float = 0.0,
    record: RecordVisited | None = None,
) -> None:
    """Build the reset states once from the source, then the sampler and bind the outputs it restores."""
    outputs = {name: _warp_view(ctx, name) for name in ctx.outputs}
    has_poses = any(value.dtype == wp.transformf for value in outputs.values())
    if shared and groups is not None:
        raise ValueError("Reset states are either shared or per group of worlds, not both.")
    if (shared or groups is not None) and "env_origins" not in ctx.inputs and has_poses:
        raise ValueError(
            "Shared or grouped reset states with poses need the env_origins input, so that poses transfer across"
            " worlds."
        )
    if record is not None and (shared or groups is not None or "terminated" not in ctx.inputs or record.num_states < 1):
        raise ValueError(
            "Recording needs reset states per world, at least one slot, and the terminated input (mdp2.terminated), so"
            " that states which end the episode are not recorded."
        )
    if not 0.0 <= default_reset_probability <= 1.0:
        raise ValueError(f"default_reset_probability must be in [0, 1], got {default_reset_probability}.")
    device = ctx.rng.device
    if shared:
        group = None
    elif groups is not None:
        # Number the groups 0, 1, ..., so that any labels work.
        group = torch.unique(torch.as_tensor(groups(ctx), device=device), return_inverse=True)[1]
        if group.shape != (ctx.num_worlds,):
            raise ValueError(f"groups must give one group per world, got shape {tuple(group.shape)}.")
    else:
        group = torch.arange(ctx.num_worlds, device=device)
    states = source(ctx, group)
    if states.keys() != outputs.keys():
        raise ValueError(f"The source returned states for {sorted(states)}, but the event writes {sorted(outputs)}.")
    leading = 1 if shared else 2
    num_groups = 0 if shared else int(group.max()) + 1
    if (
        len({value.shape[leading - 1] for value in states.values()}) != 1
        or not shared
        and any(value.shape[0] != num_groups for value in states.values())
    ):
        raise ValueError("Every output needs the same number of states, per group of worlds unless they are shared.")

    origins = ctx.inputs.get("env_origins")
    reset_states = ResetStates(
        states,
        outputs,
        sampler if sampler is not None else ResetStateSamplerCfg(history_length=1, scores=[]),
        groups=group,
        origins=None if origins is None else wp.from_torch(origins.contiguous(), dtype=wp.vec3f),
        default_reset_probability=default_reset_probability,
        record=record,
    )
    ctx.cache["reset_states"] = reset_states
    # Load every kernel and the sampler's sort and scan scratch before a graph captures the hooks: a step and a reset
    # of no worlds.
    nothing = wp.zeros(ctx.num_worlds, dtype=wp.bool, device=reset_states.sampler.device)
    rng, succeeded = _as_warp(ctx.rng), ctx.inputs.get("succeeded")
    reset_states.restore(nothing, rng, None if succeeded is None else _as_warp(succeeded))
    if record is not None:
        when = ctx.inputs.get("record_if")
        reset_states.record(rng, wp.ones_like(nothing), None if when is None else _as_warp(when))


def _restore(mask: Any, cache: dict, rng: Any, succeeded: Any = None, **_: Any) -> None:
    cache["reset_states"].restore(_as_warp(mask), _as_warp(rng), None if succeeded is None else _as_warp(succeeded))


def _record(cache: dict, rng: Any, terminated: Any = None, record_if: Any = None, **_: Any) -> None:
    reset_states: ResetStates = cache["reset_states"]
    if reset_states._record is not None:
        reset_states.record(_as_warp(rng), _as_warp(terminated), None if record_if is None else _as_warp(record_if))


_hooks = Implementation(compute=_restore, update=_record)
reset_to_states = Term(torch=_hooks, warp=_hooks, random=True, init=_init)
"""Reset the worlds to reset states drawn from a stored set, the MDP2 counterpart of MDP1's lift
``conditional_reset``.

Params:

- ``source``: a callable that returns the states, such as :class:`FromTask` or :class:`FromFile`;
- ``sampler``: a :class:`~isaaclab.envs.reset.ResetStateSamplerCfg` that weighs the states by their recent success,
  or None to draw uniformly;
- ``shared``: whether every world draws from one set of states rather than its own. Share states only between worlds
  whose states fit each other: a state stored with one object spawned does not fit a world with another;
- ``groups``: a callable that returns the group of each world, such as :func:`prototype_groups`, for one set of
  states per group: the worlds of a group draw from its states and share their success statistics, as MDP1's lift
  reset does per object. Without it, each world has its own states;
- ``default_reset_probability``: the fraction of resets that keep what the earlier reset events wrote. Those episodes
  are not credited to any slot;
- ``record``: :class:`RecordVisited`, to store states visited during training.

Inputs: ``succeeded``, the ended episode's success, credited to the slot it started from when bound;
``env_origins``, which stores the poses relative to the environment origins; ``terminated`` (``mdp2.terminated``),
required to record; and ``record_if``, a quantity above 0.5 where a state may be recorded.

Place it after the other reset events: it overwrites the outputs they wrote. The states and the sampler's history
persist across episodes; :func:`get_reset_states` returns them.
"""


@wp.kernel
def _choose_starts(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.int32),
    requests: wp.array(dtype=wp.int32),
    default_reset_probability: float,
    credit: wp.array(dtype=wp.bool),
    valid: wp.array(dtype=wp.bool),
    restored: wp.array(dtype=wp.bool),
    writes: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    writes[world] = False
    if not mask[world]:
        return
    # The ended episode counts if it started from a slot that was not recorded over since.
    valid[world] = credit[world]
    restore = requests[world] >= 0 or wp.randf(wp.rand_init(rng[world], 1)) >= default_reset_probability
    restored[world] = restore
    writes[world] = restore
    credit[world] = restore


@wp.kernel
def _choose_records(
    rng: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.bool),
    record_if: wp.array(dtype=wp.float32),
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
    if record_if:
        record = record and record_if[world] > 0.5
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


def _as_warp(value: Any) -> wp.array:
    return wp.from_torch(value) if isinstance(value, torch.Tensor) else value
