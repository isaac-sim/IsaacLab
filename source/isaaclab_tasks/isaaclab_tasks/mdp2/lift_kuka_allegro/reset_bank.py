# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The reset of the lift task: every world keeps a bank of valid start states, drawn by how often the policy solves
them.

The bank is filled once, before training, from a task of candidate start states: its reset events draw a state, its
terminations reject it, and its ``descriptor`` observation group, if it has one, measures it. Restoring a state is an
ordinary reset event that runs inside the step's CUDA graph.
"""

from __future__ import annotations

import functools

import torch
import warp as wp
from isaaclab_experimental import mdp2

from isaaclab.utils.warp import ProxyArray

MIN_WEIGHT = wp.constant(1.0e-4)
"""Smallest weight of a start state, which keeps the states the policy never or always solves in the draw."""


@wp.func
def _weight(
    world: int,
    slot: int,
    outcomes: wp.array3d(dtype=float),
    episodes: wp.array2d(dtype=float),
    target_success_rate: float,
    kappa: float,
) -> float:
    """Return the weight of a start state, which peaks where its success rate is the target."""
    history = outcomes.shape[2]
    successes = float(0.0)
    for h in range(history):
        successes += outcomes[world, slot, h]
    rate = successes / wp.clamp(episodes[world, slot], 1.0, float(history))

    weight = wp.pow(rate + MIN_WEIGHT, kappa * target_success_rate)
    weight *= wp.pow(1.0 - rate + MIN_WEIGHT, kappa * (1.0 - target_success_rate))
    return wp.max(weight, MIN_WEIGHT)


@wp.func
def _reset_from_bank(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    object_pose_w: wp.array(dtype=wp.transformf),
    object_vel_w: wp.array(dtype=wp.spatial_vectorf),
    succeeded: wp.array(dtype=float),
    bank_joint_pos: wp.array3d(dtype=float),
    bank_joint_vel: wp.array3d(dtype=float),
    bank_object_pose_w: wp.array3d(dtype=float),
    bank_object_vel_w: wp.array3d(dtype=float),
    outcomes: wp.array3d(dtype=float),
    episodes: wp.array2d(dtype=float),
    playing: wp.array(dtype=float),
    target_success_rate: float,
    kappa: float,
    rng: wp.int32,
):
    # Credit the episode that ended to the state it started from.
    played = int(playing[world])
    if played >= 0:
        position = int(episodes[world, played]) % outcomes.shape[2]
        outcomes[world, played, position] = wp.where(succeeded[world] > 0.5, 1.0, 0.0)
        episodes[world, played] = episodes[world, played] + 1.0

    # Draw a state by its weight.
    num_slots = episodes.shape[1]
    total = float(0.0)
    for s in range(num_slots):
        total += _weight(world, s, outcomes, episodes, target_success_rate, kappa)
    draw = wp.randf(wp.rand_init(rng, 0), 0.0, 1.0) * total

    slot = int(0)
    cumulative = _weight(world, 0, outcomes, episodes, target_success_rate, kappa)
    while slot < num_slots - 1 and cumulative <= draw:
        slot += 1
        cumulative += _weight(world, slot, outcomes, episodes, target_success_rate, kappa)
    playing[world] = float(slot)

    for j in range(joint_pos.shape[1]):
        joint_pos[world, j] = bank_joint_pos[world, slot, j]
        joint_vel[world, j] = bank_joint_vel[world, slot, j]
    pose = wp.transformf()
    for k in range(7):
        pose[k] = bank_object_pose_w[world, slot, k]
    velocity = wp.spatial_vectorf()
    for k in range(6):
        velocity[k] = bank_object_vel_w[world, slot, k]
    object_pose_w[world] = pose
    object_vel_w[world] = velocity


def _reset_from_bank_torch(
    mask,
    joint_pos,
    joint_vel,
    object_pose_w,
    object_vel_w,
    succeeded,
    bank_joint_pos,
    bank_joint_vel,
    bank_object_pose_w,
    bank_object_vel_w,
    outcomes,
    episodes,
    playing,
    rng,
    *,
    target_success_rate,
    kappa,
):
    worlds = torch.arange(len(mask), device=mask.device)
    history = outcomes.shape[2]

    # Credit the episode that ended to the state it started from.
    credited = mask & (playing >= 0.0)
    played = playing.long().clamp(min=0)
    position = episodes[worlds, played].long() % history
    outcome = torch.where(credited, (succeeded > 0.5).float(), outcomes[worlds, played, position])
    outcomes[worlds, played, position] = outcome
    episodes[worlds, played] += credited.float()

    # Draw a state by its weight.
    rate = outcomes.sum(-1) / episodes.clamp(1.0, history)
    weights = (rate + MIN_WEIGHT) ** (kappa * target_success_rate)
    weights = (weights * (1.0 - rate + MIN_WEIGHT) ** (kappa * (1.0 - target_success_rate))).clamp_min(MIN_WEIGHT)
    cumulative = weights.cumsum(-1)
    draw = mdp2.uniform(rng, 0.0, 1.0) * cumulative[:, -1]
    slot = (cumulative <= draw[:, None]).sum(-1).clamp(max=weights.shape[1] - 1)
    playing.copy_(torch.where(mask, slot.float(), playing))

    fields = (
        (joint_pos, bank_joint_pos),
        (joint_vel, bank_joint_vel),
        (object_pose_w, bank_object_pose_w),
        (object_vel_w, bank_object_vel_w),
    )
    for field, bank in fields:
        field.copy_(torch.where(mask[:, None], bank[worlds, slot], field))


def _farthest_points(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return, for each world, the indices of ``count`` of its points spread as far apart as possible."""
    worlds = torch.arange(len(points), device=points.device)
    chosen = torch.zeros(len(points), count, dtype=torch.long, device=points.device)
    nearest = torch.full(points.shape[:2], float("inf"), device=points.device)
    farthest = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for k in range(count):
        chosen[:, k] = farthest
        distance = (points - points[worlds, farthest][:, None]).norm(dim=-1)
        nearest = torch.minimum(nearest, distance)
        farthest = nearest.argmax(dim=-1)
    return chosen


def _fill_bank(ctx, *, candidates, bank_size, oversample_factor=2.0, max_rolls=2000, **restore_params):
    fields = ctx.outputs
    num_worlds, device = ctx.num_worlds, ctx.rng.device
    worlds = torch.arange(num_worlds, device=device)

    builder = mdp2.MDPBuilder(ctx.scene_data.schema)
    builder.add_task(candidates)
    mdp = builder.finalize(device=str(device), backend="warp")
    state, transition = mdp.state(ctx.scene_data, seed=int(ctx.rng[0])), mdp.transition()
    mdp2.startup(mdp, state)

    # With a descriptor, more states are harvested than kept, to choose the most spread ones from.
    descriptor = transition.observations.get("descriptor")
    harvest_size = bank_size if descriptor is None else round(bank_size * oversample_factor)
    states = {name: field.new_zeros(num_worlds, harvest_size, *field.shape[1:]) for name, field in fields.items()}
    if descriptor is not None:
        fields = {**fields, "descriptor": descriptor.torch}
        states["descriptor"] = descriptor.torch.new_zeros(num_worlds, harvest_size, descriptor.shape[1])

    # Roll the candidates in every world until each world has harvested enough valid states.
    mask = ProxyArray(wp.ones(num_worlds, dtype=wp.bool, device=str(device)))
    commit = functools.partial(ctx.scene_data.commit, mask.warp, mdp.reset_writes)
    filled = torch.zeros(num_worlds, dtype=torch.long, device=device)
    for _ in range(max_rolls):
        mdp2.reset(mdp, state, mask, commit)
        commit()
        mdp2.evaluate(mdp, state, transition)

        take = ~transition.terminated.torch & (filled < harvest_size)
        slot = filled.clamp(max=harvest_size - 1)
        for name, field in fields.items():
            states[name][worlds, slot] = torch.where(take[:, None], field, states[name][worlds, slot])
        filled += take
        if (filled == harvest_size).all():
            break
    else:
        short = int((filled < harvest_size).sum())
        raise RuntimeError(
            f"The reset bank found fewer than {harvest_size} valid start states in {short} of {num_worlds} worlds"
            f" after {max_rolls} rolls of the candidates."
        )

    # Keep the states most spread over the descriptor, each axis rescaled to unit span so that all axes count.
    if descriptor is not None:
        values = states.pop("descriptor")
        span = (values.amax(dim=(0, 1)) - values.amin(dim=(0, 1))).clamp(min=1e-6)
        kept = _farthest_points(values / span, bank_size)
        states = {name: value[worlds[:, None], kept] for name, value in states.items()}
    for name, value in states.items():
        ctx.state[f"bank_{name}"].copy_(value)


reset_from_bank = mdp2.Term(
    torch=_reset_from_bank_torch,
    warp=_reset_from_bank,
    state={
        "bank_joint_pos": mdp2.State(("worlds", "bank_size", "joint_pos"), persistent=True),
        "bank_joint_vel": mdp2.State(("worlds", "bank_size", "joint_vel"), persistent=True),
        "bank_object_pose_w": mdp2.State(("worlds", "bank_size", 7), persistent=True),
        "bank_object_vel_w": mdp2.State(("worlds", "bank_size", 6), persistent=True),
        "outcomes": mdp2.State(("worlds", "bank_size", "history_length"), persistent=True),
        "episodes": mdp2.State(("worlds", "bank_size"), persistent=True),
        "playing": mdp2.State(initial=-1.0, persistent=True),
    },
    random=True,
    init=_fill_bank,
)
"""Restart the reset worlds from one of their ``bank_size`` banked start states, like MDP1's ``conditional_reset``.

Each world banks its own states of the robot's joints and of the object, so they fit the object it spawned. Before
training, the reset events of the ``candidates`` task are rolled in every world, and the states that none of its
terminations reject are harvested. If the task has a ``descriptor`` observation group, ``oversample_factor`` times as
many states are harvested, and the ones most spread over the descriptor are kept.

A reset credits the episode that ended, whose success is the ``succeeded`` input, to the state it started from, in
that state's last ``history_length`` outcomes. It then draws a state with a weight that peaks where the success rate
is ``target_success_rate``, more sharply with a larger ``kappa``, and none (uniform) with zero.
"""
