# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a learned model covers the dynamic state and the actuation of an Isaac Lab Newton scene."""

from __future__ import annotations

from collections.abc import Iterator

import newton
import numpy as np
import torch
from physicsnemo_robotics.dynamics import CompositeStateCodec, StateCodec

_JOINT_FIELDS = frozenset(("joint_q", "joint_qd"))
_BODY_FIELDS = frozenset(("body_q", "body_qd"))
_PARTICLE_FIELDS = frozenset(("particle_q", "particle_qd"))
_IDENTITY_TOLERANCE = 1.0e-7


def validate_state_ownership(model: newton.Model, codec: StateCodec) -> None:
    """Require a learned codec to own every dynamic Newton state row exactly once.

    A learned solver replaces the complete state transition, so any dynamic row outside the codec
    would never advance. The codec owns either generalized joint state (joint-backed Isaac Lab assets)
    or maximal-coordinate body state (scenes without joints), optionally combined with particle state.
    Row ``w`` of every ownership table may only reference entities of Newton world ``w``.

    Joint state is turned into body state by Newton forward kinematics after each learned step, so
    the articulations must be trees that ``eval_fk`` reproduces exactly: one world root, parent-before-
    child order, enabled joints, no loop-closing or cable joints, and identity anchors on free roots.

    Args:
        model: Finalized Newton model of the scene.
        codec: State codec of the learned solver, possibly composite.

    Raises:
        ValueError: If a dynamic row is unowned, owned twice, owned by another world, or the
            articulations violate the forward-kinematics requirements.
        TypeError: If the codec does not declare its state fields or ownership tables.
    """
    world_count = model.world_count
    joint_q, joint_qd, bodies, particles = [], [], [], []
    for component in _leaf_codecs(codec):
        if component.state_fields is None:
            raise TypeError(f"{type(component).__name__} must declare state_fields for learned deployment.")
        fields = frozenset(component.state_fields)
        for pair in (_JOINT_FIELDS, _BODY_FIELDS, _PARTICLE_FIELDS):
            if fields & pair and not pair <= fields:
                raise ValueError(f"{type(component).__name__} must own {sorted(pair)} together.")
        if fields >= _JOINT_FIELDS:
            if not fields >= _BODY_FIELDS:
                raise ValueError(
                    f"{type(component).__name__} must also own body state so Isaac Lab body views stay synchronized."
                )
            joint_q.append(
                _ownership_table(component.layout.q_indices, "joint_q", model.joint_coord_count, world_count)
            )
            joint_qd.append(
                _ownership_table(component.layout.qd_indices, "joint_qd", model.joint_dof_count, world_count)
            )
        elif fields >= _BODY_FIELDS:
            bodies.append(_ownership_table(component.indices, "body", model.body_count, world_count))
        if fields >= _PARTICLE_FIELDS:
            particles.append(_ownership_table(component.indices, "particle", model.particle_count, world_count))

    if joint_q and bodies:
        raise ValueError("A learned codec must own either joint state or body state, not both.")
    if bodies and model.joint_count:
        raise ValueError("Body state cannot drive joint-backed Isaac Lab assets, whose joint state would go stale.")

    owned_bodies: set[int] = set()
    if joint_q:
        joint_worlds = model.joint_world.numpy()
        q_worlds = np.repeat(joint_worlds, np.diff(model.joint_q_start.numpy()))
        qd_worlds = np.repeat(joint_worlds, np.diff(model.joint_qd_start.numpy()))
        _require_complete(joint_q, model.joint_coord_count, "joint_q")
        _require_complete(joint_qd, model.joint_dof_count, "joint_qd")
        _require_local_worlds(joint_q, q_worlds, "joint_q")
        _require_local_worlds(joint_qd, qd_worlds, "joint_qd")
        owned_bodies = _validate_articulation_trees(model)
    if model.body_count:
        body_worlds = model.body_world.numpy()
        _require_local_worlds(bodies, body_worlds, "body")
        owned_bodies |= {int(index) for table in bodies for index in table.ravel()}
        dynamic = np.flatnonzero(model.body_flags.numpy() & int(newton.BodyFlags.DYNAMIC))
        missing = sorted(set(dynamic.tolist()) - owned_bodies)
        if missing:
            raise ValueError(f"The learned codec leaves dynamic Newton bodies {missing} outside its ownership.")
    if model.particle_count:
        _require_local_worlds(particles, model.particle_world.numpy(), "particle")
        active = np.flatnonzero(model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE))
        owned = {int(index) for table in particles for index in table.ravel()}
        missing = sorted(set(active.tolist()) - owned)
        if missing:
            raise ValueError(f"The learned codec leaves active Newton particles {missing} outside its ownership.")


def validate_effort_control(model: newton.Model) -> None:
    """Require applied joint efforts to carry every actuation command of the scene.

    Learned models in this extension are conditioned on the joint efforts in ``control.joint_f``.
    Implicit position drives (nonzero ``model.joint_target_ke``) act inside the numerical solver on
    position targets that never appear in ``joint_f``, so neither recorded data nor a deployed model
    would see the command. Use explicit actuators or effort control instead. Joint damping
    (``joint_target_kd``) with zero velocity targets is a function of the state and is allowed.

    Args:
        model: Finalized Newton model of the scene, after Isaac Lab applied its actuator gains.

    Raises:
        ValueError: If any joint uses an implicit position drive.
    """
    if model.joint_dof_count == 0:
        return
    driven = np.flatnonzero(model.joint_target_ke.numpy())
    if driven.size:
        raise ValueError(
            f"Joint DOFs {driven.tolist()} use implicit position drives, whose targets are not part of the applied"
            " joint efforts. Use explicit actuators or effort control."
        )


def _leaf_codecs(codec: StateCodec) -> Iterator[StateCodec]:
    if not isinstance(codec, CompositeStateCodec):
        yield codec
        return
    for component in codec.components:
        yield from _leaf_codecs(component)


def _ownership_table(indices: torch.Tensor | None, name: str, limit: int, world_count: int) -> np.ndarray:
    """Validate one ``[world_count, entities]`` codec ownership table."""
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.dtype != torch.long:
        raise TypeError(f"{name} ownership must be a two-dimensional torch.long tensor.")
    if indices.shape[0] != world_count:
        raise ValueError(f"{name} ownership has {indices.shape[0]} rows for {world_count} Newton worlds.")
    table = indices.cpu().numpy()
    if np.unique(table).size != table.size or table.min(initial=0) < 0 or table.max(initial=-1) >= limit:
        raise ValueError(f"{name} ownership indices must be unique and lie inside the Newton model.")
    return table


def _require_complete(tables: list[np.ndarray], count: int, name: str) -> None:
    owned = np.concatenate([table.ravel() for table in tables])
    missing = np.setdiff1d(np.arange(count), owned)
    if missing.size or owned.size != np.unique(owned).size:
        raise ValueError(f"The learned codec must own every {name} row once; unowned rows {missing.tolist()}.")


def _require_local_worlds(tables: list[np.ndarray], worlds: np.ndarray, name: str) -> None:
    for table in tables:
        mismatch = np.argwhere(worlds[table] != np.arange(table.shape[0])[:, None])
        if mismatch.size:
            row, column = mismatch[0]
            raise ValueError(
                f"Learned codec {name} row {row} owns index {table[row, column]} from Newton world"
                f" {worlds[table[row, column]]}."
            )


def _validate_articulation_trees(model: newton.Model) -> set[int]:
    """Check the forward-kinematics requirements and return the bodies produced by tree joints."""
    joint_type = model.joint_type.numpy()
    cable = np.flatnonzero(joint_type == int(newton.JointType.ROD))
    if cable.size:
        raise ValueError(f"Cable joints {cable.tolist()} keep body state that forward kinematics does not rebuild.")
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    enabled, worlds = model.joint_enabled.numpy(), model.joint_world.numpy()
    parent_xforms, child_xforms = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    starts, ends = model.articulation_start.numpy(), model.articulation_end.numpy()
    if model.articulation_count == 0 or starts[-1] != model.joint_count:
        raise ValueError("Every Newton joint must belong to an articulation tree.")

    produced: set[int] = set()
    body_worlds = model.body_world.numpy()
    for articulation in range(model.articulation_count):
        start, end, next_start = int(starts[articulation]), int(ends[articulation]), int(starts[articulation + 1])
        if end != next_start:
            raise ValueError(f"Loop-closing joints {list(range(end, next_start))} are ignored by forward kinematics.")
        tree = range(start, end)
        if sum(parents[joint] == -1 for joint in tree) != 1:
            raise ValueError(f"Articulation {articulation} must have exactly one world root.")
        for joint in tree:
            parent, child = int(parents[joint]), int(children[joint])
            if not enabled[joint]:
                raise ValueError(f"Disabled joint {joint} is not synchronized by forward kinematics.")
            if parent != -1 and parent not in produced:
                raise ValueError(f"Joint {joint} reads body {parent} before its parent joint produces it.")
            if child in produced:
                raise ValueError(f"Body {child} is produced by more than one joint.")
            if body_worlds[child] != worlds[joint]:
                raise ValueError(f"Body {child} belongs to a different Newton world than its joint {joint}.")
            if (
                parent == -1
                and joint_type[joint] == int(newton.JointType.FREE)
                and not (_is_identity(parent_xforms[joint]) and _is_identity(child_xforms[joint]))
            ):
                raise ValueError(
                    f"Free root joint {joint} must have identity anchors because Isaac Lab root views read raw"
                    " joint coordinates."
                )
            produced.add(child)
    return produced


def _is_identity(transform: np.ndarray) -> bool:
    """Return whether a Newton ``(px, py, pz, qx, qy, qz, qw)`` transform is the identity."""
    return bool(
        np.all(np.abs(transform[:6]) <= _IDENTITY_TOLERANCE) and abs(abs(transform[6]) - 1.0) <= _IDENTITY_TOLERANCE
    )
