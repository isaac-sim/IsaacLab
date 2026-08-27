# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-agnostic animated Gaussian-splat helpers for the PPISP camera demo.

Gaussian captures author motion as *tracks*: a ``ParticleField3DGaussianSplat`` prim whose
ancestors carry time-sampled xform ops (rigid motion), and/or whose per-particle
``positions``/``orientations`` arrays are themselves time sampled (deformable motion). Both
spellings of the per-particle arrays exist in the wild: the half-precision ``positionsh`` and
``orientationsh`` that NuRec exports, and the full-float ``positions`` and ``orientations``.

Playing a track back means re-stating its pose or its per-particle arrays on every rendered frame,
because the demo holds the Kit timeline at a single time code instead of playing it. Two mechanisms
are needed:

* :func:`bake_env_track_state` re-authors the sampled state on the duplicated-env USD stage. This is
  the mechanism the Isaac RTX path needs: its Fabric population re-reads every time-sampled
  attribute of an animated prim from USD before each render, at the stage's current time, so USD is
  the only authority that survives a frame. Writing the values into Fabric directly is silently
  overwritten, and scrubbing the Kit timeline is not an option either because Isaac Lab's physics
  manager owns its play state.
* :class:`GaussianTrackPlayback` prebakes the whole animation once and streams it to the GPU, for a
  renderer that ingests the Gaussians once and is then driven through its own API, which is how the
  OVRTX demo updates them. Nothing is resolved from USD while the render loop runs, so a profile of
  that loop measures the renderer rather than this module.

The Newton Warp renderer has no Gaussian-splat path, so it cannot play tracks back at all.

This module is imported by the demo scripts *after* :class:`~isaaclab.app.AppLauncher` has started
the app, because it imports :mod:`isaaclab.sim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils

GAUSSIAN_PRIM_TYPE_NAME = "ParticleField3DGaussianSplat"
"""USD prim type name of a Gaussian-splat particle field."""

POSITIONS_ATTR_NAMES = ("positionsh", "positions")
"""Candidate per-particle position attribute names, most specific first."""

ORIENTATIONS_ATTR_NAMES = ("orientationsh", "orientations")
"""Candidate per-particle orientation attribute names, most specific first."""


@dataclass(frozen=True)
class AnimatedGaussianTrack:
    """One animated Gaussian-splat prim discovered in a source stage.

    All paths are relative to the source stage's ``defaultPrim``, so they can be re-rooted under
    each duplicated env with :func:`env_prim_path`.
    """

    gaussian_rel_path: str
    """Path of the ``ParticleField3DGaussianSplat`` prim, relative to the source ``defaultPrim``."""

    animated_xform_rel_paths: tuple[str, ...]
    """Paths of every prim from the Gaussian prim up to the ``defaultPrim`` with a time-sampled xform op.

    These are the prims that carry the track's rigid motion. See :func:`bake_env_track_state`.
    """

    positions_attr_name: str | None
    """Name of the time-sampled per-particle position attribute, or ``None`` if positions are static."""

    orientations_attr_name: str | None
    """Name of the time-sampled per-particle orientation attribute, or ``None`` if orientations are static."""

    num_particles: int
    """Number of Gaussians in this track."""

    @property
    def is_rigid(self) -> bool:
        """Whether the track moves as a rigid body, i.e. through an animated ancestor xform."""
        return bool(self.animated_xform_rel_paths)

    @property
    def is_deformable(self) -> bool:
        """Whether the track deforms, i.e. time-samples its per-particle arrays."""
        return self.positions_attr_name is not None or self.orientations_attr_name is not None


def find_animated_gaussian_tracks(source_stage: Usd.Stage) -> list[AnimatedGaussianTrack]:
    """Discover every animated Gaussian-splat track in ``source_stage``.

    Static Gaussian prims are skipped: they need no per-frame work from either renderer.

    Args:
        source_stage: Opened source scene. Must have a ``defaultPrim`` so it can be referenced
            under each duplicated env.

    Returns:
        The discovered tracks, in stage traversal order.
    """
    default_prim = require_default_prim(source_stage)
    default_prefix = f"{default_prim.GetPath().pathString}/"

    tracks = []
    for prim in Usd.PrimRange(default_prim):
        if prim.GetTypeName() != GAUSSIAN_PRIM_TYPE_NAME:
            continue
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith(default_prefix):
            continue

        animated_xform_rel_paths = []
        ancestor = prim
        while ancestor and ancestor.IsValid():
            if _has_time_sampled_xform_ops(ancestor) and ancestor != default_prim:
                animated_xform_rel_paths.append(ancestor.GetPath().pathString[len(default_prefix) :])
            if ancestor == default_prim:
                break
            ancestor = ancestor.GetParent()

        positions_attr = _find_time_sampled_attr(prim, POSITIONS_ATTR_NAMES)
        orientations_attr = _find_time_sampled_attr(prim, ORIENTATIONS_ATTR_NAMES)
        if not animated_xform_rel_paths and positions_attr is None and orientations_attr is None:
            continue

        tracks.append(
            AnimatedGaussianTrack(
                gaussian_rel_path=prim_path[len(default_prefix) :],
                animated_xform_rel_paths=tuple(animated_xform_rel_paths),
                positions_attr_name=None if positions_attr is None else positions_attr.GetName(),
                orientations_attr_name=None if orientations_attr is None else orientations_attr.GetName(),
                num_particles=_get_num_particles(prim),
            )
        )
    return tracks


def collect_authored_times(source_stage: Usd.Stage, tracks: list[AnimatedGaussianTrack]) -> list[float]:
    """Return every USD time code authored by ``tracks``, sorted and de-duplicated.

    The demos union these with the camera trajectory samples so a scene that animates only its
    Gaussians still renders more than one frame.
    """
    default_prim = require_default_prim(source_stage)
    default_prefix = f"{default_prim.GetPath().pathString}/"

    times = set()
    for track in tracks:
        for xform_rel_path in track.animated_xform_rel_paths:
            prim = source_stage.GetPrimAtPath(f"{default_prefix}{xform_rel_path}")
            for xform_op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                times.update(float(value) for value in xform_op.GetAttr().GetTimeSamples())
        gaussian_prim = source_stage.GetPrimAtPath(f"{default_prefix}{track.gaussian_rel_path}")
        for attr_name in (track.positions_attr_name, track.orientations_attr_name):
            if attr_name is not None:
                times.update(float(value) for value in gaussian_prim.GetAttribute(attr_name).GetTimeSamples())
    return sorted(times)


def bake_env_track_state(
    source_stage: Usd.Stage,
    tracks: list[AnimatedGaussianTrack],
    num_envs: int,
    animation_time_code: float,
    stage_time_code: float = 0.0,
) -> None:
    """Re-author every duplicated env track's state sampled at ``animation_time_code`` onto the stage.

    The demos keep the stage parked at ``stage_time_code``, so this collapses the track's animation
    to the state the renderer resolves there. Calling it once per rendered frame plays the animation
    back, and the state stays authoritative because it masks the referenced layer's time samples:

    * rigid tracks get their sampled pose as ``translate``/``orient``/``scale`` ops, which replaces
      the referenced ``xformOpOrder`` and therefore drops the animated op from the composed pose, and
    * deformable tracks get their sampled arrays as a lone time sample, which wins over the
      referenced samples at every time code because value resolution never mixes layers.

    Args:
        source_stage: Opened source scene, sampled for the state to bake.
        tracks: Tracks to bake, as returned by :func:`find_animated_gaussian_tracks`.
        num_envs: Number of duplicated envs on the current stage.
        animation_time_code: USD time code of the animation to sample.
        stage_time_code: USD time code the renderer resolves the stage at.
    """
    stage = sim_utils.get_current_stage()
    default_prefix = f"{require_default_prim(source_stage).GetPath().pathString}/"
    source_time_code = Usd.TimeCode(animation_time_code)
    target_time_code = Usd.TimeCode(stage_time_code)

    for track in tracks:
        for xform_rel_path in track.animated_xform_rel_paths:
            source_prim = source_stage.GetPrimAtPath(f"{default_prefix}{xform_rel_path}")
            transform = Gf.Transform(UsdGeom.Xformable(source_prim).GetLocalTransformation(source_time_code))
            rotation = transform.GetRotation().GetQuat()
            imaginary = rotation.GetImaginary()
            for env_id in range(num_envs):
                sim_utils.standardize_xform_ops(
                    _require_env_prim(stage, env_id, xform_rel_path),
                    translation=tuple(transform.GetTranslation()),
                    orientation=(imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()),
                    scale=tuple(transform.GetScale()),
                )

        source_prim = source_stage.GetPrimAtPath(f"{default_prefix}{track.gaussian_rel_path}")
        for attr_name in (track.positions_attr_name, track.orientations_attr_name):
            if attr_name is None:
                continue
            # Kept in the source array's own type, half precision included, so the bake is exact.
            values = source_prim.GetAttribute(attr_name).Get(source_time_code)
            for env_id in range(num_envs):
                env_prim = _require_env_prim(stage, env_id, track.gaussian_rel_path)
                env_prim.GetAttribute(attr_name).Set(values, target_time_code)


class GaussianTrackPlayback:
    """Plays animated Gaussian tracks back on the GPU, without touching USD or the host per frame.

    Every animated column of every track is resolved from USD once, at construction, into pinned host
    staging shaped ``(num_frames, ...)``. Playing frame ``i`` then costs one asynchronous host-to-device
    copy per column into ring slot ``i % num_slots``, followed by the renderer's own asynchronous
    write of that slot. The render loop therefore does no USD value resolution, no host allocation and
    no per-particle Python -- the per-particle work happens once per column, vectorized, at construction.

    The ring lets the copy for the next frame overlap the render of the current one. Slot reuse is what
    bounds how far ahead it may run: with ``num_slots`` slots, one holds the frame being written and one
    guards the previous frame whose write the renderer may still be reading, leaving ``num_slots - 2``
    frames of prefetch. Two slots are therefore the minimum, and the default of three buys one frame of
    overlap. Slot reuse is also why playback runs forward: :meth:`play` refuses a frame the ring has
    already moved past rather than overwrite a slot the renderer may still be reading.

    Args:
        source_stage: Opened source scene, sampled once per frame time code.
        tracks: Tracks to play, as returned by :func:`find_animated_gaussian_tracks`.
        frame_time_codes: USD time codes of the frames that will be played, in play order.
        num_envs: Number of duplicated envs on the current stage.
        device: Warp device the ring buffers are allocated on.
        num_slots: Number of ring slots per column. Must be at least 2.

    Raises:
        ValueError: If ``num_slots`` is less than 2.
    """

    def __init__(
        self,
        source_stage: Usd.Stage,
        tracks: list[AnimatedGaussianTrack],
        frame_time_codes: list[float],
        num_envs: int,
        device: str,
        num_slots: int = 3,
    ):
        if num_slots < 2:
            raise ValueError(f"num_slots must be at least 2 to double-buffer the upload, received {num_slots}.")
        self._num_slots = num_slots
        self._prefetch_depth = num_slots - 2
        self._num_frames = len(frame_time_codes)
        # Frame each slot currently holds, so a frame already uploaded is not uploaded again.
        self._resident: list[int | None] = [None] * num_slots
        self._last_played = 0
        self._columns: list[_AnimationColumn] = []

        default_prefix = f"{require_default_prim(source_stage).GetPath().pathString}/"
        for track in tracks:
            for xform_rel_path in track.animated_xform_rel_paths:
                source_prim = source_stage.GetPrimAtPath(f"{default_prefix}{xform_rel_path}")
                xformable = UsdGeom.Xformable(source_prim)
                # Every env shares the track's local transform, so one sample is broadcast to all of
                # them and the renderer receives the one-matrix-per-path layout it requires.
                frames = np.stack(
                    [
                        np.broadcast_to(
                            np.asarray(xformable.GetLocalTransformation(Usd.TimeCode(time_code)), dtype=np.float64),
                            (num_envs, 4, 4),
                        )
                        for time_code in frame_time_codes
                    ]
                )
                self._columns.append(
                    _AnimationColumn(
                        prim_paths=[env_prim_path(env_id, xform_rel_path) for env_id in range(num_envs)],
                        kind="xform",
                        host=wp.array(frames, dtype=wp.float64, device="cpu", pinned=True),
                        ring=wp.zeros((num_slots, num_envs, 4, 4), dtype=wp.float64, device=device),
                    )
                )

            if not track.is_deformable:
                continue
            gaussian_prim = source_stage.GetPrimAtPath(f"{default_prefix}{track.gaussian_rel_path}")
            prim_paths = [env_prim_path(env_id, track.gaussian_rel_path) for env_id in range(num_envs)]
            for kind, attr_name, components in (
                ("positions", track.positions_attr_name, 3),
                ("orientations", track.orientations_attr_name, 4),
            ):
                if attr_name is None:
                    continue
                attribute = gaussian_prim.GetAttribute(attr_name)
                # Vt converts a quaternion array to (imaginary, real) components, which is already the
                # (x, y, z, w) order the renderers expect, so both columns are a plain vectorized cast.
                frames = np.stack(
                    [
                        np.asarray(attribute.Get(Usd.TimeCode(time_code))).astype(np.float32, copy=False)
                        for time_code in frame_time_codes
                    ]
                )
                if frames.shape[1:] != (track.num_particles, components):
                    raise RuntimeError(
                        f"{attr_name} on {track.gaussian_rel_path} sampled as shape {frames.shape[1:]}, expected"
                        f" ({track.num_particles}, {components})."
                    )
                self._columns.append(
                    _AnimationColumn(
                        prim_paths=prim_paths,
                        kind=kind,
                        host=wp.array(frames, dtype=wp.float32, device="cpu", pinned=True),
                        ring=wp.zeros((num_slots, track.num_particles, components), dtype=wp.float32, device=device),
                    )
                )

    @property
    def device_bytes(self) -> int:
        """Total size [B] of the device ring buffers, for the demo's startup report."""
        return sum(column.ring.size * wp.types.type_size_in_bytes(column.ring.dtype) for column in self._columns)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to play, i.e. no track animates anything."""
        return not self._columns

    def play(self, renderer: Any, frame_index: int) -> None:
        """Advance every track to ``frame_index`` through the renderer's Gaussian update hooks.

        Uploads whatever frames the ring is allowed to run ahead on and does not already hold, then
        writes the requested frame's slot. Both the copies and the renderer's writes are
        asynchronous, so this returns without waiting on the GPU. Replaying the frame the ring is
        already on is free, which is what lets the caller seed a pose before its warmup steps.

        Args:
            renderer: Renderer exposing the ``update_gaussian_splat_*`` hooks.
            frame_index: Frame to show, as an index into the frame time codes. Playback runs forward:
                a frame the ring has already moved past cannot be played again.

        Raises:
            ValueError: If ``frame_index`` is behind the last played frame.
        """
        if not self._columns:
            return
        if frame_index < self._last_played:
            raise ValueError(
                f"frame {frame_index} is behind the last played frame {self._last_played}: playback runs forward,"
                " because rewinding would overwrite a slot the renderer may still be reading."
            )
        for index in range(frame_index, min(frame_index + self._prefetch_depth, self._num_frames - 1) + 1):
            upload_slot = index % self._num_slots
            if self._resident[upload_slot] == index:
                continue
            for column in self._columns:
                wp.copy(column.ring[upload_slot], column.host[index])
            self._resident[upload_slot] = index
        self._last_played = frame_index

        slot = frame_index % self._num_slots
        for column in self._columns:
            values = column.ring[slot]
            if column.kind == "xform":
                renderer.update_gaussian_splat_transforms(column.prim_paths, values)
            else:
                # Every env shares the sample: the arrays are in the Gaussian prim's local frame.
                shared = [values] * len(column.prim_paths)
                renderer.update_gaussian_splat_particles(column.prim_paths, **{column.kind: shared})


@dataclass(frozen=True)
class _AnimationColumn:
    """One animated column of one track: its prebaked host frames and the device ring they stream through."""

    prim_paths: list[str]
    """Duplicated-env prim paths the column is written to."""

    kind: str
    """Which renderer hook the column feeds: ``xform``, ``positions`` or ``orientations``."""

    host: wp.array
    """Pinned host staging holding every frame, shape ``(num_frames, ...)``."""

    ring: wp.array
    """Device ring the frames are streamed through, shape ``(num_slots, ...)``."""


def env_prim_path(env_id: int, rel_path: str) -> str:
    """Return the duplicated-env path of a source path relative to the source ``defaultPrim``."""
    return f"/World/envs/env_{env_id}/Scene/{rel_path}"


def require_default_prim(source_stage: Usd.Stage) -> Usd.Prim:
    """Return the source stage ``defaultPrim``, raising if the scene cannot be referenced per env."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")
    return default_prim


def format_tracks(tracks: list[AnimatedGaussianTrack]) -> str:
    """Format the discovered tracks for demo logging."""
    if not tracks:
        return "none"
    return ", ".join(
        f"{track.gaussian_rel_path}[{track.num_particles}"
        f"{',rigid' if track.is_rigid else ''}{',deformable' if track.is_deformable else ''}]"
        for track in tracks
    )


def _require_env_prim(stage: Usd.Stage, env_id: int, rel_path: str) -> Usd.Prim:
    """Return a duplicated env's counterpart of a source prim, raising if the env stage lacks it."""
    prim_path = env_prim_path(env_id, rel_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Duplicated Gaussian track prim not found: {prim_path}")
    return prim


def _has_time_sampled_xform_ops(prim: Usd.Prim) -> bool:
    """Whether ``prim`` authors at least one time-sampled xform op."""
    if not prim.IsA(UsdGeom.Xformable):
        return False
    return any(xform_op.GetAttr().GetNumTimeSamples() > 0 for xform_op in UsdGeom.Xformable(prim).GetOrderedXformOps())


def _find_time_sampled_attr(prim: Usd.Prim, candidate_names: tuple[str, ...]) -> Usd.Attribute | None:
    """Return the first time-sampled attribute of ``prim`` among ``candidate_names``."""
    for name in candidate_names:
        attr = prim.GetAttribute(name)
        if attr and attr.GetNumTimeSamples() > 0:
            return attr
    return None


def _get_num_particles(prim: Usd.Prim) -> int:
    """Return the Gaussian count of ``prim`` from whichever position attribute it authors."""
    for name in POSITIONS_ATTR_NAMES:
        attr = prim.GetAttribute(name)
        if not attr:
            continue
        values = attr.Get(Usd.TimeCode.EarliestTime())
        if values is not None:
            return len(values)
    raise RuntimeError(f"Gaussian prim authors no position array: {prim.GetPath()}")
