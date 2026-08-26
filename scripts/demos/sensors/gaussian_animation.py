# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-agnostic animated Gaussian-splat helpers shared by the PPISP camera demos.

Gaussian captures author motion as *tracks*: a ``ParticleField3DGaussianSplat`` prim whose
ancestors carry time-sampled xform ops (rigid motion), and/or whose per-particle
``positions``/``orientations`` arrays are themselves time sampled (deformable motion). Both
spellings of the per-particle arrays exist in the wild: the half-precision ``positionsh`` and
``orientationsh`` that NuRec exports, and the full-float ``positions`` and ``orientations``.

Playing a track back means re-stating its pose or its per-particle arrays on every rendered frame,
because the demos hold the Kit timeline at a single time code instead of playing it. Two mechanisms
are needed:

* :func:`bake_env_track_state` re-authors the sampled state on the duplicated-env USD stage. This is
  the mechanism the Isaac RTX path needs: its Fabric population re-reads every time-sampled
  attribute of an animated prim from USD before each render, at the stage's current time, so USD is
  the only authority that survives a frame. Writing the values into Fabric directly is silently
  overwritten, and scrubbing the Kit timeline is not an option either because Isaac Lab's physics
  manager owns its play state.
* :func:`sample_track_transform_in_default` and :func:`sample_track_particles` return plain numeric
  samples for a renderer that ingests the Gaussians once and is then driven through its own API,
  which is how the OVRTX demo updates them.

The Newton Warp renderer has no Gaussian-splat path, so it cannot play tracks back at all.

This module is imported by the demo scripts *after* :class:`~isaaclab.app.AppLauncher` has started
the app, because it imports :mod:`isaaclab.sim`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def sample_track_transform_in_default(
    source_stage: Usd.Stage, track: AnimatedGaussianTrack, time_code: float
) -> Gf.Matrix4d:
    """Return the track's Gaussian prim transform relative to the source ``defaultPrim`` at ``time_code``.

    The whole ancestor chain is composed, so a track animated several levels above the Gaussian prim
    resolves correctly. Compose the result with the duplicated env's ``Scene`` world transform to get
    the world matrix a renderer expects.
    """
    default_prim = require_default_prim(source_stage)
    gaussian_prim = source_stage.GetPrimAtPath(f"{default_prim.GetPath().pathString}/{track.gaussian_rel_path}")
    if not gaussian_prim or not gaussian_prim.IsValid():
        raise RuntimeError(f"Gaussian prim not found in source stage: {track.gaussian_rel_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode(time_code))
    return cache.GetLocalToWorldTransform(gaussian_prim) * cache.GetLocalToWorldTransform(default_prim).GetInverse()


def sample_track_particles(
    source_stage: Usd.Stage, track: AnimatedGaussianTrack, time_code: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample a deformable track's per-particle arrays at ``time_code``.

    The values are in the Gaussian prim's local frame, so every duplicated env shares one sample.

    Args:
        source_stage: Opened source scene.
        track: Track to sample.
        time_code: USD time code to sample at. USD interpolates between authored samples.

    Returns:
        A tuple of particle positions [m] with shape (num_particles, 3) and orientation quaternions
        in ``(x, y, z, w)`` order with shape (num_particles, 4), both ``float32``. Either entry is
        ``None`` when the corresponding attribute is not animated on this track.
    """
    default_prim = require_default_prim(source_stage)
    gaussian_prim = source_stage.GetPrimAtPath(f"{default_prim.GetPath().pathString}/{track.gaussian_rel_path}")
    if not gaussian_prim or not gaussian_prim.IsValid():
        raise RuntimeError(f"Gaussian prim not found in source stage: {track.gaussian_rel_path}")

    positions = None
    if track.positions_attr_name is not None:
        values = gaussian_prim.GetAttribute(track.positions_attr_name).Get(Usd.TimeCode(time_code))
        positions = np.asarray(values, dtype=np.float32).reshape(-1, 3)

    orientations = None
    if track.orientations_attr_name is not None:
        values = gaussian_prim.GetAttribute(track.orientations_attr_name).Get(Usd.TimeCode(time_code))
        # Gf quaternions iterate as (real, imaginary); the renderers all expect (x, y, z, w).
        orientations = np.empty((len(values), 4), dtype=np.float32)
        for index, quat in enumerate(values):
            imaginary = quat.GetImaginary()
            orientations[index] = (imaginary[0], imaginary[1], imaginary[2], quat.GetReal())
    return positions, orientations


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
