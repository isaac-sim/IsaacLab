# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Animated Gaussian-splat helpers for the PPISP camera demo.

Gaussian captures author motion as *tracks*: a ``ParticleField3DGaussianSplat`` prim whose
ancestors carry time-sampled xform ops (rigid motion), and/or whose per-particle
``positions``/``orientations`` arrays are themselves time sampled (deformable motion). Both
spellings of the per-particle arrays exist in the wild: the half-precision ``positionsh`` and
``orientationsh`` that NuRec exports, and the full-float ``positions`` and ``orientations``.

The demo samples USD once while constructing a playback object, then holds the Kit timeline at one
time code. Each rendered frame therefore writes the pre-sampled state to the active renderer:

* :class:`IsaacRtxFabricTrackPlayback` writes the populated Isaac RTX Fabric columns through
  persistent USDRT selections and Warp ``fabricarray`` handles.
* :class:`OVRTXGaussianTrackPlayback` streams the same state through OVRTX's renderer API.

Neither path authors USD during playback.

The Newton Warp renderer has no Gaussian-splat path, so it cannot play tracks back at all.

This module is imported by the demo scripts *after* :class:`~isaaclab.app.AppLauncher` has started
the app, because it imports :mod:`isaaclab.sim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom

GAUSSIAN_PRIM_TYPE_NAME = "ParticleField3DGaussianSplat"
"""USD prim type name of a Gaussian-splat particle field."""

POSITIONS_ATTR_NAMES = ("positionsh", "positions")
"""Candidate per-particle position attribute names, most specific first."""

ORIENTATIONS_ATTR_NAMES = ("orientationsh", "orientations")
"""Candidate per-particle orientation attribute names, most specific first."""


@wp.kernel(enable_backward=False)
def _write_fabric_particle_array(
    values: wp.array2d(dtype=Any),
    rows: wp.fabricarray(dtype=wp.uint32),
    output: wp.fabricarrayarray(dtype=Any),
):
    """Copy one vector-typed particle column for every selected Fabric prim.

    Warp specializes this generic kernel for each of the four supported element types
    (``vec3h``, ``quath``, ``vec3f``, and ``quatf``), avoiding four identical kernels.
    """
    prim_index, particle_index = wp.tid()
    output[prim_index, particle_index] = values[rows[prim_index], particle_index]


@wp.kernel(enable_backward=False)
def _write_fabric_transforms(
    values: wp.array(dtype=wp.mat44d),
    rows: wp.fabricarray(dtype=wp.uint32),
    output: wp.fabricarray(dtype=wp.mat44d),
):
    prim_index = wp.tid()
    output[prim_index] = values[rows[prim_index]]


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

    These are the prims that carry the track's rigid motion.
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


def _sample_transform_frames(
    xformable: UsdGeom.Xformable, frame_time_codes: list[float], num_envs: int
) -> np.ndarray:
    """Sample a local xform at every frame and broadcast it to duplicated envs."""
    return np.stack(
        [
            np.broadcast_to(
                np.asarray(xformable.GetLocalTransformation(Usd.TimeCode(time_code)), dtype=np.float64),
                (num_envs, 4, 4),
            )
            for time_code in frame_time_codes
        ]
    )


def _sample_particle_frames(attribute: Usd.Attribute, frame_time_codes: list[float], dtype: np.dtype) -> np.ndarray:
    """Sample one particle-array attribute at every frame with ``dtype``."""
    return np.stack([np.asarray(attribute.Get(Usd.TimeCode(time_code))) for time_code in frame_time_codes]).astype(
        dtype, copy=False
    )


class IsaacRtxFabricTrackPlayback:
    """Play Gaussian tracks by writing their populated Fabric attributes directly.

    The Isaac RTX renderer consumes the live Fabric stage. This class creates one USDRT selection
    per animated column after scene population, retains its Warp view, and writes a pre-sampled
    device frame through that view. The caller must invoke :meth:`write_array_attribute` after
    ``sim.step()`` has refreshed Fabric and before the camera is rendered. Selections remain valid
    only while stage topology is unchanged.

    Args:
        source_stage: Source stage used to sample authored animation once.
        tracks: Animated Gaussian tracks discovered on ``source_stage``.
        frame_time_codes: USD time code of each playback frame.
        num_envs: Number of duplicated environments on the live stage.
        device: Warp device used by the Fabric selection and sampled frames.
    """

    def __init__(
        self,
        source_stage: Usd.Stage,
        tracks: list[AnimatedGaussianTrack],
        frame_time_codes: list[float],
        num_envs: int,
        device: str,
    ):
        import usdrt

        from isaaclab.sim.utils.stage import get_current_stage

        self._device = device
        self._bindings: list[_FabricAnimationBinding] = []
        self._stage = get_current_stage(fabric=True)
        self._usdrt = usdrt
        default_prefix = f"{require_default_prim(source_stage).GetPath().pathString}/"

        for track in tracks:
            for xform_rel_path in track.animated_xform_rel_paths:
                xformable = UsdGeom.Xformable(source_stage.GetPrimAtPath(f"{default_prefix}{xform_rel_path}"))
                frames = _sample_transform_frames(xformable, frame_time_codes, num_envs)
                self._bindings.append(
                    self.bind_array_attribute(
                        [env_prim_path(env_id, xform_rel_path) for env_id in range(num_envs)],
                        "omni:fabric:localMatrix",
                        wp.array(frames, dtype=wp.mat44d, device=device),
                        _write_fabric_transforms,
                        is_array=False,
                    )
                )

            source_prim = source_stage.GetPrimAtPath(f"{default_prefix}{track.gaussian_rel_path}")
            prim_paths = [env_prim_path(env_id, track.gaussian_rel_path) for env_id in range(num_envs)]
            for attr_name, half_dtype, float_dtype in (
                (track.positions_attr_name, wp.vec3h, wp.vec3f),
                (track.orientations_attr_name, wp.quath, wp.quatf),
            ):
                if attr_name is None:
                    continue
                attribute = source_prim.GetAttribute(attr_name)
                is_half = attr_name.endswith("h")
                dtype = half_dtype if is_half else float_dtype
                frames = _sample_particle_frames(attribute, frame_time_codes, np.float16 if is_half else np.float32)
                if frames.shape[1] != track.num_particles:
                    raise RuntimeError(
                        f"{attr_name} on {track.gaussian_rel_path} sampled {frames.shape[1]} particles, expected"
                        f" {track.num_particles}."
                    )
                self._bindings.append(
                    self.bind_array_attribute(
                        prim_paths,
                        attr_name,
                        wp.array(frames, dtype=dtype, device=device),
                        _write_fabric_particle_array,
                    )
                )

    @property
    def is_empty(self) -> bool:
        """Whether no animated Fabric columns were bound."""
        return not self._bindings

    def bind_array_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        frames: wp.array,
        kernel: Any,
        *,
        is_array: bool = True,
    ) -> _FabricAnimationBinding:
        """Bind one populated Fabric column for repeated device-side writes.

        A private row attribute maps the unordered Fabric selection back to the duplicated-env
        row of ``frames``. It exists only on the Fabric stage and is never authored into USD.
        """
        row_attribute = f"isaaclab:ppispGaussianRow:{len(self._bindings)}"
        first_prim = self._stage.GetPrimAtPath(prim_paths[0])
        attribute = first_prim.GetAttribute(attribute_name)
        if not attribute.IsValid():
            raise RuntimeError(f"Fabric Gaussian attribute {attribute_name!r} does not exist on {prim_paths[0]!r}.")
        for row, path in enumerate(prim_paths):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Fabric Gaussian prim {path!r} does not exist.")
            prim.CreateAttribute(row_attribute, self._usdrt.Sdf.ValueTypeNames.UInt, True).Set(row)
        selection = self._stage.SelectPrims(
            require_attrs=[
                (self._usdrt.Sdf.ValueTypeNames.UInt, row_attribute, self._usdrt.Usd.Access.Read),
                (attribute.GetTypeName(), attribute_name, self._usdrt.Usd.Access.ReadWrite),
            ],
            device=self._device,
        )
        if selection.GetCount() != len(prim_paths):
            raise RuntimeError(
                f"Fabric binding for {attribute_name!r} matched {selection.GetCount()} prims, "
                f"expected {len(prim_paths)}."
            )
        return _FabricAnimationBinding(
            selection=selection,
            rows=wp.fabricarray(selection, row_attribute),
            output=wp.fabricarray(selection, attribute_name),
            frames=frames,
            kernel=kernel,
            is_array=is_array,
        )

    def write_array_attribute(self, frame_index: int) -> None:
        """Write every bound Gaussian column for ``frame_index`` directly into Fabric.

        Args:
            frame_index: Index into the frame time codes supplied at construction.

        Raises:
            IndexError: If ``frame_index`` is outside the sampled frame range.
        """
        if self._bindings and not 0 <= frame_index < self._bindings[0].frames.shape[0]:
            raise IndexError(f"Fabric Gaussian frame index {frame_index} is out of range.")
        for binding in self._bindings:
            binding.selection.PrepareForReuse()
            values = binding.frames[frame_index]
            # ``values`` is one frame of an array whose vector dtype folds the particle components
            # into its element type, so it is one-dimensional: [num_particles].  The frame array
            # retains the [num_frames, num_particles] shape needed to size the 2-D launch.
            dim = (binding.output.size, binding.frames.shape[1]) if binding.is_array else binding.output.size
            wp.launch(binding.kernel, dim=dim, inputs=[values, binding.rows, binding.output], device=self._device)

    def close(self) -> None:
        """Release Fabric selections before the stage is torn down."""
        self._bindings.clear()
        self._stage = None


@dataclass
class _FabricAnimationBinding:
    """One persistent Fabric selection and its sampled device frames."""

    selection: Any
    rows: Any
    output: Any
    frames: wp.array
    kernel: Any
    is_array: bool


class OVRTXGaussianTrackPlayback:
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
                # Every env shares the track's local transform, but OVRTX receives one matrix per path.
                frames = _sample_transform_frames(xformable, frame_time_codes, num_envs)
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
                frames = _sample_particle_frames(attribute, frame_time_codes, np.float32)
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
        """Total size [B] of the device ring buffers"""
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
