# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay of ``.viser`` scene recordings inside the Newton GL window.

A recording stores each shape's geometry once and then the poses of its instances over time. The player draws
those shapes as separate named batches next to the live simulation, so it neither reads nor writes the Newton
model.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .newton_visualizer import NewtonViewerGL

logger = logging.getLogger(__name__)

REPLAY_PREFIX = "/replay"
"""Name prefix of every object the player adds to the viewer."""

_MESH_MESSAGES = ("BatchedMeshesMessage", "BatchedGlbMessage")


@dataclass
class ReplayShape:
    """One batch of instances of a mesh, with its poses over time."""

    name: str
    vertices: np.ndarray
    """Vertex positions [m], shape (V, 3)."""
    indices: np.ndarray
    """Triangle vertex indices, shape (3F,)."""
    positions: np.ndarray
    """Instance positions [m] per frame, shape (T, N, 3)."""
    wxyzs: np.ndarray
    """Instance orientations per frame as (w, x, y, z), shape (T, N, 4)."""
    scales: np.ndarray
    """Instance scales, shape (N, 3)."""
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    texture: np.ndarray | None = None
    color: tuple[float, float, float] | None = None
    """Base color [0, 1] used when there is no texture."""
    colors: np.ndarray | None = None
    """Per-instance colors [0, 1], shape (N, 3)."""
    roughness: float | None = None
    metallic: float | None = None
    xforms: np.ndarray = field(default=None, repr=False)
    """Newton transforms (position, then xyzw quaternion) per frame, shape (T, N, 7)."""
    frame_scales: np.ndarray | None = field(default=None, repr=False)
    """Instance scales per frame, shape (T, N, 3), or None when they never change."""
    slot_hidden: np.ndarray | None = field(default=None, repr=False)
    """Whether the slot of each instance is hidden, set while publishing a slotted recording."""
    worlds: np.ndarray | None = field(default=None, repr=False)
    """World index of each instance, shape (N,), or None when the recording has no slot information. Instances of
    terrain and other shared shapes have ``-1``."""

    @property
    def static(self) -> bool:
        """Whether no instance ever moves, as for terrain and ground planes."""
        return self.frame_scales is None and bool(np.all(self.xforms == self.xforms[:1]))


@dataclass
class RecordedSlot:
    """One Policy Ghost slot of a recorded session: a slice of environments driven by one checkpoint."""

    index: int
    """Slot index. Slot 0 is the reference."""
    label: str
    """Short name of the slot's checkpoint."""
    color: tuple[float, float, float] | None
    """Tint color of the slot's robots, or None for the reference."""


@dataclass
class ViserRecording:
    """A decoded ``.viser`` file: static geometry plus per-frame instance poses."""

    path: str
    times: np.ndarray
    """Time of each frame [s]. Frame 0 is the recorded initial state."""
    shapes: list[ReplayShape]
    slots: list[RecordedSlot] = field(default_factory=list)
    """Policy Ghost slots of the session, empty for a recording without them."""
    slice_size: int = 0
    """Number of environments per slot."""

    @property
    def duration(self) -> float:
        """Length of the recording [s]."""
        return float(self.times[-1])

    @property
    def num_frames(self) -> int:
        return len(self.times)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (min, max) corners of all instance positions in the first frame [m]."""
        points = np.concatenate([shape.positions[0] for shape in self.shapes])
        return points.min(axis=0), points.max(axis=0)

    def frame_at(self, time_s: float) -> int:
        """Return the last frame at or before ``time_s``."""
        return int(np.clip(np.searchsorted(self.times, time_s, side="right") - 1, 0, self.num_frames - 1))


def _read_container(path: str) -> tuple[dict, Any]:
    """Decompress a ``.viser`` file, returning its msgpack header and a reader for the binary buffers."""
    import msgspec
    import zstandard

    raw = Path(path).read_bytes()
    inner = zstandard.ZstdDecompressor().decompress(raw[8:], max_output_size=int.from_bytes(raw[:8], "little"))
    header_length = int.from_bytes(inner[:8], "little")
    meta = msgspec.msgpack.decode(inner[8 : 8 + header_length])
    offsets, cursor = [], 8 + header_length
    for length in meta["binaryBufferLengths"]:
        cursor += (8 - cursor % 8) % 8  # Buffers are 8-byte aligned.
        offsets.append(cursor)
        cursor += length

    def buffer(placeholder: dict) -> np.ndarray:
        index, dtype = placeholder["__binary_index"], np.dtype(placeholder["dtype"])
        return np.frombuffer(inner, dtype=dtype, count=meta["binaryBufferLengths"][index] // dtype.itemsize,
                             offset=offsets[index])  # fmt: skip

    return meta, buffer


def _glb_mesh(glb: bytes) -> dict[str, Any]:
    """Extract geometry and material from a GLB blob, merging its parts into one mesh."""
    import trimesh

    parts = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene").dump()
    mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    out: dict[str, Any] = {
        "vertices": np.asarray(mesh.vertices, np.float32),
        "indices": np.asarray(mesh.faces, np.int32).reshape(-1),
        "normals": np.asarray(mesh.vertex_normals, np.float32),
    }
    material = getattr(mesh.visual, "material", None)
    uv = getattr(mesh.visual, "uv", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is not None and uv is not None and len(parts) == 1:
        out["texture"] = np.asarray(image.convert("RGB"))
        out["uvs"] = np.asarray(uv, np.float32)
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        factor = np.asarray(factor, np.float64)
        out["color"] = tuple(float(c) for c in (factor[:3] / 255.0 if factor.max() > 1.0 else factor[:3]))
    for key in ("roughness", "metallic"):
        value = getattr(material, f"{key}Factor", None)
        if value is not None:
            out[key] = float(value)
    return out


def _scales(placeholder: dict | None, count: int, buffer: Any) -> np.ndarray:
    """Return per-instance scales, shape (count, 3); recordings may omit them or store one number per instance."""
    if placeholder is None:
        return np.ones((count, 3), np.float32)
    return np.broadcast_to(buffer(placeholder).astype(np.float32).reshape(count, -1), (count, 3)).copy()


def load_viser_recording(path: str) -> ViserRecording:
    """Decode a ``.viser`` file into shapes and per-frame poses.

    Args:
        path: Recording written by Newton's Viser viewer.

    Returns:
        The decoded recording. Messages other than mesh batches and their pose updates are ignored.

    Raises:
        ValueError: If the file has no mesh batches.
    """
    meta, buffer = _read_container(path)
    created: dict[str, dict[str, Any]] = {}
    updates: dict[float, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for time_s, message in meta["messages"]:
        kind = message["type"]
        if kind in _MESH_MESSAGES:
            props = message["props"]
            if kind == "BatchedGlbMessage":
                mesh = _glb_mesh(props["glb_data"])
            else:
                mesh = {
                    "vertices": buffer(props["vertices"]).reshape(-1, 3).astype(np.float32),
                    "indices": buffer(props["faces"]).reshape(-1).astype(np.int32),
                }
                colors = buffer(props["batched_colors"]) if props.get("batched_colors") else None
                if colors is not None and len(colors) >= 3:
                    mesh["color"] = tuple(float(c) / 255.0 for c in colors[:3])
                    mesh["colors"] = colors.reshape(-1, 3).astype(np.float32) / 255.0
            positions = buffer(props["batched_positions"]).reshape(-1, 3).astype(np.float32)
            created[message["name"]] = {
                **mesh,
                "positions": positions,
                "wxyzs": buffer(props["batched_wxyzs"]).reshape(-1, 4).astype(np.float32),
                "scales": _scales(props.get("batched_scales"), len(positions), buffer),
            }
        elif kind == "SceneNodeUpdateMessage" and message["name"] in created:
            for key in ("batched_positions", "batched_wxyzs", "batched_scales"):
                if key in message["updates"]:
                    count = len(created[message["name"]]["positions"])
                    values = buffer(message["updates"][key]).astype(np.float32)
                    if key == "batched_scales":  # One number per instance or one per axis.
                        values = np.broadcast_to(values.reshape(count, -1), (count, 3)).copy()
                    else:
                        values = values.reshape(-1, 3 if key == "batched_positions" else 4)
                    updates[round(time_s, 6)][message["name"]][key] = values
    if not created:
        raise ValueError(f"{path} contains no mesh batches to replay.")

    update_times = np.array(sorted(updates))
    # The recorder's clock starts when its viewer is created, so the first update can sit far from zero; rebase it
    # to one typical step after the initial pose, or the scrubber would show a still frame until that time.
    step = float(np.median(np.diff(update_times))) if len(update_times) > 1 else 1.0 / 60.0
    times = np.array([0.0, *(update_times - update_times[0] + step)]) if len(update_times) else np.zeros(1)
    updates = {round(float(rebased), 6): updates[round(float(raw), 6)] for raw, rebased in zip(update_times, times[1:])}
    shapes = []
    for name, data in created.items():
        count = len(data["positions"])
        positions = np.repeat(data.pop("positions")[None], len(times), axis=0)
        wxyzs = np.repeat(data.pop("wxyzs")[None], len(times), axis=0)
        animated = any("batched_scales" in updates[round(float(t), 6)].get(name, {}) for t in times[1:])
        frame_scales = np.repeat(data["scales"][None], len(times), axis=0) if animated else None
        for frame, time_s in enumerate(times[1:], start=1):
            positions[frame:], wxyzs[frame:] = positions[frame - 1], wxyzs[frame - 1]  # Carry the last pose forward.
            update = updates[round(float(time_s), 6)].get(name, {})
            if "batched_positions" in update:
                positions[frame:] = update["batched_positions"]
            if "batched_wxyzs" in update:
                wxyzs[frame:] = update["batched_wxyzs"]
            if frame_scales is not None:
                frame_scales[frame:] = frame_scales[frame - 1]
                if "batched_scales" in update:
                    frame_scales[frame:] = update["batched_scales"]
        if data.get("colors") is not None and len(data["colors"]) != count:
            data["colors"] = np.broadcast_to(data["colors"][:1], (count, 3)).copy()
        if data.get("colors") is None:
            data["colors"] = np.broadcast_to(
                np.asarray(data.get("color") or (1.0, 1.0, 1.0), np.float32), (count, 3)
            ).copy()
        xforms = np.concatenate([positions, wxyzs[..., [1, 2, 3, 0]]], axis=-1).astype(np.float32)
        assert xforms.shape == (len(times), count, 7)
        shapes.append(
            ReplayShape(name=name, positions=positions, wxyzs=wxyzs, xforms=xforms, frame_scales=frame_scales, **data)
        )
    recording = ViserRecording(path=str(path), times=times, shapes=shapes)
    _attach_slots(recording)
    return recording


def sidecar_path(path: str | Path) -> Path:
    """Return the JSON file that describes the Policy Ghost slots of the recording at ``path``."""
    return Path(path).with_suffix(".json")


def write_slot_sidecar(
    path: str | Path,
    slice_size: int,
    slots: list[RecordedSlot],
    shape_worlds: dict[str, list[int]],
) -> None:
    """Write the slot description that lets a replay tint, label, and hide each slot of a ghost session.

    Args:
        path: The ``.viser`` file the description belongs to.
        slice_size: Number of environments per slot.
        slots: The session's slots.
        shape_worlds: World index of every instance, by the name of its shape batch.
    """
    document = {
        "version": 1,
        "slice_size": slice_size,
        "slots": [{"index": slot.index, "label": slot.label, "color": slot.color} for slot in slots],
        "shapes": shape_worlds,
    }
    sidecar_path(path).write_text(json.dumps(document))


def _attach_slots(recording: ViserRecording) -> None:
    """Add the slot description next to the file to ``recording``, when there is one that matches."""
    file = sidecar_path(recording.path)
    if not file.is_file():
        return
    try:
        document = json.loads(file.read_text())
        slots = [
            RecordedSlot(item["index"], item["label"], None if item["color"] is None else tuple(item["color"]))
            for item in document["slots"]
        ]
        worlds = document["shapes"]
        slice_size = int(document["slice_size"])
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("[NewtonReplay] Ignoring the unreadable slot description %s.", file)
        return
    matched = 0
    for shape in recording.shapes:
        values = worlds.get(shape.name)
        if values is not None and len(values) == len(shape.xforms[0]):
            shape.worlds = np.asarray(values, np.int32)
            matched += 1
    if matched:
        recording.slots, recording.slice_size = slots, slice_size
    else:
        logger.warning("[NewtonReplay] The slot description %s does not match %s.", file, recording.path)


def list_recordings(folder: str) -> list[Path]:
    """Return the ``.viser`` files below ``folder``, newest first."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.viser"), key=lambda file: file.stat().st_mtime, reverse=True)


class NewtonReplayPlayer:
    """Plays a decoded recording as extra shapes in a Newton GL viewer, independent of the simulation.

    Loading decodes on a worker thread, and the meshes are registered with the viewer by the next :meth:`update`
    call, which must run inside a viewer frame.
    """

    def __init__(self, viewer_getter: Any, prefix: str = REPLAY_PREFIX):
        """Create the player.

        Args:
            viewer_getter: Callable returning the current :class:`NewtonViewerGL`, or None when it is closed.
            prefix: Name prefix of every object this player adds to the viewer; unique per player.
        """
        self._viewer_getter = viewer_getter
        self.prefix = prefix
        self.recording: ViserRecording | None = None
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.time = 0.0
        self.opacity = 1.0
        self.color: tuple[float, float, float] | None = None
        """Tint color, or None to keep the recording's colors."""
        self.tint = 0.0
        """How strongly :attr:`color` replaces the recording's colors and textures, in ``[0, 1]``."""
        self.visible = True
        self.slot_visible: dict[int, bool] = {}
        """Whether each slot of a Policy Ghost recording is shown; a missing slot is shown."""
        self.hide_static = False
        """Skip shapes that never move, which the live scene already draws."""
        self.offset = np.zeros(3, np.float32)
        self.message = ""
        self._thread: threading.Thread | None = None
        self._pending: ViserRecording | Exception | None = None
        self._registered: ViserRecording | None = None
        self._drawn: tuple | None = None
        self._last_clock = time.perf_counter()

    @property
    def loading(self) -> bool:
        """Whether a file is being decoded."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def active(self) -> bool:
        """Whether a recording is loaded."""
        return self.recording is not None

    @property
    def name(self) -> str:
        """File name of the loaded recording, or an empty string."""
        return Path(self.recording.path).name if self.recording is not None else ""

    def load(self, path: str) -> None:
        """Start decoding ``path`` in the background, replacing any loaded recording once it is ready."""
        if self.loading:
            return
        self.message = f"Loading {Path(path).name}..."

        def work() -> None:
            try:
                self._pending = load_viser_recording(path)
            except Exception as exc:
                logger.exception("[NewtonReplay] Could not load %s.", path)
                self._pending = exc

        self._thread = threading.Thread(target=work, name="viser-replay-load", daemon=True)
        self._thread.start()

    def unload(self) -> None:
        """Remove the replay from the viewer."""
        viewer = self._viewer_getter()
        if viewer is not None:
            for key in [key for key in viewer.objects if key.startswith(f"{self.prefix}/")]:
                del viewer.objects[key]
        self.recording = self._registered = self._drawn = None
        self.playing = False
        self.message = ""

    def invalidate(self) -> None:
        """Publish the current frame again, for example after the viewer cleared its instance batches."""
        self._drawn = None

    def seek(self, time_s: float) -> None:
        """Move to ``time_s`` [s] within the recording."""
        if self.recording is not None:
            self.time = float(np.clip(time_s, 0.0, self.recording.duration))

    def update(self) -> None:
        """Advance the clock and publish the current frame; call once per viewer frame, inside the frame."""
        now = time.perf_counter()
        dt, self._last_clock = now - self._last_clock, now
        viewer = self._viewer_getter()
        if viewer is None:
            return
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if isinstance(pending, Exception):
                self.message = f"Could not load: {pending}"
            else:
                self.unload()
                self.recording = pending
                self.message = (
                    f"Loaded {Path(pending.path).name} ({pending.duration:.1f} s, {pending.num_frames} frames)"
                )
        recording = self.recording
        if recording is None:
            return
        if self._registered is not recording:
            self._register(viewer, recording)
        if self.playing and recording.duration > 0.0:
            self.time += dt * self.speed
            if self.time > recording.duration:
                self.time = 0.0 if self.loop else recording.duration
                self.playing = self.loop
        frame = recording.frame_at(self.time)
        state = (
            frame, self.opacity, self.hide_static, tuple(self.offset), self.visible, self.color, self.tint,
            tuple(sorted(self.slot_visible.items())),
        )  # fmt: skip
        if state != self._drawn:
            self._publish(viewer, recording, frame)
            self._drawn = state

    def _register(self, viewer: NewtonViewerGL, recording: ViserRecording) -> None:
        for index, shape in enumerate(recording.shapes):
            viewer.log_mesh(
                f"{self.prefix}/mesh_{index}",
                wp.array(shape.vertices, dtype=wp.vec3, device=viewer.device),
                wp.array(shape.indices, dtype=wp.int32, device=viewer.device),
                normals=None if shape.normals is None else wp.array(shape.normals, dtype=wp.vec3, device=viewer.device),
                uvs=None if shape.uvs is None else wp.array(shape.uvs, dtype=wp.vec2, device=viewer.device),
                texture=shape.texture,
                hidden=True,
                color=shape.color,
                roughness=shape.roughness,
                metallic=shape.metallic,
            )
        self._registered = recording

    def _publish(self, viewer: NewtonViewerGL, recording: ViserRecording, frame: int) -> None:
        for index, shape in enumerate(recording.shapes):
            xforms = shape.xforms[frame].copy()
            xforms[:, :3] += self.offset
            count = len(xforms)
            scales = shape.scales if shape.frame_scales is None else shape.frame_scales[frame]
            tint_color, tint_amount, opacities = self._appearance(recording, shape, count)
            if tint_color is not None and shape.slot_hidden is not None:
                scales = scales * (~shape.slot_hidden)[:, None]
            # Material layout: roughness, metallic, checker, texture enable.
            textured = np.full(count, shape.texture is not None) & (tint_amount < 0.99)
            materials = np.zeros((count, 4), np.float32)
            materials[:, 0] = 0.5 if shape.roughness is None else shape.roughness
            materials[:, 1] = shape.metallic or 0.0
            materials[:, 3] = textured
            colors = np.where(textured[:, None], 1.0, shape.colors).astype(np.float32)
            if tint_color is not None:
                colors = colors + (tint_color - colors) * tint_amount[:, None]
            viewer.log_instances(
                f"{self.prefix}/instances_{index}",
                f"{self.prefix}/mesh_{index}",
                wp.array(xforms, dtype=wp.transform, device=viewer.device),
                wp.array(scales, dtype=wp.vec3, device=viewer.device),
                wp.array(colors, dtype=wp.vec3, device=viewer.device),
                wp.array(materials, dtype=wp.vec4, device=viewer.device),
                hidden=not self.visible or (self.hide_static and shape.static),
                opacities=None
                if np.all(opacities >= 1.0)
                else wp.array(opacities, dtype=wp.float32, device=viewer.device),
            )

    def _appearance(self, recording: ViserRecording, shape: ReplayShape, count: int):
        """Return the tint color, tint strength, and opacity of each instance of ``shape``.

        A recording of a Policy Ghost session tints and fades the ghost slots and leaves the reference slot and
        shared shapes alone; any other recording takes the player's single tint color.
        """
        if recording.slots and shape.worlds is not None and recording.slice_size:
            slots = np.where(shape.worlds >= 0, shape.worlds // recording.slice_size, 0)
            top = max(slot.index for slot in recording.slots)
            slots = np.clip(slots, 0, top)
            color, amount, opacity = np.zeros((count, 3), np.float32), np.zeros(count, np.float32), np.ones(count)
            for slot in recording.slots:
                if slot.color is not None:
                    mask = slots == slot.index
                    color[mask], amount[mask], opacity[mask] = slot.color, self.tint, self.opacity
            shown = np.array([self.slot_visible.get(i, True) for i in range(top + 1)])
            shape.slot_hidden = ~shown[slots] & (shape.worlds >= 0)  # Shared shapes always show.
            return color, amount, opacity.astype(np.float32)
        shape.slot_hidden = None
        if self.color is None or self.tint <= 0.0:
            return None, np.zeros(count, np.float32), np.full(count, self.opacity, np.float32)
        color = np.broadcast_to(np.asarray(self.color, np.float32), (count, 3))
        return color, np.full(count, self.tint, np.float32), np.full(count, self.opacity, np.float32)
