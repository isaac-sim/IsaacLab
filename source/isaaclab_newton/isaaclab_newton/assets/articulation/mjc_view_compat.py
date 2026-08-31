# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Back-fill the custom-frequency view API that Newton gained in newton-physics/newton#4017.

Newton 1.6 exposes ``mujoco:actuator`` through :class:`~newton.selection.ArticulationView`:
``custom_frequency_counts`` reports the per-articulation count, and ``get_attribute`` /
``set_attribute`` reach the actuator rows. On 1.5 only ``mujoco:tendon`` is admitted -- the
custom-frequency branch rejects every other namespace -- so the same calls raise.

#4017 landed on Newton's main but was deliberately not backported to the 1.5 patch series, so it
arrives with 1.6. :func:`ensure_custom_frequency_api` supplies the missing surface in the
meantime, which keeps
:mod:`~isaaclab_newton.assets.articulation.mjc_tendon_control` written against the 1.6 API on
both versions instead of carrying two code paths.

Delete this module once the pinned Newton is 1.6 or newer; the only caller reverts to passing
``articulation.root_view`` straight through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from newton import Control, Model, State
    from newton.selection import ArticulationView

__all__ = ["ensure_custom_frequency_api"]


@wp.kernel
def _scatter_into_flat_rows(
    values: wp.array3d(dtype=wp.float32),
    rows: wp.array3d(dtype=wp.int32),
    flat: wp.array(dtype=wp.float32),
) -> None:
    """Write view-shaped values into the model-global rows they belong to."""
    world, instance, column = wp.tid()
    flat[rows[world, instance, column]] = values[world, instance, column]


def ensure_custom_frequency_api(view: ArticulationView, model: Model) -> ArticulationView:
    """Return a view that exposes Newton 1.6's custom-frequency API.

    Args:
        view: Articulation view to adapt.
        model: Newton model the view selects from.

    Returns:
        *view* unchanged when Newton already provides the API, otherwise a wrapper adding it.
    """
    if hasattr(view, "custom_frequency_counts"):
        return view
    return _CustomFrequencyView(view, model)


class _CustomFrequencyView:
    """Adds ``mujoco:actuator`` support to a Newton 1.5 :class:`ArticulationView`.

    Everything except the custom-frequency surface is delegated to the wrapped view, so the
    layout, masks and every other attribute stay Newton's own.
    """

    def __init__(self, view: ArticulationView, model: Model):
        self._view = view
        self._model = model
        self._actuator_rows: wp.array | None = None

        counts = dict(getattr(model, "custom_frequency_counts", {}) or {})
        tendon_total = int(counts.get("mujoco:tendon", 0))

        # Which model-global actuator rows belong to each articulation this view holds. Ownership
        # is stated in the data -- ``actuator_target_label`` is the driven joint's prim path and
        # ``articulation_label`` is the articulation's -- so it is read, never derived from
        # strides. MEASURED: a 2-env handover model reports 80 actuators over 6 articulations,
        # 20 per hand and none for the object, so no division recovers the per-hand count.
        instances = max(int(view.count), 1)
        self._tendon_count = int(getattr(view, "tendon_count", 0)) or (tendon_total // instances if tendon_total else 0)
        self._rows_per_instance = _actuator_rows_per_instance(view, model, self._tendon_count)
        self._total_actuator_rows = int(_to_numpy(model.mujoco.actuator_world).size)
        self._actuator_count = int(self._rows_per_instance.shape[1])

        self.custom_frequency_counts: dict[str, int] = {}
        if self._actuator_count:
            self.custom_frequency_counts["mujoco:actuator"] = self._actuator_count
        if self._tendon_count:
            self.custom_frequency_counts["mujoco:tendon"] = self._tendon_count

        self.custom_frequency_labels: dict[str, list[str]] = {}
        labels = getattr(getattr(model, "mujoco", None), "actuator_target_label", None)
        if labels is not None and self._actuator_count:
            self.custom_frequency_labels["mujoco:actuator"] = [str(labels[row]) for row in self._rows_per_instance[0]]
        tendon_names = getattr(view, "tendon_names", None)
        if tendon_names:
            self.custom_frequency_labels["mujoco:tendon"] = list(tendon_names)

    def __getattr__(self, name: str) -> Any:
        """Delegate everything this wrapper does not define to the wrapped view."""
        return getattr(self._view, name)

    def get_attribute(self, name: str, source: Model | State | Control):
        """Read *name*, handling ``mujoco:actuator`` attributes Newton 1.5 rejects.

        The gather runs on the host, which synchronizes the device. That is affordable because
        the tendon adapter reads actuator attributes once while resolving its mapping; the
        per-step path is :meth:`set_attribute`, which stays on the device.
        """
        if not self._is_actuator_attribute(name):
            return self._view.get_attribute(name, source)
        rows = self._actuator_row_table().numpy()
        # Gather through numpy: Warp arrays do not accept an index array.
        flat = _to_numpy(self._flat_attribute(name, source))
        return wp.array(flat[rows], device=self._view.device)

    def set_attribute(self, name: str, source: Model | State | Control, values) -> None:
        """Write *values*, handling ``mujoco:actuator`` attributes Newton 1.5 rejects."""
        if not self._is_actuator_attribute(name):
            self._view.set_attribute(name, source, values)
            return
        rows = self._actuator_row_table()
        if tuple(values.shape) != tuple(rows.shape):
            raise ValueError(f"Expected values shaped {tuple(rows.shape)} for '{name}', got {tuple(values.shape)}.")
        wp.launch(
            _scatter_into_flat_rows,
            dim=rows.shape,
            inputs=[values, rows],
            outputs=[self._flat_attribute(name, source)],
            device=values.device,
        )

    def _is_actuator_attribute(self, name: str) -> bool:
        """Whether *name* addresses the ``mujoco:actuator`` frequency Newton 1.5 will not resolve."""
        return name.startswith("mujoco.actuator") or name == "mujoco.ctrl"

    def _flat_attribute(self, name: str, source: Model | State | Control):
        """Resolve a dotted ``mujoco.*`` name to the flat array on *source*."""
        obj: Any = source
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def _actuator_row_table(self) -> wp.array:
        """Model-global actuator row per ``(world, instance, actuator)``, built once."""
        if self._actuator_rows is not None:
            return self._actuator_rows
        view = self._view
        rows = self._rows_per_instance.reshape(view.world_count, view.count_per_world, -1)
        _assert_rows_usable(rows, self._total_actuator_rows, self._actuator_count)
        self._actuator_rows = wp.array(rows.astype(np.int32), dtype=wp.int32, device=view.device)
        return self._actuator_rows


def _actuator_rows_per_instance(view: ArticulationView, model: Model, tendon_count: int) -> np.ndarray:
    """Model-global actuator rows owned by each articulation the view selects.

    Newton 1.5 states no actuator-to-articulation mapping, and the two candidate signals are both
    unusable: ``actuator_trnid`` is left at -1 for USD imports, and ``articulation_label`` is the
    template path, so every clone reports the same string -- MEASURED, a 2-clone Shadow Hand scene
    reports ``['/World/Env_0/Robot', '/World/Env_0/Robot']`` with all 40 actuator labels under
    ``Env_0``.

    What is reliable is which world a row belongs to. Partition the actuators by
    ``actuator_world``, count the articulations in that world from its tendons, and take this
    articulation's block by its position among them. Both counts come from the same MuJoCo
    ordering, so the blocks line up.

    Args:
        view: Articulation view selecting the articulations.
        model: Newton model carrying the world assignments.
        tendon_count: Fixed tendons per articulation.

    Returns:
        Rows per instance, shape ``(view.count, actuators_per_articulation)``.

    Raises:
        ValueError: If a world does not divide evenly into equal actuator blocks.
    """
    actuator_worlds = _to_numpy(model.mujoco.actuator_world).astype(np.int64, copy=False)
    tendon_worlds = _to_numpy(model.mujoco.tendon_world).astype(np.int64, copy=False)
    ordinal = _ordinal_within_world(view, tendon_count)

    per_instance = []
    for world in range(view.world_count):
        rows = np.flatnonzero(actuator_worlds == world)
        # Tendons and actuators are emitted per articulation in the same order, so the tendon
        # count reveals how many articulations share this world.
        articulations = int(np.count_nonzero(tendon_worlds == world)) // max(tendon_count, 1)
        if articulations <= 0 or rows.size % articulations:
            raise ValueError(
                f"World {world} holds {rows.size} MuJoCo actuator rows across {articulations}"
                " articulation(s), which does not divide into equal blocks."
            )
        width = rows.size // articulations
        for instance in range(view.count_per_world):
            start = (ordinal + instance) * width
            per_instance.append(rows[start : start + width].tolist())
    return np.array(per_instance, dtype=np.int64).reshape(view.count, -1)


def _ordinal_within_world(view: ArticulationView, tendon_count: int) -> int:
    """Position of this view's articulation among those sharing a world.

    The view's tendon layout offset is this articulation's first tendon row, and tendons are
    emitted per articulation, so dividing by the per-articulation tendon count gives its index.
    """
    layout = getattr(view, "frequency_layouts", {}).get("mujoco:tendon")
    if layout is None or not tendon_count:
        return 0
    return int(layout.offset) // tendon_count


def _assert_rows_usable(rows: np.ndarray, total_rows: int, actuator_count: int) -> None:
    """Reject a row table that would silently drive the wrong actuators.

    A join that matched the wrong articulation still yields rows inside the model, so being in
    range is not enough -- every row has to be distinct, or two instances share actuators.

    Raises:
        ValueError: If any row is out of range or two slots share one.
    """
    if rows.size and (rows.min() < 0 or rows.max() >= total_rows):
        raise ValueError(
            f"MuJoCo actuator rows fall outside the model's {total_rows} rows"
            f" (got {rows.min()}..{rows.max()}); the articulation labels did not join cleanly."
        )
    if np.unique(rows).size != rows.size:
        raise ValueError(
            "MuJoCo actuator rows repeat, so two articulations would share actuators."
            f" Expected {rows.size} distinct rows across {actuator_count} actuators per instance."
        )


def _to_numpy(value) -> np.ndarray:
    """Return *value* as a numpy array, whether it is a Warp array already or a sequence."""
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)
