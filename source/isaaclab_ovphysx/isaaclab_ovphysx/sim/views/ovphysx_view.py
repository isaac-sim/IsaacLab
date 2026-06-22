# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""String-keyed view over OVPhysX ``TensorBinding`` handles.

PROTOTYPE for the design at
``docs/superpowers/specs/2026-06-17-ovphysx-view-design.md``.

OVPhysX exposes physics attributes as a loose ``dict[TensorType, TensorBinding]``
with no view object -- unlike Newton's ``selection.ArticulationView`` or PhysX's
typed tensor views. :class:`OvPhysxView` wraps those bindings for a single prim
pattern behind a string-keyed ``get_attribute`` / ``set_attribute`` surface that
mirrors Newton's selection ergonomics, but is simpler: there is no
``Model``/``State``/``Control`` source object because the :class:`TensorType`
already implies where the data lives.

Attribute names are the **lowercased** ``TensorType`` enum members, derived
directly from the wheel enum (no hand-maintained table)::

    view.get_attribute("articulation_dof_stiffness")
    view.set_attribute("rigid_body_pose", values, mask=env_mask)

Known follow-ups (see design doc §10 and §7):

* Buffers are :class:`warp.array` for consistency with the existing
  ``_binding_read`` / ``_binding_write`` helpers; a torch-returning convenience
  can wrap this (``wp.to_torch(view.get_attribute(...))``).
* ``_CPU_ONLY_NAMES`` mirrors ``isaaclab_ovphysx.tensor_types._CPU_ONLY_TYPES``;
  these should be consolidated, ideally sourced from wheel writability/placement
  metadata once it is exposed.
* The read-only set is hard-coded pending a wheel ``is_writable`` query.
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from isaaclab_ovphysx._runtime import import_ovphysx

logger = logging.getLogger(__name__)

# Pure-Python enum (no native dependency); safe to import regardless of USD state.
TensorType = import_ovphysx("ovphysx.types").TensorType

# Tensor types that are read-only on the wheel today. The first group is marked
# read-only in the enum; the computed-dynamics group is read-only in practice.
# TODO(ovphysx): source this from a wheel writability query (design doc §7 ask 3).
_READ_ONLY_NAMES: frozenset[str] = frozenset(
    {
        "rigid_body_acceleration",
        "rigid_body_inv_mass",
        "rigid_body_inv_inertia",
        "articulation_link_acceleration",
        "articulation_body_inv_mass",
        "articulation_body_inv_inertia",
        "articulation_dof_projected_joint_force",
        # computed dynamics -- read-only in practice (confirm via wheel metadata)
        "articulation_jacobian",
        "articulation_mass_matrix",
        "articulation_coriolis_and_centrifugal_force",
        "articulation_gravity_force",
    }
)

# DOF/body property tensor types are CPU-resident even on GPU sims, so their
# bindings must be read/written through a CPU buffer. Mirrors
# ``isaaclab_ovphysx.tensor_types._CPU_ONLY_TYPES`` (consolidate later).
_CPU_ONLY_NAMES: frozenset[str] = frozenset(
    {
        "articulation_dof_stiffness",
        "articulation_dof_damping",
        "articulation_dof_limit",
        "articulation_dof_max_velocity",
        "articulation_dof_max_force",
        "articulation_dof_armature",
        "articulation_dof_friction_properties",
        "articulation_body_mass",
        "articulation_body_com_pose",
        "articulation_body_inertia",
        "articulation_body_inv_mass",
        "articulation_body_inv_inertia",
        "rigid_body_mass",
        "rigid_body_com_pose",
        "rigid_body_inertia",
        "rigid_body_inv_mass",
        "rigid_body_inv_inertia",
    }
)


class OvPhysxViewError(RuntimeError):
    """Raised for unknown attribute names, read-only writes, or prims a name cannot bind to."""


# -----------------------------------------------------------------------------
# Pure helpers (no native simulation required; testable against ``ovphysx.types``)
# -----------------------------------------------------------------------------


def attribute_vocabulary() -> list[str]:
    """Return every valid attribute name (sorted lowercased ``TensorType`` members)."""
    return sorted(t.name.lower() for t in TensorType if t.name != "INVALID")


def resolve_tensor_type(name: str) -> Any:
    """Resolve a lowercased attribute name to its :class:`TensorType` member.

    Args:
        name: Lowercased enum name, e.g. ``"articulation_dof_stiffness"``.

    Returns:
        The matching :class:`TensorType` member.

    Raises:
        OvPhysxViewError: If the name is not a valid ``TensorType`` member.
    """
    try:
        tt = TensorType[name.upper()]
    except KeyError:
        raise OvPhysxViewError(
            f"Unknown attribute {name!r}. Valid names are the lowercased TensorType members, "
            f"e.g. {attribute_vocabulary()[:4]} ... ({len(attribute_vocabulary())} total)."
        ) from None
    if tt.name == "INVALID":
        raise OvPhysxViewError(f"{name!r} is not an addressable attribute.")
    return tt


def tensor_type_name(tensor_type: Any) -> str:
    """Return the canonical lowercased name of a :class:`TensorType` member."""
    return tensor_type.name.lower()


def is_read_only(name: str) -> bool:
    """Return whether an attribute name is read-only (cannot be written)."""
    return name.lower() in _READ_ONLY_NAMES


# -----------------------------------------------------------------------------
# The view
# -----------------------------------------------------------------------------


class OvPhysxView:
    """A string-keyed, generic view over OVPhysX ``TensorBinding`` handles for one prim pattern.

    Args:
        physx: The OVPhysX ``PhysX`` instance exposing ``create_tensor_binding``.
        pattern: An fnmatch glob selecting the prims this view addresses.
        device: Simulation device (e.g. ``"cuda:0"`` or ``"cpu"``) used to size buffers.
        tensor_types: Optional explicit set of :class:`TensorType` members to instantiate
            eagerly. Ignored unless ``eager`` is set; defaults to "all applicable" when eager.
        eager: If ``True``, create bindings up front (see design doc §4/§8). Defaults to ``False``
            (lazy: bindings are created on first access).
    """

    def __init__(
        self,
        physx: Any,
        pattern: str,
        device: str,
        *,
        tensor_types: list | None = None,
        eager: bool = False,
    ) -> None:
        self._physx = physx
        self._pattern = pattern
        self._device = device
        self._bindings: dict[Any, Any] = {}
        self._read_buffers: dict[Any, wp.array] = {}
        self._staging_buffers: dict[Any, wp.array] = {}

        if eager:
            requested = tensor_types if tensor_types is not None else [t for t in TensorType if t.name != "INVALID"]
            for tt in requested:
                try:
                    self._binding(tt)
                except OvPhysxViewError:
                    # Not applicable to these prims; skip silently (matches asset eager-create).
                    logger.debug("eager binding skipped for %s on %s", tt, pattern)

    # -- core: string-keyed get / set ----------------------------------------

    def get_attribute(self, name: str, *, out: wp.array | None = None) -> wp.array:
        """Read the full attribute tensor.

        Reads are full-array (the wheel exposes no selective read); index into the
        returned tensor for a subset.

        Args:
            name: Lowercased ``TensorType`` name.
            out: Optional destination buffer (shape ``binding.shape``); receives a copy.
                If omitted, a view-owned buffer is returned that is **overwritten on the
                next read of the same attribute**.

        Returns:
            A :class:`warp.array` holding the attribute values.
        """
        tt = resolve_tensor_type(name)
        binding = self._binding(tt)
        buf = self._read_buffers.get(tt)
        if buf is None:
            buf = wp.zeros(tuple(binding.shape), dtype=wp.float32, device=self._read_device(name))
            self._read_buffers[tt] = buf
        binding.read(buf)
        if out is not None:
            wp.copy(out, buf)
            return out
        return buf

    def set_attribute(
        self,
        name: str,
        values: wp.array,
        *,
        indices: wp.array | None = None,
        mask: wp.array | None = None,
    ) -> None:
        """Write a full-shaped attribute tensor; ``indices``/``mask`` select which rows apply.

        If both ``indices`` and ``mask`` are given, ``mask`` wins (and the wheel warns),
        matching ``TensorBinding.write``.

        Args:
            name: Lowercased ``TensorType`` name.
            values: Source buffer with shape ``binding.shape``.
            indices: Optional integer row indices to write.
            mask: Optional boolean row mask to write.

        Raises:
            OvPhysxViewError: If the attribute is read-only or the shape mismatches.
        """
        if is_read_only(name):
            raise OvPhysxViewError(f"Attribute {name!r} is read-only and cannot be written.")
        tt = resolve_tensor_type(name)
        binding = self._binding(tt)
        src = self._as_wp(values)
        if tuple(src.shape) != tuple(binding.shape):
            raise OvPhysxViewError(
                f"Shape mismatch writing {name!r}: got {tuple(src.shape)}, expected {tuple(binding.shape)}."
            )
        if self._needs_cpu_staging(name):
            staging = self._staging_buffers.get(tt)
            if staging is None:
                staging = wp.zeros(tuple(binding.shape), dtype=wp.float32, device="cpu", pinned=True)
                self._staging_buffers[tt] = staging
            wp.copy(staging, src)
            binding.write(staging, indices=indices, mask=mask)
        else:
            binding.write(src, indices=indices, mask=mask)

    # -- discoverability -------------------------------------------------------

    @property
    def attribute_names(self) -> list[str]:
        """All valid attribute names addressable by this view (the full vocabulary)."""
        return attribute_vocabulary()

    @property
    def available_attributes(self) -> list[str]:
        """Names with a live binding instantiated for this view's prims."""
        return sorted(tensor_type_name(tt) for tt in self._bindings)

    def has_attribute(self, name: str) -> bool:
        """Return whether ``name`` is a valid attribute name (resolvable to a ``TensorType``)."""
        try:
            resolve_tensor_type(name)
        except OvPhysxViewError:
            return False
        return True

    def __contains__(self, name: str) -> bool:
        return self.has_attribute(name)

    # -- metadata passthrough (from a sample binding) --------------------------

    @property
    def count(self) -> int:
        """Number of prims matched by this view's pattern."""
        return self._sample().count

    @property
    def prim_paths(self) -> list[str]:
        """USD paths of the prims matched by this view's pattern."""
        return list(self._sample().prim_paths)

    @property
    def dof_names(self) -> list[str]:
        """Per-articulation DOF names (articulation views only)."""
        return list(self._sample().dof_names)

    @property
    def body_names(self) -> list[str]:
        """Per-articulation body (link) names (articulation views only)."""
        return list(self._sample().body_names)

    @property
    def dof_count(self) -> int:
        return self._sample().dof_count

    @property
    def body_count(self) -> int:
        return self._sample().body_count

    # -- internals -------------------------------------------------------------

    def _binding(self, tensor_type: Any) -> Any:
        """Return the cached :class:`TensorBinding` for ``tensor_type``, creating it on first use."""
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        try:
            binding = self._physx.create_tensor_binding(pattern=self._pattern, tensor_type=tensor_type)
        except Exception as exc:  # noqa: BLE001 -- wheel raises bare exceptions; re-wrapped below
            raise OvPhysxViewError(
                f"Attribute {tensor_type_name(tensor_type)!r} is not available for prims matching {self._pattern!r}."
            ) from exc
        # The wheel returns a 0-count binding when the pattern matches nothing applicable.
        if binding is None or getattr(binding, "count", 0) == 0:
            raise OvPhysxViewError(
                f"Attribute {tensor_type_name(tensor_type)!r} is not available for prims matching "
                f"{self._pattern!r} (no matching prims)."
            )
        self._bindings[tensor_type] = binding
        return binding

    def _sample(self) -> Any:
        """Return any instantiated binding to read view-level metadata from."""
        if not self._bindings:
            raise OvPhysxViewError(
                "No bindings instantiated yet; access an attribute (or construct with eager=True) "
                "before reading view metadata."
            )
        return next(iter(self._bindings.values()))

    def _read_device(self, name: str) -> str:
        """Device a read buffer for ``name`` should live on (CPU for CPU-only types on GPU sims)."""
        if name.lower() in _CPU_ONLY_NAMES and self._device != "cpu":
            return "cpu"
        return self._device

    def _needs_cpu_staging(self, name: str) -> bool:
        return name.lower() in _CPU_ONLY_NAMES and self._device != "cpu"

    def _as_wp(self, values: Any) -> wp.array:
        """Coerce ``values`` to a :class:`warp.array` (accepts warp or torch input)."""
        if isinstance(values, wp.array):
            return values
        # torch tensor -> warp (zero-copy view) without importing torch eagerly
        if hasattr(values, "__dlpack__") or values.__class__.__module__.startswith("torch"):
            return wp.from_torch(values)
        return wp.array(values)

    def __repr__(self) -> str:
        return f"OvPhysxView(pattern={self._pattern!r}, device={self._device!r}, instantiated={len(self._bindings)})"
