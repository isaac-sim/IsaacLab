# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""String-keyed view over OVPhysX ``TensorBinding`` handles.

OVPhysX exposes physics attributes as a loose ``dict[TensorType, TensorBinding]``
with no view object -- unlike Newton's ``selection.ArticulationView`` or PhysX's
typed tensor views. :class:`OvPhysxView` wraps those bindings for one prim pattern
(or an explicit ``prim_paths`` list) behind a string-keyed surface that mirrors
Newton's selection ergonomics, but is simpler: there is no ``Model``/``State``/
``Control`` source object because the :class:`TensorType` already implies where the
data lives.

Attributes are addressed by the lowercased ``TensorType`` enum member name (derived
directly from the wheel enum -- no hand-maintained table) or by the enum member
itself::

    view.get_attribute("articulation_dof_stiffness")  # allocates and returns
    view.read_into("articulation_root_pose", root_pose_buf)  # zero-copy into a caller buffer
    view.set_attribute("rigid_body_pose", values, mask=env_mask)

Design intent: be usable as the binding-management layer *inside* the OVPhysX asset
classes (see ``docs/superpowers/specs/2026-06-17-ovphysx-view-design.md`` §6), so it
exposes a raw :meth:`binding_for` accessor and a zero-copy :meth:`read_into` that
fills a caller-owned, possibly structured-dtype buffer via a ``float32`` reinterpret
view -- the same mechanism the data containers use today.

**Device policy: no implicit CPU<->GPU conversion.** OVPhysX serves DOF/body
*property* tensor types from CPU memory even on a GPU sim (see :data:`_CPU_ONLY_NAMES`),
while *state* tensor types are device-resident. This view reads/writes each binding on
its native device and **raises** :class:`OvPhysxView.DeviceMismatch` if a caller hands
it a buffer on the wrong device. Staging a CPU property to/from the simulation device
is the caller's explicit responsibility, never hidden here.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, runtime_checkable

import warp as wp

from isaaclab_ovphysx._runtime import import_ovphysx
from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES

logger = logging.getLogger(__name__)

# Pure-Python enum (no native dependency); safe to import regardless of USD state.
TensorType = import_ovphysx("ovphysx.types").TensorType

# Tensor types that cannot be written. The first group is read-only by PhysX
# convention (accelerations, inverse mass/inertia, projected joint force); the
# computed-dynamics group (jacobian, mass matrix, coriolis, gravity) is read-only
# in practice. The wheel enum exposes no writability flag today, so this set is
# hand-maintained.
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
        "articulation_jacobian",
        "articulation_mass_matrix",
        "articulation_coriolis_and_centrifugal_force",
        "articulation_gravity_force",
    }
)

# DOF/body property tensor types that are CPU-resident even on a GPU sim. Derived
# from the canonical, wheel-availability-gated set in ``isaaclab_ovphysx.tensor_types``
# so the two never drift.
_CPU_ONLY_NAMES: frozenset[str] = frozenset(tt.name.lower() for tt in _CPU_ONLY_TYPES)


@runtime_checkable
class _BindingLike(Protocol):
    """Structural type of an ovphysx ``TensorBinding`` as used by this view."""

    shape: tuple[int, ...]
    count: int
    prim_paths: list[str]
    dof_names: list[str]
    body_names: list[str]
    joint_names: list[str]
    dof_count: int
    body_count: int
    joint_count: int
    is_fixed_base: bool
    fixed_tendon_count: int
    spatial_tendon_count: int

    def read(self, tensor: wp.array) -> None: ...

    def write(self, tensor: wp.array, indices: wp.array | None = None, mask: wp.array | None = None) -> None: ...


class _PhysXLike(Protocol):
    """Structural type of the ovphysx ``PhysX`` instance this view depends on."""

    def create_tensor_binding(self, *, tensor_type: Any, pattern: str = ..., prim_paths: list[str] = ...) -> Any: ...


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
        OvPhysxView.UnknownAttribute: If the name is not an addressable ``TensorType``.
    """
    try:
        tt = TensorType[name.upper()]
    except KeyError:
        raise OvPhysxView.UnknownAttribute(
            f"Unknown attribute {name!r}. Valid names are the lowercased TensorType members, "
            f"e.g. {attribute_vocabulary()[:4]} ... ({len(attribute_vocabulary())} total)."
        ) from None
    if tt.name == "INVALID":
        raise OvPhysxView.UnknownAttribute(f"{name!r} is not an addressable attribute.")
    return tt


def tensor_type_name(tensor_type: Any) -> str:
    """Return the canonical lowercased name of a :class:`TensorType` member."""
    return tensor_type.name.lower()


def is_read_only(name: str) -> bool:
    """Return whether an attribute name is read-only (cannot be written)."""
    return name.lower() in _READ_ONLY_NAMES


def is_cpu_only(name: str) -> bool:
    """Return whether an attribute is CPU-resident even on a GPU simulation."""
    return name.lower() in _CPU_ONLY_NAMES


# -----------------------------------------------------------------------------
# The view
# -----------------------------------------------------------------------------


class OvPhysxView:
    """A string-keyed, generic view over OVPhysX ``TensorBinding`` handles for one prim set.

    Args:
        physx: The OVPhysX ``PhysX`` instance exposing ``create_tensor_binding``.
        pattern: An fnmatch glob selecting the prims this view addresses. Mutually
            exclusive with ``prim_paths``.
        device: Simulation device (e.g. ``"cuda:0"`` or ``"cpu"``). State bindings are
            read/written on this device; CPU-only property bindings always use ``"cpu"``.
        prim_paths: An explicit list of fnmatch globs for the fused multi-prim binding
            form (``create_tensor_binding(prim_paths=[...])``). Mutually exclusive with
            ``pattern``.
        key_aliases: Optional mapping ``requested_type -> created_type`` so a binding can
            be stored under a different :class:`TensorType` key than the one created
            (e.g. a ``RigidObjectCollection`` stores ``rigid_body_pose`` under ``link_pose``).
        tensor_types: Explicit set of :class:`TensorType` members to instantiate eagerly.
            Used only when ``eager`` is set; defaults to every applicable type.
        eager: If ``True``, create bindings up front and raise if none could be created.
            Defaults to ``False`` (lazy: bindings are created on first access).
    """

    class OvPhysxViewError(RuntimeError):
        """Base class for all errors raised by :class:`OvPhysxView`."""

    class UnknownAttribute(OvPhysxViewError):
        """The attribute name does not resolve to an addressable ``TensorType``."""

    class ReadOnlyAttribute(OvPhysxViewError):
        """A write was attempted on a read-only attribute."""

    class AttributeUnavailable(OvPhysxViewError):
        """No binding could be created for the attribute on this view's prims."""

    class ShapeMismatch(OvPhysxViewError):
        """A supplied buffer does not match the binding's element count."""

    class DeviceMismatch(OvPhysxViewError):
        """A supplied buffer is on a different device than the binding requires."""

    def __init__(
        self,
        physx: _PhysXLike,
        pattern: str | None = None,
        device: str = "cpu",
        *,
        prim_paths: list[str] | None = None,
        key_aliases: dict[Any, Any] | None = None,
        tensor_types: list[Any] | None = None,
        eager: bool = False,
    ) -> None:
        if (pattern is None) == (prim_paths is None):
            raise ValueError("Provide exactly one of 'pattern' or 'prim_paths'.")
        self._physx = physx
        self._pattern = pattern
        self._prim_paths = prim_paths
        self._device = device
        self._key_aliases: dict[Any, Any] = dict(key_aliases or {})
        self._bindings: dict[Any, Any] = {}

        if eager:
            requested = tensor_types if tensor_types is not None else [t for t in TensorType if t.name != "INVALID"]
            for tt in requested:
                try:
                    self._binding(tt)
                except OvPhysxView.AttributeUnavailable:
                    # Not applicable to these prims; skip and keep going.
                    logger.debug("eager binding skipped for %s", tt)
            if not self._bindings:
                raise OvPhysxView.AttributeUnavailable(
                    f"Could not create any bindings for {self._target_repr()}; "
                    "the pattern/prim_paths likely match no prims."
                )

    # -- core: string-keyed get / set / read-into ------------------------------

    def get_attribute(self, name: str | Any, *, out: wp.array | None = None) -> wp.array:
        """Read the full attribute tensor.

        Reads are full-array (the wheel exposes no selective read); index into the
        returned tensor for a subset.

        Args:
            name: Lowercased ``TensorType`` name or the member itself.
            out: Optional destination buffer to fill (must be on the binding's native
                device and match its element count). If omitted, a freshly allocated
                ``float32`` :class:`warp.array` on the native device is returned.

        Returns:
            A :class:`warp.array` holding the attribute values. When ``out`` is omitted
            this is a fresh, caller-owned array (no aliasing of view state).
        """
        tt = self._resolve(name)
        binding = self._binding(tt)
        device = self._native_device(tt)
        if out is not None:
            self._check_device(out, device, tensor_type_name(tt), "destination")
            binding.read(self._as_binding_view(out, binding, "destination"))
            return out
        buf = wp.zeros(tuple(binding.shape), dtype=wp.float32, device=device)
        binding.read(buf)
        return buf

    def read_into(self, name: str | Any, dst: wp.array) -> None:
        """Fill ``dst`` in place from the attribute binding (zero-copy).

        ``dst`` may be a structured-dtype buffer (e.g. ``wp.transformf``); it is read
        through a ``float32`` reinterpret view that matches the binding's flat shape, so
        the structured GPU/CPU buffer is filled directly with no extra copy. This is the
        path the asset data containers use.

        Args:
            name: Lowercased ``TensorType`` name or the member itself.
            dst: Caller-owned buffer on the binding's native device whose element count
                matches the binding.

        Raises:
            OvPhysxView.DeviceMismatch: If ``dst`` is not on the binding's native device.
            OvPhysxView.ShapeMismatch: If ``dst``'s element count does not match.
        """
        tt = self._resolve(name)
        binding = self._binding(tt)
        self._check_device(dst, self._native_device(tt), tensor_type_name(tt), "destination")
        binding.read(self._as_binding_view(dst, binding, "destination"))

    def set_attribute(
        self,
        name: str | Any,
        values: wp.array,
        *,
        indices: wp.array | None = None,
        mask: wp.array | None = None,
    ) -> None:
        """Write a full attribute tensor; ``indices``/``mask`` select which rows apply.

        ``values`` may be a structured-dtype buffer (read through a ``float32``
        reinterpret view). If both ``indices`` and ``mask`` are given, ``mask`` wins and
        the wheel emits a ``UserWarning`` -- this view forwards both verbatim to
        ``TensorBinding.write`` and does not implement the precedence itself.

        Args:
            name: Lowercased ``TensorType`` name or the member itself.
            values: Source buffer on the binding's native device, matching its element count.
            indices: Optional integer row indices to write.
            mask: Optional boolean row mask to write.

        Raises:
            OvPhysxView.ReadOnlyAttribute: If the attribute is read-only.
            OvPhysxView.DeviceMismatch: If ``values`` is not on the binding's native device.
            OvPhysxView.ShapeMismatch: If ``values``' element count does not match.
        """
        tt = self._resolve(name)
        attr = tensor_type_name(tt)
        if attr in _READ_ONLY_NAMES:
            raise OvPhysxView.ReadOnlyAttribute(f"Attribute {attr!r} is read-only and cannot be written.")
        binding = self._binding(tt)
        device = self._native_device(tt)
        src = self._as_wp(values, device)
        self._check_device(src, device, attr, "source")
        binding.write(self._as_binding_view(src, binding, "source"), indices=indices, mask=mask)

    # -- raw binding access (for asset/data-container adoption) ----------------

    def binding_for(self, name: str | Any) -> _BindingLike:
        """Return the underlying ``TensorBinding`` for an attribute, creating it on first use."""
        return self._binding(self._resolve(name))

    # -- discoverability -------------------------------------------------------

    @property
    def attribute_names(self) -> list[str]:
        """All valid attribute names addressable by this view (the full vocabulary)."""
        return attribute_vocabulary()

    @property
    def available_attributes(self) -> list[str]:
        """Names with a live binding instantiated for this view's prims."""
        return sorted(tensor_type_name(tt) for tt in self._bindings)

    def has_attribute(self, name: str | Any) -> bool:
        """Return whether ``name`` resolves to a valid ``TensorType``."""
        try:
            self._resolve(name)
        except OvPhysxView.UnknownAttribute:
            return False
        return True

    def __contains__(self, name: str | Any) -> bool:
        return self.has_attribute(name)

    # -- metadata passthrough (from a sample binding) --------------------------

    @property
    def count(self) -> int:
        """Number of prims matched by this view."""
        return self._sample().count

    @property
    def prim_paths(self) -> list[str]:
        """USD paths of the prims matched by this view."""
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
    def joint_names(self) -> list[str]:
        """Per-articulation joint names (articulation views only)."""
        return list(self._sample().joint_names)

    @property
    def dof_count(self) -> int:
        return self._sample().dof_count

    @property
    def body_count(self) -> int:
        return self._sample().body_count

    @property
    def joint_count(self) -> int:
        return self._sample().joint_count

    @property
    def is_fixed_base(self) -> bool:
        return self._sample().is_fixed_base

    @property
    def fixed_tendon_count(self) -> int:
        return self._sample().fixed_tendon_count

    @property
    def spatial_tendon_count(self) -> int:
        return self._sample().spatial_tendon_count

    # -- internals -------------------------------------------------------------

    def _resolve(self, name: str | Any) -> Any:
        """Resolve a string name or a ``TensorType`` member to a ``TensorType``."""
        if isinstance(name, str):
            return resolve_tensor_type(name)
        return name

    def _binding(self, tensor_type: Any) -> Any:
        """Return the cached ``TensorBinding`` for ``tensor_type``, creating it on first use."""
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        create_type = self._key_aliases.get(tensor_type, tensor_type)
        kwargs: dict[str, Any] = {"tensor_type": create_type}
        if self._prim_paths is not None:
            kwargs["prim_paths"] = self._prim_paths
        else:
            kwargs["pattern"] = self._pattern
        try:
            binding = self._physx.create_tensor_binding(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- wheel raises bare exceptions; re-wrapped below
            raise OvPhysxView.AttributeUnavailable(
                f"Attribute {tensor_type_name(tensor_type)!r} is not available for {self._target_repr()}."
            ) from exc
        # The wheel returns a 0-count binding when nothing matches.
        if binding is None or getattr(binding, "count", 0) == 0:
            raise OvPhysxView.AttributeUnavailable(
                f"Attribute {tensor_type_name(tensor_type)!r} is not available for {self._target_repr()} "
                "(no matching prims)."
            )
        self._bindings[tensor_type] = binding
        return binding

    def _sample(self) -> Any:
        """Return any instantiated binding to read view-level metadata from."""
        if not self._bindings:
            raise OvPhysxView.AttributeUnavailable(
                "No bindings instantiated yet; access an attribute (or construct with eager=True) "
                "before reading view metadata."
            )
        return next(iter(self._bindings.values()))

    def _native_device(self, tensor_type: Any) -> str:
        """Device a buffer for ``tensor_type`` must live on (CPU for CPU-only types)."""
        return "cpu" if tensor_type in _CPU_ONLY_TYPES else self._device

    def _check_device(self, arr: wp.array, device: str, attr: str, role: str) -> None:
        """Raise if ``arr`` is not on the binding's native device (no implicit conversion)."""
        if str(arr.device) != device:
            raise OvPhysxView.DeviceMismatch(
                f"{role} for {attr!r} must be on device {device!r}, got {str(arr.device)!r}. "
                "OvPhysxView does not stage between CPU and GPU; move the buffer yourself."
            )

    def _as_binding_view(self, arr: wp.array, binding: Any, role: str) -> wp.array:
        """Return a ``float32`` view of ``arr`` matching the binding's flat shape.

        Validates that ``arr``'s element count matches the binding, then returns ``arr``
        directly if it is already ``float32`` with the binding's shape, or a zero-copy
        reinterpret view otherwise (for structured dtypes such as ``wp.transformf``).
        """
        expected = math.prod(tuple(binding.shape))
        actual = arr.size * (wp.types.type_size_in_bytes(arr.dtype) // 4)
        if actual != expected:
            raise OvPhysxView.ShapeMismatch(
                f"Shape mismatch for {role}: {actual} float32-equivalent elements, "
                f"binding expects {expected} (shape {tuple(binding.shape)})."
            )
        if arr.dtype == wp.float32 and tuple(arr.shape) == tuple(binding.shape):
            return arr
        return wp.array(ptr=arr.ptr, shape=tuple(binding.shape), dtype=wp.float32, device=str(arr.device), copy=False)

    def _as_wp(self, values: Any, device: str) -> wp.array:
        """Coerce ``values`` to a :class:`warp.array`.

        Device-bearing inputs (warp / torch) keep their own device and are validated by
        the caller (a mismatch raises -- never staged). Device-less host data (numpy
        arrays, lists) carries no device, so it is materialized directly on ``device``.
        """
        if isinstance(values, wp.array):
            return values
        if type(values).__module__.split(".")[0] == "torch":
            return wp.from_torch(values)
        return wp.array(values, device=device)

    def _target_repr(self) -> str:
        return f"prim_paths={self._prim_paths!r}" if self._prim_paths is not None else f"pattern={self._pattern!r}"

    def __repr__(self) -> str:
        return f"OvPhysxView({self._target_repr()}, device={self._device!r}, instantiated={len(self._bindings)})"


# Backward-compatible module-level alias for the error base class.
OvPhysxViewError = OvPhysxView.OvPhysxViewError
