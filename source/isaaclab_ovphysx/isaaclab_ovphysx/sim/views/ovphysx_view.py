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

**Dtype: float32 only (interim).** The wheel's ``TensorBinding.read``/``write`` are
float32-only and expose no dtype metadata, so this view treats every binding as ``float32``
(the structured dtypes in :data:`_ATTR_DTYPE`, e.g. ``wp.transformf``, are byte-compatible
views over float32, not a different scalar type). Every ``TensorType`` in the current wheel
is float32; a future non-float binding (e.g. a ``uint8``/``bool`` control tensor) could not be
read correctly here until the wheel exposes dtype metadata. Rather than guess a dtype-supported
subset, the public surface (:attr:`~OvPhysxView.attribute_names`) deliberately stays at
name-validity; narrowing it to a dtype-aware subset is deferred to wheel dtype metadata
(design doc §7 ask).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

import warp as wp

from isaaclab_ovphysx._runtime import import_ovphysx
from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES

logger = logging.getLogger(__name__)

# Pure-Python enum (no native dependency); safe to import regardless of USD state.
TensorType = import_ovphysx("ovphysx.types").TensorType

# Tensor types that cannot be written. The first group is read-only by PhysX
# convention (accelerations, inverse mass/inertia, projected joint force); the
# computed-dynamics group (jacobian, mass matrix, coriolis, gravity) is read-only
# in practice.
#
# TEMPORARY: this is a hand-maintained table and WILL drift as ``TensorType`` grows.
# The wheel has at least three access modes in practice -- read/write, read-only, and
# write-only control tensors -- and exposes no access metadata today. Replace this whole
# table once the wheel exposes a per-type ``access_mode`` enum (preferred over a boolean
# ``is_writable`` flag, so write-only control tensors stay distinguishable).
# TODO(ovphysx): source access mode from a wheel ``access_mode`` query (design doc §7 ask 3).
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

# Structured Warp dtype for attributes whose flat trailing dimension has a fixed semantic
# layout: 7-float poses -> ``wp.transformf``, 6-float spatial vectors -> ``wp.spatial_vectorf``.
# :meth:`OvPhysxView.get_attribute` returns an array of this dtype, so callers get a typed
# ``[N, ...]`` array rather than a flat ``[N, ..., k]`` float32 one. Attributes absent from this
# map default to flat ``float32``. The wheel exposes only flat float32 shapes, so this map is
# hand-maintained.
# TODO(ovphysx): source structured layouts from a wheel dtype query if one is added.
_ATTR_DTYPE: dict[str, Any] = {
    "articulation_root_pose": wp.transformf,
    "articulation_link_pose": wp.transformf,
    "articulation_body_com_pose": wp.transformf,
    "rigid_body_pose": wp.transformf,
    "rigid_body_com_pose": wp.transformf,
    "articulation_root_velocity": wp.spatial_vectorf,
    "articulation_link_velocity": wp.spatial_vectorf,
    "articulation_link_acceleration": wp.spatial_vectorf,
    "articulation_link_incoming_joint_force": wp.spatial_vectorf,
    "rigid_body_velocity": wp.spatial_vectorf,
    "rigid_body_acceleration": wp.spatial_vectorf,
}


class _BindingLike(Protocol):
    """Structural type of an ovphysx ``TensorBinding`` as used by this view.

    ``read`` fills the passed array in place; ``write`` consumes ``indices``/``mask``
    for partial writes. ``shape`` is the binding's flat tensor shape; ``count`` is the
    number of matched prims.
    """

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
            This is an **internal IsaacLab adapter** for the fused-collection binding path, not
            a general public API: the requested key and the created binding type deliberately
            differ, so a caller reasoning from the visible key can get different runtime
            semantics. A public form would instead carry descriptor metadata (requested key,
            source tensor type, shape, native device, access mode); that is deferred to
            wheel-exposed metadata (design doc §7 ask). Prefer not to rely on it outside the
            collection adapter.
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

    class DtypeMismatch(OvPhysxViewError):
        """A supplied buffer's scalar element type is not ``float32``."""

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
        if pattern is not None and not pattern:
            raise ValueError("'pattern' must be a non-empty glob string.")
        if prim_paths is not None and not prim_paths:
            raise ValueError("'prim_paths' must contain at least one glob.")
        if tensor_types is not None and not eager:
            raise ValueError("'tensor_types' is only honored with eager=True; pass eager=True or omit it.")
        self._physx = physx
        self._pattern = pattern
        self._prim_paths = prim_paths
        # Canonicalize the device so a "cuda" alias compares equal to a buffer's "cuda:0"
        # (warp canonicalizes buffer devices). Fall back to the raw string when the device
        # cannot be resolved here (e.g. constructing a cuda view on a CPU-only CI box) -- the
        # string is only used for comparison, so construction must not fail on it.
        try:
            self._device = str(wp.get_device(device))
        except Exception:  # noqa: BLE001 -- unresolvable device: keep the raw string for comparison
            self._device = device
        # Normalize key_aliases to TensorType members (accepts str names too) so string keys are
        # honored rather than silently dropped, and reject aliases that cross the CPU/GPU residency
        # or read-only boundary -- the device and read-only guards key on the requested type.
        self._key_aliases: dict[Any, Any] = {}
        for requested_type, created_type in (key_aliases or {}).items():
            req_tt, made_tt = self._resolve(requested_type), self._resolve(created_type)
            if (req_tt in _CPU_ONLY_TYPES) != (made_tt in _CPU_ONLY_TYPES):
                raise ValueError(
                    f"key_alias {tensor_type_name(req_tt)!r} -> {tensor_type_name(made_tt)!r} crosses the "
                    "CPU/GPU residency boundary; the device policy would apply to the wrong type."
                )
            if is_read_only(tensor_type_name(req_tt)) != is_read_only(tensor_type_name(made_tt)):
                raise ValueError(
                    f"key_alias {tensor_type_name(req_tt)!r} -> {tensor_type_name(made_tt)!r} mixes a read-only "
                    "and a writable type."
                )
            self._key_aliases[req_tt] = made_tt
        self._bindings: dict[Any, Any] = {}
        # Cache of float32 reinterpret views for read_into / get_attribute, keyed by the
        # destination buffer's id(). Reusing the same reinterpret object across calls keeps the
        # wheel's object-identity read cache (the TensorBinding.read fast path) warm.
        self._read_views: dict[int, wp.array] = {}

        if eager:
            explicit = tensor_types is not None
            requested = tensor_types if explicit else [t for t in TensorType if t.name != "INVALID"]
            for tt in requested:
                try:
                    self._binding(self._resolve(tt))
                except OvPhysxView.AttributeUnavailable:
                    if explicit:
                        raise  # caller named this exact type; surface the failure rather than drop it
                    logger.debug("eager binding skipped for %s", tt)  # default sweep: skip inapplicable types
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
                :class:`warp.array` on the native device is returned.

        Returns:
            A :class:`warp.array` holding the attribute values, on the attribute's native
            device -- ``cpu`` for CPU-only property types even on a GPU sim (see
            :func:`is_cpu_only`). When ``out`` is omitted this is a fresh, caller-owned array;
            its dtype is the attribute's structured Warp dtype when it has one (e.g.
            ``wp.transformf`` for poses, ``wp.spatial_vectorf`` for velocities) and flat
            ``float32`` otherwise (see :data:`_ATTR_DTYPE`).
        """
        tt = self._resolve(name)
        binding = self._binding(tt)
        device = self._native_device(tt)
        if out is not None:
            self._check_device(out, device, tensor_type_name(tt), "destination")
            binding.read(self._read_view(out, binding))
            return out
        alloc_shape, dtype = self._attribute_dtype(tt, binding)
        buf = wp.zeros(alloc_shape, dtype=dtype, device=device)
        # ``buf`` is freshly allocated here, so it is never a persistent destination: route it
        # through ``_as_binding_view`` directly rather than ``_read_view``. Caching by ``id(buf)``
        # could never hit on a later call and would leak one entry (and keep ``buf`` alive) per
        # call in a step loop -- the read cache only pays off for a reused ``out``/``dst`` buffer.
        binding.read(self._as_binding_view(buf, binding, "destination"))
        return buf

    def read_into(self, name: str | Any, dst: wp.array) -> None:
        """Fill ``dst`` in place from the attribute binding (zero-copy).

        ``dst`` may be a structured-dtype buffer (e.g. ``wp.transformf``); it is read
        through a ``float32`` reinterpret view that matches the binding's flat shape, so
        the structured GPU/CPU buffer is filled directly with no extra copy. This is the
        path the asset data containers use. The reinterpret view for a given ``dst`` is
        built once and reused across calls (see :meth:`_read_view`) so the wheel's
        object-identity read cache stays warm -- callers can pass the structured buffer
        directly each step without maintaining their own reinterpret cache.

        Args:
            name: Lowercased ``TensorType`` name or the member itself.
            dst: Caller-owned buffer on the binding's native device whose element count
                matches the binding.

        Raises:
            OvPhysxView.DeviceMismatch: If ``dst`` is not on the binding's native device.
            OvPhysxView.DtypeMismatch: If ``dst``'s scalar element type is not ``float32``.
            OvPhysxView.ShapeMismatch: If ``dst`` is non-contiguous or its element count does not match.
        """
        tt = self._resolve(name)
        binding = self._binding(tt)
        self._check_device(dst, self._native_device(tt), tensor_type_name(tt), "destination")
        binding.read(self._read_view(dst, binding))

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
            OvPhysxView.DtypeMismatch: If ``values``' scalar element type is not ``float32``.
            OvPhysxView.ShapeMismatch: If ``values`` is non-contiguous or its element count does not match.
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
        """Return the underlying ``TensorBinding`` for an attribute, creating it on first use.

        This is a raw escape hatch for asset-internal binding management: the returned
        binding's ``read``/``write`` **bypass** the view's device, dtype-reinterpret, shape,
        and read-only guards. Prefer :meth:`get_attribute` / :meth:`read_into` /
        :meth:`set_attribute` unless you are deliberately managing bindings directly.
        """
        return self._binding(self._resolve(name))

    def try_binding_for(self, name: str | Any) -> _BindingLike | None:
        """Like :meth:`binding_for`, but return ``None`` instead of raising when the attribute
        is valid yet **not available for this view's prims** (e.g. tendon types on a
        tendon-less articulation, or a not-yet-created optional binding).

        An invalid *name* still raises :class:`UnknownAttribute` -- that is a programming
        error, not an availability question. Use this for the asset's ``binding or None``
        pattern over optional bindings.
        """
        try:
            return self._binding(self._resolve(name))
        except OvPhysxView.AttributeUnavailable:
            return None

    # -- discoverability -------------------------------------------------------

    @property
    def attribute_names(self) -> list[str]:
        """Every valid attribute name (the full ``TensorType`` vocabulary).

        This is name *validity*, not availability for this view's prims -- a rigid-body view
        still lists ``"articulation_*"`` names. Use :attr:`available_attributes` for what is
        actually instantiated.

        .. note::
            A listed name is **not** a promise of correct dtype handling. The view is
            float32-only (see the module docstring); every ``TensorType`` in the current wheel
            is float32, but a future non-float binding would still be listed here yet not be
            correctly readable until the wheel exposes dtype metadata. Filtering this to a
            dtype-aware supported subset is deferred to that metadata (design doc §7 ask).
        """
        return attribute_vocabulary()

    @property
    def available_attributes(self) -> list[str]:
        """Names with a live binding instantiated for this view's prims."""
        return sorted(tensor_type_name(tt) for tt in self._bindings)

    def has_attribute(self, name: str | Any) -> bool:
        """Return whether ``name`` is a valid attribute name (resolves to a ``TensorType``).

        This checks name *validity* for any view, not availability for these prims: it can
        return ``True`` for a name whose binding does not apply to this view's prims (in which
        case :meth:`get_attribute` raises :class:`AttributeUnavailable`). It likewise does not
        promise dtype support -- the view is float32-only (see :attr:`attribute_names`).
        """
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
        """Number of DOFs per articulation (articulation views only)."""
        return self._sample().dof_count

    @property
    def body_count(self) -> int:
        """Number of bodies (links) per articulation (articulation views only)."""
        return self._sample().body_count

    @property
    def joint_count(self) -> int:
        """Number of joints per articulation (articulation views only)."""
        return self._sample().joint_count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation has a fixed base (articulation views only)."""
        return self._sample().is_fixed_base

    @property
    def fixed_tendon_count(self) -> int:
        """Number of fixed tendons per articulation (articulation views only)."""
        return self._sample().fixed_tendon_count

    @property
    def spatial_tendon_count(self) -> int:
        """Number of spatial tendons per articulation (articulation views only)."""
        return self._sample().spatial_tendon_count

    # -- internals -------------------------------------------------------------

    def _resolve(self, name: str | Any) -> Any:
        """Resolve a string name or a ``TensorType`` member to a ``TensorType``."""
        if isinstance(name, str):
            return resolve_tensor_type(name)
        if isinstance(name, TensorType):
            if name.name == "INVALID":  # mirror the string path's INVALID rejection
                raise OvPhysxView.UnknownAttribute(f"{name!r} is not an addressable attribute.")
            return name
        raise OvPhysxView.UnknownAttribute(
            f"Attribute key must be a str name or a TensorType member, got {type(name).__name__}."
        )

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
        except Exception as exc:  # noqa: BLE001 -- wheel raises bare exceptions; surface the cause below
            # The wheel raises both for "type not applicable to these prims" and for genuine
            # failures (init/ABI/OOM); we can't tell them apart without a wheel-side exception
            # type, so the underlying error is surfaced in the message (and chained) rather than
            # hidden behind a generic "not available". TODO(ovphysx): a typed no-match error.
            raise OvPhysxView.AttributeUnavailable(
                f"Could not create the {tensor_type_name(tensor_type)!r} binding for "
                f"{self._target_repr()}: create_tensor_binding raised {type(exc).__name__}: {exc}"
            ) from exc
        # The wheel returns a 0-count binding when nothing matches. Access ``count`` directly so a
        # malformed binding (missing ``count``) surfaces as an error rather than a phantom no-match.
        if binding is None or binding.count == 0:
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

        ``arr`` must have a ``float32`` scalar element type (``float32`` itself or a
        composite built on it, e.g. ``wp.transformf``/``wp.vec3f``): the view
        **reinterprets bits, not values**, so a non-``float32`` dtype (``int32``,
        ``float64``, ``float16``) would corrupt the data and is rejected. Given a matching
        scalar, validates the flat ``float32`` element count and returns ``arr`` directly
        when it is already ``float32`` with the binding's shape, else a zero-copy
        reinterpret view.
        """
        scalar = getattr(arr.dtype, "_wp_scalar_type_", arr.dtype)
        if scalar is not wp.float32:
            raise OvPhysxView.DtypeMismatch(
                f"{role} must have float32 scalar elements (got dtype "
                f"{getattr(arr.dtype, '__name__', arr.dtype)}); the view reinterprets bits, "
                "not values, so a non-float32 dtype would silently corrupt the buffer."
            )
        if not arr.is_contiguous:
            raise OvPhysxView.ShapeMismatch(
                f"{role} must be a contiguous array; the view reinterprets the buffer's raw memory, "
                "so a strided/sliced view would read or write the wrong elements."
            )
        expected = math.prod(tuple(binding.shape))
        actual = arr.size * (wp.types.type_size_in_bytes(arr.dtype) // 4)  # scalar is float32 -> exact
        if actual != expected:
            raise OvPhysxView.ShapeMismatch(
                f"Shape mismatch for {role}: {actual} float32 elements, "
                f"binding expects {expected} (shape {tuple(binding.shape)})."
            )
        if arr.dtype == wp.float32 and tuple(arr.shape) == tuple(binding.shape):
            return arr
        return wp.array(ptr=arr.ptr, shape=tuple(binding.shape), dtype=wp.float32, device=str(arr.device), copy=False)

    def _read_view(self, dst: wp.array, binding: Any) -> wp.array:
        """Return the ``float32`` view of ``dst`` to hand to ``binding.read``, reused across calls.

        The wheel's ``TensorBinding.read`` has an object-identity read cache: it skips DLPack
        acquisition and the attribute-chain lookup when handed the *same* tensor object as the
        previous read. To keep that cache warm, the ``float32`` reinterpret of a structured
        ``dst`` is built once and reused for that destination buffer; a pointer-staleness guard
        rebuilds it if the buffer's backing storage moved. A ``dst`` that is already flat
        ``float32`` is its own stable identity, so it is returned directly (and not cached).
        """
        cached = self._read_views.get(id(dst))
        if cached is not None and cached.ptr == dst.ptr:
            return cached
        view = self._as_binding_view(dst, binding, "destination")
        if view is not dst:  # structured dst -> cache the reinterpret; a flat float32 dst caches nothing
            self._read_views[id(dst)] = view
        return view

    def _attribute_dtype(self, tensor_type: Any, binding: Any) -> tuple[tuple[int, ...], Any]:
        """Return ``(alloc_shape, dtype)`` for :meth:`get_attribute`.

        Maps an attribute to its structured Warp dtype (see :data:`_ATTR_DTYPE`) when the
        binding's trailing dimension matches that dtype's ``float32`` count, dropping the
        trailing dimension from the allocation shape (e.g. ``[N, 7] -> ([N], wp.transformf)``).
        Falls back to the flat ``float32`` shape for unmapped attributes or a mismatched layout.
        """
        dtype = _ATTR_DTYPE.get(tensor_type_name(tensor_type))
        shape = tuple(binding.shape)
        if dtype is not None and shape and shape[-1] == wp.types.type_size_in_bytes(dtype) // 4:
            return shape[:-1], dtype
        return shape, wp.float32

    def _as_wp(self, values: Any, device: str) -> wp.array:
        """Coerce ``values`` to a :class:`warp.array`.

        A :class:`warp.array` is used as-is, keeping its own device (validated by the caller;
        a mismatch raises and is never staged). Device-less host data (numpy arrays, lists)
        carries no device, so it is materialized directly on ``device``.

        This view is Warp-native and does **not** special-case framework tensors: bridge a
        Torch tensor on the caller side with ``view.set_attribute(name, wp.from_torch(t))``.
        This keeps the device policy explicit and avoids an optional Torch dependency and the
        fragile detection a built-in conversion would require.
        """
        if isinstance(values, wp.array):
            return values
        return wp.array(values, device=device)

    def _target_repr(self) -> str:
        return f"prim_paths={self._prim_paths!r}" if self._prim_paths is not None else f"pattern={self._pattern!r}"

    def __repr__(self) -> str:
        return f"OvPhysxView({self._target_repr()}, device={self._device!r}, instantiated={len(self._bindings)})"


# Backward-compatible module-level alias for the error base class.
OvPhysxViewError = OvPhysxView.OvPhysxViewError
