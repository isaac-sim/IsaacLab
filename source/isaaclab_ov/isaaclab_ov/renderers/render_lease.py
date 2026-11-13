# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Exclusive borrowing, thread handoff and publication ordering for OVRTX/OVStage."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Hashable, Iterator
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ovrtx import Renderer
    from ovstage import Stage as OvStage

    from .ovrtx_renderer import OVRTXRenderer
_NO_PHYSICS_STEP_ID = object()


class _OvLibrariesRenderLeaseState(Enum):
    """Internal lifecycle state for a renderer/scene pair lease."""

    UNBOUND = auto()
    BOUND = auto()
    RELEASED = auto()


class OvLibrariesRenderLease:
    """Exclusive non-owning lease over one exact attached OVRTX/OVStage pair.

    The lease is acquired before stage preparation. Stage preparation then
    materializes the renderer-owned scene without requiring a camera sensor and
    binds this object to the exact live Python ``Renderer`` and ``Stage``
    instances. External consumers must release the lease before the renderer is
    closed; they must never detach or destroy either borrowed object.
    """

    _controller: RenderLeaseController
    """Renderer-owned controller that created this lease."""

    def __init__(self) -> None:
        raise RuntimeError("OvLibrariesRenderLease must be acquired from OVRTXRenderer")

    def __enter__(self) -> OvLibrariesRenderLease:
        self._validate_active()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    @property
    def ovrtx_renderer(self) -> Renderer:
        """Return the exact retained application renderer."""
        self.validate_live()
        return cast("Renderer", self._renderer)

    @property
    def ovstage(self) -> OvStage:
        """Return the exact retained renderer-attached OVStage."""
        self.validate_live()
        return cast("OvStage", self._stage)

    def offer_owner_transfer(self) -> object:
        """Freeze one unbound lease and return its one-shot owner capability."""
        owner = self._controller
        return owner.offer_owner_transfer(self)

    def claim_owner_transfer(self, capability: object) -> None:
        """Make the calling thread the render owner using the offered capability.

        Args:
            capability: The object returned by :meth:`offer_owner_transfer`.

        Raises:
            RuntimeError: If the capability is stale, already claimed, or the lease is no longer unbound.
        """
        self._controller.claim_owner_transfer(self, capability)

    def cancel_owner_transfer(self, capability: object) -> None:
        """Withdraw an unclaimed offer on the original owner thread.

        Args:
            capability: The object returned by :meth:`offer_owner_transfer`.

        Raises:
            RuntimeError: If called off the original owner thread or with a stale capability.
        """
        self._controller.cancel_owner_transfer(self, capability)

    def defer_stage_until_owner_transfer(self) -> None:
        """Defer Stage materialization requested before the owner transfer is claimed.

        Camera initialization calls ``prepare_stage`` for every registered renderer on the
        application thread. With deferral, that call only records the request; the claimed
        render owner completes it with :meth:`materialize_deferred_stage`.
        """
        self._validate_active()
        if self._state is not _OvLibrariesRenderLeaseState.UNBOUND:
            raise RuntimeError("Stage deferral must be selected before Stage materialization")
        self._controller.defer_until_transfer = True

    def materialize_deferred_stage(self, stage: Any, num_envs: int) -> bool:
        """Complete a deferred Stage preparation on the claimed render owner.

        Args:
            stage: The environment USD stage.
            num_envs: Number of environments; the lease supports exactly one.

        Returns:
            Whether a deferred preparation was completed. ``False`` means nothing was deferred.

        Raises:
            RuntimeError: If called off the render-owner thread, or deferral is selected and
                the owner transfer has not been claimed.
        """
        return self._controller.materialize_deferred_stage(self, stage, num_envs)

    def capture_scene_publisher(self) -> OvLibrariesScenePublisher:
        """Capture the application-thread scene publisher before render ownership transfers.

        Call this on the owner thread while the lease is unbound and before
        :meth:`offer_owner_transfer`. The publisher stays bound to the calling thread; see
        :class:`~isaaclab_ov.renderers.OvLibrariesScenePublisher` for the publication
        protocol.

        Returns:
            The lease's only scene publisher.

        Raises:
            RuntimeError: If called off the owner thread or while a transfer is offered, after the
                Stage is bound, or when the lease already has a publisher.
        """
        return self._controller.capture_scene_publisher(self)

    @contextlib.contextmanager
    def ovstage_owner_lane(self) -> Iterator[None]:
        """Guard one non-reentrant operation on the render-owner thread.

        The application's publication guard must separately exclude concurrent
        producer writes while this operation uses the borrowed stage. A failed
        producer write poisons publication only: the lane stays available so the
        borrower can destroy its consumers and retire the pair.
        """
        self._validate_live(allow_poisoned=True)
        owner = self._controller
        with owner.owner_lane("external renderer operation"):
            yield

    def release(self) -> None:
        """Release the borrowing intent without destroying either pair member."""
        if self._state is _OvLibrariesRenderLeaseState.RELEASED:
            return
        owner = self._controller
        owner.release(self)
        self._renderer = None
        self._stage = None
        self._state = _OvLibrariesRenderLeaseState.RELEASED

    def validate_live(self) -> None:
        """Verify that this lease still refers to its live, unpoisoned renderer and stage."""
        self._validate_live(allow_poisoned=False)

    def _validate_live(self, *, allow_poisoned: bool) -> None:
        self._validate_active()
        if self._state is not _OvLibrariesRenderLeaseState.BOUND:
            raise RuntimeError("OvLibrariesRenderLease is not bound to a prepared Stage")
        owner = self._controller
        owner.validate_live(
            self,
            cast("Renderer", self._renderer),
            cast("OvStage", self._stage),
            self._generation,
            allow_poisoned=allow_poisoned,
        )

    @classmethod
    def _create_unbound(cls, owner: RenderLeaseController) -> OvLibrariesRenderLease:
        lease = object.__new__(cls)
        lease._controller = owner
        lease._state = _OvLibrariesRenderLeaseState.UNBOUND
        lease._renderer = None
        lease._stage = None
        lease._generation = 0
        return lease

    def _bind(self, renderer: Renderer, stage: OvStage, generation: int) -> None:
        if self._state is not _OvLibrariesRenderLeaseState.UNBOUND:
            raise RuntimeError("OvLibrariesRenderLease can only bind once")
        self._renderer = renderer
        self._stage = stage
        self._generation = generation
        self._state = _OvLibrariesRenderLeaseState.BOUND

    def _validate_active(self) -> None:
        if self._state is _OvLibrariesRenderLeaseState.RELEASED:
            raise RuntimeError("OvLibrariesRenderLease is released")
        owner = self._controller
        owner.validate_active(self)


class OvLibrariesScenePublisher:
    """Application-thread writer for complete provider-derived OVStage batches.

    Obtain it with :meth:`~isaaclab_ov.renderers.OvLibrariesRenderLease.capture_scene_publisher`.
    While the lease is active, the renderer skips its own eager transform, geometry and material
    writes, and the borrower publishes each complete physics state through this object instead:

    1. On the application thread, call :meth:`publication_required` with the identity of the
       current physics state.
    2. If it returns ``True``, reserve a nonzero Stage ordinal and call
       :meth:`write_complete_scene` with it. The write does not advance the Stage write floor:
       the borrower seals that ordinal afterwards, or abandons it if the write fails.
    3. Hold one application publication guard around both calls and around every operation the
       render owner performs in :meth:`~isaaclab_ov.renderers.OvLibrariesRenderLease.ovstage_owner_lane`, so that scene
       writes and native frame operations never interleave.

    A failed write poisons the pair: later publication and ordinary renderer use are rejected,
    but the borrower can still retire its consumers and release the lease. The renderer must then
    be recreated. :meth:`close` stops publication; releasing the lease also closes the publisher.
    The OVXR teleoperation adapter, ``isaaclab_teleop.OvLibrariesXrSession``, follows this protocol.
    """

    def __init__(self, owner: RenderLeaseController, lease: OvLibrariesRenderLease) -> None:
        self._controller = owner
        self._lease = lease
        self._application_thread = threading.current_thread()
        self._lock = threading.Lock()
        self._closed = False

    def publication_required(self, physics_step_id: Hashable) -> bool:
        """Return whether a new physics step or dirty material requires a scene write.

        Args:
            physics_step_id: Hashable identity of the complete physics state to publish. Only the
                last written identity is remembered, so the identity must change whenever the
                state does, including after a reset that does not advance the physics step count.

        Returns:
            Whether ``physics_step_id`` differs from the last written batch or a visual material
            has pending updates.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed
                or inactive, the lease is not bound, or a failed write poisoned the pair.
            TypeError: If ``physics_step_id`` is not hashable.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            return self._controller.producer_publication_required(self._lease, physics_step_id)

    def write_complete_scene(self, ordinal: int, physics_step_id: Hashable, *, force: bool = False) -> None:
        """Write, but do not publish, one complete batch at ``ordinal``.

        Writes the current transforms, geometries and pending visual materials at ``ordinal`` and
        records ``physics_step_id`` as written. The caller seals the ordinal afterwards.

        Args:
            ordinal: Nonzero Stage ordinal reserved by the caller for this batch.
            physics_step_id: The identity passed to the preceding :meth:`publication_required` call.
            force: Publish an independently changed anchor even when physics is unchanged.

        Raises:
            ValueError: If ``ordinal`` is not positive.
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, the state needs no publication, or the write failed. A failed write
                poisons the pair.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            self._controller.write_producer_scene(self._lease, ordinal, physics_step_id, force=force)

    def invalidate(self) -> None:
        """Reject further use of a pair after a borrower write or seal failed."""
        with self._lock:
            self._closed = True
            if self._controller.active is self._lease:
                self._controller._poisoned = True

    def close(self) -> None:
        """Stop scene publication after any in-flight batch finishes. Safe to call repeatedly."""
        with self._lock:
            self._closed = True

    def _require_application_thread(self) -> None:
        if threading.current_thread() is not self._application_thread:
            raise RuntimeError("OVStage scene publisher must run on its captured application thread")


class RenderLeaseController:
    """Coordinate external borrowing without owning the renderer's native resources.

    Lease state, thread transfer and publication ordering live here. The renderer
    retains scene/camera storage and supplies the component-write operations.
    """

    def __init__(self, renderer: OVRTXRenderer) -> None:
        self._renderer = renderer
        self._owner_thread = threading.current_thread()
        self._transfer_lock = threading.Lock()
        self._transfer_capability: object | None = None
        self._transfer_claimed = False
        self.active: OvLibrariesRenderLease | None = None
        self._publisher: OvLibrariesScenePublisher | None = None
        self._generation = 0
        self._lane_operation: str | None = None
        self._last_step_id: Hashable = _NO_PHYSICS_STEP_ID
        self._poisoned = False
        self.defer_until_transfer = False
        self.stage_deferred = False

    def acquire(self) -> OvLibrariesRenderLease:
        """Reserve the exact OVRTX/OVStage pair before stage preparation."""
        self.require_owner_thread("acquire a render lease")
        if os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1").strip() != "1":
            raise RuntimeError("OvLibrariesRenderLease requires ISAAC_LAB_OVRTX_USE_OVSTAGE=1")
        if os.environ.get("ISAAC_LAB_SHARE_OVSTAGE") == "1":
            raise RuntimeError("The render lease requires a private rendering OVStage")
        if self._renderer._closed:
            raise RuntimeError("Cannot acquire OvLibrariesRenderLease from a closed renderer")
        if self.active is not None:
            raise RuntimeError("An OvLibrariesRenderLease is already active")
        if self._renderer.backend is not None or self._renderer._initialized_scene or self._generation != 0:
            raise RuntimeError("OvLibrariesRenderLease must be acquired before clone preparation")
        lease = OvLibrariesRenderLease._create_unbound(self)
        self.active = lease
        return lease

    def require_owner_thread(self, operation: str) -> None:
        """Reject lease operations outside the renderer construction thread."""
        if self._transfer_capability is not None:
            raise RuntimeError("OVRTXRenderer owner transfer is pending")
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError(f"OVRTXRenderer owner thread must {operation}")

    def offer_owner_transfer(self, lease: OvLibrariesRenderLease) -> object:
        self.require_owner_thread("offer an owner transfer")
        with self._transfer_lock:
            if (
                self.active is not lease
                or lease._state is not _OvLibrariesRenderLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
                or self._transfer_capability is not None
                or self._transfer_claimed
            ):
                raise RuntimeError("OVRTXRenderer is not eligible for a pre-use owner transfer")
            capability = object()
            self._transfer_capability = capability
            return capability

    def claim_owner_transfer(self, lease: OvLibrariesRenderLease, capability: object) -> None:
        with self._transfer_lock:
            if (
                capability is None
                or capability is not self._transfer_capability
                or self._transfer_claimed
                or self.active is not lease
                or lease._state is not _OvLibrariesRenderLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
            ):
                raise RuntimeError("OVRTXRenderer owner transfer capability is invalid")
            self._owner_thread = threading.current_thread()
            self._transfer_claimed = True
            self._transfer_capability = None

    def cancel_owner_transfer(self, lease: OvLibrariesRenderLease, capability: object) -> None:
        with self._transfer_lock:
            # Cancellation is the one original-owner operation allowed while an offer is pending.
            if threading.current_thread() is not self._owner_thread:
                raise RuntimeError("OVRTXRenderer owner thread must cancel an owner transfer")
            if capability is None or capability is not self._transfer_capability or self.active is not lease:
                raise RuntimeError("OVRTXRenderer owner transfer capability is invalid")
            self._transfer_capability = None

    def capture_scene_publisher(self, lease: OvLibrariesRenderLease) -> OvLibrariesScenePublisher:
        """Create the scene publisher before render ownership transfers."""
        self.require_owner_thread("capture an OVStage scene publisher")
        if (
            self.active is not lease
            or lease._state is not _OvLibrariesRenderLeaseState.UNBOUND
            or self._publisher is not None
        ):
            raise RuntimeError("OVRTXRenderer is not eligible to capture an OVStage scene publisher")
        publisher = OvLibrariesScenePublisher(self, lease)
        self._publisher = publisher
        return publisher

    @contextlib.contextmanager
    def owner_lane(self, operation: str) -> Iterator[None]:
        """Reject nested operations and access outside the render-owner thread.

        This lane is not a cross-thread lock. The external consumer must share
        its application publication guard with the captured scene publisher.
        """
        self.require_owner_thread(f"enter the OVStage owner lane for {operation}")
        if self._lane_operation is not None:
            raise RuntimeError(
                f"OVStage owner lane is already active for {self._lane_operation}; nested writers are forbidden"
            )
        self._lane_operation = operation
        try:
            yield
        finally:
            self._lane_operation = None

    def validate_active(self, lease: OvLibrariesRenderLease) -> None:
        self.require_owner_thread("access a render lease")
        if self.active is not lease:
            raise RuntimeError("OvLibrariesRenderLease is not active for this renderer")
        if self._renderer._closed:
            raise RuntimeError("OvLibrariesRenderLease owner is closed or no longer uses OVStage")

    def validate_live(
        self,
        lease: OvLibrariesRenderLease,
        ovrtx_renderer: Renderer,
        ovstage_instance: OvStage,
        generation: int,
        *,
        allow_poisoned: bool = False,
    ) -> None:
        """Validate retained Python-object identity for the live attached pair.

        ``allow_poisoned`` admits retirement work after a failed publication.
        """
        self.validate_active(lease)
        if self._poisoned and not allow_poisoned:
            raise RuntimeError("OvLibrariesRenderLease owner pair is poisoned")
        if generation != self._generation:
            raise RuntimeError("OvLibrariesRenderLease has a stale Stage generation")
        if ovrtx_renderer is not self._renderer.backend.renderer or ovstage_instance is not self._renderer.scene.stage:
            raise RuntimeError("OvLibrariesRenderLease no longer matches its exact renderer-owned pair")
        if not self._renderer._initialized_scene or not self._renderer.backend.attached:
            raise RuntimeError("OvLibrariesRenderLease pair is no longer initialized and attached")

    def release(self, lease: OvLibrariesRenderLease) -> None:
        """Retire the lease after borrower destruction and rebase the dormant ordinal."""
        self.validate_active(lease)
        if self._lane_operation is not None:
            raise RuntimeError("Cannot release OvLibrariesRenderLease while the owner lane is active")
        if lease._state is _OvLibrariesRenderLeaseState.BOUND and (not self._poisoned):
            with self.owner_lane("lease-release ordinal rebase"):
                self._renderer.scene.ordinal = self._next_ordinal()
        self._last_step_id = _NO_PHYSICS_STEP_ID
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        self._transfer_claimed = False
        self.defer_until_transfer = False
        self.stage_deferred = False
        self.active = None
        if self._renderer.backend is not None:
            self._renderer.backend._render_lease_owner = None
        if self._renderer.scene is not None:
            self._renderer.scene._render_lease_owner = None

    def reject_ordinary_operation(self, operation: str, *, retiring: bool = False) -> None:
        """Reject ordinary OVRTX operations while an external lease is active.

        A pair poisoned by a failed leased write keeps rejecting ordinary use after
        release, because its ordinal is not rebased; only ``retiring`` operations
        (closing the renderer) are admitted so the pair can be recreated.
        """
        if self.active is not None or (
            self._renderer.backend is not None and self._renderer.backend._render_lease_owner is not None
        ):
            raise RuntimeError(f"Cannot {operation} while an OvLibrariesRenderLease is active")
        if self._poisoned and not retiring:
            raise RuntimeError(
                f"Cannot {operation}: a failed leased publication poisoned this OVRTX/OVStage pair; recreate it"
            )

    def owns_scene_updates(self) -> bool:
        """Whether an active lease's scene publisher owns transform and geometry writes."""
        return self.active is not None

    def producer_publication_required(self, lease: OvLibrariesRenderLease, physics_step_id: Hashable) -> bool:
        """Check publication on the application thread without entering the render thread."""
        if self.active is not lease or self._publisher is None:
            raise RuntimeError("OVStage scene publisher is not active for this lease")
        if self._poisoned or lease._state is not _OvLibrariesRenderLeaseState.BOUND:
            raise RuntimeError("OVStage scene publisher pair is unavailable")
        try:
            hash(physics_step_id)
        except TypeError as exc:
            raise TypeError("physics_step_id must be hashable") from exc
        writer_ref = self._renderer._visual_material_writer_ref
        writer = writer_ref() if writer_ref is not None else None
        return (
            self._last_step_id != physics_step_id
            or bool(self._renderer.marker_groups)
            or bool(self._renderer.markers)
            or (writer is not None and writer.has_pending_updates)
        )

    def write_producer_scene(
        self, lease: OvLibrariesRenderLease, ordinal: int, physics_step_id: Hashable, *, force: bool = False
    ) -> None:
        """Write a complete scene batch without advancing its reserved ordinal."""
        if ordinal <= 0:
            raise ValueError("OVStage publication ordinal must be nonzero")
        if not self.producer_publication_required(lease, physics_step_id) and not force:
            raise RuntimeError("OVStage scene publication is not required for this complete state")
        self._renderer.scene.ordinal = ordinal
        try:
            with self._renderer._publish_visual_materials():
                self._renderer._update_transforms()
                self._renderer._update_geometries()
        except Exception as exc:
            self._poisoned = True
            raise RuntimeError("Producer scene publication failed; recreate the exact pair") from exc
        self._last_step_id = physics_step_id

    def prepare_stage(self, num_envs: int) -> bool:
        """Validate or defer leased preparation; ordinary rendering is immediately allowed.

        Returns:
            Whether the renderer should materialize the Stage now.
        """
        lease = self.active
        if lease is not None:
            if self.defer_until_transfer and not self._transfer_claimed:
                if lease._state is not _OvLibrariesRenderLeaseState.UNBOUND:
                    raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
                self.stage_deferred = True
                return False
            self.require_owner_thread("prepare the leased Stage")
            if lease._state is not _OvLibrariesRenderLeaseState.UNBOUND:
                raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
            if num_envs != 1:
                raise RuntimeError("The render lease requires exactly one environment")
            if not self._renderer._use_ovstage or self._renderer.backend._render_lease_owner is not self._renderer:
                raise RuntimeError("The render lease requires its reserved private OVStage renderer")
        return True

    def materialize_deferred_stage(self, lease: OvLibrariesRenderLease, stage: Any, num_envs: int) -> bool:
        """Materialize a deferred Stage after render-thread ownership transfer."""
        self.require_owner_thread("materialize a deferred Stage")
        # Unclaimed, a fallback prepare_stage would only defer again; reject even if nothing is deferred yet.
        if self.defer_until_transfer and not self._transfer_claimed:
            raise RuntimeError("Deferred Stage materialization requires a claimed owner transfer")
        if not self.stage_deferred:
            return False
        if self.active is not lease or lease._state is not _OvLibrariesRenderLeaseState.UNBOUND:
            raise RuntimeError("Deferred Stage is not eligible for materialization")
        self._renderer.prepare_stage(stage, num_envs)
        return True

    def bind_stage(self) -> None:
        """Bind the prepared native pair after renderer initialization succeeds."""
        lease = self.active
        if lease is not None:
            self.stage_deferred = False
            self._generation += 1
            lease._bind(self._renderer.backend.renderer, self._renderer.scene.stage, self._generation)

    def reserve_pair(self) -> None:
        """Reserve a newly acquired private native pair during clone preparation."""
        renderer = self._renderer
        if renderer.backend._render_lease_owner is not None or renderer.scene._render_lease_owner is not None:
            raise RuntimeError("The native pair already has a render lease")
        with renderer.scene.stage.get_attribute_write_floor() as floor:
            floor.wait()
            if int(floor.fetch()) != 0:
                raise RuntimeError("The render lease requires an unused private OVStage")
        renderer.backend._render_lease_owner = renderer
        renderer.scene._render_lease_owner = renderer

    def _next_ordinal(self) -> int:
        """Return ``floor + 1`` from the exact renderer-owned Stage."""
        if self._renderer.scene.stage is None:
            raise RuntimeError("Cannot query an OVStage floor before scene materialization")
        with self._renderer.scene.stage.get_attribute_write_floor() as floor_query:
            floor_query.wait()
            floor = int(floor_query.fetch())
        if floor >= (1 << 64) - 1:
            raise RuntimeError("OVStage ordinal range is exhausted")
        return floor + 1
