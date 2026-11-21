# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that publishes the simulation's scene to an OVStage that the OVXR session renders.

``OvxrRenderer`` implements :class:`~isaaclab.renderers.BaseRenderer` for an OVXR session. It creates
its own OVStage and OVRTX engine and writes the simulation's scene to them, while the session's render
worker owns rendering. Its lease handles the render-thread handoff, deferred Stage preparation and
scene publication ordering. It renders no camera images; cameras keep their own renderers.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import tempfile
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

import numpy as np
import ovstage
import warp as wp
from isaaclab_ov.stage import (
    OvstageBackendCfg,
    export_stage_to_string,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

from pxr import Gf, Usd, UsdGeom

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.scene_data import SceneDataFormat
from isaaclab.utils import configclass

from .ovxr_scene import (
    _OvxrEngineCfg,
    _OvxrMarkers,
    _OvxrVisualMaterialWriter,
    _resolve_env_light_partitions,
)

if TYPE_CHECKING:
    from isaaclab_ov.stage import OvstageBackend
    from ovrtx import Renderer
    from ovstage import Stage as OvStage

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.utils.warp import ProxyArray

    from .ovxr_scene import _OvxrEngine

logger = logging.getLogger(__name__)

_NO_PHYSICS_STEP_ID = object()
_GPU_CAMERA_TRANSFORMS_ATTRIBUTE = "omni:rtx:scene:hydra:readGPUCameraTransformsFromFabricInRenderDelegate"


class _OvxrLeaseState(Enum):
    """Internal lifecycle state for a renderer/scene pair lease."""

    UNBOUND = auto()
    BOUND = auto()
    RELEASED = auto()


class OvxrRenderLease:
    """Exclusive non-owning lease of the OVXR session over one attached OVRTX/OVStage pair.

    The session acquires the lease before environment construction. The renderer then defers
    Stage preparation until Isaac Capture's render worker claims the single-use owner transfer
    and materializes the Stage, which binds this object to the exact live Python ``Renderer``
    and ``Stage`` instances. The session releases the lease before the renderer is closed and
    never detaches or destroys either borrowed object.
    """

    _controller: _OvxrLeaseController
    """Renderer-owned controller that created this lease."""

    def __init__(self) -> None:
        raise RuntimeError("OvxrRenderLease must be acquired from OvxrRenderer")

    def __enter__(self) -> OvxrRenderLease:
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

    def create_native_pair(self) -> None:
        """Create the session's own OVStage and the OVRTX engine that attaches to it.

        Call this after process-wide renderer settings are applied and before the environment is
        built, so that clone preparation delivers the scene copies to the Stage. Calling it again
        on the same renderer is a no-op.

        Raises:
            RuntimeError: If called off the owner thread or on an inactive lease.
        """
        self._controller.create_native_pair(self)

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

    def materialize_deferred_stage(self, stage: Any, num_envs: int) -> bool:
        """Complete a deferred Stage preparation on the claimed render owner.

        Args:
            stage: The environment USD stage.
            num_envs: Number of environments; the lease supports exactly one.

        Returns:
            Whether a deferred preparation was completed. ``False`` means nothing was deferred.

        Raises:
            RuntimeError: If called off the render-owner thread, or before the owner transfer is claimed.
        """
        return self._controller.materialize_deferred_stage(self, stage, num_envs)

    def capture_scene_publisher(self) -> OvxrScenePublisher:
        """Capture the application-thread scene publisher before render ownership transfers.

        Call this on the owner thread while the lease is unbound and before
        :meth:`offer_owner_transfer`. The publisher stays bound to the calling thread; see
        :class:`OvxrScenePublisher` for the publication
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

        Isaac Capture's render worker runs its native frame operations here. The application's
        publication guard separately excludes scene publication while they use the borrowed stage.
        A failed publication poisons publication only: the lane stays available so the session can
        destroy its OVXR context and retire the pair.
        """
        self._validate_live(allow_poisoned=True)
        owner = self._controller
        with owner.owner_lane("render-worker operation"):
            yield

    def release(self) -> None:
        """Release the borrowing intent without destroying either pair member."""
        if self._state is _OvxrLeaseState.RELEASED:
            return
        owner = self._controller
        owner.release(self)
        self._renderer = None
        self._stage = None
        self._state = _OvxrLeaseState.RELEASED

    def validate_live(self) -> None:
        """Verify that this lease still refers to its live, unpoisoned renderer and stage."""
        self._validate_live(allow_poisoned=False)

    def _validate_live(self, *, allow_poisoned: bool) -> None:
        self._validate_active()
        if self._state is not _OvxrLeaseState.BOUND:
            raise RuntimeError("OvxrRenderLease is not bound to a prepared Stage")
        owner = self._controller
        owner.validate_live(
            self,
            cast("Renderer", self._renderer),
            cast("OvStage", self._stage),
            self._generation,
            allow_poisoned=allow_poisoned,
        )

    @classmethod
    def _create_unbound(cls, owner: _OvxrLeaseController) -> OvxrRenderLease:
        lease = object.__new__(cls)
        lease._controller = owner
        lease._state = _OvxrLeaseState.UNBOUND
        lease._renderer = None
        lease._stage = None
        lease._generation = 0
        return lease

    def _bind(self, renderer: Renderer, stage: OvStage, generation: int) -> None:
        if self._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("OvxrRenderLease can only bind once")
        self._renderer = renderer
        self._stage = stage
        self._generation = generation
        self._state = _OvxrLeaseState.BOUND

    def _validate_active(self) -> None:
        if self._state is _OvxrLeaseState.RELEASED:
            raise RuntimeError("OvxrRenderLease is released")
        owner = self._controller
        owner.validate_active(self)


class OvxrScenePublisher:
    """Application-thread publisher of complete provider-derived OVStage scenes.

    Obtain it with :meth:`OvxrRenderLease.capture_scene_publisher`. The renderer makes no eager
    transform, geometry or material writes of its own; the OVXR session publishes each complete
    physics state through :meth:`publish`, one transaction per scene:

    1. Skip a state that needs no publication (see :meth:`publication_required`).
    2. Run the simulation's visualization-marker callbacks, then the session's preparation, such as
       capturing the inputs that the seal publishes.
    3. Reserve an ordinal, write every scene component at it, and seal it.
    4. If the write or seal fails, poison the pair, abandon the ordinal and re-raise.

    :meth:`~isaaclab.sim.SimulationContext.update_visualizers` runs the marker callbacks only for
    marker-capable visualizers, so without one, step 2 is what moves debug markers before they are drawn.

    Call :meth:`publish` under the application's publication guard, which the render owner also holds
    around every operation in :meth:`OvxrRenderLease.ovstage_owner_lane`, so that scene writes and
    native frame operations never interleave. A poisoned pair rejects later publication, but the
    session can still retire its OVXR context and release the lease; the renderer must then be
    recreated. :meth:`close` stops publication; releasing the lease also closes
    the publisher. ``isaaclab_teleop.OvLibrariesXrSession`` publishes through this method.
    """

    def __init__(self, owner: _OvxrLeaseController, lease: OvxrRenderLease) -> None:
        self._controller = owner
        self._lease = lease
        self._application_thread = threading.current_thread()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether publication stopped, by :meth:`close`, a failed publication or :meth:`invalidate`."""
        return self._closed

    def publication_required(self, physics_step_id: Hashable) -> bool:
        """Return whether a new physics step, changed markers or dirty materials require a scene write.

        Args:
            physics_step_id: Hashable identity of the complete physics state to publish. Only the
                last written identity is remembered, so the identity must change whenever the
                state does, including after a reset that does not advance the physics step count.

        Returns:
            Whether ``physics_step_id`` differs from the last written state, the drawn markers
            changed, or a visual material has pending updates.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, the lease is not bound, or a failed write poisoned the pair.
            TypeError: If ``physics_step_id`` is not hashable.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            return self._controller.producer_publication_required(self._lease, physics_step_id)

    def publish(
        self,
        physics_step_id: Hashable,
        ordinals: Any,
        *,
        prepare: Callable[[], None] | None = None,
        seal: Callable[[int], None] | None = None,
        force: bool = False,
    ) -> int | None:
        """Publish one complete scene as a single transaction.

        Args:
            physics_step_id: Hashable identity of the complete physics state, as for
                :meth:`publication_required`.
            ordinals: Stage ordinal ledger with ``reserve()``, ``publish(ordinal)`` and
                ``abandon(ordinal)``, such as Isaac Capture's ``StageOrdinalLedger``.
            prepare: Called once a publication is required, after the marker callbacks and before an
                ordinal is reserved.
            seal: Seals the written ordinal. Defaults to ``ordinals.publish``.
            force: Publish even when the state needs no publication, for example for an
                independently changed XR anchor.

        Returns:
            The sealed ordinal, or ``None`` when the state needs no publication.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, or the write or seal failed. A failed write or seal poisons the pair,
                abandons the ordinal and closes the publisher.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            controller = self._controller
            if not force and not controller.producer_publication_required(self._lease, physics_step_id):
                return None
            # Callbacks only update marker state, so a failure here leaves the pair publishable.
            controller.dispatch_marker_callbacks()
            if prepare is not None:
                prepare()
            controller.require_publishable(self._lease)
            ordinal = ordinals.reserve()
            try:
                controller.write_scene_components(self._lease, ordinal, physics_step_id)
                (ordinals.publish if seal is None else seal)(ordinal)
            except BaseException as error:
                self._invalidate_locked()
                try:
                    ordinals.abandon(ordinal)
                except BaseException as abandon_error:
                    # Cleanup follows the failure; retain its original cause.
                    error.add_note(f"Abandon after failed scene publication also failed: {abandon_error!r}")
                raise
            return ordinal

    def invalidate(self) -> None:
        """Reject further use of a pair after the session's own write to it failed."""
        with self._lock:
            self._invalidate_locked()

    def close(self) -> None:
        """Stop scene publication after any in-flight scene finishes. Safe to call repeatedly."""
        with self._lock:
            self._closed = True

    def _invalidate_locked(self) -> None:
        self._closed = True
        if self._controller.active is self._lease:
            self._controller._poisoned = True

    def _require_application_thread(self) -> None:
        if threading.current_thread() is not self._application_thread:
            raise RuntimeError("OVStage scene publisher must run on its captured application thread")


class _OvxrLeaseController:
    """Coordinate the OVXR session's lease without owning the renderer's native resources.

    Lease state, thread transfer and publication ordering live here. The renderer
    retains scene/camera storage and supplies the component-write operations.
    """

    def __init__(self, renderer: OvxrRenderer) -> None:
        self._renderer = renderer
        self._owner_thread = threading.current_thread()
        self._transfer_lock = threading.Lock()
        self._transfer_capability: object | None = None
        self._transfer_claimed = False
        self.active: OvxrRenderLease | None = None
        self._publisher: OvxrScenePublisher | None = None
        self._generation = 0
        self._lane_operation: str | None = None
        self._last_step_id: Hashable = _NO_PHYSICS_STEP_ID
        self._published_marker_revision: int | None = None
        self._poisoned = False
        self.stage_deferred = False
        self._marker_consumer: object | None = None
        self._previous_marker_groups: Iterable = ()
        self._created_pair = False

    def acquire(self) -> OvxrRenderLease:
        """Reserve the renderer before stage preparation; :meth:`create_native_pair` creates its pair."""
        self.require_owner_thread("acquire a render lease")
        if self._renderer._closed:
            raise RuntimeError("Cannot acquire OvxrRenderLease from a closed renderer")
        if self.active is not None:
            raise RuntimeError("An OvxrRenderLease is already active")
        if self._renderer._initialized_scene or self._generation != 0:
            raise RuntimeError("OvxrRenderLease must be acquired before Stage preparation")
        lease = OvxrRenderLease._create_unbound(self)
        registry = self._renderer._sim.vis_marker_registry
        self._marker_consumer = registry.register_consumer()
        self._previous_marker_groups = self._renderer.marker_groups
        self._renderer.marker_groups = registry.get_groups().values()
        self.active = lease
        return lease

    def create_native_pair(self, lease: OvxrRenderLease) -> None:
        """Create the Stage and engine this lease lends, once, before clone preparation."""
        self.validate_active(lease)
        self.require_owner_thread("create the native pair")
        if self._created_pair:
            return
        # The simulation's registry owns both. Clone preparation delivers the scene copies to every
        # registered OVStage, and teardown closes the engine, registered second, before the Stage.
        renderer = self._renderer
        sim = renderer._sim
        scene = sim.get_or_create_backend(OvstageBackendCfg(scene_key=renderer.cfg))
        renderer.backend = sim.get_or_create_backend(
            _OvxrEngineCfg(
                scene_key=renderer.cfg,
                log_file_path=renderer.cfg.log_file_path,
                log_level=renderer.cfg.log_level,
                device=sim.device,
            )
        )
        renderer.scene = scene
        self._created_pair = True

    def require_owner_thread(self, operation: str) -> None:
        """Reject lease operations outside the renderer construction thread."""
        if self._transfer_capability is not None:
            raise RuntimeError("OvxrRenderer owner transfer is pending")
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError(f"OvxrRenderer owner thread must {operation}")

    def offer_owner_transfer(self, lease: OvxrRenderLease) -> object:
        self.require_owner_thread("offer an owner transfer")
        with self._transfer_lock:
            if (
                self.active is not lease
                or lease._state is not _OvxrLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
                or self._transfer_capability is not None
                or self._transfer_claimed
            ):
                raise RuntimeError("OvxrRenderer is not eligible for a pre-use owner transfer")
            capability = object()
            self._transfer_capability = capability
            return capability

    def claim_owner_transfer(self, lease: OvxrRenderLease, capability: object) -> None:
        with self._transfer_lock:
            if (
                capability is None
                or capability is not self._transfer_capability
                or self._transfer_claimed
                or self.active is not lease
                or lease._state is not _OvxrLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
            ):
                raise RuntimeError("OvxrRenderer owner transfer capability is invalid")
            self._owner_thread = threading.current_thread()
            self._transfer_claimed = True
            self._transfer_capability = None

    def cancel_owner_transfer(self, lease: OvxrRenderLease, capability: object) -> None:
        with self._transfer_lock:
            # Cancellation is the one original-owner operation allowed while an offer is pending.
            if threading.current_thread() is not self._owner_thread:
                raise RuntimeError("OvxrRenderer owner thread must cancel an owner transfer")
            if capability is None or capability is not self._transfer_capability or self.active is not lease:
                raise RuntimeError("OvxrRenderer owner transfer capability is invalid")
            self._transfer_capability = None

    def capture_scene_publisher(self, lease: OvxrRenderLease) -> OvxrScenePublisher:
        """Create the scene publisher before render ownership transfers."""
        self.require_owner_thread("capture an OVStage scene publisher")
        if self.active is not lease or lease._state is not _OvxrLeaseState.UNBOUND or self._publisher is not None:
            raise RuntimeError("OvxrRenderer is not eligible to capture an OVStage scene publisher")
        publisher = OvxrScenePublisher(self, lease)
        self._publisher = publisher
        return publisher

    @contextlib.contextmanager
    def owner_lane(self, operation: str) -> Iterator[None]:
        """Reject nested operations and access outside the render-owner thread.

        This lane is not a cross-thread lock. The OVXR session holds its application publication
        guard around the lane and around scene publication.
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

    def validate_active(self, lease: OvxrRenderLease) -> None:
        self.require_owner_thread("access a render lease")
        if self.active is not lease:
            raise RuntimeError("OvxrRenderLease is not active for this renderer")
        if self._renderer._closed:
            raise RuntimeError("OvxrRenderLease owner is closed or no longer uses OVStage")

    def validate_live(
        self,
        lease: OvxrRenderLease,
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
            raise RuntimeError("OvxrRenderLease owner pair is poisoned")
        if generation != self._generation:
            raise RuntimeError("OvxrRenderLease has a stale Stage generation")
        if ovrtx_renderer is not self._renderer.backend.renderer or ovstage_instance is not self._renderer.scene.stage:
            raise RuntimeError("OvxrRenderLease no longer matches its exact renderer-owned pair")
        if not self._renderer._initialized_scene or not self._renderer.backend.attached:
            raise RuntimeError("OvxrRenderLease pair is no longer initialized and attached")

    def release(self, lease: OvxrRenderLease) -> None:
        """Retire the lease after the OVXR context is destroyed and rebase the dormant ordinal."""
        self.validate_active(lease)
        if self._lane_operation is not None:
            raise RuntimeError("Cannot release OvxrRenderLease while the owner lane is active")
        if self._publisher is not None:
            self._publisher.close()
        if lease._state is _OvxrLeaseState.BOUND and (not self._poisoned):
            with self.owner_lane("lease-release ordinal rebase"):
                self._renderer.scene.ordinal = self._next_ordinal()
        self._last_step_id = _NO_PHYSICS_STEP_ID
        self._published_marker_revision = None
        self._publisher = None
        self._transfer_claimed = False
        self.stage_deferred = False
        self._renderer._sim.vis_marker_registry.unregister_consumer(self._marker_consumer)
        self._marker_consumer = None
        self._renderer.marker_groups = self._previous_marker_groups
        self._previous_marker_groups = ()
        self.active = None

    def reject_ordinary_operation(self, operation: str) -> None:
        """Reject releasing renderer resources while the session's lease borrows them."""
        if self.active is not None:
            raise RuntimeError(f"Cannot {operation} while an OvxrRenderLease is active")

    def require_publishable(self, lease: OvxrRenderLease) -> None:
        """Raise unless ``lease`` holds the active, bound and unpoisoned publisher pair."""
        if self.active is not lease or self._publisher is None:
            raise RuntimeError("OVStage scene publisher is not active for this lease")
        if self._poisoned or lease._state is not _OvxrLeaseState.BOUND:
            raise RuntimeError("OVStage scene publisher pair is unavailable")

    def producer_publication_required(self, lease: OvxrRenderLease, physics_step_id: Hashable) -> bool:
        """Check publication on the application thread without entering the render thread."""
        self.require_publishable(lease)
        try:
            hash(physics_step_id)
        except TypeError as exc:
            raise TypeError("physics_step_id must be hashable") from exc
        writer_ref = self._renderer._visual_material_writer_ref
        writer = writer_ref() if writer_ref is not None else None
        return (
            self._last_step_id != physics_step_id
            or self._marker_revision() != self._published_marker_revision
            or (writer is not None and writer.has_pending_updates)
        )

    def dispatch_marker_callbacks(self) -> None:
        """Run the simulation's visualization-marker callbacks so the next scene write draws current markers."""
        self._renderer._sim.vis_marker_registry.dispatch_callbacks()

    def _marker_revision(self) -> int | None:
        """Revision of the simulation's marker groups that this renderer draws, or ``None`` without markers.

        The registry revision changes when a group is added, updated, hidden or removed.
        """
        if not self._renderer.marker_groups and not self._renderer.markers:
            return None
        return self._renderer._sim.vis_marker_registry.revision

    def write_scene_components(self, lease: OvxrRenderLease, ordinal: int, physics_step_id: Hashable) -> None:
        """Write every scene component at a reserved ordinal; the publisher validates and seals."""
        if ordinal <= 0:
            raise ValueError("OVStage publication ordinal must be nonzero")
        self.require_publishable(lease)
        try:
            self._write_components(ordinal)
        except Exception as exc:
            self._poisoned = True
            raise RuntimeError("Producer scene publication failed; recreate the exact pair") from exc
        self._last_step_id = physics_step_id
        self._published_marker_revision = self._marker_revision()

    def prepare_stage(self, num_envs: int) -> bool:
        """Validate or defer leased preparation; the renderer prepares its Stage only under a lease.

        Camera initialization prepares every registered renderer on the application thread. A leased
        pair only records that request until Capture's render worker claims the owner transfer.

        Returns:
            Whether the renderer should materialize the Stage now.

        Raises:
            RuntimeError: Without an active lease, or when the lease cannot prepare its Stage.
        """
        lease = self.active
        if lease is None:
            raise RuntimeError("OvxrRenderer prepares its Stage only under the OVXR session's OvxrRenderLease")
        if not self._transfer_claimed:
            if lease._state is not _OvxrLeaseState.UNBOUND:
                raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
            self.stage_deferred = True
            return False
        self.require_owner_thread("prepare the leased Stage")
        if lease._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
        if num_envs != 1:
            raise RuntimeError("The render lease requires exactly one environment")
        if not self._created_pair:
            raise RuntimeError("The render lease requires the pair that create_native_pair() creates")
        with self._renderer.scene.stage.get_attribute_write_floor() as floor:
            floor.wait()
            # A new OVStage is sealed through the minimum ordinal; any published write raises the floor.
            if int(floor.fetch()) != ovstage.ORDINAL_MIN:
                raise RuntimeError("The render lease requires an unused OVStage")
        return True

    def materialize_deferred_stage(self, lease: OvxrRenderLease, stage: Any, num_envs: int) -> bool:
        """Materialize a deferred Stage after render-thread ownership transfer."""
        self.require_owner_thread("materialize a deferred Stage")
        # Unclaimed, a fallback prepare_stage would only defer again; reject even if nothing is deferred yet.
        if not self._transfer_claimed:
            raise RuntimeError("Deferred Stage materialization requires a claimed owner transfer")
        if not self.stage_deferred:
            return False
        if self.active is not lease or lease._state is not _OvxrLeaseState.UNBOUND:
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

    def _next_ordinal(self) -> int:
        """Return ``floor + 1`` from the exact renderer-owned Stage."""
        if self._renderer.scene.stage is None:
            raise RuntimeError("Cannot query an OVStage floor before scene materialization")
        with self._renderer.scene.stage.get_attribute_write_floor() as floor_query:
            floor_query.wait()
            floor = int(floor_query.fetch())
        if floor >= ovstage.ORDINAL_MAX:
            raise RuntimeError("OVStage ordinal range is exhausted")
        return floor + 1

    def _write_components(self, ordinal: int) -> None:
        """Write selected scene components; the caller owns validation and commit."""
        self._renderer.scene.ordinal = ordinal
        with self._renderer._publish_visual_materials():
            self._renderer._publish_transforms()
            self._renderer._publish_geometries()


@configclass
class OvxrRendererCfg(RendererCfg):
    """Configuration of the renderer that publishes the simulation's scene for an OVXR session.

    The session's :class:`OvxrRenderer` creates its own OVStage and OVRTX engine and lends them to
    Isaac Capture's render worker. It renders no camera images, so cameras keep their own renderer
    configurations, such as :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`.
    """

    class_type: type[OvxrRenderer] | str = "{DIR}.ovxr_renderer:OvxrRenderer"
    """Renderer implementation class."""

    renderer_type: str = "ovxr"
    """Type identifier for the OVXR renderer."""

    launcher_type: ClassVar[str] = "isaaclab_ov.app:OvrtxLauncher"
    """The launcher that registers the OVRTX runtime before the stage is created."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab_ov.cloner:OvstageReplicateContext",)
    """Deliver the scene copies to the renderer's own OVStage."""

    # FIXME: revert once OVRTX stops warning on every cloned attribute without data.
    log_level: str = "error"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for the OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Return no camera outputs: the renderer produces no camera images."""
        return {}


@final
class OvxrRenderer(BaseRenderer):
    """Renderer that publishes the simulation's scene to the OVStage that the OVXR session renders.

    It works only under the session's :class:`OvxrRenderLease`: the lease creates the renderer's own
    OVStage and OVRTX engine, the session's render worker prepares the Stage and renders it, and the
    session's :class:`OvxrScenePublisher` writes each complete scene. The renderer produces no camera
    images and rejects camera sensors.

    This class is final and is not designed for subclassing.
    """

    cfg: OvxrRendererCfg

    def __init__(self, cfg: OvxrRendererCfg, *, sim: Any):
        super().__init__(cfg, sim=sim)
        self._warp_device = wp.get_device(self._sim.device)
        self._device = str(self._warp_device)
        self.markers: dict[Any, _OvxrMarkers] = {}
        self.marker_groups: Iterable = ()
        self._sdp = self._sim.get_scene_data_provider()
        self._transforms_timestamp = -1
        self._transforms = SceneDataFormat.TransposedMatrix44d()
        self._object_scales: wp.array | None = None
        self._geometry_paths: list[str] = []
        self._geometry_timestamp = -1
        self._initialized_scene = False
        self._visual_material_writer_ref: weakref.ReferenceType[_OvxrVisualMaterialWriter] | None = None
        self.scene: OvstageBackend | None = None
        """The renderer's own OVStage, registered with the simulation when the lease creates the pair."""
        self.backend: _OvxrEngine | None = None
        """Native engine that attaches to :attr:`scene`, registered with the simulation after it."""
        self._object_xform_binding: ovstage.Query | None = None
        self._geometry_points_binding: ovstage.Query | None = None
        self._bindings = contextlib.ExitStack()
        self._closed = False
        self._render_lease = _OvxrLeaseController(self)

    def acquire_render_lease(self) -> OvxrRenderLease:
        """Reserve the native renderer/scene pair before stage preparation."""
        return self._render_lease.acquire()

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """See :meth:`OvxrRendererCfg.supported_output_types`."""
        return self.cfg.supported_output_types()

    @property
    def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], Any] | None:
        """Return a writer factory while the session holds the lease.

        The writer exists before the render worker prepares the deferred Stage, because environment
        reset finalizes material consumers first; addresses compile on the first publication after
        population.
        """
        return self._create_visual_material_writer if self._render_lease.active is not None else None

    def _create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> _OvxrVisualMaterialWriter:
        writer = _OvxrVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    def prepare_stage(self, stage: Usd.Stage, num_envs: int) -> None:
        """Prepare the Stage on the render owner, or defer it until the session's render worker claims the pair.

        Raises:
            RuntimeError: Without the session's lease, or when the lease cannot prepare its Stage.
        """
        if not self._render_lease.prepare_stage(num_envs):
            return
        self._populate_scene(stage)
        self._render_lease.bind_stage()

    # The session's scene publisher writes every scene; ordinary scene updates write nothing.

    def update_transforms(self) -> None:
        """Do nothing: the session's scene publisher writes transforms with each complete scene."""

    def update_geometries(self) -> None:
        """Do nothing: the session's scene publisher writes geometry with each complete scene."""

    # Camera sensors are rejected; this renderer produces no camera images.

    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        raise RuntimeError("OvxrRenderer renders no camera sensors; give the camera its own renderer configuration")

    def set_outputs(self, render_data: Any, output_data: dict[str, ProxyArray]) -> None:
        """Do nothing: this renderer writes no camera outputs."""

    def prepare_capture(self, render_data: Any, camera_data: Any, frame: ProxyArray) -> None:
        """Do nothing: this renderer captures no camera images."""

    def reset(self, render_data: Any, env_ids: Sequence[int] | None = None) -> None:
        """Do nothing: this renderer keeps no per-camera state."""

    def update_camera(
        self, render_data: Any, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray
    ) -> None:
        raise RuntimeError("OvxrRenderer has no camera to update")

    def update_camera_intrinsics(self, render_data: Any, intrinsics: wp.array, parameters: wp.array) -> None:
        raise RuntimeError("OvxrRenderer produces no camera images; the OVXR session renders its Stage")

    def render(self, render_data: Any) -> None:
        raise RuntimeError("OvxrRenderer produces no camera images; the OVXR session renders its Stage")

    def render_batch(self, render_data: Sequence[Any]) -> None:
        """Reject a camera batch; an empty batch performs no work."""
        if render_data:
            raise RuntimeError("OvxrRenderer produces no camera images; the OVXR session renders its Stage")

    def read_output(self, render_data: Any, camera_data: Any) -> None:
        raise RuntimeError("OvxrRenderer produces no camera images; the OVXR session renders its Stage")

    def cleanup(self, render_data: Any) -> None:
        """Reject releasing camera render data while the session holds the lease; there is none to release."""
        if render_data is not None:
            self._render_lease.reject_ordinary_operation("release camera render data")

    def close(self) -> None:
        """Release scene queries and markers once the lease is released; the registry owns the engine and Stage."""
        self._render_lease.reject_ordinary_operation("close the renderer")
        try:
            with contextlib.ExitStack() as resources:
                resources.callback(self._bindings.close)
                for marker in self.markers.values():
                    resources.callback(marker.close)
        finally:
            self._object_xform_binding = self._geometry_points_binding = None
            self._object_scales = None
            self._geometry_paths = []
            self._geometry_timestamp = -1
            self._transforms_timestamp = -1
            self._initialized_scene = False
            self._visual_material_writer_ref = None
            self.markers.clear()
            self._closed = True

    # Stage preparation on the render owner, following the OVStage path of Isaac Lab's OVRTX renderer.

    def _populate_scene(self, stage: Usd.Stage) -> None:
        """Populate, clone and bind the renderer's own OVStage, then attach the engine to it."""
        scene = self.scene
        logger.info("Preparing the OVXR Stage...")
        env_paths = scene.clone_env_paths
        light_partitions = _resolve_env_light_partitions(stage, env_paths, copies=scene.clone_copies)
        usd = export_stage_to_string(
            stage, env_paths, source_paths=tuple(source for source, _ in scene.clone_copies),
            keep_env_roots=False, env_paths=scene.population_env_paths, preserve_references=True,
        )  # fmt: skip
        scene.populate(usd)
        partition_writes = (
            (env_paths, "primvars:omni:scenePartition", [f"env_{i}" for i in range(len(env_paths))]),
            (list(light_partitions), "omni:scenePartition", list(light_partitions.values())),
        )
        for paths, attribute_name, tokens in partition_writes:
            if not paths:
                continue
            with scene.query(paths) as query:
                scene.stage.write_attribute(
                    query, attribute_name, ordinal=scene.ordinal,
                    tensors=np.array([scene.paths.intern_token(token).raw_id for token in tokens], dtype=np.uint64),
                    is_array=False, semantic=ovstage.AttributeSemantic.TOKEN_ID,
                ).wait()  # fmt: skip
        self._setup_xform_bindings(stage)
        self._setup_geometry_bindings()
        # OVRTX 0.7's GPU camera cache mixes tiled views and misses pose updates. Disable only camera
        # reads before attaching the OVStage, retaining GPU instance transforms for large articulated scenes.
        settings_path = str(stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings")
        with scene.query([settings_path]) as query:
            scene.stage.write_attribute(
                query, _GPU_CAMERA_TRANSFORMS_ATTRIBUTE, ordinal=scene.ordinal,
                tensors=np.array([False], dtype=np.bool_), is_array=False,
            ).wait()  # fmt: skip
        scene.commit()
        if not self.backend.attached:
            self.backend.renderer.attach_ovstage(scene.stage)
            self.backend.attached = True
            scene.resources.callback(self.backend.detach)
        scene.resources.callback(self._sim.render_context.reset_stage_prepare_flag)
        scene.resources.callback(self.close)
        self._initialized_scene = True

    def _setup_xform_bindings(self, stage: Usd.Stage) -> None:
        """Bind SDP body paths and preserve their authored scales when writing world poses."""
        object_paths = self._sdp.backend.transform_paths
        if not object_paths:
            return
        # SDP poses omit scale. Include native copies that do not exist on the authored USD stage.
        scales_by_path = {}
        xform_cache = UsdGeom.XformCache()
        for root, targets in self.scene.clone_copies:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if not prim.IsA(UsdGeom.Xformable):
                    continue
                scale = tuple(map(float, Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()))
                if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                    path = str(prim.GetPath())
                    scales_by_path[path] = scale
                    scales_by_path.update((target + path[len(root) :], scale) for target in targets)
        self._object_scales = None
        # No override lets renderers share SDP's publication instead of multiplying by ones.
        if any(path in scales_by_path for path in object_paths):
            scales = [scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
            self._object_scales = wp.array(scales, dtype=wp.vec3f, device=self._device)
        self._object_xform_binding = self._bindings.enter_context(self.scene.query(object_paths))
        self.scene.stage.write_attribute(
            self._object_xform_binding,
            "omni:resetXformStack",
            ordinal=self.scene.ordinal,
            tensors=np.ones(len(object_paths), dtype=np.bool_),
            is_array=False,
        ).wait()

    def _setup_geometry_bindings(self) -> None:
        """Bind SDP's world-space point prims without applying inherited transforms again."""
        self._geometry_paths = list(self._sdp.get_geometry_points())
        if not self._geometry_paths:
            return
        prim_count = len(self._geometry_paths)
        self._geometry_points_binding = self._bindings.enter_context(self.scene.query(self._geometry_paths))
        self.scene.stage.write_attribute(
            self._geometry_points_binding,
            "omni:resetXformStack",
            ordinal=self.scene.ordinal,
            tensors=np.ones(prim_count, dtype=np.bool_),
            is_array=False,
        ).wait()
        self.scene.stage.write_attribute(
            self._geometry_points_binding,
            "omni:xform",
            ordinal=self.scene.ordinal,
            tensors=xform_tensor_from_numpy(np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1))),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

    # Scene writes for the session's publisher, at the ordinal it reserved.

    @contextlib.contextmanager
    def _publish_visual_materials(self) -> Iterator[None]:
        """Publish dirty materials and complete their reads before buffers may change."""
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            yield
        finally:
            if material_writer is not None:
                # When another exception is already propagating, log the drain failure instead of
                # replacing it. Losing it silently would hide a failed material write.
                primary_active = sys.exc_info()[0] is not None
                try:
                    material_writer.drain()
                except Exception as e:
                    if not primary_active:
                        raise
                    logger.warning("Error draining material writes after a failed scene update: %s", e, exc_info=True)

    def _publish_transforms(self) -> None:
        """Write changed SDP transforms and wait for their consumption."""
        binding = self._object_xform_binding
        if binding is None:
            return
        timestamp = self._sdp.backend.transforms_timestamp
        if self._transforms_timestamp == timestamp:
            return
        if not self._sdp.get_transforms(self._transforms, allow_passthrough=True, scales=self._object_scales):
            return
        # The producing CUDA stream orders access to the borrowed buffer.
        self.scene.stage.write_attribute(
            binding,
            "omni:xform",
            ordinal=self.scene.ordinal,
            tensors=xform_tensor_from_warp(self._transforms.matrices),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=self._cuda_stream(),
        ).wait()
        self._transforms_timestamp = timestamp

    def _publish_geometries(self) -> None:
        """Draw the current markers, then write changed SDP points and wait for their consumption."""
        for marker in tuple(self.markers):
            if marker not in self.marker_groups:
                self.markers.pop(marker).close()
        for marker in self.marker_groups:
            if marker.translations is None:
                continue
            if marker not in self.markers:
                self.markers[marker] = _OvxrMarkers(self, marker.cfg)
            self.markers[marker].update(
                marker.translations, marker.orientations, marker.scales, marker.marker_indices,
                visible=marker.is_visible(),
            )  # fmt: skip
        binding = self._geometry_points_binding
        if binding is None:
            return
        points = self._sdp.get_geometry_points()
        points = [points[path] for path in self._geometry_paths]
        timestamp = self._sdp.backend.geometry_timestamp
        if self._geometry_timestamp == timestamp:
            return
        self.scene.stage.write_attribute(
            binding,
            "points",
            ordinal=self.scene.ordinal,
            tensors=[points_tensor_from_warp(array) for array in points],
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            cuda_stream=self._cuda_stream(),
        ).wait()
        self._geometry_timestamp = timestamp

    def _cuda_stream(self) -> int | None:
        """Return the simulation device's CUDA stream handle, or ``None`` to write host memory on a CPU device."""
        return (self._warp_device.stream.cuda_stream or 1) if self._warp_device.is_cuda else None
