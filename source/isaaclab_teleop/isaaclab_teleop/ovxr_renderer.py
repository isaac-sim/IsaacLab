# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that publishes the simulation's scene to an OVStage that the OVXR session renders.

``OvxrRenderer`` implements :class:`~isaaclab.renderers.BaseRenderer` for an OVXR session. It creates
its own OVStage and OVRTX engine and writes the simulation's scene to them, while the session's render
worker owns rendering. Its lease handles the render-thread handoff, deferred Stage preparation, scene
publication ordering and the selected camera panels, whose render products the session presents. It
produces no camera images; other cameras keep their own renderers.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import tempfile
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

import numpy as np
import ovstage
import warp as wp
from isaaclab_ov.cloner import ovstage_replicate
from isaaclab_ov.renderers import OVRTXBackendCfg, OVRTXRendererCfg
from isaaclab_ov.renderers.ovrtx_usd import build_render_product_as_string
from isaaclab_ov.stage import (
    OvstageBackendCfg,
    export_stage_to_string,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.scene_data import SceneDataFormat
from isaaclab.utils import configclass

from .ovxr_gpu_publication import _excludes_simulation_gpu, _GeometryPublicationBuffers, parse_active_cuda_gpus
from .ovxr_scene import (
    _camera_transforms_kernel,
    _OvxrEngineCfg,
    _OvxrMarkers,
    _OvxrPanelRenderData,
    _OvxrVisualMaterialWriter,
    _remove_usd_reference,
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


@dataclass
class _AuxiliaryCamera:
    cfg: Any
    max_fps: float | None
    spec: CameraRenderSpec | None = None
    render_data: _OvxrPanelRenderData | None = None
    state: tuple[Any, ...] | None = None
    product_path: str | None = None


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
    def auxiliary_render_product_paths(self) -> tuple[str, ...]:
        """RenderProduct paths of the selected auxiliary cameras, in configuration order.

        Raises:
            RuntimeError: If the lease is not bound and live, or a selected camera did not
                materialize exactly one product.
        """
        self.validate_live()
        return self._controller.product_paths(self)

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

    def configure_auxiliary_camera(self, camera_cfg: Any, *, max_fps: float | None = None) -> None:
        """Authorize an exact camera before Stage materialization.

        The leased renderer then accepts the one camera sensor whose specification matches
        ``camera_cfg`` exactly: a single camera view with the same width, height and ordered
        ``data_types``, at ``prim_path`` (``{ENV_REGEX_NS}`` matches any environment) or its
        ``camera`` child prim. Every other camera on the leased renderer remains rejected.

        Args:
            camera_cfg: Configuration of the camera sensor to authorize, such as a
                :class:`~isaaclab.sensors.camera.CameraCfg`. Its ``prim_path`` must be unique among
                the lease's cameras.
            max_fps: Maximum native rate of the camera's frames [Hz], authored on its prim in the
                exported Stage. ``None`` authors no rate, so the camera renders with every frame of
                its render product.

        Raises:
            RuntimeError: If the lease is released or already bound to a Stage, or ``prim_path`` is
                already configured.
            ValueError: If ``camera_cfg`` is ``None``, or ``max_fps`` is not finite and positive.
        """
        self._validate_active()
        if self._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("Auxiliary camera configuration must precede Stage materialization")
        self._controller._renderer._panels.configure(camera_cfg, max_fps=max_fps)

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
       refreshing camera panel poses.
    3. Reject incomplete camera panel samples before reserving an ordinal.
    4. Reserve an ordinal, write every scene component at it, and seal it.
    5. If the write or seal fails, poison the pair, abandon the ordinal and re-raise.

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
            prepare: Called once a publication is required, after the marker callbacks and before camera
                panel samples are validated, for example to refresh their poses.
            seal: Seals the written ordinal. Defaults to ``ordinals.publish``.
            force: Publish even when the state needs no publication, for example for an
                independently changed XR anchor.

        Returns:
            The sealed ordinal, or ``None`` when the state needs no publication.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, a camera panel lacks a live pose sample, or the write or seal failed. A
                failed write or seal poisons the pair, abandons the ordinal and closes the publisher;
                a rejected sample changes nothing.
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
            controller._renderer._panels.validate_samples()
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
        self._renderer._panels.clear()
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
        renderer._reject_ordinary_ovrtx_renderers()
        sim = renderer._sim
        scene = sim.get_or_create_backend(OvstageBackendCfg(scene_key=renderer.cfg))
        renderer.backend = sim.get_or_create_backend(
            _OvxrEngineCfg(
                scene_key=renderer.cfg,
                log_file_path=renderer.cfg.log_file_path,
                log_level=renderer.cfg.log_level,
                device=sim.device,
                active_cuda_gpus=renderer.cfg.active_cuda_gpus,
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
        # Keep the lease active until every retained producer/transfer completes, so
        # environment teardown keeps the pair after a failed drain.
        if self._renderer._geometry_publication is not None:
            self._renderer._geometry_publication.close()
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
        self._renderer._panels.clear()

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

    def product_paths(self, lease: OvxrRenderLease) -> tuple[str, ...]:
        """Return products in application selection order, independently of sensor initialization order."""
        self.validate_live(lease, lease.ovrtx_renderer, lease.ovstage, lease._generation)
        return self._renderer._panels.product_paths()

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
        """Materialize a deferred Stage, with any selected cameras, after render-thread ownership transfer."""
        self.require_owner_thread("materialize a deferred Stage")
        # Unclaimed, a fallback prepare_stage would only defer again; reject even if nothing is deferred yet.
        if not self._transfer_claimed:
            raise RuntimeError("Deferred Stage materialization requires a claimed owner transfer")
        if not self.stage_deferred:
            return False
        if self.active is not lease or lease._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("Deferred Stage is not eligible for materialization")
        if not self._renderer._panels.prepared():
            raise RuntimeError("Deferred auxiliary camera Stage is not eligible for materialization")
        self._renderer.prepare_stage(stage, num_envs)
        return True

    def bind_stage(self) -> None:
        """Bind the prepared native pair after renderer initialization succeeds."""
        lease = self.active
        if lease is not None:
            self._renderer._panels.materialize()
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
            self._renderer._panels.publish_poses()


class _OvxrPanelCameras:
    """Camera panels selected for the OVXR session, owned by ``OvxrRenderer``.

    They match the selected camera sensors, create their render products once the session's render
    worker prepares the Stage, author native frame rates on the export, and hold each camera's latest
    pose sample until scene publication writes it. Product creation and pose writes use the borrowed
    native pair, so they run under the lease, which itself handles only ownership, retirement and
    publication ordering.
    """

    def __init__(self, renderer: OvxrRenderer):
        self._renderer = renderer
        self.cameras: list[_AuxiliaryCamera] = []

    def clear(self) -> None:
        self.cameras = []

    def configure(self, camera_cfg: Any, *, max_fps: float | None = None) -> None:
        """Select an exact camera for the current lease; see :meth:`OvxrRenderLease.configure_auxiliary_camera`."""
        if camera_cfg is None:
            raise ValueError("Auxiliary camera configuration cannot be None")
        if any(camera.cfg.prim_path == camera_cfg.prim_path for camera in self.cameras):
            raise RuntimeError("This auxiliary camera is already configured")
        if max_fps is not None and (not math.isfinite(max_fps) or max_fps <= 0.0):
            raise ValueError("Auxiliary camera max_fps must be finite and positive [Hz]")
        self.cameras.append(_AuxiliaryCamera(camera_cfg, max_fps))

    def for_spec(self, spec: CameraRenderSpec) -> _AuxiliaryCamera | None:
        """Return the selected camera that ``spec`` describes, or ``None`` for an ordinary camera."""
        if not self.cameras or not spec.camera_prim_paths:
            return None
        for camera in self.cameras:
            expected = camera.cfg
            pattern = str(expected.prim_path)
            if "{ENV_REGEX_NS}" in pattern:
                roots = self._renderer.scene.clone_env_paths
                if len(roots) != 1:
                    raise RuntimeError("An external camera requires exactly one cloned environment root")
                pattern = pattern.replace("{ENV_REGEX_NS}", re.escape(roots[0]))
            if (
                int(spec.cfg.width) == int(expected.width)
                and int(spec.cfg.height) == int(expected.height)
                and tuple(spec.cfg.data_types) == tuple(expected.data_types)
                and re.fullmatch(f"(?:{pattern})(?:/camera)?", str(spec.camera_prim_paths[0])) is not None
            ):
                # Only a matched camera is external; other specifications are not panels and are rejected.
                if spec.num_instances != 1 or spec.view_count != 1 or len(spec.camera_prim_paths) != 1:
                    raise RuntimeError("An external camera requires exactly one camera view")
                return camera
        return None

    def accept(self, spec: CameraRenderSpec) -> _AuxiliaryCamera | None:
        """Return the selected camera for ``spec`` after recording its specification."""
        camera = self.for_spec(spec)
        if camera is None:
            return None
        if camera.spec is None:
            camera.spec = spec
        elif camera.spec != spec:
            raise RuntimeError("Auxiliary camera specification changed after authorization")
        return camera

    def prepared(self) -> bool:
        """Whether every selected camera has its specification and render data."""
        return all(camera.spec is not None and camera.render_data is not None for camera in self.cameras)

    def product_paths(self) -> tuple[str, ...]:
        """Return products in application selection order, independently of sensor initialization order."""
        paths = tuple(camera.product_path for camera in self.cameras)
        if any(not isinstance(path, str) or not path.startswith("/") for path in paths) or len(set(paths)) != len(
            paths
        ):
            raise RuntimeError("Auxiliary camera did not materialize exactly one RenderProduct per selection")
        return paths

    def populate(self, usd: str) -> None:
        """Populate the renderer's scene with the selected cameras' native frame rates.

        Without a rate-limited camera this is the stage holder's own population. Otherwise the
        rendering population also selects the cameras' rate properties, which it would not copy,
        and then clones and seals the scene as the stage holder does.
        """
        usd = self._author_schedules(usd)
        scene = self._renderer.scene
        scheduled_paths = [camera.spec.camera_prim_paths[0] for camera in self.cameras if camera.max_fps is not None]
        if not scheduled_paths:
            scene.populate(usd)
            return
        population = ovstage.population
        description = population.Desc(
            domains=scene.cfg.population_domains,
            selectors=[
                population.Selector(
                    population.PrimPredicate.has_path(*scheduled_paths),
                    population.PropertyPredicate.has_name("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs"),
                )
            ],
        )
        # Ordinal 0 is the empty state; population and cloning form the first committed write.
        population.open_usd_from_string_with_desc(
            scene.stage,
            usd,
            ordinal=1,
            time=population.DEFAULT_TIME,
            descs=(population.Desc(domains=scene.cfg.population_domains), description),
        )
        ovstage_replicate(
            scene.stage, scene.paths, scene.clone_copies, scene.clone_env_paths, scene.clone_positions, ordinal=1
        )
        scene.ordinal = 1
        scene.commit()

    def materialize(self) -> None:
        """Register the selected cameras' products in the prepared Stage."""
        for camera in self.cameras:
            if camera.spec is None or camera.render_data is None:
                raise RuntimeError("Selected auxiliary camera was not prepared")
            self._renderer._register_panel(camera.spec, camera.render_data)
            camera.product_path = camera.render_data.render_product_path
        if self.cameras:
            self._renderer.scene.commit()

    def queue_update(self, render_data, positions, orientations, intrinsics) -> bool:
        """Retain the latest pose sample of a selected camera until publication."""
        for camera in self.cameras:
            if camera.render_data is render_data:
                camera.state = (render_data, positions, orientations, intrinsics)
                return True
        return False

    def validate_samples(self) -> None:
        """Validate the whole camera batch before any scene component starts writing."""
        for camera in self.cameras:
            if camera.state is None or camera.render_data is None or camera.render_data.camera_xform_binding is None:
                raise RuntimeError("Auxiliary camera has no live world-pose sample")

    def publish_poses(self) -> None:
        """Write each selected camera's latest pose at the publication ordinal."""
        for camera in self.cameras:
            if camera.state is None or camera.render_data.camera_xform_binding is None:
                raise RuntimeError("Auxiliary camera has no live world-pose sample")
            self._renderer._publish_camera_pose(*camera.state)

    def _author_schedules(self, usd: str) -> str:
        """Author selected native camera rates on the private export, preserving the application stage.

        The rates are authored on the export's layer, so composition arcs that the export
        preserved are not flattened. An export without rate-limited cameras is returned unchanged.
        """
        scheduled = [camera for camera in self.cameras if camera.max_fps is not None]
        if not scheduled:
            return usd
        layer = Sdf.Layer.CreateAnonymous("borrowed-renderer.usda")
        if not layer.ImportFromString(usd):
            raise RuntimeError("Cannot parse the exported leased Stage")
        stage = None
        for camera in scheduled:
            if camera.spec is None:
                raise RuntimeError("Native-rate auxiliary camera was not prepared")
            path = Sdf.Path(camera.spec.camera_prim_paths[0])
            if layer.GetPrimAtPath(path) is None:
                # A camera inside a preserved reference has no spec of its own; check the composed export.
                stage = stage or Usd.Stage.Open(layer)
                if not stage.GetPrimAtPath(path).IsValid():
                    raise RuntimeError("Native-rate auxiliary camera is absent from the exported Stage")
            prim = Sdf.CreatePrimInLayer(layer, path)
            for name, value_type, value in (
                ("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2, Gf.Vec2d(camera.max_fps, 1.0)),
                ("omni:sensor:Core:partialOutputs", Sdf.ValueTypeNames.Bool, False),
            ):
                attribute = prim.attributes.get(name) or Sdf.AttributeSpec(prim, name, value_type)
                attribute.default = value
        return layer.ExportToString()


@configclass
class OvxrRendererCfg(RendererCfg):
    """Configuration of the renderer that publishes the simulation's scene for an OVXR session.

    The session's :class:`OvxrRenderer` creates its own OVStage and OVRTX engine and lends them to
    Isaac Capture's render worker. Give the session and every camera it shows as a panel the same
    configuration, so they share one ``OvxrRenderer``. Other cameras keep their own renderer
    configurations, such as :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`, which renders on the
    simulation device. OVRTX selects rendering GPUs once per process, so such cameras can only share
    the simulation when :attr:`active_cuda_gpus` is unset or names just the simulation GPU.
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

    active_cuda_gpus: str | None = None
    """Comma-separated CUDA device indices that render the session (e.g. ``"1"`` or ``"0,1"``).

    Defaults to the simulation CUDA device; CPU simulations use native GPU selection.
    The first OVXR engine fixes this selection for the process, even after renderer teardown or
    failed startup. Later OVXR renderers must request the same selection; conflicting requests raise
    :class:`ValueError`. Start a new process to change rendering GPUs. This does not change physics.
    When the set excludes a CUDA simulation GPU, rigid transforms and deformable points produced on
    the simulation GPU are copied to the first listed GPU before OVStage consumes them. Other inputs,
    such as camera poses, are read from the simulation GPU, and camera panel products use automatic
    device assignment.

    Other OVRTX renderers render on the simulation device. A selection other than the simulation GPU
    alone is therefore rejected when an ordinary OVRTX renderer or engine is registered in the same
    simulation, both when the session creates its engine and when the Stage is prepared.
    """

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Return the outputs a camera panel's render product may provide, with the OVRTX renderer's layouts.

        The OVXR session presents a panel's colour output; ``depth`` is also accepted, for example to
        check a panel's rendered scene directly. The renderer itself writes none of these buffers.
        """
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
        }

    def __post_init__(self) -> None:
        # Reject malformed GPU selections before any environment or native resource exists.
        if self.active_cuda_gpus is not None:
            parse_active_cuda_gpus(self.active_cuda_gpus)


@final
class OvxrRenderer(BaseRenderer):
    """Renderer that publishes the simulation's scene to the OVStage that the OVXR session renders.

    It works only under the session's :class:`OvxrRenderLease`: the lease creates the renderer's own
    OVStage and OVRTX engine, the session's render worker prepares the Stage and renders it, and the
    session's :class:`OvxrScenePublisher` writes each complete scene. Camera sensors that the lease
    selected as camera panels get render products in that Stage, which the session renders and
    presents; the renderer produces no camera images and rejects every other camera sensor.

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
        self._camera_render_data: list[_OvxrPanelRenderData] = []
        """Camera panels' render data, released once the lease is released."""
        self._panels = _OvxrPanelCameras(self)
        self._render_lease = _OvxrLeaseController(self)
        self._geometry_publication: _GeometryPublicationBuffers | None = None

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
            ValueError: If :attr:`OvxrRendererCfg.active_cuda_gpus` selects GPUs other than the simulation
                GPU alone while an ordinary OVRTX renderer is registered in the same simulation.
        """
        self._reject_ordinary_ovrtx_renderers()
        if not self._render_lease.prepare_stage(num_envs):
            return
        if self._geometry_publication is None:
            publication_device = self._geometry_publication_device()
            if publication_device is not None:
                self._geometry_publication = _GeometryPublicationBuffers(publication_device)
        self._populate_scene(stage)
        self._render_lease.bind_stage()

    # Split-GPU rendering.

    def _excludes_simulation_gpu(self) -> bool:
        """Whether :attr:`OvxrRendererCfg.active_cuda_gpus` omits the CUDA simulation GPU."""
        return _excludes_simulation_gpu(self.cfg.active_cuda_gpus, self._device)

    def _selects_other_than_simulation_gpu(self) -> bool:
        """Whether an explicit selection differs from the simulation GPU that ordinary OVRTX rendering uses."""
        if self.cfg.active_cuda_gpus is None:
            return False
        ordinary = (self._warp_device.ordinal,) if self._warp_device.is_cuda else ()
        return parse_active_cuda_gpus(self.cfg.active_cuda_gpus) != ordinary

    def _reject_ordinary_ovrtx_renderers(self) -> None:
        """Reject a GPU selection that ordinary OVRTX renderers in the same simulation cannot share.

        Ordinary OVRTX renderers render on the simulation device, and OVRTX selects rendering GPUs once
        per process.
        """
        if not self._selects_other_than_simulation_gpu():
            return
        if self._sim.get_backends(OVRTXBackendCfg) or self._sim.get_backends(OVRTXRendererCfg):
            raise ValueError(
                "An ordinary OVRTX camera cannot share a simulation with an OVXR session that renders on other"
                f" GPUs: active_cuda_gpus={self.cfg.active_cuda_gpus!r}, simulation device {self._device!r}."
                " Remove the OVRTX camera, or render the session on the simulation GPU alone."
            )

    def _geometry_publication_device(self) -> str | None:
        """Rendering GPU that receives completed geometry copies, or ``None`` to publish in place.

        When OVStage renders on GPUs that exclude the CUDA simulation GPU, geometry produced on the
        simulation GPU is copied to the first listed rendering GPU before OVStage consumes it.
        """
        if not self._excludes_simulation_gpu():
            return None
        return f"cuda:{parse_active_cuda_gpus(self.cfg.active_cuda_gpus)[0]}"

    # The session's scene publisher writes every scene; ordinary scene updates write nothing.

    def update_transforms(self) -> None:
        """Do nothing: the session's scene publisher writes transforms with each complete scene."""

    def update_geometries(self) -> None:
        """Do nothing: the session's scene publisher writes geometry with each complete scene."""

    # Only the session's camera panels are accepted; this renderer produces no camera images.

    def create_render_data(self, spec: CameraRenderSpec) -> _OvxrPanelRenderData:
        """Reserve a camera panel that the lease selected; its product is registered when the Stage is prepared.

        Raises:
            ValueError: If the camera does not use the simulation device.
            RuntimeError: If the camera is not a selected camera panel, or its render data already exists.
        """
        if str(wp.get_device(spec.device)) != self._device:
            raise ValueError("Camera panels must use the simulation device.")
        camera = self._panels.accept(spec)
        if camera is None:
            raise RuntimeError(
                "OvxrRenderer renders only the OVXR session's camera panels; give other cameras their own renderer"
                " configuration"
            )
        if camera.render_data is not None:
            raise RuntimeError("Auxiliary camera render data already exists")
        data = _OvxrPanelRenderData(spec, f"RenderCamera_{self.scene.next_camera_id}", self._device)
        self.scene.next_camera_id += 1
        camera.render_data = data
        self._camera_render_data.append(data)
        return data

    def set_outputs(self, render_data: Any, output_data: dict[str, ProxyArray]) -> None:
        """Do nothing: this renderer writes no camera outputs."""

    def prepare_capture(self, render_data: Any, camera_data: Any, frame: ProxyArray) -> None:
        """Do nothing: this renderer captures no camera images."""

    def reset(self, render_data: Any, env_ids: Sequence[int] | None = None) -> None:
        """Do nothing: this renderer keeps no per-camera image state."""

    def update_camera(
        self, render_data: Any, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray
    ) -> None:
        """Keep a camera panel's latest pose sample; the next scene publication writes it.

        Raises:
            RuntimeError: If ``render_data`` is not a selected camera panel's.
        """
        if not self._panels.queue_update(render_data, positions, orientations, intrinsics):
            raise RuntimeError("OvxrRenderer has no such camera panel to update")

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

    def cleanup(self, render_data: _OvxrPanelRenderData | None) -> None:
        """Release a camera panel's render data once the lease is released.

        Releasing a product removes it from the borrowed Stage, so a panel keeps its resources until the
        lease is released; :meth:`close` releases them afterwards.
        """
        if render_data is None:
            return
        self._render_lease.reject_ordinary_operation("release camera render data")
        if render_data not in self._camera_render_data:
            return
        try:
            render_data.cleanup()
        finally:
            self._camera_render_data.remove(render_data)

    def close(self) -> None:
        """Release panels, queries and markers once the lease is released; the registry owns the engine and Stage."""
        self._render_lease.reject_ordinary_operation("close the renderer")
        # A failed drain must preserve publication storage and its scene bindings for retry.
        if self._geometry_publication is not None:
            self._geometry_publication.close()
        try:
            with contextlib.ExitStack() as resources:
                resources.callback(self._bindings.close)
                for marker in self.markers.values():
                    resources.callback(marker.close)
                for render_data in tuple(self._camera_render_data):
                    resources.callback(self.cleanup, render_data)
        finally:
            self._object_xform_binding = self._geometry_points_binding = None
            self._object_scales = None
            self._geometry_paths = []
            self._geometry_timestamp = -1
            self._transforms_timestamp = -1
            self._initialized_scene = False
            self._visual_material_writer_ref = None
            self.markers.clear()
            self._geometry_publication = None
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
        self._panels.populate(usd)
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

    def _register_panel(self, spec: CameraRenderSpec, render_data: _OvxrPanelRenderData) -> None:
        """Add a camera panel's render product and pose binding to the populated Stage on the render owner."""
        scene = self.scene
        camera_paths = render_data.camera_paths
        env_partitions = {Sdf.Path(path): f"env_{i}" for i, path in enumerate(scene.clone_env_paths)}
        source_root = next((root for root in Sdf.Path(camera_paths[0]).GetPrefixes() if root in env_partitions), None)
        camera_partitions = []
        if source_root is not None:
            camera_partitions = [
                next(env_partitions[root] for root in Sdf.Path(path).GetPrefixes() if root in env_partitions)
                for path in camera_paths
            ]
        scope, product_path = render_data.render_scope_name, render_data.render_product_path
        usd = self._panel_product_usd(spec, render_data)
        try:
            reference = ovstage.population.add_usd_reference_from_string(scene.stage, usd, f"/{scope}")
            render_data.resources.callback(_remove_usd_reference, scene, reference)
            ovstage.population.apply_usd_changes(scene.stage, ordinal=scene.ordinal)
            with scene.query([product_path]) as query:
                # USD references drop external camera targets; author the relationship in Fabric.
                scene.stage.write_attribute(
                    query, "camera", ordinal=scene.ordinal,
                    tensors=np.array([scene.paths.intern_path(p).raw_id for p in camera_paths], dtype=np.uint64),
                    is_array=True, semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
                ).wait()  # fmt: skip
            render_data.camera_xform_binding = render_data.resources.enter_context(scene.query(camera_paths))
            scene.stage.write_attribute(
                render_data.camera_xform_binding,
                "omni:resetXformStack",
                ordinal=scene.ordinal,
                tensors=np.full(len(camera_paths), True, dtype=np.bool_),
                is_array=False,
            ).wait()
            if camera_partitions:
                scene.stage.write_attribute(
                    render_data.camera_xform_binding, "omni:scenePartition", ordinal=scene.ordinal,
                    tensors=np.array(
                        [scene.paths.intern_token(partition).raw_id for partition in camera_partitions], dtype=np.uint64
                    ),
                    is_array=False, semantic=ovstage.AttributeSemantic.TOKEN_ID,
                ).wait()  # fmt: skip
        except Exception:
            render_data.cleanup()
            raise
        self.backend.render_products.add(product_path)
        render_data.resources.callback(self.backend.render_products.discard, product_path)

    def _panel_product_usd(self, spec: CameraRenderSpec, render_data: _OvxrPanelRenderData) -> str:
        """Build a panel's render product, pinned to the simulation GPU when that GPU renders.

        A CPU simulation, or rendering GPUs that exclude the simulation GPU, leave the product's GPU choice
        to OVRTX, which would otherwise skip a product request for a GPU outside the rendering set on every
        frame.
        """
        device_id = self._warp_device.ordinal if self._warp_device.is_cuda else 0
        usd = build_render_product_as_string(spec, render_data, device_id=device_id)
        if self._warp_device.is_cuda and not self._excludes_simulation_gpu():
            return usd
        # The device index above is removed with the product's only device pin.
        usd, removed = re.subn(r"^[ \t]*uint\[\] deviceIds = \[[^\]\n]*\]\n", "", usd, flags=re.MULTILINE)
        if removed != 1:
            raise RuntimeError(f"Expected one deviceIds line in the camera panel's render product; found {removed}")
        return usd

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
        matrices = self._transforms.matrices
        if self._geometry_publication is not None:
            # Rendering excludes the simulation GPU: publish a completed copy on the first rendering GPU.
            self._geometry_publication.write_transforms(
                self.scene.stage, binding, matrices, self.scene.ordinal, self._producer_stream()
            )
        else:
            # The producing CUDA stream orders access to the borrowed buffer.
            self.scene.stage.write_attribute(
                binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_warp(matrices),
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
        if self._geometry_publication is not None:
            # Rendering excludes the simulation GPU: publish completed copies on the first rendering GPU.
            self._geometry_publication.write(
                self.scene.stage, binding, self._geometry_paths, points, self.scene.ordinal, self._producer_stream()
            )
        else:
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

    def _publish_camera_pose(
        self, render_data: _OvxrPanelRenderData, positions: ProxyArray, orientations: ProxyArray, intrinsics: Any
    ) -> None:
        """Write a camera panel's world pose; panels keep their authored intrinsics."""
        binding = render_data.camera_xform_binding
        if binding is None:
            return
        matrices = render_data.camera_transforms
        wp.launch(_camera_transforms_kernel, len(matrices), [positions, orientations, matrices], device=self._device)
        # On a CUDA device the stream orders the kernel before the write; a CPU device writes host memory.
        self.scene.stage.write_attribute(
            binding,
            "omni:xform",
            ordinal=self.scene.ordinal,
            tensors=xform_tensor_from_warp(matrices),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=self._cuda_stream(),
        ).wait()

    def _producer_stream(self) -> wp.Stream | None:
        """Return the simulation device's current Warp stream, which produced the published data; ``None`` on a CPU."""
        return self._warp_device.stream if self._warp_device.is_cuda else None

    def _cuda_stream(self) -> int | None:
        """Return the simulation device's CUDA stream handle, or ``None`` to write host memory on a CPU device."""
        stream = self._producer_stream()
        return (stream.cuda_stream or 1) if stream is not None else None
