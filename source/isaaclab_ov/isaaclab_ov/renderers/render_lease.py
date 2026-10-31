# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Exclusive borrowing, thread handoff and publication ordering for OVRTX/OVStage."""

from __future__ import annotations

import contextlib
import math
import os
import re
import threading
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import ovstage
import torch

from pxr import Gf, Sdf, Usd

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

if TYPE_CHECKING:
    from ovrtx import Renderer
    from ovstage import Stage as OvStage

    from .ovrtx_renderer import OVRTXCameraRenderData, OVRTXRenderer
_NO_PHYSICS_STEP_ID = object()


class _OvLibrariesRenderLeaseState(Enum):
    """Internal lifecycle state for a renderer/scene pair lease."""

    UNBOUND = auto()
    BOUND = auto()
    RELEASED = auto()


@dataclass
class _AuxiliaryCamera:
    cfg: Any
    max_fps: float | None
    spec: CameraRenderSpec | None = None
    render_data: OVRTXCameraRenderData | None = None
    state: tuple[Any, ...] | None = None
    product_path: str | None = None


class OvLibrariesRenderLease:
    """Exclusive non-owning lease over one exact attached OVRTX/OVStage pair.

    The lease is acquired before stage preparation. Stage preparation then
    materializes the renderer-owned scene without requiring a camera sensor and
    binds this object to the exact live Python ``Renderer`` and ``Stage``
    instances. External consumers must release the lease before the renderer is
    closed; they must never detach or destroy either borrowed object.
    """

    def __init__(self) -> None:
        raise RuntimeError("OvLibrariesRenderLease must be acquired from OVRTXRenderer")

    def __enter__(self) -> OvLibrariesRenderLease:
        self._validate_active()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    @property
    def auxiliary_render_product_paths(self) -> tuple[str, ...]:
        self.validate_live()
        return cast("RenderLeaseController", self._controller).product_paths(self)

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

    def configure_auxiliary_camera(self, camera_cfg: Any, *, max_fps: float | None = None) -> None:
        """Authorize an exact camera before Stage materialization."""
        self._validate_active()
        if self._state is not _OvLibrariesRenderLeaseState.UNBOUND:
            raise RuntimeError("Auxiliary camera configuration must precede Stage materialization")
        if camera_cfg is None:
            raise ValueError("Auxiliary camera configuration cannot be None")
        if any(camera.cfg.prim_path == camera_cfg.prim_path for camera in self._auxiliary_cameras):
            raise RuntimeError("This auxiliary camera is already configured")
        if max_fps is not None and (not math.isfinite(max_fps) or max_fps <= 0.0):
            raise ValueError("Auxiliary camera max_fps must be finite and positive [Hz]")
        self._auxiliary_cameras.append(_AuxiliaryCamera(camera_cfg, max_fps))

    def offer_owner_transfer(self) -> object:
        """Freeze one unbound lease and return its one-shot owner capability."""
        owner = cast("RenderLeaseController", self._controller)
        return owner.offer_owner_transfer(self)

    def claim_owner_transfer(self, capability: object) -> None:
        cast("RenderLeaseController", self._controller).claim_owner_transfer(self, capability)

    def cancel_owner_transfer(self, capability: object) -> None:
        cast("RenderLeaseController", self._controller).cancel_owner_transfer(self, capability)

    def capture_scene_publisher(self) -> OvLibrariesScenePublisher:
        """Capture the application-thread scene publisher before render ownership transfers."""
        return cast("RenderLeaseController", self._controller).capture_scene_publisher(self)

    def materialize_deferred_auxiliary_camera_stage(self, stage: Any, num_envs: int) -> bool:
        """Complete the selected camera's deferred Stage on the claimed owner."""
        return cast("RenderLeaseController", self._controller).materialize_deferred_stage(self, stage, num_envs)

    @contextlib.contextmanager
    def ovstage_owner_lane(self) -> Iterator[None]:
        """Guard one non-reentrant operation on the render-owner thread.

        The application's publication guard must separately exclude concurrent
        producer writes while this operation uses the borrowed stage.
        """
        self.validate_live()
        owner = cast("RenderLeaseController", self._controller)
        with owner.owner_lane("external renderer operation", lease=self):
            yield

    def configure_external_camera_scene_partitions(
        self, environment_root_path: str, camera_paths: Sequence[str]
    ) -> str:
        """Assign the live environment partition token to external cameras.

        This method must be called inside :meth:`ovstage_owner_lane` after the
        external camera prims exist. The environment root's resolved token is
        authoritative; empty and token-zero values fail closed.
        """
        self.validate_live()
        owner = cast("RenderLeaseController", self._controller)
        return owner.configure_camera_partitions(self, environment_root_path, camera_paths)

    def validate_external_scene_partitions(self, ordinal: int) -> dict[str, str]:
        """Read back and prove environment/camera partition equality."""
        self.validate_live()
        owner = cast("RenderLeaseController", self._controller)
        return owner.validate_partitions(self, ordinal)

    def release(self) -> None:
        """Release the borrowing intent without destroying either pair member."""
        if self._state is _OvLibrariesRenderLeaseState.RELEASED:
            return
        owner = cast("RenderLeaseController", self._controller)
        owner.release(self)
        self._renderer = None
        self._stage = None
        self._state = _OvLibrariesRenderLeaseState.RELEASED

    def validate_live(self) -> None:
        """Verify that this lease still refers to its live renderer and stage."""
        self._validate_active()
        if self._state is not _OvLibrariesRenderLeaseState.BOUND:
            raise RuntimeError("OvLibrariesRenderLease is not bound to a prepared Stage")
        owner = cast("RenderLeaseController", self._controller)
        owner.validate_live(
            self,
            cast("Renderer", self._renderer),
            cast("OvStage", self._stage),
            self._generation,
        )

    @classmethod
    def _create_unbound(cls, owner: RenderLeaseController) -> OvLibrariesRenderLease:
        lease = object.__new__(cls)
        lease._controller = owner
        lease._state = _OvLibrariesRenderLeaseState.UNBOUND
        lease._renderer = None
        lease._stage = None
        lease._generation = 0
        lease._required_environment_scene_partitions = ()
        lease._external_camera_scene_partitions = ()
        lease._external_camera_scene_partitions_pending = False
        lease._auxiliary_cameras = []
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
        owner = cast("RenderLeaseController", self._controller)
        owner.validate_active(self)


class OvLibrariesScenePublisher:
    """Application-thread writer for complete provider-derived OVStage batches."""

    def __init__(self, owner: RenderLeaseController, lease: OvLibrariesRenderLease) -> None:
        self._controller = owner
        self._lease = lease
        self._application_thread = threading.current_thread()
        self._lock = threading.Lock()
        self._closed = False

    def publication_required(self, physics_step_id: Hashable) -> bool:
        self._require_application_thread()
        return self._controller.producer_publication_required(self._lease, physics_step_id)

    def write_complete_scene(self, ordinal: int, physics_step_id: Hashable) -> None:
        """Write, but do not publish, one complete batch at ``ordinal``."""
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            self._controller.write_producer_scene(self._lease, ordinal, physics_step_id)

    def close(self) -> None:
        """Stop scene publication and wait for an in-flight batch to finish."""
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
        self._lane_lease: OvLibrariesRenderLease | None = None
        self._last_step_id: Hashable = _NO_PHYSICS_STEP_ID
        self._poisoned = False
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
    def owner_lane(self, operation: str, *, lease: OvLibrariesRenderLease | None = None) -> Iterator[None]:
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
        self._lane_lease = lease
        try:
            yield
        finally:
            self._lane_lease = None
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
    ) -> None:
        """Validate retained Python-object identity for the live attached pair."""
        self.validate_active(lease)
        if self._poisoned:
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
        self.active = None
        if self._renderer.backend is not None:
            self._renderer.backend._render_lease_owner = None
        if self._renderer.scene is not None:
            self._renderer.scene._render_lease_owner = None

    def reject_ordinary_operation(self, operation: str) -> None:
        """Reject ordinary OVRTX operations while an external lease is active."""
        if self.active is not None or (
            self._renderer.backend is not None and self._renderer.backend._render_lease_owner is not None
        ):
            raise RuntimeError(f"Cannot {operation} while an OvLibrariesRenderLease is active")

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
            lease._external_camera_scene_partitions_pending
            or self._last_step_id != physics_step_id
            or (writer is not None and writer.has_pending_updates)
        )

    def write_producer_scene(self, lease: OvLibrariesRenderLease, ordinal: int, physics_step_id: Hashable) -> None:
        """Write a complete scene batch without advancing its reserved ordinal."""
        if ordinal <= 0:
            raise ValueError("OVStage publication ordinal must be nonzero")
        if not self.producer_publication_required(lease, physics_step_id):
            raise RuntimeError("OVStage scene publication is not required for this complete state")
        publish_camera_partitions = lease._external_camera_scene_partitions_pending
        try:
            self._write_scene_components(ordinal, partition_lease=lease if publish_camera_partitions else None)
        except Exception as exc:
            self._poisoned = True
            raise RuntimeError("Producer scene publication failed; recreate the exact pair") from exc
        self._last_step_id = physics_step_id
        if publish_camera_partitions:
            lease._external_camera_scene_partitions_pending = False

    def camera_for_spec(self, spec: CameraRenderSpec) -> _AuxiliaryCamera | None:
        lease = self.active
        if lease is None:
            return None
        for camera in lease._auxiliary_cameras:
            expected = camera.cfg
            if spec.num_instances != 1 or spec.view_count != 1 or len(spec.camera_prim_paths) != 1:
                raise RuntimeError("An external camera requires exactly one camera view")
            pattern = str(expected.prim_path).replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+")
            if (
                int(spec.cfg.width) == int(expected.width)
                and int(spec.cfg.height) == int(expected.height)
                and tuple(spec.cfg.data_types) == tuple(expected.data_types)
                and re.fullmatch(f"(?:{pattern})(?:/camera)?", str(spec.camera_prim_paths[0])) is not None
            ):
                return camera
        return None

    def accept_camera_spec(self, spec: CameraRenderSpec) -> bool:
        camera = self.camera_for_spec(spec)
        if camera is None:
            return False
        if camera.spec is None:
            camera.spec = spec
        elif camera.spec != spec:
            raise RuntimeError("Auxiliary camera specification changed after authorization")
        return True

    def materialize_deferred_stage(self, lease: OvLibrariesRenderLease, stage: Any, num_envs: int) -> bool:
        """Materialize the selected camera after render-thread ownership transfer."""
        self.require_owner_thread("materialize the auxiliary camera Stage")
        if not self.stage_deferred:
            return False
        if (
            self.active is not lease
            or lease._state is not _OvLibrariesRenderLeaseState.UNBOUND
            or not self._transfer_claimed
            or not lease._auxiliary_cameras
            or any(camera.spec is None or camera.render_data is None for camera in lease._auxiliary_cameras)
        ):
            raise RuntimeError("Deferred auxiliary camera Stage is not eligible for materialization")
        self._renderer.prepare_stage(stage, num_envs)
        return True

    def product_paths(self, lease: OvLibrariesRenderLease) -> tuple[str, ...]:
        """Return products in application selection order, independently of sensor initialization order."""
        self.validate_live(lease, lease.ovrtx_renderer, lease.ovstage, lease._generation)
        paths = tuple(camera.product_path for camera in lease._auxiliary_cameras)
        if any(not isinstance(path, str) or not path.startswith("/") for path in paths) or len(set(paths)) != len(
            paths
        ):
            raise RuntimeError("Auxiliary camera did not materialize exactly one RenderProduct per selection")
        return paths

    def _write_scene_components(self, ordinal: int, *, partition_lease: OvLibrariesRenderLease | None) -> None:
        """Write selected scene components; the caller owns validation and commit."""
        self._renderer.scene.ordinal = ordinal
        with self._renderer._publish_visual_materials():
            self._renderer._update_transforms()
            self._renderer._update_geometries()
            self._update_cameras()
            if partition_lease is not None:
                self._write_camera_partitions(partition_lease, ordinal)

    def configure_camera_partitions(
        self,
        lease: OvLibrariesRenderLease,
        environment_root_path: str,
        camera_paths: Sequence[str],
    ) -> str:
        """Read the authoritative environment token and arm matching camera writes."""
        self.validate_active(lease)
        lease.validate_live()
        if self._lane_lease is not lease:
            raise RuntimeError("configure_external_camera_scene_partitions() needs the active lease owner lane")
        if not environment_root_path or not camera_paths or any(not path for path in camera_paths):
            raise ValueError("One environment root and at least one non-empty external camera path are required")

        floor = self._next_ordinal() - 1
        attribute = "primvars:omni:scenePartition"
        actual = self._read_partition_tokens([environment_root_path], (attribute,), floor)
        token = actual.get((environment_root_path, attribute), "")
        if token in {"", "0"}:
            raise RuntimeError(
                f"Environment root {environment_root_path!r} resolved to invalid scene-partition token {token!r}"
            )

        lease._required_environment_scene_partitions = ((environment_root_path, token),)
        lease._external_camera_scene_partitions = tuple((str(path), token) for path in camera_paths)
        lease._external_camera_scene_partitions_pending = True
        return token

    def _write_camera_partitions(self, lease: OvLibrariesRenderLease, ordinal: int) -> None:
        """Write the external camera tokens inside the complete scene batch."""
        if self._renderer.scene.stage is None or self._renderer.scene.paths is None:
            raise RuntimeError("Cannot write external camera partitions before Stage materialization")
        camera_scene_partitions = lease._external_camera_scene_partitions
        if not camera_scene_partitions:
            raise RuntimeError("No external camera scene partitions are configured")
        camera_paths = [path for path, _token in camera_scene_partitions]
        token_ids = np.array(
            [self._renderer.scene.paths.intern_token(token) for _path, token in camera_scene_partitions],
            dtype=np.uint64,
        )
        camera_paths_list = self._renderer.scene.paths.create_path_list_from_strings(camera_paths)
        camera_query = None
        try:
            camera_query = self._renderer.scene.stage.query_from_path_list(camera_paths_list)
            if camera_query is None:
                raise RuntimeError("External camera scene-partition query is None")
            self._renderer.scene.stage.write_attribute(
                camera_query,
                "omni:scenePartition",
                ordinal=ordinal,
                tensors=token_ids,
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ).wait()
        finally:
            if camera_query is not None:
                self._renderer.scene.stage.release_query(camera_query).wait()
            self._renderer.scene.paths.destroy_path_list(camera_paths_list)

    def _read_partition_tokens(
        self, paths: list[str], attributes: tuple[str, ...], ordinal: int
    ) -> dict[tuple[str, str], str]:
        """Read resolved partition tokens from the exact renderer-owned Stage."""
        if self._renderer.scene.stage is None or self._renderer.scene.paths is None:
            raise RuntimeError("Cannot read scene partitions before Stage materialization")
        path_list = self._renderer.scene.paths.create_path_list_from_strings(paths)
        attribute_tokens = [self._renderer.scene.paths.intern_token(attribute) for attribute in attributes]
        resolved: dict[tuple[str, str], str] = {}
        try:
            with self._renderer.scene.stage.query_from_path_list(path_list) as query:
                query.wait()
                with self._renderer.scene.stage.read_attributes(
                    query, attribute_tokens, ovstage.OrdinalRange.latest(ordinal)
                ) as read:
                    read.wait()
                    for group in read.groups():
                        try:
                            if group.is_delete or group.is_array or group.tensor_count != 1:
                                continue
                            attribute = self._renderer.scene.paths.token_to_string(group.attribute)
                            if attribute not in attributes:
                                continue
                            group_paths = self._renderer.scene.paths.get_path_strings(group.prim_list)
                            try:
                                values = np.asarray(group.array(0).copy()).reshape(-1)
                            except NotImplementedError:
                                values = torch.from_dlpack(group.dlpack(0)).detach().cpu().numpy().copy().reshape(-1)
                            logical_count = min(group.prim_count, group.data_count)
                            for local_index in range(logical_count):
                                prim_index = group.prim_index(local_index)
                                data_row_index = group.data_row_index(local_index)
                                if prim_index >= len(group_paths) or data_row_index >= len(values):
                                    raise RuntimeError("OVStage scene-partition read escaped its group-owned rows")
                                token_id = int(values[data_row_index])
                                token = self._renderer.scene.paths.token_to_string(token_id) if token_id else ""
                                key = (group_paths[prim_index], attribute)
                                if key in resolved and resolved[key] != token:
                                    raise RuntimeError(f"OVStage returned conflicting scene-partition rows for {key}")
                                resolved[key] = token
                        finally:
                            self._renderer.scene.stage.release_group(group)
        finally:
            self._renderer.scene.paths.destroy_path_list(path_list)
        return resolved

    def validate_partitions(self, lease: OvLibrariesRenderLease, ordinal: int) -> dict[str, str]:
        """Prove the configured root and external cameras resolve to one valid token."""
        self.validate_active(lease)
        lease.validate_live()
        if self._lane_lease is not lease:
            raise RuntimeError("validate_external_scene_partitions() needs the active lease owner lane")
        expected_rows = [
            *(
                (path, "primvars:omni:scenePartition", token)
                for path, token in lease._required_environment_scene_partitions
            ),
            *((path, "omni:scenePartition", token) for path, token in lease._external_camera_scene_partitions),
        ]
        if len(lease._required_environment_scene_partitions) != 1:
            raise RuntimeError("Exactly one authoritative environment scene partition is required")
        if not lease._external_camera_scene_partitions:
            raise RuntimeError("At least one external camera scene partition is required")
        paths = list(dict.fromkeys(path for path, _attribute, _token in expected_rows))
        attributes = tuple(dict.fromkeys(attribute for _path, attribute, _token in expected_rows))
        actual = self._read_partition_tokens(paths, attributes, ordinal)
        partitions: dict[str, str] = {}
        mismatches = []
        for path, attribute, expected_token in expected_rows:
            actual_token = actual.get((path, attribute), "")
            partitions[path] = actual_token
            if actual_token in {"", "0"} or actual_token != expected_token:
                mismatches.append(f"{path}.{attribute}: expected {expected_token!r}, got {actual_token!r}")
        if mismatches:
            raise RuntimeError("External scene-partition proof failed: " + "; ".join(mismatches))
        return partitions

    def _author_camera_schedules(self, usd: str, lease: OvLibrariesRenderLease) -> str:
        """Author selected native camera rates on the private export, preserving the application stage."""

        layer = Sdf.Layer.CreateAnonymous("borrowed-renderer.usda")
        if not layer.ImportFromString(usd):
            raise RuntimeError("Cannot parse the exported leased Stage")
        stage = Usd.Stage.Open(layer)
        for camera in lease._auxiliary_cameras:
            if camera.max_fps is None:
                continue
            if camera.spec is None:
                raise RuntimeError("Native-rate auxiliary camera was not prepared")
            prim = stage.GetPrimAtPath(camera.spec.camera_prim_paths[0])
            if not prim.IsValid():
                raise RuntimeError("Native-rate auxiliary camera is absent from the exported Stage")
            prim.CreateAttribute("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(camera.max_fps, 1.0))
            prim.CreateAttribute("omni:sensor:Core:partialOutputs", Sdf.ValueTypeNames.Bool).Set(False)
        return stage.ExportToString()

    def _materialize_camera_products(self, lease: OvLibrariesRenderLease) -> None:
        for camera in lease._auxiliary_cameras:
            if camera.spec is None or camera.render_data is None:
                raise RuntimeError("Selected auxiliary camera was not prepared")
            self._renderer._register_camera(camera.spec, camera.render_data)
            camera.product_path = camera.render_data.render_product_path
        if lease._auxiliary_cameras:
            self._renderer.scene.commit()

    def _update_cameras(self) -> None:
        lease = self.active
        if lease is None:
            return
        for camera in lease._auxiliary_cameras:
            if camera.state is None or camera.render_data.camera_xform_binding is None:
                raise RuntimeError("Auxiliary camera has no live world-pose sample")
            self._renderer._update_camera(*camera.state)

    def prepare_stage(self, num_envs: int) -> bool:
        """Validate or defer preparation; ordinary rendering is immediately allowed."""
        lease = self.active
        if lease is not None:
            if lease._auxiliary_cameras and not self._transfer_claimed:
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

    def bind_stage(self) -> None:
        """Bind the prepared native pair after renderer initialization succeeds."""
        lease = self.active
        if lease is not None:
            self._materialize_camera_products(lease)
            self.stage_deferred = False
            self._generation += 1
            lease._bind(self._renderer.backend.renderer, self._renderer.scene.stage, self._generation)

    def populate_stage(self, usd: str) -> None:
        """Populate the borrowed scene with the authorized camera schedule."""
        lease = self.active
        if lease is not None and lease._auxiliary_cameras:
            usd = self._author_camera_schedules(usd, lease)
        scheduled_paths = (
            [camera.spec.camera_prim_paths[0] for camera in lease._auxiliary_cameras if camera.max_fps is not None]
            if lease is not None
            else []
        )
        if scheduled_paths:
            population = ovstage.population
            description = population.Desc(
                domains=self._renderer.scene.cfg.population_domains,
                selectors=[
                    population.Selector(
                        population.PrimPredicate.has_path(*scheduled_paths),
                        population.PropertyPredicate.has_name(
                            "omni:sensor:frameRate", "omni:sensor:Core:partialOutputs"
                        ),
                    )
                ],
            )
            self._renderer.scene.populate(usd, population_desc=description)
        else:
            self._renderer.scene.populate(usd)

    def queue_camera_update(self, render_data, positions, orientations, intrinsics) -> bool:
        """Retain the latest authorized external-camera sample until publication."""
        if self.active is not None:
            for camera in self.active._auxiliary_cameras:
                if camera.render_data is render_data:
                    camera.state = (render_data, positions, orientations, intrinsics)
                    return True
        return False

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
