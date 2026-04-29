# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for Newton physics backend."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

import warp as wp

from pxr import UsdGeom

from isaaclab.physics.base_scene_data_provider import BaseSceneDataProvider
from isaaclab.physics.capabilities import GpuTransformBuffer, UsdFabric
from isaaclab.physics.scene_data_buffers import TransformBufferPool
from isaaclab.physics.scene_data_types import (
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformData,
    TransformFormat,
)

from isaaclab_newton.physics.capabilities import NewtonState

logger = logging.getLogger(__name__)

_ENV_ID_RE = re.compile(r"/World/envs/env_(\d+)")


class _NewtonGpuTransformBufferCap:
    """Adapter exposing :class:`GpuTransformBuffer` for the Newton provider."""

    def __init__(self, provider: NewtonSceneDataProvider) -> None:
        self._provider = provider

    def get_body_transforms(
        self,
        target_format: TransformFormat,
        *,
        env_ids: list[int] | None = None,
        quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
        matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
        double_precision: bool = False,
        stream: wp.Stream | None = None,
        allow_passthrough: bool = True,
        index_map: wp.array | None = None,
    ) -> TransformData | None:
        return self._provider.get_body_transforms(
            target_format,
            env_ids=env_ids,
            quat_convention=quat_convention,
            matrix_layout=matrix_layout,
            double_precision=double_precision,
            stream=stream,
            allow_passthrough=allow_passthrough,
            index_map=index_map,
        )

    def get_source_format(self) -> TransformFormat | None:
        return self._provider.get_source_format()


class _NewtonUsdFabricCap:
    """Adapter exposing :class:`UsdFabric` for the Newton provider.

    Newton does not write USD natively; ``ensure_current`` runs the
    sync-to-USD bridge owned by :class:`NewtonManager`.
    """

    def __init__(self, provider: NewtonSceneDataProvider) -> None:
        self._provider = provider

    def ensure_current(self, stream: wp.Stream | None = None) -> None:
        # Sync handled inside NewtonSceneDataProvider.update() when
        # _needs_usd_sync is True; this is the explicit consumer-facing
        # entry point.
        self._provider._sync_transforms_to_usd_if_needed()


class _NewtonStateCap:
    """Adapter exposing :class:`NewtonState` for the Newton provider."""

    def __init__(self, provider: NewtonSceneDataProvider) -> None:
        self._provider = provider

    def get_state(self) -> Any:
        return self._provider.get_newton_state()

    def get_model(self) -> Any:
        return self._provider.get_newton_model()


class NewtonSceneDataProvider(BaseSceneDataProvider):
    """Scene data provider for Newton physics backend.

    Provides access to Newton model, state, and USD stage for visualizers and renderers.
    Unlike PhysxSceneDataProvider which must build its own Newton model from USD and sync
    PhysX transforms into it, this provider delegates directly to NewtonManager since the
    Newton backend already owns the authoritative model and state.
    """

    def __init__(self, stage, simulation_context) -> None:
        """Initialize the Newton scene data provider.

        Args:
            stage: USD stage handle.
            simulation_context: Active simulation context.
        """
        super().__init__()
        self._simulation_context = simulation_context
        self._stage = stage
        self._metadata = {"physics_backend": "newton"}
        self._num_envs: int | None = None
        self._warned_once: set[str] = set()

        # Determine if usd stage sync is required for selected renderers and visualizers
        requirements = self._simulation_context.get_scene_data_requirements()
        self._needs_usd_sync = bool(requirements.requires_usd_stage)

        # Typed transform API state
        self._transform_buffer_pool = TransformBufferPool()
        self._generation: int = 0

        # Capability registration (ADR-0002).
        self._register_capability(GpuTransformBuffer, _NewtonGpuTransformBufferCap(self))
        self._register_capability(NewtonState, _NewtonStateCap(self))
        if self._needs_usd_sync:
            self._register_capability(UsdFabric, _NewtonUsdFabricCap(self))

    def _warn_once(self, key: str, message: str, *args) -> None:
        """Emit a warning once per unique key.

        Args:
            key: Unique warning key.
            message: Warning message format string.
            *args: Optional formatting arguments.
        """
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        logger.warning(message, *args)

    # ---- Environment discovery ---------------------------------------------------------------

    def get_num_envs(self) -> int:
        """Return discovered environment count.

        Returns:
            Number of environments discovered from stage prim paths.
        """
        if self._num_envs is not None and self._num_envs > 0:
            return self._num_envs
        discovered = self._determine_num_envs_in_scene()
        if discovered > 0:
            self._num_envs = discovered
            return discovered
        return 0

    def _determine_num_envs_in_scene(self) -> int:
        """Infer environment count from ``/World/envs/env_<id>`` prim names.

        Returns:
            Number of environments inferred from the stage.
        """
        if self._stage is None:
            return 0
        max_env_id = -1
        env_name_re = re.compile(r"^env_(\d+)$")
        envs_root = self._stage.GetPrimAtPath("/World/envs")
        if envs_root.IsValid():
            for child in envs_root.GetChildren():
                match = env_name_re.match(child.GetName())
                if match:
                    max_env_id = max(max_env_id, int(match.group(1)))
        return max_env_id + 1 if max_env_id >= 0 else 0

    # ---- Core provider API -------------------------------------------------------------------

    def update(self) -> None:
        """Sync Newton body transforms to USD Fabric when a Kit viewport is active.

        Called at render cadence by :meth:`~isaaclab.sim.SimulationContext.update_scene_data_provider`,
        after forward kinematics have been evaluated.  Only calls
        :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd` when a Kit
        (or other USD-based) visualizer is in use. When both sim and rendering backend
        are Newton (or Rerun), the sync is skipped to avoid unnecessary slowdown.
        """
        self._generation += 1
        self._sync_transforms_to_usd_if_needed()

    def _sync_transforms_to_usd_if_needed(self) -> None:
        """Run the USD-Fabric sync when a USD-reading consumer is registered.

        Idempotent and safe to call multiple times per frame.
        """
        if not self._needs_usd_sync:
            return
        try:
            from isaaclab_newton.physics import NewtonManager

            NewtonManager.sync_transforms_to_usd()
        except Exception:
            pass

    def get_newton_model(self) -> Any | None:
        """Return Newton model from ``NewtonManager``.

        Returns:
            Newton model object, or ``None`` when unavailable.
        """
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_model()

    def get_newton_state(self) -> Any | None:
        """Return Newton state from NewtonManager.

        Returns:
            The current Newton state (state_0) from NewtonManager.
        """
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_state_0()

    def get_model(self) -> Any | None:
        """Alias for :meth:`get_newton_model` for visualizer compatibility.

        Returns:
            Newton model object, or ``None`` when unavailable.
        """
        return self.get_newton_model()

    def get_state(self) -> Any | None:
        """Alias for :meth:`get_newton_state` for visualizer compatibility."""
        return self.get_newton_state()

    def get_usd_stage(self) -> Any | None:
        """Return the USD stage handle.

        Returns:
            USD stage object, or ``None`` when unavailable.
        """
        if self._stage is not None:
            return self._stage
        return getattr(self._simulation_context, "stage", None)

    def get_metadata(self) -> dict[str, Any]:
        """Return provider metadata.

        Returns:
            Metadata dictionary with backend and synchronization information.
        """
        out = dict(self._metadata)
        out["num_envs"] = self.get_num_envs()
        out["needs_usd_sync"] = self._needs_usd_sync
        return out

    def get_transforms(self) -> dict[str, Any] | None:
        """Return body transforms from Newton state.

        Reads body_q from the authoritative Newton state and splits it into
        positions (vec3) and orientations (quaternion xyzw).

        Returns:
            Dictionary containing positions and orientations, or ``None`` when unavailable.
        """
        try:
            import warp as wp

            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None or state.body_q is None:
                return None

            body_q_t = wp.to_torch(state.body_q)
            positions = body_q_t[:, :3]
            orientations = body_q_t[:, 3:7]
            return {"positions": positions, "orientations": orientations}
        except Exception as exc:
            self._warn_once(
                "get-transforms-failed",
                "[NewtonSceneDataProvider] get_transforms() failed: %s",
                exc,
            )
            return None

    def get_velocities(self) -> dict[str, Any] | None:
        """Return body velocities from Newton state.

        Returns:
            Dictionary containing linear and angular velocities, or ``None`` when unavailable.
        """
        try:
            import warp as wp

            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None:
                return None

            body_qd = getattr(state, "body_qd", None)
            if body_qd is None:
                return None

            body_qd_t = wp.to_torch(body_qd)
            linear = body_qd_t[:, :3]
            angular = body_qd_t[:, 3:6]
            return {"linear": linear, "angular": angular, "source": "newton"}
        except Exception as exc:
            self._warn_once(
                "get-velocities-failed",
                "[NewtonSceneDataProvider] get_velocities() failed: %s",
                exc,
            )
            return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Return contact data for Newton provider.

        Returns:
            ``None`` because contacts are not currently implemented in this provider.
        """
        return None

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-environment transforms.

        Returns:
            Dictionary containing camera order, positions, orientations, and environment count,
            or ``None`` when unavailable.
        """
        if self._stage is None:
            return None

        import isaaclab.sim as isaaclab_sim

        env_pattern = re.compile(r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)")
        shared_paths: list[str] = []
        instances: dict[str, list[tuple[int, str]]] = {}
        num_envs = -1

        stage_prims = deque([self._stage.GetPseudoRoot()])
        while stage_prims:
            prim = stage_prims.popleft()
            prim_path = prim.GetPath().pathString

            world_id = 0
            template_path = prim_path
            if match := env_pattern.match(prim_path):
                world_id = int(match.group("id"))
                template_path = match.group("root") + "%d" + match.group("path")
                if world_id > num_envs:
                    num_envs = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            if prim.IsA(UsdGeom.Camera):
                if template_path not in instances:
                    instances[template_path] = []
                instances[template_path].append((world_id, prim_path))
                if template_path not in shared_paths:
                    shared_paths.append(template_path)

            if hasattr(UsdGeom, "TraverseInstanceProxies"):
                child_prims = prim.GetFilteredChildren(UsdGeom.TraverseInstanceProxies())
            else:
                child_prims = prim.GetChildren()
            if child_prims:
                stage_prims.extend(child_prims)

        num_envs += 1
        positions: list[list[list[float] | None]] = []
        orientations: list[list[list[float] | None]] = []

        for template_path in shared_paths:
            per_world_pos: list[list[float] | None] = [None] * num_envs
            per_world_ori: list[list[float] | None] = [None] * num_envs
            for world_id, prim_path in instances.get(template_path, []):
                if world_id < 0 or world_id >= num_envs:
                    continue
                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                pos, ori = isaaclab_sim.resolve_prim_pose(prim)
                per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
                per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            positions.append(per_world_pos)
            orientations.append(per_world_ori)

        return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}

    # ---- Typed transform API -----------------------------------------------------------------

    def get_source_format(self) -> TransformFormat:
        """Return Newton's native transform format.

        Returns:
            :attr:`TransformFormat.TRANSFORM` — Newton stores body poses as
            packed ``wp.transformf`` arrays in ``state.body_q``.
        """
        return TransformFormat.TRANSFORM

    def get_body_transforms(
        self,
        target_format: TransformFormat,
        *,
        env_ids: list[int] | None = None,
        quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
        matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
        double_precision: bool = False,
        stream: wp.Stream | None = None,
        allow_passthrough: bool = True,
        index_map: wp.array | None = None,
    ) -> TransformData | None:
        """Return body transforms in the requested format.

        When *target_format* is :attr:`TransformFormat.TRANSFORM` and
        *allow_passthrough* is ``True``, this returns Newton's own
        ``state.body_q`` array directly — zero copies, zero kernel launches.

        Args:
            target_format: Desired output transform representation.
            env_ids: Optional environment subset (currently unused).
            quat_convention: Quaternion component ordering for VEC3_QUAT output.
            matrix_layout: Matrix memory layout for MAT44 / VEC3_MAT33 output.
            double_precision: Use 64-bit floats for MAT44 output.
            stream: CUDA stream for deferred kernel execution.
            allow_passthrough: If ``True`` and formats match, return source
                data directly without copying.
            index_map: Optional index remapping for subset scatter writes.

        Returns:
            Typed transform data, or ``None`` when state is unavailable.
        """
        try:
            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None or state.body_q is None:
                return None

            body_q = state.body_q
            device = body_q.device.alias if hasattr(body_q.device, "alias") else str(body_q.device)
            count = body_q.shape[0]

            source = TransformArrayData(
                count=count,
                device=device,
                transforms=body_q,
            )

            return self._transform_buffer_pool.get_or_convert(
                source,
                target_format,
                generation=self._generation,
                quat_convention=quat_convention,
                matrix_layout=matrix_layout,
                double_precision=double_precision,
                stream=stream,
                allow_passthrough=allow_passthrough,
                index_map=index_map,
            )
        except Exception as exc:
            self._warn_once(
                "get-body-transforms-failed",
                "[NewtonSceneDataProvider] get_body_transforms() failed: %s",
                exc,
            )
            return None
