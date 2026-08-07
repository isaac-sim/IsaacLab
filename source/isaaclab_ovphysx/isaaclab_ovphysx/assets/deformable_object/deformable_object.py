# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed deformable object asset."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from pxr import UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager
from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView

from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    set_kinematic_flags_to_one,
    vec6f,
    write_nodal_vec3f_to_buffer,
    write_nodal_vec4f_to_buffer,
)
from .views import OvPhysxDeformableBodyView

if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)


class DeformableObject(BaseDeformableObject):
    """OVPhysX-backed volume or surface deformable object asset.

    The state of a deformable object comprises the world-frame positions and
    velocities of its simulation nodes. Volume deformables additionally expose
    per-node kinematic targets. OVPhysX surface deformables do not support those
    targets and reject target writes with the same error as the PhysX backend.

    OVPhysX deformable tensor bindings are supported only on CUDA simulation
    devices.
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg) -> None:
        """Initialize the deformable object.

        Args:
            cfg: Configuration instance for the deformable object.
        """
        super().__init__(cfg)
        OvPhysxManager.require_full_stage()
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}
        self._deformable_type: str | None = None
        self._root_physx_view: OvPhysxDeformableBodyView | None = None
        self._material_physx_view: OvPhysxView | None = None

    @property
    def data(self) -> DeformableObjectData:
        """Data container for the deformable object."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of deformable object instances matched by the asset."""
        return self.root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always one because each object is a single deformable body.
        """
        return 1

    @property
    def root_view(self) -> OvPhysxDeformableBodyView:
        """Deformable body view for direct OVPhysX tensor access.

        .. note::
            Use this view with caution. OVPhysX indexed writes require complete
            first-dimension buffers even when only selected rows are applied.
        """
        if self._root_physx_view is None:
            raise RuntimeError("The OVPhysX deformable body view is not initialized.")
        return self._root_physx_view

    @property
    def root_physx_view(self) -> OvPhysxDeformableBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        logger.warning(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead."
        )
        return self.root_view

    @property
    def material_physx_view(self) -> OvPhysxView | None:
        """Optional deformable material view for direct OVPhysX tensor access."""
        return self._material_physx_view

    @property
    def max_sim_elements_per_body(self) -> int:
        """Maximum number of simulation mesh elements per deformable body."""
        return self.root_view.max_simulation_elements_per_body

    @property
    def max_collision_elements_per_body(self) -> int:
        """Maximum number of collision mesh elements per deformable body."""
        return self.root_view.max_collision_elements_per_body

    @property
    def max_sim_vertices_per_body(self) -> int:
        """Maximum number of simulation mesh vertices per deformable body."""
        return self.root_view.max_simulation_nodes_per_body

    @property
    def max_collision_vertices_per_body(self) -> int:
        """Maximum number of collision mesh vertices per deformable body."""
        return self.root_view.max_collision_nodes_per_body

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Reset the deformable object.

        Args:
            env_ids: Environment indices. If None, all indices are used.
            env_mask: Environment mask. If None, all instances are used.
        """
        pass

    def write_data_to_sim(self) -> None:
        """Write pending deformable commands to the simulator."""
        pass

    def update(self, dt: float) -> None:
        """Update the internal simulation timestamp.

        Args:
            dt: Time elapsed since the previous update [s].
        """
        self._data.update(dt)

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array(dtype=vec6f) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal positions and velocities over selected environments.

        Args:
            nodal_state: Nodal state in simulation frame [m, m/s]. Shape is
                ``(len(env_ids), max_sim_vertices_per_body, 6)`` or the full
                ``(num_instances, max_sim_vertices_per_body, 6)``.
            env_ids: Environment indices. If None, all indices are used.
            full_data: Whether :paramref:`nodal_state` contains all instances.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.warp
        expected_instances = self.num_instances if full_data else env_ids.shape[0]
        self.assert_shape_and_dtype(
            nodal_state, (expected_instances, self.max_sim_vertices_per_body), vec6f, "nodal_state"
        )
        if isinstance(nodal_state, wp.array):
            nodal_state = wp.to_torch(nodal_state)
        self.write_nodal_pos_to_sim_index(nodal_state[..., :3], env_ids=env_ids, full_data=full_data)
        self.write_nodal_velocity_to_sim_index(nodal_state[..., 3:], env_ids=env_ids, full_data=full_data)

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array(dtype=wp.vec3f) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal positions over selected environment indices.

        Args:
            nodal_pos: Nodal positions in simulation frame [m]. Shape is
                ``(len(env_ids), max_sim_vertices_per_body, 3)`` or the full
                ``(num_instances, max_sim_vertices_per_body, 3)``.
            env_ids: Environment indices. If None, all indices are used.
            full_data: Whether :paramref:`nodal_pos` contains all instances.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        expected_instances = self.num_instances if full_data else env_ids.shape[0]
        self.assert_shape_and_dtype(
            nodal_pos, (expected_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_pos"
        )
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)
        if (
            env_ids.shape[0] < self.num_instances
            and self._data._nodal_pos_w.timestamp < self._data._sim_timestamp
            and self._arrays_overlap(nodal_pos, self._data._nodal_pos_w.data)
        ):
            nodal_pos = wp.clone(nodal_pos)
        if env_ids.shape[0] < self.num_instances:
            _ = self._data.nodal_pos_w

        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_pos, env_ids, full_data],
            outputs=[self._data._nodal_pos_w.data],
            device=self.device,
        )
        self._data._nodal_pos_w.timestamp = self._data._sim_timestamp
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0
        self.root_view.set_attribute(self._sim_nodal_position_type, self._get_nodal_pos_w_f32(), indices=env_ids)

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array(dtype=wp.vec3f) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal velocities over selected environment indices.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s]. Shape is
                ``(len(env_ids), max_sim_vertices_per_body, 3)`` or the full
                ``(num_instances, max_sim_vertices_per_body, 3)``.
            env_ids: Environment indices. If None, all indices are used.
            full_data: Whether :paramref:`nodal_vel` contains all instances.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        expected_instances = self.num_instances if full_data else env_ids.shape[0]
        self.assert_shape_and_dtype(
            nodal_vel, (expected_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_vel"
        )
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)
        if (
            env_ids.shape[0] < self.num_instances
            and self._data._nodal_vel_w.timestamp < self._data._sim_timestamp
            and self._arrays_overlap(nodal_vel, self._data._nodal_vel_w.data)
        ):
            nodal_vel = wp.clone(nodal_vel)
        if env_ids.shape[0] < self.num_instances:
            _ = self._data.nodal_vel_w

        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_vel, env_ids, full_data],
            outputs=[self._data._nodal_vel_w.data],
            device=self.device,
        )
        self._data._nodal_vel_w.timestamp = self._data._sim_timestamp
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0
        self.root_view.set_attribute(self._sim_nodal_velocity_type, self._get_nodal_vel_w_f32(), indices=env_ids)

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array(dtype=wp.vec4f) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
        full_data: bool = False,
    ) -> None:
        """Set volume-deformable kinematic targets over selected environments.

        Args:
            targets: Nodal target positions and free-node flags [m, dimensionless].
                Shape is ``(len(env_ids), max_sim_vertices_per_body, 4)`` or the
                full ``(num_instances, max_sim_vertices_per_body, 4)``.
            env_ids: Environment indices. If None, all indices are used.
            full_data: Whether :paramref:`targets` contains all instances.

        Raises:
            ValueError: If this is a surface deformable body.
        """
        if self._deformable_type != "volume":
            raise ValueError("Kinematic targets can only be set for volume deformable bodies.")

        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(targets, ProxyArray):
            targets = targets.warp
        expected_instances = self.num_instances if full_data else env_ids.shape[0]
        self.assert_shape_and_dtype(targets, (expected_instances, self.max_sim_vertices_per_body), wp.vec4f, "targets")
        if isinstance(targets, torch.Tensor):
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)
        target_buffer = self._data.nodal_kinematic_target
        if target_buffer is None:
            raise RuntimeError("Volume deformable kinematic targets are not initialized.")
        wp.launch(
            write_nodal_vec4f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[targets, env_ids, full_data],
            outputs=[target_buffer.warp],
            device=self.device,
        )
        self.root_view.set_attribute(
            self._sim_kinematic_target_type, target_buffer.warp.view(wp.float32), indices=env_ids
        )

    def _initialize_impl(self) -> None:
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        self._device = OvPhysxManager.get_device()
        if not wp.get_device(self._device).is_cuda:
            raise RuntimeError(
                f"OVPhysX deformable tensors require a CUDA simulation device; received {self._device!r}."
            )

        def has_deformable_body_api(prim) -> bool:
            return "OmniPhysicsDeformableBodyAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()

        asset_prim, root_expr = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)[0]
        walk_root = asset_prim.GetPath().pathString
        resolve_kwargs = {"predicate": has_deformable_body_api, "expected_num_matches": 1}
        deformable_prims = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)
        root_prim, root_path_expr = deformable_prims[0]

        material_prim = None
        if root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
            for material_path in material_paths:
                candidate = root_prim.GetStage().GetPrimAtPath(material_path)
                schemas = candidate.GetPrimTypeInfo().GetAppliedAPISchemas()
                if "OmniPhysicsDeformableMaterialAPI" not in schemas:
                    continue
                material_prim = candidate
                if "PhysxSurfaceDeformableMaterialAPI" in schemas:
                    self._deformable_type = "surface"
                elif "PhysxDeformableMaterialAPI" in schemas:
                    self._deformable_type = "volume"
                break

        if material_prim is None:
            logger.warning(
                f"Failed to find a deformable material binding for '{root_prim.GetPath().pathString}'. "
                "The material properties will use defaults and cannot be modified at runtime."
            )

        if self._deformable_type is None:
            has_tetmesh = bool(
                sim_utils.get_all_matching_child_prims(
                    root_prim.GetPath(), lambda prim: prim.GetTypeName() == "TetMesh"
                )
            )
            if has_tetmesh:
                self._deformable_type = "volume"
            else:
                has_mesh = bool(
                    sim_utils.get_all_matching_child_prims(
                        root_prim.GetPath(), lambda prim: prim.GetTypeName() == "Mesh"
                    )
                )
                if has_mesh:
                    self._deformable_type = "surface"

        if self._deformable_type == "volume":
            self._sim_nodal_position_type = TT.DEFORMABLE_SIM_NODAL_POSITION
            self._sim_nodal_velocity_type = TT.DEFORMABLE_SIM_NODAL_VELOCITY
            self._sim_kinematic_target_type = TT.DEFORMABLE_SIM_KINEMATIC_TARGET
            self._rest_nodal_position_type = TT.DEFORMABLE_REST_NODAL_POSITION
            self._sim_element_indices_type = TT.DEFORMABLE_SIM_ELEMENT_INDICES
            self._collision_element_indices_type = TT.DEFORMABLE_COLLISION_ELEMENT_INDICES
            body_tensor_types = [
                self._sim_nodal_position_type,
                self._sim_nodal_velocity_type,
                self._sim_kinematic_target_type,
                self._rest_nodal_position_type,
                self._sim_element_indices_type,
                self._collision_element_indices_type,
            ]
        elif self._deformable_type == "surface":
            self._sim_nodal_position_type = TT.SURFACE_DEFORMABLE_SIM_POSITION
            self._sim_nodal_velocity_type = TT.SURFACE_DEFORMABLE_SIM_VELOCITY
            self._sim_kinematic_target_type = None
            self._rest_nodal_position_type = TT.SURFACE_DEFORMABLE_REST_POSITION
            self._sim_element_indices_type = TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES
            self._collision_element_indices_type = None
            body_tensor_types = [
                self._sim_nodal_position_type,
                self._sim_nodal_velocity_type,
                self._rest_nodal_position_type,
                self._sim_element_indices_type,
            ]
        else:
            raise RuntimeError(
                f"Failed to determine deformable type for '{root_prim.GetPath().pathString}'. "
                "Ensure that a deformable material is bound or a valid TetMesh or Mesh exists below the body."
            )

        root_pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", root_path_expr)
        root_pattern = re.sub(r"\.\*", "*", root_pattern)
        try:
            self._root_physx_view = OvPhysxDeformableBodyView(
                physx_instance,
                pattern=root_pattern,
                device=self._device,
                tensor_types=body_tensor_types,
                eager=True,
                simulation_nodal_position_type=self._sim_nodal_position_type,
                simulation_element_indices_type=self._sim_element_indices_type,
                collision_element_indices_type=self._collision_element_indices_type,
            )
        except Exception as error:
            raise RuntimeError(
                f"OVPhysX could not create a {self._deformable_type} deformable body view for pattern "
                f"{root_pattern!r}: {error}"
            ) from error

        if material_prim is not None:
            material_path = material_prim.GetPath().pathString
            material_path_expr = (
                root_expr + material_path[len(walk_root) :]
                if material_prim.GetPath().HasPrefix(asset_prim.GetPath())
                else material_path
            )
            material_pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", material_path_expr)
            material_pattern = re.sub(r"\.\*", "*", material_pattern)
            material_tensor_types = [
                TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION,
                TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS,
                TT.DEFORMABLE_MATERIAL_POISSONS_RATIO,
                TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING,
                TT.DEFORMABLE_MATERIAL_BENDING_STIFFNESS,
                TT.DEFORMABLE_MATERIAL_THICKNESS,
                TT.DEFORMABLE_MATERIAL_BENDING_DAMPING,
            ]
            try:
                self._material_physx_view = OvPhysxView(
                    physx_instance,
                    pattern=material_pattern,
                    device="cpu",
                    tensor_types=material_tensor_types,
                    eager=True,
                )
            except Exception as error:
                self._root_physx_view = None
                raise RuntimeError(
                    f"OVPhysX could not create a deformable material view for pattern {material_pattern!r}."
                ) from error
        else:
            self._material_physx_view = None

        logger.info("Deformable body initialized at: %s", root_pattern)
        logger.info("Deformable type: %s", self._deformable_type)
        logger.info("Number of instances: %s", self.num_instances)
        if self._material_physx_view is not None:
            logger.info("Number of deformable materials: %s", self._material_physx_view.count)

        self._data = DeformableObjectData(
            self.root_view,
            self.device,
            position_tensor_type=self._sim_nodal_position_type,
            velocity_tensor_type=self._sim_nodal_velocity_type,
        )
        self._create_buffers()
        self.update(0.0)

        if self._debug_vis_handle is None:
            self.set_debug_vis(self.cfg.debug_vis)

    def _create_buffers(self) -> None:
        """Create stable default-state and kinematic-target buffers."""
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)
        self._nodal_pos_w_f32: wp.array | None = None
        self._nodal_vel_w_f32: wp.array | None = None

        nodal_positions = self._data.nodal_pos_w.warp
        nodal_velocities = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec3f, device=self.device
        )
        default_nodal_state = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=vec6f, device=self.device
        )
        wp.launch(
            compute_nodal_state_w,
            dim=(self.num_instances, self.max_sim_vertices_per_body),
            inputs=[nodal_positions, nodal_velocities],
            outputs=[default_nodal_state],
            device=self.device,
        )
        self._data.default_nodal_state_w = ProxyArray(default_nodal_state)

        if self._deformable_type == "volume":
            target_values = self.root_view.get_attribute(self._sim_kinematic_target_type)
            target_values = target_values.view(wp.vec4f).reshape((self.num_instances, self.max_sim_vertices_per_body))
            kinematic_targets = wp.zeros(
                (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec4f, device=self.device
            )
            wp.copy(kinematic_targets, target_values)
            wp.launch(
                set_kinematic_flags_to_one,
                dim=self.num_instances * self.max_sim_vertices_per_body,
                inputs=[kinematic_targets.reshape((self.num_instances * self.max_sim_vertices_per_body,))],
                device=self.device,
            )
            self._data.nodal_kinematic_target = ProxyArray(kinematic_targets)
            self.root_view.set_attribute(self._sim_kinematic_target_type, kinematic_targets.view(wp.float32))
        else:
            self._data.nodal_kinematic_target = None

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None) -> wp.array(
        dtype=wp.int32
    ):
        """Resolve environment indices to a device-resident int32 Warp array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            values = env_ids.to(device=self.device, dtype=torch.int32).contiguous()
            return wp.from_torch(values, dtype=wp.int32)
        if isinstance(env_ids, wp.array):
            if env_ids.dtype != wp.int32:
                values = wp.to_torch(env_ids).to(device=self.device, dtype=torch.int32).contiguous()
                return wp.from_torch(values, dtype=wp.int32)
            if str(env_ids.device) != self.device:
                return wp.clone(env_ids, device=self.device)
            return env_ids
        return wp.array(list(env_ids), dtype=wp.int32, device=self.device)

    @staticmethod
    def _arrays_overlap(first: wp.array, second: wp.array) -> bool:
        """Return whether two contiguous Warp arrays share any storage."""
        first_start = int(first.ptr)
        first_end = first_start + first.size * wp.types.type_size_in_bytes(first.dtype)
        second_start = int(second.ptr)
        second_end = second_start + second.size * wp.types.type_size_in_bytes(second.dtype)
        return first_start < second_end and second_start < first_end

    def _get_nodal_pos_w_f32(self) -> wp.array(dtype=wp.float32):
        """Return the stable scalar view used for OVPhysX position writes [m]."""
        if self._nodal_pos_w_f32 is None:
            self._nodal_pos_w_f32 = self._data._nodal_pos_w.data.view(wp.float32)
        return self._nodal_pos_w_f32

    def _get_nodal_vel_w_f32(self) -> wp.array(dtype=wp.float32):
        """Return the stable scalar view used for OVPhysX velocity writes [m/s]."""
        if self._nodal_vel_w_f32 is None:
            self._nodal_vel_w_f32 = self._data._nodal_vel_w.data.view(wp.float32)
        return self._nodal_vel_w_f32

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        num_enabled = 0
        if self._deformable_type == "volume" and self.data.nodal_kinematic_target is not None:
            kinematic_targets = self.data.nodal_kinematic_target.torch
            targets_enabled = kinematic_targets[..., 3] == 0.0
            num_enabled = int(torch.sum(targets_enabled).item())
        if num_enabled == 0:
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = kinematic_targets[targets_enabled][..., :3]
        self.target_visualizer.visualize(positions)

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidate the OVPhysX deformable views."""
        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None
        self._material_physx_view = None
