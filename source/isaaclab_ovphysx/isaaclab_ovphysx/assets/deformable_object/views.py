# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX-style OVPhysX tensor views for deformable bodies and materials."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import torch
import warp as wp

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView

_BODY_TENSORS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "volume": MappingProxyType(
            {
                "position": TT.DEFORMABLE_SIM_NODAL_POSITION,
                "velocity": TT.DEFORMABLE_SIM_NODAL_VELOCITY,
                "target": TT.DEFORMABLE_SIM_KINEMATIC_TARGET,
                "rest": TT.DEFORMABLE_REST_NODAL_POSITION,
                "elements": TT.DEFORMABLE_SIM_ELEMENT_INDICES,
                "collision_elements": TT.DEFORMABLE_COLLISION_ELEMENT_INDICES,
            }
        ),
        "surface": MappingProxyType(
            {
                "position": TT.SURFACE_DEFORMABLE_SIM_POSITION,
                "velocity": TT.SURFACE_DEFORMABLE_SIM_VELOCITY,
                "rest": TT.SURFACE_DEFORMABLE_REST_POSITION,
                "elements": TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES,
            }
        ),
    }
)

_MATERIAL_TENSORS: Mapping[str, Any] = MappingProxyType(
    {
        "dynamic_friction": TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION,
        "youngs_modulus": TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS,
        "poissons_ratio": TT.DEFORMABLE_MATERIAL_POISSONS_RATIO,
        "elasticity_damping": TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING,
        "bending_stiffness": TT.DEFORMABLE_MATERIAL_BENDING_STIFFNESS,
        "thickness": TT.DEFORMABLE_MATERIAL_THICKNESS,
        "bending_damping": TT.DEFORMABLE_MATERIAL_BENDING_DAMPING,
    }
)


class OvPhysxDeformableBodyView:
    """PhysX-style tensor view for volume and surface deformable bodies.

    Args:
        physx_instance: OVPhysX instance that creates tensor bindings.
        pattern: Glob pattern selecting deformable body prims.
        device: Simulation device for body state tensors.
        deformable_type: Whether the bodies are ``"volume"`` or ``"surface"`` deformables.
    """

    def __init__(
        self,
        physx_instance: Any,
        pattern: str,
        device: str,
        deformable_type: Literal["volume", "surface"],
    ) -> None:
        if deformable_type not in _BODY_TENSORS:
            raise ValueError(f"Unsupported deformable type {deformable_type!r}; expected 'volume' or 'surface'.")
        self._device = str(wp.get_device(device))
        self._deformable_type = deformable_type
        self._tensor_map = _BODY_TENSORS[deformable_type]
        self._view: OvPhysxView | None = OvPhysxView(
            physx_instance,
            pattern=pattern,
            device=device,
            tensor_types=list(self._tensor_map.values()),
            eager=True,
        )

    @property
    def count(self) -> int:
        """Number of deformable bodies matched by this view."""
        return self._require_view().count

    @property
    def prim_paths(self) -> list[str]:
        """USD paths of deformable bodies matched by this view."""
        return self._require_view().prim_paths

    @property
    def max_simulation_nodes_per_body(self) -> int:
        """Maximum simulation nodes per deformable body."""
        return self._tensor_shape("position")[1]

    @property
    def max_simulation_elements_per_body(self) -> int:
        """Maximum simulation elements per deformable body."""
        return self._tensor_shape("elements")[1]

    @property
    def max_collision_elements_per_body(self) -> int:
        """Maximum collision elements per body, or zero for surface deformables."""
        if "collision_elements" not in self._tensor_map:
            return 0
        return self._tensor_shape("collision_elements")[1]

    @property
    def max_collision_nodes_per_body(self) -> int:
        """Maximum collision nodes per body, or zero when unavailable."""
        return 0

    def get_simulation_nodal_positions(self) -> wp.array(dtype=wp.float32):
        """Return simulation nodal positions [m]."""
        return self._get("position")

    def read_simulation_nodal_positions_into(self, values: wp.array(dtype=wp.float32)) -> None:
        """Read simulation nodal positions [m] into a caller-owned buffer.

        Args:
            values: Destination buffer [m]. Shape is
                ``(count, max_simulation_nodes_per_body, 3)``.
        """
        self._require_view().read_into(self._tensor_map["position"], values)

    def get_simulation_nodal_velocities(self) -> wp.array(dtype=wp.float32):
        """Return simulation nodal velocities [m/s]."""
        return self._get("velocity")

    def read_simulation_nodal_velocities_into(self, values: wp.array(dtype=wp.float32)) -> None:
        """Read simulation nodal velocities [m/s] into a caller-owned buffer.

        Args:
            values: Destination buffer [m/s]. Shape is
                ``(count, max_simulation_nodes_per_body, 3)``.
        """
        self._require_view().read_into(self._tensor_map["velocity"], values)

    def get_simulation_nodal_kinematic_targets(self) -> wp.array(dtype=wp.float32):
        """Return simulation nodal kinematic targets [m, dimensionless]."""
        self._require_volume_targets()
        return self._get("target")

    def get_rest_nodal_positions(self) -> wp.array(dtype=wp.float32):
        """Return rest nodal positions [m]."""
        return self._get("rest")

    def get_simulation_element_indices(self) -> wp.array(dtype=wp.int32):
        """Return simulation element connectivity indices."""
        return self._get("elements")

    def get_collision_element_indices(self) -> wp.array(dtype=wp.int32):
        """Return collision element connectivity indices."""
        return self._get("collision_elements")

    def set_simulation_nodal_positions(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set simulation nodal positions [m]."""
        self._set("position", values, indices=indices, mask=mask)

    def set_simulation_nodal_velocities(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set simulation nodal velocities [m/s]."""
        self._set("velocity", values, indices=indices, mask=mask)

    def set_simulation_nodal_kinematic_targets(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set simulation nodal kinematic targets [m, dimensionless]."""
        self._require_volume_targets()
        self._set("target", values, indices=indices, mask=mask)

    get_sim_nodal_positions = get_simulation_nodal_positions
    get_sim_nodal_velocities = get_simulation_nodal_velocities
    get_sim_nodal_kinematic_targets = get_simulation_nodal_kinematic_targets
    get_rest_nodal_positions = get_rest_nodal_positions
    get_sim_element_indices = get_simulation_element_indices
    get_collision_element_indices = get_collision_element_indices
    set_sim_nodal_positions = set_simulation_nodal_positions
    set_sim_nodal_velocities = set_simulation_nodal_velocities
    set_sim_nodal_kinematic_targets = set_simulation_nodal_kinematic_targets

    def destroy(self) -> None:
        """Destroy all cached OVPhysX tensor bindings held by this view."""
        view = self._view
        if view is None:
            return
        for binding in view._bindings.values():
            destroy = getattr(binding, "destroy", None)
            if destroy is not None:
                destroy()
        view._bindings.clear()
        view._read_views.clear()
        self._view = None

    def _get(self, name: str) -> wp.array:
        return self._require_view().get_attribute(self._tensor_map[name])

    def _set(
        self,
        name: str,
        values: wp.array | torch.Tensor,
        *,
        indices: wp.array | torch.Tensor | None,
        mask: wp.array | torch.Tensor | None,
    ) -> None:
        self._require_view().set_attribute(
            self._tensor_map[name],
            self._normalize_values(values),
            indices=self._normalize_selection(indices, wp.int32),
            mask=self._normalize_selection(mask, wp.bool),
        )

    def _tensor_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._require_view().binding_for(self._tensor_map[name]).shape)

    @staticmethod
    def _normalize_values(values: wp.array | torch.Tensor) -> wp.array:
        if isinstance(values, torch.Tensor):
            return wp.from_torch(values.contiguous(), dtype=wp.float32)
        return values

    def _normalize_selection(self, values: wp.array | torch.Tensor | None, dtype: Any) -> wp.array | None:
        if values is None:
            return None
        if isinstance(values, wp.array) and values.dtype == dtype and str(values.device) == self._device:
            return values
        if isinstance(values, torch.Tensor):
            torch_dtype = torch.int32 if dtype == wp.int32 else torch.bool
            values = values.to(device=self._device, dtype=torch_dtype).contiguous()
            return wp.from_torch(values, dtype=dtype)
        return wp.array(values, dtype=dtype, device=self._device)

    def _require_volume_targets(self) -> None:
        if self._deformable_type != "volume":
            raise ValueError("Kinematic targets can only be set for volume deformable bodies.")

    def _require_view(self) -> OvPhysxView:
        if self._view is None:
            raise RuntimeError("The deformable body view has been destroyed.")
        return self._view


class OvPhysxDeformableMaterialView:
    """PhysX-style CPU tensor view for deformable material properties.

    Args:
        physx_instance: OVPhysX instance that creates tensor bindings.
        pattern: Glob pattern selecting deformable material prims.
    """

    def __init__(self, physx_instance: Any, pattern: str) -> None:
        self._view: OvPhysxView | None = OvPhysxView(
            physx_instance,
            pattern=pattern,
            device="cpu",
            tensor_types=list(_MATERIAL_TENSORS.values()),
            eager=True,
        )

    @property
    def count(self) -> int:
        """Number of deformable materials matched by this view."""
        return self._require_view().count

    @property
    def prim_paths(self) -> list[str]:
        """USD paths of deformable materials matched by this view."""
        return self._require_view().prim_paths

    def get_dynamic_frictions(self) -> wp.array(dtype=wp.float32):
        """Return dynamic friction coefficients [dimensionless]."""
        return self._get("dynamic_friction")

    def set_dynamic_frictions(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set dynamic friction coefficients [dimensionless]."""
        self._set("dynamic_friction", values, indices=indices, mask=mask)

    def get_youngs_moduli(self) -> wp.array(dtype=wp.float32):
        """Return Young's moduli [Pa]."""
        return self._get("youngs_modulus")

    def set_youngs_moduli(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set Young's moduli [Pa]."""
        self._set("youngs_modulus", values, indices=indices, mask=mask)

    def get_poissons_ratios(self) -> wp.array(dtype=wp.float32):
        """Return Poisson's ratios [dimensionless]."""
        return self._get("poissons_ratio")

    def set_poissons_ratios(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set Poisson's ratios [dimensionless]."""
        self._set("poissons_ratio", values, indices=indices, mask=mask)

    def get_elasticity_dampings(self) -> wp.array(dtype=wp.float32):
        """Return elasticity damping coefficients [dimensionless]."""
        return self._get("elasticity_damping")

    def set_elasticity_dampings(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set elasticity damping coefficients [dimensionless]."""
        self._set("elasticity_damping", values, indices=indices, mask=mask)

    def get_bending_stiffnesses(self) -> wp.array(dtype=wp.float32):
        """Return bending stiffnesses [N·m]."""
        return self._get("bending_stiffness")

    def set_bending_stiffnesses(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set bending stiffnesses [N·m]."""
        self._set("bending_stiffness", values, indices=indices, mask=mask)

    def get_thicknesses(self) -> wp.array(dtype=wp.float32):
        """Return material thicknesses [m]."""
        return self._get("thickness")

    def set_thicknesses(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set material thicknesses [m]."""
        self._set("thickness", values, indices=indices, mask=mask)

    def get_bending_dampings(self) -> wp.array(dtype=wp.float32):
        """Return bending damping coefficients [dimensionless]."""
        return self._get("bending_damping")

    def set_bending_dampings(
        self,
        values: wp.array(dtype=wp.float32) | torch.Tensor,
        indices: wp.array(dtype=wp.int32) | torch.Tensor | None = None,
        mask: wp.array(dtype=wp.bool) | torch.Tensor | None = None,
    ) -> None:
        """Set bending damping coefficients [dimensionless]."""
        self._set("bending_damping", values, indices=indices, mask=mask)

    def destroy(self) -> None:
        """Destroy all cached OVPhysX tensor bindings held by this view."""
        view = self._view
        if view is None:
            return
        for binding in view._bindings.values():
            destroy = getattr(binding, "destroy", None)
            if destroy is not None:
                destroy()
        view._bindings.clear()
        view._read_views.clear()
        self._view = None

    def _get(self, name: str) -> wp.array:
        return self._require_view().get_attribute(_MATERIAL_TENSORS[name])

    def _set(
        self,
        name: str,
        values: wp.array | torch.Tensor,
        *,
        indices: wp.array | torch.Tensor | None,
        mask: wp.array | torch.Tensor | None,
    ) -> None:
        self._require_view().set_attribute(
            _MATERIAL_TENSORS[name],
            self._normalize_values(values),
            indices=self._normalize_selection(indices, wp.int32),
            mask=self._normalize_selection(mask, wp.bool),
        )

    @staticmethod
    def _normalize_values(values: wp.array | torch.Tensor) -> wp.array:
        if isinstance(values, torch.Tensor):
            values = values.to(device="cpu", dtype=torch.float32).contiguous()
            return wp.from_torch(values, dtype=wp.float32)
        return values

    @staticmethod
    def _normalize_selection(values: wp.array | torch.Tensor | None, dtype: Any) -> wp.array | None:
        if values is None:
            return None
        if isinstance(values, wp.array) and values.dtype == dtype and str(values.device) == "cpu":
            return values
        if isinstance(values, torch.Tensor):
            torch_dtype = torch.int32 if dtype == wp.int32 else torch.bool
            values = values.to(device="cpu", dtype=torch_dtype).contiguous()
            return wp.from_torch(values, dtype=dtype)
        return wp.array(values, dtype=dtype, device="cpu")

    def _require_view(self) -> OvPhysxView:
        if self._view is None:
            raise RuntimeError("The deformable material view has been destroyed.")
        return self._view
