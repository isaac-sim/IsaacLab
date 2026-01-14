# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for visualizers and renderers.

This module provides a unified interface for accessing scene data from different physics backends
and synchronizing that data to rendering backends (USD Fabric, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from newton import Model, State


# Lazy import warp dependencies for OV visualizer support
def _get_warp_kernel():
    """Get the warp kernel for syncing fabric transforms.

    Returns the kernel only when warp is available, avoiding import errors
    when warp is not installed.
    """
    try:
        import warp as wp

        # Define warp kernel for syncing transforms
        @wp.kernel(enable_backward=False)
        def set_vec3d_array(
            fabric_vals: wp.fabricarray(dtype=wp.mat44d),
            indices: wp.fabricarray(dtype=wp.uint32),
            newton_vals: wp.array(ndim=1, dtype=wp.transformf),
        ):
            i = int(wp.tid())
            idx = int(indices[i])
            new_val = newton_vals[idx]
            fabric_vals[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(new_val)))

        return set_vec3d_array
    except ImportError:
        return None


class SceneDataProvider:
    """Unified scene data provider for all physics and rendering backends.

    This class provides:
    - Access to physics state and model for visualizers (Newton OpenGL, Rerun)
    - Synchronization of physics state to USD Fabric for OV visualizer
    - Conditional updates based on active visualizers to avoid unnecessary work

    The provider handles both Newton-specific data (state, model) and OV-specific
    synchronization (fabric transforms) within a single class to keep the abstraction
    simple while supporting multiple backends.
    """

    def __init__(self, visualizer_cfgs: list[Any] | None = None):
        """Initialize the scene data provider with visualizer configurations.

        Args:
            visualizer_cfgs: List of visualizer configurations to determine which backends are active.
        """
        self._has_newton_visualizer = False
        self._has_rerun_visualizer = False
        self._has_ov_visualizer = False
        self._is_initialized = False

        # Determine which visualizers are enabled from configs
        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                viz_type = getattr(cfg, "visualizer_type", None)
                if viz_type == "newton":
                    self._has_newton_visualizer = True
                elif viz_type == "rerun":
                    self._has_rerun_visualizer = True
                elif viz_type == "omniverse":
                    self._has_ov_visualizer = True

        self._is_initialized = True

    def update(self) -> None:
        """Update scene data for visualizers.

        This method:
        - Syncs Newton transforms to USD Fabric (if OV visualizer is active)
        - Can be extended for other backend-specific updates

        Note:
            Only performs work if necessary visualizers are active to avoid overhead.
        """
        if not self._is_initialized:
            return

        # Sync fabric transforms only if OV visualizer is active
        if self._has_ov_visualizer:
            self._sync_fabric_transforms()

    def get_state(self) -> State | None:
        """Get physics state for Newton-based visualizers.

        Returns:
            Newton State object, or None if not available.
        """
        # State is needed by Newton OpenGL and Rerun visualizers
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None

        # Lazy import to avoid loading Newton if not needed
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._state_0
        except ImportError:
            return None

    def get_model(self) -> Model | None:
        """Get physics model for Newton-based visualizers.

        Returns:
            Newton Model object, or None if not available.
        """
        # Model is needed by Newton OpenGL and Rerun visualizers
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None

        # Lazy import to avoid loading Newton if not needed
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._model
        except ImportError:
            return None

    def _sync_fabric_transforms(self) -> None:
        """Sync Newton transforms to USD Fabric for OV visualizer.

        This method updates the USD Fabric with the latest physics transforms from Newton.
        It uses a warp kernel to efficiently copy transform data on the GPU.

        The sync process:
        1. Selects USD prims that have both worldMatrix and newton index attributes
        2. Creates fabric arrays for transforms and indices
        3. Launches warp kernel to copy Newton body transforms to USD Fabric

        Note:
            This is only called when OV visualizer is active to avoid unnecessary GPU work.
        """
        # Get warp kernel (lazy loaded)
        set_vec3d_array = _get_warp_kernel()
        if set_vec3d_array is None:
            return

        try:
            import usdrt
            import warp as wp

            from isaaclab.sim._impl.newton_manager import NewtonManager

            # Select all prims with required attributes
            selection = NewtonManager._usdrt_stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
                    (usdrt.Sdf.ValueTypeNames.UInt, NewtonManager._newton_index_attr, usdrt.Usd.Access.Read),
                ],
                device="cuda:0",
            )

            # Create fabric arrays for indices and transforms
            fabric_newton_indices = wp.fabricarray(selection, NewtonManager._newton_index_attr)
            current_transforms = wp.fabricarray(selection, "omni:fabric:worldMatrix")

            # Launch warp kernel to sync transforms
            wp.launch(
                set_vec3d_array,
                dim=(fabric_newton_indices.shape[0]),
                inputs=[current_transforms, fabric_newton_indices, NewtonManager._state_0.body_q],
                device="cuda:0",
            )
        except (ImportError, AttributeError):
            # Silently fail if Newton isn't initialized or attributes are missing
            pass
