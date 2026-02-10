# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Manager for PhysX to Newton Warp model conversion.

This manager creates a Newton model for rendering purposes while PhysX handles physics simulation.
It builds the Newton model from the USD stage and synchronizes rigid body states from PhysX.
"""

from __future__ import annotations

import logging

import warp as wp

logger = logging.getLogger(__name__)


class NewtonManager:
    """Manages Newton Warp model for rendering with PhysX simulation.
    
    This is a simplified version of Newton-Warp's NewtonManager that only handles rendering.
    PhysX is used for physics simulation, and Newton is used only for Warp-based ray tracing.
    
    Key differences from full Newton simulation:
    - No physics solver (PhysX handles that)
    - Only maintains model geometry and rigid body poses
    - State is synchronized from PhysX each frame
    
    Lifecycle:
        1. initialize() - Build Newton model from USD stage
        2. Each frame: update_state() with PhysX rigid body poses
        3. Renderer calls get_model() and get_state_0() for ray tracing
    """

    _builder = None
    _model = None
    _state_0 = None
    _device: str = "cuda:0"
    _is_initialized: bool = False
    _num_envs: int = None
    _up_axis: str = "Z"

    @classmethod
    def clear(cls):
        """Clear all Newton manager state."""
        cls._builder = None
        cls._model = None
        cls._state_0 = None
        cls._is_initialized = False

    @classmethod
    def initialize(cls, num_envs: int, device: str = "cuda:0"):
        """Initialize Newton model from USD stage for rendering.
        
        Creates a Newton model that mirrors the PhysX scene structure but is used
        only for rendering, not physics simulation.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run Newton on ("cuda:0", etc.)
        
        Raises:
            ImportError: If Newton package is not installed
            RuntimeError: If USD stage is not available
        """
        if cls._is_initialized:
            logger.warning("NewtonManager already initialized. Skipping.")
            return

        cls._num_envs = num_envs
        cls._device = device

        try:
            import omni.usd
            from newton import Axis, ModelBuilder
            from pxr import UsdGeom
        except ImportError as e:
            raise ImportError(
                f"Failed to import required packages for Newton: {e}\n"
                "Please install newton:\n"
                "  pip install git+https://github.com/newton-physics/newton.git"
            ) from e

        logger.info(f"[NewtonManager] Initializing Newton model for rendering on device: {device}")

        # Get USD stage
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not available. Cannot initialize Newton model.")

        # Get stage up axis
        up_axis_str = UsdGeom.GetStageUpAxis(stage)
        cls._up_axis = up_axis_str
        logger.info(f"[NewtonManager] Stage up axis: {up_axis_str}")

        # Create Newton model builder from USD stage
        logger.info("[NewtonManager] Building Newton model from USD stage...")
        cls._builder = ModelBuilder(up_axis=up_axis_str)
        cls._builder.add_usd(stage)

        # Finalize model on device
        logger.info(f"[NewtonManager] Finalizing Newton model on {device}...")
        cls._builder.up_axis = Axis.from_string(cls._up_axis)
        cls._model = cls._builder.finalize(device=device)
        cls._model.num_envs = num_envs

        # Create state for rigid body poses
        cls._state_0 = cls._model.state()

        # Do forward kinematics to initialize body transforms
        from newton import eval_fk

        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

        cls._is_initialized = True
        logger.info(
            f"[NewtonManager] Initialized successfully: "
            f"{cls._model.body_count} bodies, "
            f"{cls._model.shape_count} shapes, "
            f"{num_envs} environments"
        )

    @classmethod
    def get_model(cls):
        """Get the Newton model for rendering.
        
        Returns:
            Newton Model instance containing scene geometry
            
        Raises:
            RuntimeError: If not initialized
        """
        if not cls._is_initialized:
            raise RuntimeError("NewtonManager not initialized. Call initialize() first.")
        return cls._model

    @classmethod
    def get_state_0(cls):
        """Get the current Newton state for rendering.
        
        Returns:
            Newton State instance with current rigid body poses
            
        Raises:
            RuntimeError: If not initialized
        """
        if not cls._is_initialized:
            raise RuntimeError("NewtonManager not initialized. Call initialize() first.")
        return cls._state_0

    @classmethod
    def update_state_from_usdrt(cls):
        """Update Newton state from USD runtime (USDRT) stage.
        
        This reads the current rigid body transforms from the USDRT fabric stage
        and updates the Newton state for rendering. This allows Newton's renderer
        to use the latest PhysX simulation results.
        
        Note: This is the key synchronization point between PhysX and Newton.
        
        TODO: Implement proper PhysX to Newton state synchronization.
        Currently this is a placeholder that doesn't update state.
        Newton will render using the initial model configuration.
        """
        if not cls._is_initialized:
            return

        # TODO: Implement USDRT fabric stage access and state synchronization
        # This requires:
        # 1. Access to fabric stage: usdrt.Usd.Stage.Attach(stage_id)
        # 2. Extract rigid body world transforms from fabric
        # 3. Map USD prim paths to Newton body indices
        # 4. Update Newton state (body_q for positions, quaternions stored elsewhere)
        # 5. Handle multi-environment scenarios (cloned environments)
        
        logger.debug("[NewtonManager] State synchronization placeholder (not yet implemented)")

    @classmethod
    def reset(cls):
        """Reset Newton state to initial configuration.
        
        This should be called when environments are reset in PhysX.
        """
        if not cls._is_initialized:
            return

        # Re-run forward kinematics to reset body transforms
        from newton import eval_fk

        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def shutdown(cls):
        """Shutdown and cleanup Newton manager."""
        logger.info("[NewtonManager] Shutting down")
        cls.clear()

    @staticmethod
    def _matrix_to_quaternion(rot_matrix):
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z).
        
        Args:
            rot_matrix: 3x3 rotation matrix as list of lists
            
        Returns:
            tuple: Quaternion as (w, x, y, z)
        """
        # Shoemake's algorithm for matrix to quaternion conversion
        # Based on: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        
        m = rot_matrix
        trace = m[0][0] + m[1][1] + m[2][2]
        
        if trace > 0:
            s = 0.5 / (trace + 1.0) ** 0.5
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
            
        return (w, x, y, z)
