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


@wp.kernel
def _copy_physx_poses_to_newton_kernel(
    physx_positions: wp.array(dtype=wp.vec3),
    physx_quaternions: wp.array(dtype=wp.vec4),
    newton_body_indices: wp.array(dtype=wp.int32),
    newton_body_q: wp.array(dtype=wp.transformf),
):
    """GPU kernel to copy PhysX poses to Newton body_q array.
    PhysX quaternions are (w, x, y, z); Warp transformf uses vec3 + quat (x, y, z, w).
    """
    i = wp.tid()
    newton_idx = newton_body_indices[i]
    if newton_idx < 0:
        return
    pos = physx_positions[i]
    quat = physx_quaternions[i]  # (w, x, y, z) from PhysX
    q = wp.quatf(quat[1], quat[2], quat[3], quat[0])  # (x, y, z, w) for Warp
    newton_body_q[newton_idx] = wp.transformf(pos, q)


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
    _scene = None  # InteractiveScene reference for PhysX tensor access
    _body_path_to_newton_idx: dict = {}  # Map USD path -> Newton body index

    @classmethod
    def clear(cls):
        """Clear all Newton manager state."""
        cls._builder = None
        cls._model = None
        cls._state_0 = None
        cls._is_initialized = False
        cls._scene = None
        cls._body_path_to_newton_idx = {}

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

        # Build mapping from USD path to Newton body index for fast lookup
        cls._body_path_to_newton_idx = {}
        for newton_idx, body_path in enumerate(cls._model.body_key):
            cls._body_path_to_newton_idx[body_path] = newton_idx

        cls._is_initialized = True
        logger.info(
            f"[NewtonManager] Initialized successfully: "
            f"{cls._model.body_count} bodies, "
            f"{cls._model.shape_count} shapes, "
            f"{num_envs} environments"
        )
        # Build PhysX->Newton mapping if scene was set (e.g. by env before camera init)
        cls._build_physx_to_newton_mapping()

    @classmethod
    def set_scene(cls, scene):
        """Set the InteractiveScene reference for PhysX tensor access."""
        cls._scene = scene
        num_arts = len(scene.articulations)
        num_objs = len(getattr(scene, "rigid_objects", None) or [])
        logger.info(f"[NewtonManager] Scene reference set with {num_arts} articulations, {num_objs} rigid objects")
        cls._build_physx_to_newton_mapping()

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
        """
        if not cls._is_initialized:
            return

        if cls._model.body_count == 0:
            # No rigid bodies in the model, nothing to sync
            return

        try:
            import omni.usd
            import usdrt
            from pxr import UsdGeom
        except ImportError as e:
            logger.error(f"Failed to import USDRT for state synchronization: {e}")
            return

        # Get USDRT fabric stage
        try:
            stage_id = omni.usd.get_context().get_stage_id()
            fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
            if fabric_stage is None:
                logger.warning("[NewtonManager] USDRT fabric stage not available for state sync")
                return
        except Exception as e:
            logger.debug(f"[NewtonManager] Could not attach to fabric stage: {e}")
            return

        # Newton's body_q stores 7-DOF poses: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        # Get the state array as numpy for efficient updates
        body_q_np = cls._state_0.body_q.numpy()
        
        # Track how many bodies we successfully updated
        updated_count = 0
        
        # Update each rigid body transform from USDRT
        for body_idx, body_prim_path in enumerate(cls._model.body_key):
            try:
                # Get prim from fabric stage
                prim = fabric_stage.GetPrimAtPath(body_prim_path)
                if not prim or not prim.IsValid():
                    continue

                # Get world transform from USDRT
                xformable = usdrt.Rt.Xformable(prim)
                if not xformable.HasWorldXform():
                    continue

                # Get 4x4 world transform matrix (row-major: [m00, m01, m02, m03, m10, ...])
                world_xform = xformable.GetWorldXform()
                
                # Extract translation from last column [m03, m13, m23]
                pos_x = world_xform[3]
                pos_y = world_xform[7]
                pos_z = world_xform[11]
                
                # Extract rotation matrix (top-left 3x3)
                rot_matrix = [
                    [world_xform[0], world_xform[1], world_xform[2]],    # row 0
                    [world_xform[4], world_xform[5], world_xform[6]],    # row 1
                    [world_xform[8], world_xform[9], world_xform[10]]    # row 2
                ]
                
                # Convert rotation matrix to quaternion (xyzw format for Newton)
                quat = cls._matrix_to_quaternion(rot_matrix)
                
                # Update Newton state: body_q[body_idx] = [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
                body_q_np[body_idx, 0] = pos_x
                body_q_np[body_idx, 1] = pos_y
                body_q_np[body_idx, 2] = pos_z
                body_q_np[body_idx, 3] = quat[1]  # x
                body_q_np[body_idx, 4] = quat[2]  # y
                body_q_np[body_idx, 5] = quat[3]  # z
                body_q_np[body_idx, 6] = quat[0]  # w
                
                updated_count += 1
                
            except Exception as e:
                logger.debug(f"[NewtonManager] Failed to update transform for {body_prim_path}: {e}")
                continue

        # Copy updated transforms back to Warp array
        if updated_count > 0:
            cls._state_0.body_q.assign(body_q_np)
            logger.debug(f"[NewtonManager] Updated {updated_count}/{cls._model.body_count} body transforms from PhysX")

    @classmethod
    def _body_path_to_newton_idx_lookup(cls, body_path: str, root_path: str, body_name: str) -> int:
        """Resolve Newton body index: try exact path, then match body_key by path components."""
        idx = cls._body_path_to_newton_idx.get(body_path, -1)
        if idx >= 0:
            return idx
        # Newton's body_key may use different path format; match by root + body_name as last component
        suffix = "/" + body_name
        for key, newton_idx in cls._body_path_to_newton_idx.items():
            if key.startswith(root_path) and key.endswith(suffix):
                return newton_idx
        # Also try: key ends with body_name (no extra slash) or key path parts end with body_name
        for key, newton_idx in cls._body_path_to_newton_idx.items():
            if not key.startswith(root_path):
                continue
            parts = key.split("/")
            if parts and parts[-1] == body_name:
                return newton_idx
        return -1

    @classmethod
    def _build_physx_to_newton_mapping(cls):
        """Build mapping arrays for GPU kernel (called once during setup)."""
        if cls._scene is None or not cls._is_initialized:
            return
        import torch
        cls._physx_to_newton_maps = {}

        # One-time debug: log sample Newton body_key paths vs our paths (remove after fixing)
        _debug_done = getattr(cls, "_build_mapping_debug_done", False)
        if not _debug_done:
            newton_keys = list(cls._body_path_to_newton_idx.keys())
            logger.warning("[NewtonManager] DEBUG sample Newton body_key (first 15): %s", newton_keys[:15])
            if len(newton_keys) > 20:
                logger.warning("[NewtonManager] DEBUG Newton body_key (last 5): %s", newton_keys[-5:])

        for art_name, articulation in cls._scene.articulations.items():
            num_bodies = articulation.num_bodies
            num_instances = articulation.num_instances
            total_bodies = num_bodies * num_instances
            mapping = torch.full((total_bodies,), -1, dtype=torch.int32, device=articulation.device)
            root_paths = articulation._root_physx_view.prim_paths
            body_names = articulation.body_names
            if not _debug_done:
                logger.warning(
                    "[NewtonManager] DEBUG articulation %r: root_path[0]=%r, body_names[:8]=%r",
                    art_name, root_paths[0], body_names[:8],
                )
            # Newton body_key uses link paths under the articulation root (e.g. /World/envs/env_0/Robot/iiwa7_link_0).
            # PhysX root_path is often the root joint prim (e.g. .../Robot/root_joint); use its parent as base.
            flat_idx = 0
            for env_idx in range(num_instances):
                root_path = root_paths[env_idx]
                base_path = root_path.rsplit("/", 1)[0] if "/" in root_path else root_path
                for body_local_idx, body_name in enumerate(body_names):
                    body_path = f"{base_path}/{body_name}"
                    mapping[flat_idx] = cls._body_path_to_newton_idx_lookup(body_path, base_path, body_name)
                    flat_idx += 1
            num_matched = (mapping >= 0).sum().item()
            cls._physx_to_newton_maps[art_name] = mapping
            logger.info(f"[NewtonManager] Built GPU mapping for articulation '{art_name}': {num_matched}/{total_bodies} bodies matched")
            if num_matched == 0:
                logger.warning(
                    "[NewtonManager] DEBUG no matches for %r; sample our path=%r/%r",
                    art_name, root_paths[0], body_names[0],
                )
        if hasattr(cls._scene, "rigid_objects") and cls._scene.rigid_objects:
            for obj_name, rigid_object in cls._scene.rigid_objects.items():
                num_instances = rigid_object.num_instances
                mapping = torch.full((num_instances,), -1, dtype=torch.int32, device=rigid_object.device)
                root_paths = rigid_object._root_physx_view.prim_paths
                for env_idx in range(num_instances):
                    mapping[env_idx] = cls._body_path_to_newton_idx.get(root_paths[env_idx], -1)
                cls._physx_to_newton_maps[obj_name] = mapping
                logger.info(f"[NewtonManager] Built GPU mapping for rigid object '{obj_name}': {num_instances} instances")
        if not _debug_done:
            cls._build_mapping_debug_done = True

    @classmethod
    def update_state_from_physx_tensors_gpu(cls):
        """Update Newton body poses from PhysX tensors using GPU kernels. Use this before render so robots/cube move."""
        if not cls._is_initialized:
            logger.warning("[NewtonManager] Not initialized, cannot update state")
            return
        if cls._model.body_count == 0:
            return
        if cls._scene is None or not hasattr(cls, "_physx_to_newton_maps"):
            cls.update_state_from_usdrt()
            return
        import torch
        for art_name, articulation in cls._scene.articulations.items():
            if art_name not in cls._physx_to_newton_maps:
                continue
            body_pos_w = articulation.data.body_pos_w
            body_quat_w = articulation.data.body_quat_w
            flat_pos = body_pos_w.reshape(-1, 3)
            flat_quat = body_quat_w.reshape(-1, 4)
            physx_positions_wp = wp.from_torch(flat_pos, dtype=wp.vec3)
            physx_quaternions_wp = wp.from_torch(flat_quat, dtype=wp.vec4)
            mapping_wp = wp.from_torch(cls._physx_to_newton_maps[art_name], dtype=wp.int32)
            num_bodies = flat_pos.shape[0]
            wp.launch(
                kernel=_copy_physx_poses_to_newton_kernel,
                dim=num_bodies,
                inputs=[physx_positions_wp, physx_quaternions_wp, mapping_wp, cls._state_0.body_q],
                device=cls._device,
            )
        if hasattr(cls._scene, "rigid_objects") and cls._scene.rigid_objects:
            for obj_name, rigid_object in cls._scene.rigid_objects.items():
                if obj_name not in cls._physx_to_newton_maps:
                    continue
                root_pos_w = rigid_object.data.root_pos_w
                root_quat_w = rigid_object.data.root_quat_w
                physx_positions_wp = wp.from_torch(root_pos_w, dtype=wp.vec3)
                physx_quaternions_wp = wp.from_torch(root_quat_w, dtype=wp.vec4)
                mapping_wp = wp.from_torch(cls._physx_to_newton_maps[obj_name], dtype=wp.int32)
                num_instances = root_pos_w.shape[0]
                wp.launch(
                    kernel=_copy_physx_poses_to_newton_kernel,
                    dim=num_instances,
                    inputs=[physx_positions_wp, physx_quaternions_wp, mapping_wp, cls._state_0.body_q],
                    device=cls._device,
                )
        wp.synchronize()
        logger.debug("[NewtonManager] Updated body transforms from PhysX tensors (GPU kernel)")

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
