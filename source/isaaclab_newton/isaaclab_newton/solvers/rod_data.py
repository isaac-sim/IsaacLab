# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Data structures for the Direct Position-Based Solver for Stiff Rods.

This module implements the data structures needed for rod simulation based on:
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"
Computer Graphics Forum, Vol. 37, No. 8

The solver uses the XPBD (Extended Position-Based Dynamics) framework with
a direct solver that exploits the tree structure of rod constraints for
linear-time complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import warp as wp

# Initialize warp
wp.init()


@dataclass
class RodMaterialConfig:
    """Material configuration for Cosserat rod model.

    Based on the Cosserat rod model, material properties are defined
    using physically meaningful parameters like Young's modulus and
    shear modulus, plus normalized stiffness multipliers for fine control.

    Attributes:
        young_modulus: Young's modulus E [Pa]. Controls bending stiffness.
        shear_modulus: Shear modulus G [Pa]. Controls torsion stiffness.
            If None, computed as E / (2 * (1 + poisson_ratio)).
        poisson_ratio: Poisson's ratio ν. Used to compute shear modulus if not given.
        density: Material density ρ [kg/m³].
        damping: Damping coefficient for velocity damping.
        
        # Normalized stiffness multipliers (0.0 to 1.0 scale, like Newton Viewer)
        stretch_stiffness: Stretch stiffness multiplier (1.0 = inextensible).
        shear_stiffness: Shear stiffness multiplier (1.0 = no shear deformation).
        bend_stiffness: Bending stiffness multiplier (0.0-1.0, lower = more flexible).
        twist_stiffness: Twist/torsion stiffness multiplier (0.0-1.0).
    """

    young_modulus: float = 1e9  # Steel-like stiffness
    shear_modulus: float | None = None
    poisson_ratio: float = 0.3
    density: float = 7800.0  # Steel density
    damping: float = 0.01
    
    # Normalized stiffness multipliers (Newton Viewer style)
    stretch_stiffness: float = 1.0   # 1.0 = fully inextensible
    shear_stiffness: float = 1.0     # 1.0 = no shear deformation  
    bend_stiffness: float = 0.1      # Lower = more flexible (catheter-like)
    twist_stiffness: float = 0.4     # Moderate torsional resistance

    def __post_init__(self):
        """Compute shear modulus if not provided."""
        if self.shear_modulus is None:
            self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class RodTipConfig:
    """Configuration for catheter/guidewire tip shaping.
    
    Allows defining a curved tip region with different rest curvature,
    simulating J-tip or angled catheters commonly used in interventional procedures.
    
    Attributes:
        num_tip_segments: Number of segments at the tip with custom curvature.
        rest_bend_omega1: Rest bending curvature around local X axis [rad/m].
        rest_bend_omega2: Rest bending curvature around local Y axis [rad/m].
        rest_twist: Rest twist around local Z axis [rad/m].
    """
    num_tip_segments: int = 20  # Last N particles for tip shaping
    rest_bend_omega1: float = 0.0  # Curvature in XZ plane
    rest_bend_omega2: float = 0.0  # Curvature in YZ plane
    rest_twist: float = 0.0  # Axial twist


@dataclass
class RodGeometryConfig:
    """Geometry configuration for rod segments.

    Attributes:
        num_segments: Number of rigid segments in the rod.
        segment_length: Length of each segment [m]. If None, computed from rest_length.
        rest_length: Total rest length of the rod [m].
        radius: Cross-section radius [m]. Can be a single value or per-segment array.
        cross_section: Cross-section type for computing area moments.
        tip: Tip shaping configuration for catheter-like behavior.
    """

    num_segments: int = 10
    segment_length: float | None = None
    rest_length: float = 1.0
    radius: float | list[float] = 0.01
    cross_section: Literal["circle", "rectangle"] = "circle"
    tip: RodTipConfig = field(default_factory=RodTipConfig)

    def __post_init__(self):
        """Compute segment length if not provided."""
        if self.segment_length is None:
            self.segment_length = self.rest_length / self.num_segments


@dataclass
class FrictionConfig:
    """Configuration for contact friction model.
    
    Attributes:
        method: Friction method ("none", "coulomb", "viscous", "static_dynamic").
        static_coefficient: Static friction coefficient μ_s.
        dynamic_coefficient: Dynamic friction coefficient μ_d.
        viscous_coefficient: Viscous friction coefficient for velocity-dependent friction.
        stiction_velocity: Velocity threshold for static/dynamic transition [m/s].
    """
    method: Literal["none", "coulomb", "viscous", "static_dynamic"] = "none"
    static_coefficient: float = 0.5
    dynamic_coefficient: float = 0.3
    viscous_coefficient: float = 0.1
    stiction_velocity: float = 0.01


@dataclass
class CollisionMeshConfig:
    """Configuration for collision mesh (e.g., vessel geometry).
    
    Attributes:
        mesh_path: Path to USD/OBJ mesh file for collision geometry.
        use_bvh: Use BVH acceleration for collision queries.
        collision_radius: Additional collision radius for particles.
        restitution: Coefficient of restitution for bouncing.
    """
    mesh_path: str | None = None
    use_bvh: bool = True
    collision_radius: float = 0.001
    restitution: float = 0.0


@dataclass
class RodSolverConfig:
    """Solver configuration for the direct position-based solver.

    Attributes:
        dt: Time step size [s].
        num_substeps: Number of substeps per simulation step.
        newton_iterations: Maximum number of Newton iterations per substep.
        newton_tolerance: Convergence tolerance for Newton iterations.
        use_direct_solver: Use direct solver (True) or Gauss-Seidel (False).
        gravity: Gravity vector [m/s²].
        enable_collisions: Enable collision detection and response.
        collision_margin: Collision detection margin [m].
        friction: Friction model configuration.
        collision_mesh: Collision mesh configuration for vessel/environment.
    """

    dt: float = 1.0 / 60.0
    num_substeps: int = 1
    newton_iterations: int = 4
    newton_tolerance: float = 1e-6
    use_direct_solver: bool = True
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    enable_collisions: bool = False
    collision_margin: float = 0.001
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    collision_mesh: CollisionMeshConfig = field(default_factory=CollisionMeshConfig)


@dataclass
class RodConfig:
    """Complete configuration for a rod simulation.

    Attributes:
        material: Material properties configuration.
        geometry: Geometry configuration.
        solver: Solver configuration.
        device: Device for computation ("cuda" or "cpu").
    """

    material: RodMaterialConfig = field(default_factory=RodMaterialConfig)
    geometry: RodGeometryConfig = field(default_factory=RodGeometryConfig)
    solver: RodSolverConfig = field(default_factory=RodSolverConfig)
    device: str = "cuda"


class RodData:
    """Runtime data for rod simulation.

    This class holds all the state data needed for simulating a rod
    or tree of rods using the direct position-based solver.

    The rod is discretized as a chain of rigid segments connected by
    constraints. Each segment has:
    - Position (center of mass)
    - Orientation (quaternion)
    - Linear velocity
    - Angular velocity
    - Mass and inertia

    Constraints between segments:
    - Zero-stretch constraint (inextensibility)
    - Bending constraint (Cosserat model)
    - Twisting constraint (Cosserat model)

    Attributes:
        config: Rod configuration.
        num_envs: Number of parallel environments.
        num_segments: Number of segments per rod.
        positions: Segment center positions. Shape: (num_envs, num_segments, 3).
        orientations: Segment orientations as quaternions (x, y, z, w).
            Shape: (num_envs, num_segments, 4).
        velocities: Segment linear velocities. Shape: (num_envs, num_segments, 3).
        angular_velocities: Segment angular velocities. Shape: (num_envs, num_segments, 3).
        masses: Segment masses. Shape: (num_envs, num_segments).
        inertias: Segment inertia tensors. Shape: (num_envs, num_segments, 3, 3).
        rest_darboux: Rest Darboux vector (curvature/twist). Shape: (num_envs, num_segments-1, 3).
        segment_lengths: Segment lengths. Shape: (num_envs, num_segments).
        radii: Segment radii. Shape: (num_envs, num_segments).
        bending_stiffness: Bending stiffness per constraint. Shape: (num_envs, num_segments-1, 2).
        torsion_stiffness: Torsion stiffness per constraint. Shape: (num_envs, num_segments-1).
        stretch_compliance: Stretch constraint compliance. Shape: (num_envs, num_segments-1).
        bend_twist_compliance: Bend/twist constraint compliance. Shape: (num_envs, num_segments-1, 3).
        lambda_stretch: Lagrange multipliers for stretch. Shape: (num_envs, num_segments-1).
        lambda_bend_twist: Lagrange multipliers for bend/twist. Shape: (num_envs, num_segments-1, 3).
        fixed_segments: Mask for fixed segments. Shape: (num_envs, num_segments).
        parent_indices: Parent segment index for tree structure. -1 for root.
            Shape: (num_envs, num_segments).
    """

    def __init__(
        self,
        config: RodConfig,
        num_envs: int = 1,
        device: str | None = None,
    ):
        """Initialize rod data.

        Args:
            config: Rod configuration.
            num_envs: Number of parallel environments.
            device: Device for computation. If None, uses config.device.
        """
        self.config = config
        self.num_envs = num_envs
        self.device = device or config.device
        self.num_segments = config.geometry.num_segments

        # Initialize state arrays
        self._init_state()

        # Initialize material properties
        self._init_material()

        # Initialize constraint data
        self._init_constraints()

        # Initialize tree structure (linear chain by default)
        self._init_tree_structure()

        # Create Warp arrays for GPU computation
        self._create_warp_arrays()

    def _init_state(self):
        """Initialize state arrays with default values."""
        n, s = self.num_envs, self.num_segments
        cfg = self.config

        # Segment length
        seg_len = cfg.geometry.segment_length

        # Positions: segments placed along x-axis by default
        self.positions = torch.zeros((n, s, 3), device=self.device, dtype=torch.float32)
        for i in range(s):
            self.positions[:, i, 0] = (i + 0.5) * seg_len

        # Orientations: identity quaternion (x, y, z, w)
        self.orientations = torch.zeros((n, s, 4), device=self.device, dtype=torch.float32)
        self.orientations[:, :, 3] = 1.0  # w = 1 for identity

        # Velocities
        self.velocities = torch.zeros((n, s, 3), device=self.device, dtype=torch.float32)
        self.angular_velocities = torch.zeros((n, s, 3), device=self.device, dtype=torch.float32)

        # Previous positions for velocity update (XPBD)
        self.prev_positions = self.positions.clone()
        self.prev_orientations = self.orientations.clone()

    def _init_material(self):
        """Initialize material property arrays."""
        n, s = self.num_envs, self.num_segments
        cfg = self.config
        mat = cfg.material
        geo = cfg.geometry

        # Get radius (can be scalar or list)
        if isinstance(geo.radius, (int, float)):
            radii = torch.full((n, s), geo.radius, device=self.device, dtype=torch.float32)
        else:
            radii = torch.tensor(geo.radius, device=self.device, dtype=torch.float32)
            radii = radii.unsqueeze(0).expand(n, -1)
        self.radii = radii

        # Segment lengths
        self.segment_lengths = torch.full(
            (n, s), geo.segment_length, device=self.device, dtype=torch.float32
        )

        # Compute cross-section properties
        if geo.cross_section == "circle":
            # Area = π * r²
            area = torch.pi * self.radii**2
            # Second moment of area I = π * r⁴ / 4
            I_xx = torch.pi * self.radii**4 / 4.0
            I_yy = I_xx
            # Polar moment J = π * r⁴ / 2
            J = torch.pi * self.radii**4 / 2.0
        else:
            raise NotImplementedError(f"Cross section {geo.cross_section} not implemented")

        # Masses: ρ * A * L
        self.masses = mat.density * area * self.segment_lengths

        # Inertias (diagonal approximation for cylinder)
        self.inertias = torch.zeros((n, s, 3, 3), device=self.device, dtype=torch.float32)
        for i in range(s):
            m = self.masses[:, i]
            r = self.radii[:, i]
            L = self.segment_lengths[:, i]
            # Cylinder inertia: Ixx = Iyy = m*(3r² + L²)/12, Izz = m*r²/2
            I_trans = m * (3 * r**2 + L**2) / 12.0
            I_axial = m * r**2 / 2.0
            self.inertias[:, i, 0, 0] = I_trans
            self.inertias[:, i, 1, 1] = I_trans
            self.inertias[:, i, 2, 2] = I_axial

        # Bending stiffness: E * I (for x and y directions)
        # Shape: (n, s-1, 2) for constraints between segments
        self.bending_stiffness = torch.zeros((n, s - 1, 2), device=self.device, dtype=torch.float32)
        self.bending_stiffness[:, :, 0] = mat.young_modulus * I_xx[:, :-1]
        self.bending_stiffness[:, :, 1] = mat.young_modulus * I_yy[:, :-1]

        # Torsion stiffness: G * J
        self.torsion_stiffness = mat.shear_modulus * J[:, :-1]

        # Inverse masses and inertias for constraint solving
        self.inv_masses = torch.zeros_like(self.masses)
        nonzero_mass = self.masses > 1e-10
        self.inv_masses[nonzero_mass] = 1.0 / self.masses[nonzero_mass]

        # Inverse inertias (diagonal)
        self.inv_inertias = torch.zeros_like(self.inertias)
        for i in range(3):
            nonzero = self.inertias[:, :, i, i] > 1e-10
            self.inv_inertias[:, :, i, i][nonzero] = 1.0 / self.inertias[:, :, i, i][nonzero]

    def _init_constraints(self):
        """Initialize constraint-related arrays."""
        n, s = self.num_envs, self.num_segments
        cfg = self.config
        mat = cfg.material
        dt = cfg.solver.dt

        # Rest Darboux vector (zero for straight rod, except tip region)
        self.rest_darboux = torch.zeros((n, s - 1, 3), device=self.device, dtype=torch.float32)
        
        # Apply tip shaping to last N segments
        tip_cfg = cfg.geometry.tip
        if tip_cfg.num_tip_segments > 0:
            tip_start = max(0, s - 1 - tip_cfg.num_tip_segments)
            self.rest_darboux[:, tip_start:, 0] = tip_cfg.rest_bend_omega1
            self.rest_darboux[:, tip_start:, 1] = tip_cfg.rest_bend_omega2
            self.rest_darboux[:, tip_start:, 2] = tip_cfg.rest_twist

        # Compliance values (inverse stiffness)
        # α = 1 / (stiffness * dt²) for XPBD
        dt2 = dt**2

        # Stretch compliance - use normalized multiplier
        # stretch_stiffness = 1.0 means nearly inextensible
        base_stretch_stiffness = 1e12
        effective_stretch = base_stretch_stiffness * mat.stretch_stiffness
        self.stretch_compliance = torch.full(
            (n, s - 1), 1.0 / (effective_stretch * dt2 + 1e-10), device=self.device, dtype=torch.float32
        )
        
        # Shear compliance (new!) - resistance to shear deformation
        base_shear_stiffness = mat.shear_modulus * torch.pi * self.radii[:, :-1]**2
        effective_shear = base_shear_stiffness * mat.shear_stiffness
        self.shear_compliance = torch.zeros((n, s - 1, 2), device=self.device, dtype=torch.float32)
        self.shear_compliance[:, :, 0] = 1.0 / (effective_shear * dt2 + 1e-10)
        self.shear_compliance[:, :, 1] = 1.0 / (effective_shear * dt2 + 1e-10)
        
        # Lagrange multipliers for shear
        self.lambda_shear = torch.zeros((n, s - 1, 2), device=self.device, dtype=torch.float32)

        # Bend/twist compliance - use normalized multipliers
        self.bend_twist_compliance = torch.zeros(
            (n, s - 1, 3), device=self.device, dtype=torch.float32
        )
        # Bending compliance (x, y) - scaled by bend_stiffness multiplier
        for i in range(2):
            base_stiffness = self.bending_stiffness[:, :, i]
            effective_stiffness = base_stiffness * mat.bend_stiffness
            self.bend_twist_compliance[:, :, i] = 1.0 / (effective_stiffness * dt2 + 1e-10)
        # Torsion compliance (z) - scaled by twist_stiffness multiplier
        effective_torsion = self.torsion_stiffness * mat.twist_stiffness
        self.bend_twist_compliance[:, :, 2] = 1.0 / (effective_torsion * dt2 + 1e-10)

        # Lagrange multipliers (accumulated over Newton iterations)
        self.lambda_stretch = torch.zeros((n, s - 1), device=self.device, dtype=torch.float32)
        self.lambda_bend_twist = torch.zeros((n, s - 1, 3), device=self.device, dtype=torch.float32)

        # Fixed segment mask
        self.fixed_segments = torch.zeros((n, s), device=self.device, dtype=torch.bool)
        
        # Collision mesh data (initialized if mesh path provided)
        self.collision_mesh_vertices = None
        self.collision_mesh_indices = None
        self.collision_bvh = None

    def _init_tree_structure(self):
        """Initialize tree structure for the direct solver.

        By default, creates a linear chain where segment i's parent is i-1.
        The root segment (index 0) has parent -1.
        """
        n, s = self.num_envs, self.num_segments

        # Parent indices: -1 for root, otherwise i-1
        self.parent_indices = torch.zeros((n, s), device=self.device, dtype=torch.int32)
        self.parent_indices[:, 0] = -1
        for i in range(1, s):
            self.parent_indices[:, i] = i - 1

        # Children lists (for tree traversal)
        # This is computed on-the-fly during solving

    def _create_warp_arrays(self):
        """Create Warp arrays from torch tensors for GPU kernels."""
        # Convert to Warp arrays - these are views when possible
        self.wp_positions = wp.from_torch(self.positions.view(-1, 3), dtype=wp.vec3f)
        self.wp_orientations = wp.from_torch(self.orientations.view(-1, 4), dtype=wp.quatf)
        self.wp_velocities = wp.from_torch(self.velocities.view(-1, 3), dtype=wp.vec3f)
        self.wp_angular_velocities = wp.from_torch(
            self.angular_velocities.view(-1, 3), dtype=wp.vec3f
        )
        self.wp_prev_positions = wp.from_torch(self.prev_positions.view(-1, 3), dtype=wp.vec3f)
        self.wp_prev_orientations = wp.from_torch(
            self.prev_orientations.view(-1, 4), dtype=wp.quatf
        )
        self.wp_masses = wp.from_torch(self.masses.view(-1), dtype=wp.float32)
        self.wp_inv_masses = wp.from_torch(self.inv_masses.view(-1), dtype=wp.float32)
        self.wp_segment_lengths = wp.from_torch(self.segment_lengths.view(-1), dtype=wp.float32)
        self.wp_fixed_segments = wp.from_torch(self.fixed_segments.view(-1), dtype=wp.bool)
        self.wp_lambda_stretch = wp.from_torch(self.lambda_stretch.view(-1), dtype=wp.float32)
        self.wp_lambda_bend_twist = wp.from_torch(
            self.lambda_bend_twist.view(-1, 3), dtype=wp.vec3f
        )
        self.wp_lambda_shear = wp.from_torch(
            self.lambda_shear.view(-1, 2), dtype=wp.vec2f
        )
        self.wp_stretch_compliance = wp.from_torch(
            self.stretch_compliance.view(-1), dtype=wp.float32
        )
        self.wp_shear_compliance = wp.from_torch(
            self.shear_compliance.view(-1, 2), dtype=wp.vec2f
        )
        self.wp_bend_twist_compliance = wp.from_torch(
            self.bend_twist_compliance.view(-1, 3), dtype=wp.vec3f
        )
        self.wp_rest_darboux = wp.from_torch(self.rest_darboux.view(-1, 3), dtype=wp.vec3f)
        self.wp_parent_indices = wp.from_torch(self.parent_indices.view(-1), dtype=wp.int32)

    def sync_from_warp(self):
        """Synchronize data from Warp arrays back to torch tensors."""
        # The arrays are views, so they should be synced automatically
        # This is mainly for ensuring GPU operations are complete
        wp.synchronize()

    def sync_to_warp(self):
        """Synchronize data from torch tensors to Warp arrays."""
        # Recreate Warp arrays if torch tensors were modified
        self._create_warp_arrays()

    def fix_segment(self, env_idx: int | slice, segment_idx: int | slice):
        """Mark segments as fixed (infinite mass).

        Args:
            env_idx: Environment index or slice.
            segment_idx: Segment index or slice.
        """
        self.fixed_segments[env_idx, segment_idx] = True
        self.inv_masses[env_idx, segment_idx] = 0.0
        self.inv_inertias[env_idx, segment_idx] = 0.0

    def reset(self, env_indices: torch.Tensor | None = None):
        """Reset rod state to initial configuration.

        Args:
            env_indices: Indices of environments to reset. If None, reset all.
        """
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)

        cfg = self.config
        seg_len = cfg.geometry.segment_length

        # Reset positions to straight rod along x-axis
        for i in range(self.num_segments):
            self.positions[env_indices, i, 0] = (i + 0.5) * seg_len
            self.positions[env_indices, i, 1] = 0.0
            self.positions[env_indices, i, 2] = 0.0

        # Reset orientations to identity
        self.orientations[env_indices, :, :3] = 0.0
        self.orientations[env_indices, :, 3] = 1.0

        # Reset velocities
        self.velocities[env_indices] = 0.0
        self.angular_velocities[env_indices] = 0.0

        # Reset previous state
        self.prev_positions[env_indices] = self.positions[env_indices]
        self.prev_orientations[env_indices] = self.orientations[env_indices]

        # Reset Lagrange multipliers
        self.lambda_stretch[env_indices] = 0.0
        self.lambda_bend_twist[env_indices] = 0.0
        self.lambda_shear[env_indices] = 0.0

        # Sync to Warp
        self.sync_to_warp()
    
    def set_tip_curvature(
        self, 
        bend_omega1: float = 0.0, 
        bend_omega2: float = 0.0, 
        twist: float = 0.0,
        num_tip_segments: int | None = None
    ):
        """Dynamically update tip rest curvature.
        
        This allows interactive control of the catheter tip shape,
        similar to Numpad +/- in Newton Viewer.
        
        Args:
            bend_omega1: Bending curvature around X axis [rad/m].
            bend_omega2: Bending curvature around Y axis [rad/m].
            twist: Twist around Z axis [rad/m].
            num_tip_segments: Number of tip segments to affect. If None, uses config.
        """
        s = self.num_segments
        tip_count = num_tip_segments or self.config.geometry.tip.num_tip_segments
        tip_start = max(0, s - 1 - tip_count)
        
        self.rest_darboux[:, tip_start:, 0] = bend_omega1
        self.rest_darboux[:, tip_start:, 1] = bend_omega2
        self.rest_darboux[:, tip_start:, 2] = twist
        
        # Update Warp array
        self.wp_rest_darboux = wp.from_torch(self.rest_darboux.view(-1, 3), dtype=wp.vec3f)
    
    def load_collision_mesh(self, mesh_path: str):
        """Load a collision mesh for vessel/environment collision.
        
        Args:
            mesh_path: Path to USD or OBJ mesh file.
        """
        import os
        if not os.path.exists(mesh_path):
            print(f"Warning: Collision mesh not found: {mesh_path}")
            return
            
        # Try to load mesh based on extension
        ext = os.path.splitext(mesh_path)[1].lower()
        
        if ext == ".obj":
            self._load_obj_mesh(mesh_path)
        elif ext in [".usd", ".usda", ".usdc"]:
            self._load_usd_mesh(mesh_path)
        else:
            print(f"Warning: Unsupported mesh format: {ext}")
    
    def _load_obj_mesh(self, mesh_path: str):
        """Load OBJ mesh for collision."""
        vertices = []
        faces = []
        
        with open(mesh_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    # Handle face indices (1-based to 0-based)
                    face = []
                    for p in parts[1:4]:  # Only first 3 vertices (triangles)
                        idx = int(p.split('/')[0]) - 1
                        face.append(idx)
                    faces.append(face)
        
        self.collision_mesh_vertices = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        self.collision_mesh_indices = torch.tensor(faces, device=self.device, dtype=torch.int32)
        
        # Create Warp mesh for BVH queries
        self._create_collision_bvh()
        print(f"Loaded collision mesh: {len(vertices)} vertices, {len(faces)} triangles")
    
    def _load_usd_mesh(self, mesh_path: str):
        """Load USD mesh for collision."""
        try:
            from pxr import Usd, UsdGeom
            stage = Usd.Stage.Open(mesh_path)
            
            # Find first mesh in stage
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    
                    # Convert to triangles if needed
                    vertices = [[p[0], p[1], p[2]] for p in points]
                    faces = []
                    idx = 0
                    for count in counts:
                        if count == 3:
                            faces.append([indices[idx], indices[idx+1], indices[idx+2]])
                        elif count == 4:
                            # Triangulate quad
                            faces.append([indices[idx], indices[idx+1], indices[idx+2]])
                            faces.append([indices[idx], indices[idx+2], indices[idx+3]])
                        idx += count
                    
                    self.collision_mesh_vertices = torch.tensor(vertices, device=self.device, dtype=torch.float32)
                    self.collision_mesh_indices = torch.tensor(faces, device=self.device, dtype=torch.int32)
                    self._create_collision_bvh()
                    print(f"Loaded USD mesh: {len(vertices)} vertices, {len(faces)} triangles")
                    return
                    
            print("Warning: No mesh found in USD file")
        except ImportError:
            print("Warning: pxr (USD) not available, cannot load USD mesh")
    
    def _create_collision_bvh(self):
        """Create Warp BVH for accelerated collision queries."""
        if self.collision_mesh_vertices is None:
            return
            
        # Create Warp mesh with BVH
        vertices_wp = wp.from_torch(self.collision_mesh_vertices, dtype=wp.vec3f)
        indices_wp = wp.from_torch(self.collision_mesh_indices.view(-1), dtype=wp.int32)
        
        self.collision_bvh = wp.Mesh(
            points=vertices_wp,
            indices=indices_wp,
        )
        print("Created BVH for collision mesh")

    def get_endpoint_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get positions of rod endpoints.

        Returns:
            Tuple of (start_positions, end_positions), each with shape (num_envs, 3).
        """
        seg_len = self.config.geometry.segment_length

        # Start point: position of first segment - half segment length in local x
        q0 = self.orientations[:, 0]  # (n, 4)
        local_offset = torch.tensor([[-seg_len / 2, 0, 0]], device=self.device).expand(
            self.num_envs, -1
        )
        start_offset = self._rotate_vector(local_offset, q0)
        start_pos = self.positions[:, 0] + start_offset

        # End point: position of last segment + half segment length in local x
        q_last = self.orientations[:, -1]
        local_offset = torch.tensor([[seg_len / 2, 0, 0]], device=self.device).expand(
            self.num_envs, -1
        )
        end_offset = self._rotate_vector(local_offset, q_last)
        end_pos = self.positions[:, -1] + end_offset

        return start_pos, end_pos

    @staticmethod
    def _rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q (x, y, z, w format)."""
        # Extract quaternion components
        qv = q[:, :3]  # vector part
        qw = q[:, 3:4]  # scalar part

        # Rodrigues rotation formula
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

