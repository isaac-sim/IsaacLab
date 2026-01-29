# Bring Your Own Solver (BYOS) to Newton in Isaac Lab

## Overview

This tutorial explains how to implement your own physics solver in Isaac Lab using the Newton-style architecture. We'll use the existing Rod Solver (for catheters/guidewires) as a reference implementation.

**Reference Paper:** The rod solver implements [Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"](https://animation.rwth-aachen.de/publication/0557/) from Computer Graphics Forum.

## Architecture Overview

```
isaaclab_newton/
├── solvers/
│   ├── __init__.py          # Module exports 
│   ├── rod_data.py          # Data structures & configuration 
│   ├── rod_kernels.py       # Warp GPU kernels 
│   └── rod_solver.py        # Main solver class 
└── examples/
    └── your_example.py      # Usage example
```


## Step 1: Define Configuration Classes

Create dataclasses for your solver's configuration. These define the parameters users can adjust.

**Actual rod_data.py pattern:**

```python
# my_solver_data.py
from dataclasses import dataclass, field
from typing import Literal
import torch
import warp as wp

wp.init()

@dataclass
class MySolverMaterialConfig:
    """Material properties for your physics object.
    
    Based on the Cosserat rod model, using physically meaningful
    parameters plus normalized stiffness multipliers for fine control.
    """
    
    # Physical properties (SI units)
    young_modulus: float = 1e9      # [Pa] - Bending stiffness
    shear_modulus: float | None = None  # [Pa] - Computed if None
    poisson_ratio: float = 0.3
    density: float = 7800.0         # [kg/m³] - Steel density
    damping: float = 0.01           # Velocity damping coefficient
    
    # Normalized stiffness multipliers (0.0 to 1.0, like Newton Viewer)
    stretch_stiffness: float = 1.0   # 1.0 = fully inextensible
    shear_stiffness: float = 1.0     # 1.0 = no shear deformation
    bend_stiffness: float = 0.1      # Lower = more flexible (catheter-like)
    twist_stiffness: float = 0.4     # Moderate torsional resistance
    
    def __post_init__(self):
        """Compute derived properties."""
        if self.shear_modulus is None:
            # G = E / (2 * (1 + ν))
            self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class MySolverTipConfig:
    """Tip shaping configuration (for catheter J-tips, angled tips)."""
    num_tip_segments: int = 20
    rest_bend_omega1: float = 0.0  # Curvature in XZ plane [rad/m]
    rest_bend_omega2: float = 0.0  # Curvature in YZ plane [rad/m]
    rest_twist: float = 0.0        # Axial twist [rad/m]


@dataclass
class MySolverGeometryConfig:
    """Geometry configuration."""
    
    num_segments: int = 10          # Renamed from num_elements
    segment_length: float | None = None  # Computed if None
    rest_length: float = 1.0
    radius: float | list[float] = 0.01  # Per-segment radius supported
    cross_section: Literal["circle", "rectangle"] = "circle"
    tip: MySolverTipConfig = field(default_factory=MySolverTipConfig)

    def __post_init__(self):
        """Compute segment length if not provided."""
        if self.segment_length is None:
            self.segment_length = self.rest_length / self.num_segments


@dataclass
class FrictionConfig:
    """Friction model configuration."""
    method: Literal["none", "coulomb", "viscous", "static_dynamic"] = "none"
    static_coefficient: float = 0.5
    dynamic_coefficient: float = 0.3
    viscous_coefficient: float = 0.1
    stiction_velocity: float = 0.01


@dataclass
class CollisionMeshConfig:
    """Collision mesh configuration (for vessel geometry)."""
    mesh_path: str | None = None
    use_bvh: bool = True            # BVH acceleration for collision queries
    collision_radius: float = 0.001
    restitution: float = 0.0


@dataclass  
class MySolverConfig:
    """Solver algorithm configuration."""
    
    dt: float = 1.0 / 60.0
    num_substeps: int = 1           # Reduced from 4 (direct solver is stable)
    newton_iterations: int = 4      # Renamed from num_iterations
    newton_tolerance: float = 1e-6
    use_direct_solver: bool = True  # O(n) direct solver vs O(n²) Gauss-Seidel
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    enable_collisions: bool = False
    collision_margin: float = 0.001
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    collision_mesh: CollisionMeshConfig = field(default_factory=CollisionMeshConfig)


@dataclass
class MyFullConfig:
    """Complete configuration combining all sub-configs."""
    
    material: MySolverMaterialConfig = field(default_factory=MySolverMaterialConfig)
    geometry: MySolverGeometryConfig = field(default_factory=MySolverGeometryConfig)
    solver: MySolverConfig = field(default_factory=MySolverConfig)
    device: str = "cuda"
```


## Step 2: Create the Data Structure

The data class holds all simulation state (positions, velocities, constraints, etc.).

```python


class MySolverData:
    """Holds all simulation state data."""
    
    def __init__(self, config: MyFullConfig, num_envs: int = 1):
        """Initialize data arrays.
        
        Args:
            config: Complete solver configuration.
            num_envs: Number of parallel environments (for batch simulation).
        """
        self.config = config
        self.num_envs = num_envs
        self.device = config.device
        n = config.geometry.num_elements
        
        # ===========================================
        # PyTorch tensors for main state variables
        # ===========================================
        
        # Positions: [num_envs, num_elements, 3]
        self.positions = torch.zeros(
            (num_envs, n, 3), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Velocities: [num_envs, num_elements, 3]
        self.velocities = torch.zeros(
            (num_envs, n, 3), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Orientations (quaternions): [num_envs, num_elements, 4]
        # Format: [x, y, z, w] where w is scalar
        self.orientations = torch.zeros(
            (num_envs, n, 4), 
            dtype=torch.float32, 
            device=self.device
        )
        self.orientations[:, :, 3] = 1.0  # Identity quaternion
        
        # Angular velocities: [num_envs, num_elements, 3]
        self.angular_velocities = torch.zeros(
            (num_envs, n, 3), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Fixed/constraint flags
        self.fixed_mask = torch.zeros(
            (num_envs, n), 
            dtype=torch.bool, 
            device=self.device
        )
        
        # Inverse masses (0 = fixed)
        self.inv_masses = torch.ones(
            (num_envs, n), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # ===========================================
        # Warp arrays for GPU kernel access
        # ===========================================
        
        # Create Warp arrays that share memory with PyTorch
        self._create_warp_arrays()
        
        # Initialize geometry
        self._initialize_geometry()
    
    def _create_warp_arrays(self):
        """Create Warp arrays from PyTorch tensors."""
        # Warp can wrap PyTorch tensors directly
        self.wp_positions = wp.from_torch(
            self.positions.view(-1, 3), 
            dtype=wp.vec3f
        )
        self.wp_velocities = wp.from_torch(
            self.velocities.view(-1, 3), 
            dtype=wp.vec3f
        )
        self.wp_orientations = wp.from_torch(
            self.orientations.view(-1, 4), 
            dtype=wp.quatf
        )
    
    def _initialize_geometry(self):
        """Set initial positions based on geometry config."""
        n = self.config.geometry.num_elements
        length = self.config.geometry.total_length
        seg_length = length / n
        
        # Initialize along X-axis
        for i in range(n):
            self.positions[:, i, 0] = (i + 0.5) * seg_length
            self.positions[:, i, 1] = 0
            self.positions[:, i, 2] = 0
    
    def sync_to_warp(self):
        """Synchronize PyTorch changes to Warp arrays."""
        # Warp arrays are views, but explicit sync can be needed
        wp.synchronize()
    
    def sync_from_warp(self):
        """Synchronize Warp changes back to PyTorch."""
        wp.synchronize()
    
    def fix_element(self, env_idx: int, element_idx: int):
        """Fix an element in place (make it immovable)."""
        self.fixed_mask[env_idx, element_idx] = True
        self.inv_masses[env_idx, element_idx] = 0.0
    
    def unfix_element(self, env_idx: int, element_idx: int):
        """Unfix an element (make it movable)."""
        self.fixed_mask[env_idx, element_idx] = False
        # Restore mass based on geometry
        self.inv_masses[env_idx, element_idx] = 1.0 / self._compute_mass(element_idx)
    
    def _compute_mass(self, element_idx: int) -> float:
        """Compute mass for a single element."""
        seg_length = self.config.geometry.total_length / self.config.geometry.num_elements
        radius = self.config.geometry.radius
        volume = 3.14159 * radius * radius * seg_length
        return volume * self.config.material.density
```

## Step 3: Implement Warp GPU Kernels

Warp kernels run on the GPU for parallel computation.

**Actual rod_kernels.py pattern:**

```python
# my_solver_kernels.py
import warp as wp

# ============================================================================
# Quaternion Helper Functions 
# ============================================================================

@wp.func
def quat_mul(q1: wp.quatf, q2: wp.quatf) -> wp.quatf:
    """Multiply two quaternions."""
    return q1 * q2


@wp.func
def quat_conjugate(q: wp.quatf) -> wp.quatf:
    """Compute quaternion conjugate."""
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_rotate_vec(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector v by quaternion q."""
    return wp.quat_rotate(q, v)


@wp.func
def quat_to_omega(q: wp.quatf, q_prev: wp.quatf, dt: float) -> wp.vec3f:
    """Compute angular velocity from quaternion change.
    ω = 2 * Im(q * q_prev^(-1)) / dt
    """
    dq = quat_mul(q, quat_conjugate(q_prev))
    return wp.vec3f(2.0 * dq[0] / dt, 2.0 * dq[1] / dt, 2.0 * dq[2] / dt)


@wp.func
def omega_to_quat_delta(omega: wp.vec3f, dt: float) -> wp.quatf:
    """Convert angular velocity to quaternion increment.
    Δq = [sin(|ω|*dt/2) * ω/|ω|, cos(|ω|*dt/2)]
    """
    angle = wp.length(omega) * dt
    if angle < 1e-8:
        # Small angle approximation
        return wp.quatf(omega[0] * dt * 0.5, omega[1] * dt * 0.5, omega[2] * dt * 0.5, 1.0)
    
    half_angle = angle * 0.5
    s = wp.sin(half_angle) / (angle / dt)
    return wp.quatf(s * omega[0], s * omega[1], s * omega[2], wp.cos(half_angle))


# ============================================================================
# Cosserat Rod Model Functions 
# ============================================================================

@wp.func
def compute_darboux_vector(q1: wp.quatf, q2: wp.quatf, segment_length: float) -> wp.vec3f:
    """Compute the Darboux vector (curvature + twist) between two segments.
    
    The Darboux vector Ω represents the angular rate of change of the
    material frame along the rod: Ω = 2 * Im(q1^(-1) * q2) / L
    
    Returns: Darboux vector [bend_x, bend_y, twist_z] in local frame.
    """
    q_rel = quat_mul(quat_conjugate(q1), q2)
    inv_L = 1.0 / segment_length
    return wp.vec3f(2.0 * q_rel[0] * inv_L, 2.0 * q_rel[1] * inv_L, 2.0 * q_rel[2] * inv_L)


# ============================================================================
# XPBD Integration Kernels
# ============================================================================

@wp.kernel
def predict_positions_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    prev_positions: wp.array(dtype=wp.vec3f),  # Store for velocity update
    masses: wp.array(dtype=wp.float32),        # Use mass, not inv_mass
    fixed: wp.array(dtype=wp.bool),            # Boolean fixed flag
    gravity: wp.vec3f,
    dt: float,
    damping: float,
):
    """Predict positions using explicit Euler.
    x̃ = x + dt * v + dt² * f_ext / m
    """
    idx = wp.tid()
    
    if fixed[idx]:
        return
    
    # Store previous position (needed for velocity update)
    prev_positions[idx] = positions[idx]
    
    m = masses[idx]
    if m > 1e-10:
        # Apply damping to velocity BEFORE integration
        v = velocities[idx] * (1.0 - damping)
        positions[idx] = positions[idx] + dt * v + dt * dt * gravity


@wp.kernel
def predict_orientations_kernel(
    orientations: wp.array(dtype=wp.quatf),
    angular_velocities: wp.array(dtype=wp.vec3f),
    prev_orientations: wp.array(dtype=wp.quatf),
    fixed: wp.array(dtype=wp.bool),
    dt: float,
    damping: float,
):
    """Predict orientations using quaternion integration.
    q̃ = q + dt/2 * [ω, 0] * q
    """
    idx = wp.tid()
    
    if fixed[idx]:
        return
    
    prev_orientations[idx] = orientations[idx]
    
    # Apply damping
    omega = angular_velocities[idx] * (1.0 - damping)
    
    # Integrate orientation
    dq = omega_to_quat_delta(omega, dt)
    q_new = quat_mul(dq, orientations[idx])
    
    # Normalize quaternion (CRITICAL for stability)
    q_len = wp.sqrt(q_new[0]*q_new[0] + q_new[1]*q_new[1] + q_new[2]*q_new[2] + q_new[3]*q_new[3])
    if q_len > 1e-8:
        orientations[idx] = wp.quatf(q_new[0]/q_len, q_new[1]/q_len, q_new[2]/q_len, q_new[3]/q_len)


# ============================================================================
# XPBD Constraint Kernels (with Lagrange multiplier tracking)
# ============================================================================

@wp.kernel
def solve_stretch_constraints_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    inv_masses: wp.array(dtype=wp.float32),
    segment_lengths: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),      # α = 1/stiffness
    lambda_stretch: wp.array(dtype=wp.float32),  # Accumulated λ for XPBD
    parent_indices: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    dt: float,
):
    """Solve zero-stretch constraints (inextensibility).
    
    XPBD formulation: (α + ∂C^T M^(-1) ∂C) Δλ = -C - α*λ
    """
    idx = wp.tid()
    segment_idx = idx + 1
    if segment_idx >= num_segments:
        return
    
    parent_idx = parent_indices[segment_idx]
    if parent_idx < 0:
        return
    
    w1 = inv_masses[parent_idx]
    w2 = inv_masses[segment_idx]
    
    if w1 < 1e-10 and w2 < 1e-10:
        return  # Both fixed
    
    # Compute attachment points (end of parent, start of child)
    x1 = positions[parent_idx]
    x2 = positions[segment_idx]
    q1 = orientations[parent_idx]
    q2 = orientations[segment_idx]
    
    L1 = segment_lengths[parent_idx]
    L2 = segment_lengths[segment_idx]
    
    local_end = wp.vec3f(L1 * 0.5, 0.0, 0.0)
    local_start = wp.vec3f(-L2 * 0.5, 0.0, 0.0)
    
    p1 = x1 + quat_rotate_vec(q1, local_end)
    p2 = x2 + quat_rotate_vec(q2, local_start)
    
    diff = p2 - p1
    dist = wp.length(diff)
    
    if dist < 1e-8:
        return
    
    n = diff / dist
    
    # XPBD: Δλ = (-C - α*λ) / (w1 + w2 + α/dt²)
    alpha = compliance[idx]
    w_sum = w1 + w2 + alpha / (dt * dt)
    
    if w_sum < 1e-10:
        return
    
    C = dist
    delta_lambda = (-C - alpha * lambda_stretch[idx]) / w_sum
    lambda_stretch[idx] = lambda_stretch[idx] + delta_lambda
    
    p_corr = delta_lambda * n
    
    if not fixed[parent_idx]:
        positions[parent_idx] = positions[parent_idx] - w1 * p_corr
    if not fixed[segment_idx]:
        positions[segment_idx] = positions[segment_idx] + w2 * p_corr


@wp.kernel
def update_velocities_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    prev_positions: wp.array(dtype=wp.vec3f),
    prev_orientations: wp.array(dtype=wp.quatf),
    velocities: wp.array(dtype=wp.vec3f),
    angular_velocities: wp.array(dtype=wp.vec3f),
    fixed: wp.array(dtype=wp.bool),
    dt: float,
):
    """Update velocities from position/orientation changes.
    v = (x - x_prev) / dt
    ω = 2 * Im(q * q_prev^(-1)) / dt
    """
    idx = wp.tid()
    
    if fixed[idx]:
        velocities[idx] = wp.vec3f(0.0, 0.0, 0.0)
        angular_velocities[idx] = wp.vec3f(0.0, 0.0, 0.0)
        return
    
    inv_dt = 1.0 / dt
    velocities[idx] = (positions[idx] - prev_positions[idx]) * inv_dt
    angular_velocities[idx] = quat_to_omega(orientations[idx], prev_orientations[idx], dt)


@wp.kernel
def normalize_quaternions_kernel(orientations: wp.array(dtype=wp.quatf)):
    """Normalize quaternions to prevent drift."""
    idx = wp.tid()
    q = orientations[idx]
    q_len = wp.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if q_len > 1e-8:
        orientations[idx] = wp.quatf(q[0]/q_len, q[1]/q_len, q[2]/q_len, q[3]/q_len)


@wp.kernel
def reset_lambda_kernel(
    lambda_stretch: wp.array(dtype=wp.float32),
    lambda_bend_twist: wp.array(dtype=wp.vec3f),
):
    """Reset Lagrange multipliers at start of each substep."""
    idx = wp.tid()
    lambda_stretch[idx] = 0.0
    lambda_bend_twist[idx] = wp.vec3f(0.0, 0.0, 0.0)
```

**Key patterns:**
- XPBD uses accumulated Lagrange multipliers (`lambda_stretch`, `lambda_bend_twist`)
- Compliance `α = 1/stiffness` enables time-step independent behavior
- `prev_positions` and `prev_orientations` stored for velocity computation
- Quaternion normalization is a separate kernel (called after all constraints)
- `parent_indices` array enables tree structures (not just linear chains)

## Step 4: Create the Main Solver Class

The solver orchestrates the simulation loop.

**Actual rod_solver.py pattern :**

```python
# my_solver.py
from __future__ import annotations
from typing import Callable
import torch
import warp as wp

from .my_solver_data import MyFullConfig, MySolverData
from .my_solver_kernels import (
    predict_positions_kernel,
    predict_orientations_kernel,
    solve_stretch_constraints_kernel,
    solve_bend_twist_constraints_kernel,
    solve_shear_constraints_kernel,
    solve_ground_collision_kernel,
    solve_mesh_collision_kernel,
    apply_coulomb_friction_kernel,
    update_velocities_kernel,
    normalize_quaternions_kernel,
    reset_lambda_kernel,
)


class DirectTreeSolver:
    """Linear-time direct solver for tree-structured constraints.
    
    Implements Section 4 of Deul et al. 2018.
    For linear chain: tridiagonal system solved with Thomas algorithm.
    """
    
    def __init__(self, num_segments: int, num_envs: int, device: str = "cuda"):
        self.num_segments = num_segments
        self.num_envs = num_envs
        self.num_constraints = num_segments - 1
        self._allocate_temp_arrays(device)
    
    def _allocate_temp_arrays(self, device: str):
        """Allocate arrays for direct solve."""
        n = self.num_envs
        c = self.num_constraints
        
        # Block system matrices
        self.diag_blocks = torch.zeros((n, c, 6, 6), device=device)
        self.off_diag_blocks = torch.zeros((n, c - 1, 6, 6), device=device)
        self.rhs = torch.zeros((n, c, 6), device=device)
        self.delta_lambda = torch.zeros((n, c, 6), device=device)
    
    def solve(self, data, dt: float) -> torch.Tensor:
        """Solve constraint system using direct method."""
        self._build_system(data, dt)
        self._solve_tridiagonal()
        return self.delta_lambda


class MySolver:
    """Direct Position-Based Solver for Stiff Rods.
    
    Implements XPBD with direct solver for improved convergence.
    Based on Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods".
    
    Simulation loop:
    1. Predict positions/orientations (explicit Euler)
    2. Reset Lagrange multipliers
    3. Newton iterations (direct solver or Gauss-Seidel)
    4. Handle collisions
    5. Update velocities
    """
    
    def __init__(
        self, 
        config: MyFullConfig | None = None, 
        num_envs: int = 1,
        device: str | None = None,
    ):
        """Initialize solver.
        
        Args:
            config: Solver configuration. If None, uses defaults.
            num_envs: Number of parallel environments.
            device: Computation device. If None, uses config.device.
        """
        self.config = config or MyFullConfig()
        self.device = device or self.config.device
        self.num_envs = num_envs
        
        # Create data structure
        self.data = MySolverData(self.config, num_envs, self.device)
        
        # Initialize direct solver (optional)
        if self.config.solver.use_direct_solver:
            self.direct_solver = DirectTreeSolver(
                self.config.geometry.num_segments, num_envs, self.device
            )
        else:
            self.direct_solver = None
        
        # Pre-compute constants
        self.gravity = wp.vec3f(*self.config.solver.gravity)
        self.time = 0.0
        
        # External force callback (for RL, teleoperation, etc.)
        self._external_force_callback: Callable[[MySolverData], None] | None = None
    
    def step(self, dt: float | None = None):
        """Advance simulation by one frame.
        
        Args:
            dt: Timestep. If None, uses config.solver.dt.
        """
        dt = dt or self.config.solver.dt
        num_substeps = self.config.solver.num_substeps
        sub_dt = dt / num_substeps
        
        for _ in range(num_substeps):
            self._substep(sub_dt)
        
        self.time += dt
    
    def _substep(self, dt: float):
        """Perform one simulation substep."""
        # Apply external forces if callback set
        if self._external_force_callback is not None:
            self._external_force_callback(self.data)
        
        # Sync PyTorch → Warp
        self.data.sync_to_warp()
        
        num_segments = self.config.geometry.num_segments
        total_segments = self.num_envs * num_segments
        num_constraints = num_segments - 1
        total_constraints = self.num_envs * num_constraints
        
        # Step 1: Predict positions and orientations
        wp.launch(
            kernel=predict_positions_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_velocities,
                self.data.wp_prev_positions,
                self.data.wp_masses,
                self.data.wp_fixed_segments,
                self.gravity,
                dt,
                self.config.material.damping,
            ],
        )
        
        wp.launch(
            kernel=predict_orientations_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_orientations,
                self.data.wp_angular_velocities,
                self.data.wp_prev_orientations,
                self.data.wp_fixed_segments,
                dt,
                self.config.material.damping,
            ],
        )
        
        # Step 2: Reset Lagrange multipliers
        wp.launch(
            kernel=reset_lambda_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_lambda_stretch,
                self.data.wp_lambda_bend_twist,
            ],
        )
        
        # Step 3: Newton iterations
        newton_iters = self.config.solver.newton_iterations
        
        if self.config.solver.use_direct_solver and self.direct_solver:
            for _ in range(newton_iters):
                self._direct_solve_iteration(dt)
        else:
            for _ in range(newton_iters):
                self._gauss_seidel_iteration(dt)
        
        # Step 4: Handle collisions
        if self.config.solver.enable_collisions:
            self._solve_collisions()
        
        # Normalize quaternions (prevent drift)
        wp.launch(
            kernel=normalize_quaternions_kernel,
            dim=total_segments,
            inputs=[self.data.wp_orientations],
        )
        
        # Step 5: Update velocities
        wp.launch(
            kernel=update_velocities_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_prev_positions,
                self.data.wp_prev_orientations,
                self.data.wp_velocities,
                self.data.wp_angular_velocities,
                self.data.wp_fixed_segments,
                dt,
            ],
        )
        
        # Sync Warp → PyTorch
        self.data.sync_from_warp()
    
    def _gauss_seidel_iteration(self, dt: float):
        """Gauss-Seidel constraint projection (one iteration)."""
        num_segments = self.config.geometry.num_segments
        total_constraints = self.num_envs * (num_segments - 1)
        
        # Stretch constraints (inextensibility)
        wp.launch(
            kernel=solve_stretch_constraints_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_inv_masses,
                self.data.wp_segment_lengths,
                self.data.wp_stretch_compliance,
                self.data.wp_lambda_stretch,
                self.data.wp_parent_indices,
                self.data.wp_fixed_segments,
                num_segments,
                dt,
            ],
        )
        
        # Shear constraints (optional)
        if self.config.material.shear_stiffness > 0:
            wp.launch(
                kernel=solve_shear_constraints_kernel,
                dim=total_constraints,
                inputs=[...],  # Similar pattern
            )
        
        # Bend/twist constraints (Cosserat model)
        wp.launch(
            kernel=solve_bend_twist_constraints_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_inv_masses,
                self.data.wp_inv_inertias_diag,
                self.data.wp_segment_lengths,
                self.data.wp_rest_darboux,
                self.data.wp_bend_twist_compliance,
                self.data.wp_lambda_bend_twist,
                self.data.wp_parent_indices,
                self.data.wp_fixed_segments,
                num_segments,
                dt,
            ],
        )
    
    def _direct_solve_iteration(self, dt: float):
        """Direct solve iteration (Newton method)."""
        self.data.sync_from_warp()
        delta_lambda = self.direct_solver.solve(self.data, dt)
        self._apply_corrections(delta_lambda, dt)
        self.data.sync_to_warp()
    
    def _solve_collisions(self):
        """Collision detection and response."""
        num_segments = self.config.geometry.num_segments
        total_segments = self.num_envs * num_segments
        
        # Ground collision
        wp.launch(
            kernel=solve_ground_collision_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_segment_lengths,
                self.data.wp_radii,
                self.data.wp_fixed_segments,
                0.0,  # ground height
                self.config.solver.collision_mesh.restitution,
            ],
        )
        
        # Mesh collision (BVH-accelerated)
        if self.data.collision_bvh is not None:
            wp.launch(
                kernel=solve_mesh_collision_kernel,
                dim=total_segments,
                inputs=[
                    self.data.wp_positions,
                    self.data.wp_velocities,
                    self.data.wp_radii,
                    self.data.wp_fixed_segments,
                    self.data.collision_bvh.id,
                    self._contact_normals,
                    self._contact_depths,
                    self.config.solver.collision_mesh.restitution,
                    self.config.solver.collision_mesh.collision_radius,
                ],
            )
            
            # Apply friction
            if self.config.solver.friction.method == "coulomb":
                wp.launch(
                    kernel=apply_coulomb_friction_kernel,
                    dim=total_segments,
                    inputs=[...],
                )
    
    def set_external_force_callback(self, callback: Callable[[MySolverData], None]):
        """Set callback for applying external forces (RL, teleoperation)."""
        self._external_force_callback = callback
    
    def reset(self, env_indices: torch.Tensor | None = None):
        """Reset simulation state."""
        self.data.reset(env_indices)
        self.time = 0.0
```

**Key patterns :**
- `DirectTreeSolver` for O(n) constraint solving
- External force callback for RL integration
- Sync points at substep boundaries only
- Optional constraint kernels based on stiffness settings
- BVH mesh collision with friction models

## Step 5: Export from __init__.py

**Actual solvers/__init__.py :**

```python
# solvers/__init__.py
"""
Solvers module for position-based dynamics simulations.

Features:
- XPBD (Extended Position-Based Dynamics) framework
- Cosserat rod model for bending and twisting
- Separate stiffness controls (stretch, shear, bend, twist)
- Tip shaping for catheter/guidewire simulation
- BVH-accelerated mesh collision
- Friction models (Coulomb, viscous, static/dynamic)
"""

from .my_solver_data import (
    MyFullConfig,
    MySolverData,
    MySolverMaterialConfig,
    MySolverGeometryConfig,
    MySolverConfig,
    MySolverTipConfig,
    FrictionConfig,
    CollisionMeshConfig,
)
from .my_solver import MySolver

__all__ = [
    # Main classes
    "MySolver",
    "MySolverData",
    
    # Configuration classes
    "MyFullConfig",
    "MySolverMaterialConfig",
    "MySolverGeometryConfig",
    "MySolverConfig",
    "MySolverTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
]
```

**Key pattern:** Export ALL config classes so users can customize every aspect.

## Step 6: Create an Example

```python
# examples/my_solver_example.py
import torch
from isaaclab_newton.solvers import (
    MySolver,
    MyFullConfig,
    MySolverMaterialConfig,
    MySolverGeometryConfig,
    MySolverConfig,
)

def main():
    # Configure solver
    config = MyFullConfig(
        material=MySolverMaterialConfig(
            young_modulus=1e7,
            density=1000.0,
            damping=0.05,
            stiffness_param1=0.8,
        ),
        geometry=MySolverGeometryConfig(
            num_elements=50,
            total_length=1.0,
            radius=0.01,
        ),
        solver=MySolverConfig(
            dt=1/60,
            num_substeps=4,
            num_iterations=6,
            gravity=(0, 0, -9.81),
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create solver
    solver = MySolver(config, num_envs=1)
    
    # Fix the first element
    solver.data.fix_element(0, 0)
    
    # Apply initial velocity to tip
    solver.data.velocities[0, -1, :] = torch.tensor([0, 1.0, 0.5])
    
    # Simulation loop
    for frame in range(1000):
        solver.step()
        
        # Get positions for visualization
        positions = solver.data.positions[0].cpu().numpy()
        
        if frame % 60 == 0:
            tip = positions[-1]
            print(f"Frame {frame}: Tip at ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")


if __name__ == "__main__":
    main()
```

## Key Concepts Summary

### 1. Configuration Pattern (from actual implementation)
- Use `@dataclass` for clean, typed configuration
- Separate concerns: material, geometry, solver, tip, friction, collision
- Use `field(default_factory=...)` for mutable defaults
- Use `__post_init__()` for computed properties
- Use `Literal` types for constrained string options

### 2. Data Structure Pattern
- Store both PyTorch tensors AND Warp arrays
- PyTorch for easy manipulation, ML integration, CPU access
- Warp for GPU kernels (zero-copy via `wp.from_torch()`)
- Store `prev_positions` for velocity computation
- Store `lambda_*` arrays for XPBD accumulated multipliers
- Sync at substep boundaries: `sync_to_warp()` at start, `sync_from_warp()` at end

### 3. Kernel Pattern
- `@wp.func` for helper functions (inlined at compile time)
- `@wp.kernel` for GPU kernels (launched with `wp.launch`)
- Use `wp.tid()` to get global thread ID
- Use `fixed[idx]` check instead of `if inv_mass < epsilon`
- Normalize quaternions explicitly (numerical drift)

### 4. Solver Pattern (XPBD from Deul et al. 2018)
```
for each frame:
    for each substep:
        1. apply_external_forces()   # Callback for RL/teleoperation
        2. sync_to_warp()
        3. predict_positions()       # x̃ = x + dt*v + dt²*g
        4. predict_orientations()    # q̃ = q + dt/2 * [ω,0] * q
        5. reset_lambda()            # λ = 0 for new substep
        6. for newton_iterations:    # 4 iterations typical
              solve_stretch()        # Inextensibility
              solve_shear()          # Optional (if stiffness > 0)
              solve_bend_twist()     # Cosserat bending/twisting
        7. handle_collisions()       # Ground, mesh (BVH), friction
        8. normalize_quaternions()   # Prevent numerical drift
        9. update_velocities()       # v = (x - x_prev) / dt
        10. sync_from_warp()
```

### 5. XPBD Constraint Formulation
```
Standard PBD:    Δx = -λ * ∇C    where λ = C / |∇C|²
XPBD (better):   Δλ = (-C - α*λ) / (w₁ + w₂ + α/dt²)
                 α = compliance = 1/stiffness
                 Δx = w * Δλ * ∇C
```

Benefits of XPBD:
- Time-step independent behavior (α scaled by dt²)
- Better convergence (Newton-like with accumulated λ)
- Separate stiffness from iteration count

### 6. Direct Solver (O(n) for linear chains)
```
For tree-structured constraints:
1. Build block-tridiagonal system: A * Δλ = b
2. Solve with Thomas algorithm (forward elimination + back substitution)
3. Apply corrections: Δx = W * J^T * Δλ

Result: O(n) complexity vs O(n²) for dense systems
```

## Advanced Topics

### Adding Custom Constraints (XPBD Pattern)

```python
# XPBD constraint pattern from actual implementation:
# 1. Compute constraint value C (should be 0 when satisfied)
# 2. Compute compliance α = 1 / stiffness
# 3. Compute XPBD update: Δλ = (-C - α*λ) / (w1 + w2 + α/dt²)
# 4. Update accumulated λ
# 5. Apply position correction: Δx = w * Δλ * ∇C

@wp.kernel
def solve_my_constraint_kernel(
    positions: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_accum: wp.array(dtype=wp.float32),  # Accumulated multiplier
    dt: float,
):
    idx = wp.tid()
    
    # 1. Compute constraint value
    C = compute_my_constraint(positions, idx)
    
    # 2. Get compliance
    alpha = compliance[idx]
    
    # 3. XPBD delta lambda
    w_sum = get_effective_mass(inv_masses, idx) + alpha / (dt * dt)
    delta_lambda = (-C - alpha * lambda_accum[idx]) / w_sum
    
    # 4. Update accumulated lambda
    lambda_accum[idx] = lambda_accum[idx] + delta_lambda
    
    # 5. Apply correction
    apply_correction(positions, delta_lambda, idx)
```

### Mesh Collision with BVH (Actual Pattern)

```python
# From rod_data.py: CollisionMeshConfig
@dataclass
class CollisionMeshConfig:
    mesh_path: str | None = None
    use_bvh: bool = True           # BVH acceleration
    collision_radius: float = 0.001
    restitution: float = 0.0

# From rod_solver.py: Loading collision mesh
def load_collision_mesh(self, mesh_vertices, mesh_indices):
    """Load mesh for BVH-accelerated collision."""
    self.data.collision_bvh = wp.Mesh(
        points=wp.array(mesh_vertices, dtype=wp.vec3f),
        indices=wp.array(mesh_indices.flatten(), dtype=wp.int32),
    )

# From rod_kernels.py: Mesh collision query
@wp.kernel
def solve_mesh_collision_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    mesh: wp.uint64,  # BVH mesh ID
    contact_normals: wp.array(dtype=wp.vec3f),
    contact_depths: wp.array(dtype=wp.float32),
    restitution: float,
    collision_radius: float,
):
    idx = wp.tid()
    if fixed[idx]:
        return
    
    pos = positions[idx]
    radius = radii[idx] + collision_radius
    
    # Query closest point on mesh (O(log n) with BVH)
    face_idx = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    
    result = wp.mesh_query_point(mesh, pos, radius * 2.0, sign, face_idx, face_u, face_v)
    
    if result:
        closest = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)
        normal = wp.normalize(pos - closest)
        depth = radius - wp.length(pos - closest)
        
        if depth > 0.0:
            # Push out of mesh
            positions[idx] = pos + normal * depth
            contact_normals[idx] = normal
            contact_depths[idx] = depth
```

### Parallel Environments

```python
# Array shape convention: [num_envs, num_segments, dimensions]
positions = torch.zeros((num_envs, num_segments, 3), device="cuda")

# Kernel processes all environments in parallel
@wp.kernel
def my_kernel(..., num_segments: int):
    tid = wp.tid()  # Global thread ID across all envs
    
    # Compute env and segment indices
    env_idx = tid // num_segments
    segment_idx = tid % num_segments
    
    # Access element
    pos = positions[tid]  # Flattened view works because contiguous
```

### Direct Solver vs Gauss-Seidel

The rod solver supports two modes:

| Mode | Complexity | Convergence | Use Case |
|------|-----------|-------------|----------|
| **Direct Solver** | O(n) per iteration | Fast (Newton) | Stiff rods, few iterations needed |
| **Gauss-Seidel** | O(n) per iteration | Slow (linear) | Soft rods, more iterations |

```python
# From rod_solver.py
@dataclass
class RodSolverConfig:
    use_direct_solver: bool = True   # O(n) direct solver
    newton_iterations: int = 4       # Fewer iterations needed

# Direct solver exploits tree structure of constraints
# For a linear chain: tridiagonal system solved with Thomas algorithm
```

---

## Optimization 

### 1. ✅ Reduce `wp.from_torch()` Calls

**Problem:** `_gauss_seidel_iteration()` calls `wp.from_torch()` repeatedly.

**Solution (now in rod_solver.py):**

```python
def __init__(self, config, num_envs):
    ...
    # Performance optimization: Cache Warp arrays to avoid repeated creation
    self._cached_wp_inv_inertias_diag = None
    self._cached_wp_radii = None
    self._cache_dirty = True  # Flag to invalidate cache when data changes
    
    # Pre-compute constraint indices for vectorized operations
    num_constraints = self.config.geometry.num_segments - 1
    self._parent_indices_vec = torch.arange(num_constraints, device=self.device)
    self._child_indices_vec = self._parent_indices_vec + 1

def _update_warp_cache(self):
    """Update cached Warp arrays if data has changed."""
    if not self._cache_dirty:
        return
        
    inv_inertia_diag = torch.diagonal(
        self.data.inv_inertias, dim1=-2, dim2=-1
    ).contiguous()
    self._cached_wp_inv_inertias_diag = wp.from_torch(
        inv_inertia_diag.flatten(), dtype=wp.float32
    )
    self._cache_dirty = False

def invalidate_cache(self):
    """Mark cache as dirty, forcing update on next use."""
    self._cache_dirty = True
```

### 2. Reduce Synchronization

**Problem:** `sync_to_warp()` and `sync_from_warp()` called frequently:

```python
# Current pattern
def _substep(self, dt):
    self.data.sync_to_warp()    # Sync 1
    # ... kernels ...
    self.data.sync_from_warp()  # Sync 2

def _direct_solve_iteration(self, dt):
    self.data.sync_from_warp()  # Sync 3
    # ... PyTorch ops ...
    self.data.sync_to_warp()    # Sync 4
```

**Optimization:** Batch syncs at substep boundaries only:

```python
def _substep(self, dt):
    # One sync at start
    self.data.sync_to_warp()
    
    # All kernels
    self._predict()
    for _ in range(self.config.solver.newton_iterations):
        self._solve_constraints(dt)
    self._update_velocities(dt)
    
    # One sync at end
    self.data.sync_from_warp()
```

### 3. ✅ Vectorize Python Loops

**Problem:** `_apply_corrections()` used Python loop.

**Solution (now in rod_solver.py):**

```python
def _apply_corrections(self, delta_lambda: torch.Tensor, dt: float):
    """Apply position and orientation corrections from delta lambda.
    
    OPTIMIZED: Uses vectorized PyTorch operations instead of Python loops.
    """
    # Use pre-computed indices for vectorized operations
    parent_idx = self._parent_indices_vec  # [0, 1, 2, ..., n-2]
    child_idx = self._child_indices_vec    # [1, 2, 3, ..., n-1]

    # Extract stretch and bend/twist for ALL constraints at once
    d_lambda_stretch = delta_lambda[:, :, :3]  # (num_envs, num_constraints, 3)
    d_lambda_bend = delta_lambda[:, :, 3:]     # (num_envs, num_constraints, 3)

    # Vectorized position corrections
    w1 = self.data.inv_masses[:, parent_idx].unsqueeze(-1)
    w2 = self.data.inv_masses[:, child_idx].unsqueeze(-1)

    self.data.positions[:, parent_idx] -= w1 * d_lambda_stretch
    self.data.positions[:, child_idx] += w2 * d_lambda_stretch

    # Vectorized orientation corrections using batched quaternion ops
    corr1_world = self._quat_rotate_batch(q1, corr1)
    dq1 = self._omega_to_quat_batch(-corr1_world)
    self.data.orientations[:, parent_idx] = self._quat_multiply_batch(dq1, q1)
```

**Batched quaternion operations added:**

```python
@staticmethod
def _quat_rotate_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q (batched)."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)

@staticmethod
def _quat_multiply_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions (batched)."""
    # ... vectorized implementation
```

### 4. ✅ Contact Stiffness Parameters

**Added to CollisionMeshConfig (rod_data.py):**

```python
@dataclass
class CollisionMeshConfig:
    """Configuration for collision mesh (e.g., vessel geometry)."""
    mesh_path: str | None = None
    use_bvh: bool = True
    collision_radius: float = 0.001
    restitution: float = 0.0
    contact_stiffness: float = 1e4   # N/m - stiff contacts by default
    contact_damping: float = 100.0   # N·s/m - moderate damping
```

**Benefits:**
- Configurable contact response stiffness
- Separate damping for velocity correction
- Prevents jittering in stiff contact scenarios

### 4b. Optional Constraint Kernels (pattern)

**Good pattern from rod_solver.py:** Shear constraints are optional:

```python
if self.config.material.shear_stiffness > 0:
    wp.launch(solve_shear_constraints_kernel, ...)
```

**Apply to other constraints:** Skip kernels when stiffness is 0

### 5. ✅ Fuse Constraint Kernels

**Problem:** Separate kernels for stretch and bend/twist constraints cause:
- Two GPU kernel launches per iteration
- Memory read/write of positions and orientations twice
- Synchronization overhead between launches

**Solution (now in rod_kernels.py):**

```python
@wp.kernel
def solve_stretch_bend_fused_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertias_diag: wp.array(dtype=wp.vec3f),
    segment_lengths: wp.array(dtype=wp.float32),
    stretch_compliance: wp.array(dtype=wp.float32),
    bend_twist_compliance: wp.array(dtype=wp.vec3f),
    rest_darboux: wp.array(dtype=wp.vec3f),
    lambda_stretch: wp.array(dtype=wp.float32),
    lambda_bend_twist: wp.array(dtype=wp.vec3f),
    parent_indices: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    dt: float,
):
    """Fused kernel for solving both stretch and bend/twist constraints.
    
    PERFORMANCE BENEFITS:
    - Single kernel launch instead of two
    - Better memory locality (positions/orientations read once)
    - Reduced GPU synchronization overhead
    """
    idx = wp.tid()
    # ... PART 1: Stretch constraint ...
    # ... PART 2: Bend/twist constraint ...
```

**Enable in config:**

```python
from isaaclab_newton.solvers import RodConfig, RodSolverConfig

config = RodConfig(
    solver=RodSolverConfig(
        use_fused_kernel=True,  # Enable fused constraint kernel
    )
)
```

---

## Performance Summary

| Optimization | Status | Impact |
|--------------|--------|--------|
| Warp array caching | ✅ | Reduced `wp.from_torch()` overhead |
| Vectorized corrections | ✅ | Eliminated Python loop in `_apply_corrections()` |
| Fused constraint kernel | ✅ | Single kernel for stretch+bend (optional) |
| Contact stiffness params | ✅ | Configurable contact response |
| Batched quaternion ops | ✅ | Vectorized rotation operations |

---

## Next Steps

1. **Start simple:** Particle system with distance constraints
2. **Add XPBD:** Use compliance and accumulated λ
3. **Add rotations:** Quaternion integration + bending constraints
4. **Add collision:** Ground plane → BVH mesh
5. **Add friction:** Coulomb or viscous
6. **Optimize:** Cache Warp arrays, reduce syncs, vectorize
7. **Integrate:** Connect to Isaac Sim visualization

