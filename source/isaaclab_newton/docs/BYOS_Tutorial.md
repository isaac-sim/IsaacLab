# Bring Your Own Solver (BYOS) to Newton in Isaac Lab

## Overview

This tutorial explains how to implement your own physics solver in Isaac Lab using the Newton-style architecture. We'll use the existing Rod Solver (for catheters/guidewires) as a reference implementation.

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

```python
# my_solver_data.py
from dataclasses import dataclass, field
import torch
import warp as wp

wp.init()

@dataclass
class MySolverMaterialConfig:
    """Material properties for your physics object."""
    
    # Physical properties
    young_modulus: float = 1e9      # [Pa]
    density: float = 1000.0          # [kg/m³]
    damping: float = 0.01            # Velocity damping
    
    # Normalized stiffness multipliers (0.0 to 1.0)
    stiffness_param1: float = 1.0
    stiffness_param2: float = 0.5
    
    def __post_init__(self):
        """Validate or compute derived properties."""
        assert 0.0 <= self.stiffness_param1 <= 1.0


@dataclass
class MySolverGeometryConfig:
    """Geometry configuration."""
    
    num_elements: int = 100
    total_length: float = 1.0
    radius: float = 0.01


@dataclass  
class MySolverConfig:
    """Solver algorithm configuration."""
    
    dt: float = 1/60                 # Timestep
    num_substeps: int = 4            # Substeps per frame
    num_iterations: int = 6          # Solver iterations
    gravity: tuple = (0, 0, -9.81)   # Gravity vector
    device: str = "cuda"


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
# my_solver_data.py (continued)

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

```python
# my_solver_kernels.py
import warp as wp

# ===========================================
# Helper Functions (must be @wp.func)
# ===========================================

@wp.func
def safe_normalize(v: wp.vec3f) -> wp.vec3f:
    """Safely normalize a vector."""
    length = wp.length(v)
    if length > 1e-8:
        return v / length
    return wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def quat_rotate_vec(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector by quaternion."""
    return wp.quat_rotate(q, v)


# ===========================================
# Simulation Kernels (must be @wp.kernel)
# ===========================================

@wp.kernel
def predict_positions_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    predicted_positions: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    gravity: wp.vec3f,
    dt: wp.float32,
    num_elements: wp.int32,
):
    """Predict positions using semi-implicit Euler.
    
    This is the first step of Position-Based Dynamics:
    1. Apply external forces (gravity)
    2. Integrate velocities to get predicted positions
    """
    tid = wp.tid()  # Thread ID
    
    # Bounds check
    if tid >= num_elements:
        return
    
    # Get current state
    pos = positions[tid]
    vel = velocities[tid]
    inv_mass = inv_masses[tid]
    
    # Skip fixed elements
    if inv_mass < 1e-8:
        predicted_positions[tid] = pos
        return
    
    # Apply gravity
    vel = vel + gravity * dt
    
    # Predict new position
    predicted_positions[tid] = pos + vel * dt


@wp.kernel
def solve_distance_constraint_kernel(
    positions: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    rest_lengths: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    num_constraints: wp.int32,
):
    """Solve distance constraints between adjacent elements.
    
    This is a Position-Based Dynamics constraint:
    C(x1, x2) = |x2 - x1| - L₀ = 0
    """
    tid = wp.tid()
    
    if tid >= num_constraints:
        return
    
    # Indices of connected elements
    i = tid
    j = tid + 1
    
    # Get positions and masses
    p1 = positions[i]
    p2 = positions[j]
    w1 = inv_masses[i]
    w2 = inv_masses[j]
    
    # Compute constraint
    diff = p2 - p1
    dist = wp.length(diff)
    rest_length = rest_lengths[tid]
    
    # Constraint value (should be 0 when satisfied)
    C = dist - rest_length
    
    # Skip if no correction needed or both fixed
    w_sum = w1 + w2
    if wp.abs(C) < 1e-8 or w_sum < 1e-8:
        return
    
    # Compute correction
    direction = safe_normalize(diff)
    delta = stiffness * C / w_sum
    
    # Apply corrections (position update)
    positions[i] = p1 + w1 * delta * direction
    positions[j] = p2 - w2 * delta * direction


@wp.kernel
def update_velocities_kernel(
    positions: wp.array(dtype=wp.vec3f),
    old_positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    damping: wp.float32,
    dt: wp.float32,
    num_elements: wp.int32,
):
    """Update velocities from position changes.
    
    v = (x_new - x_old) / dt
    """
    tid = wp.tid()
    
    if tid >= num_elements:
        return
    
    # Compute velocity from position change
    new_vel = (positions[tid] - old_positions[tid]) / dt
    
    # Apply damping
    new_vel = new_vel * (1.0 - damping)
    
    velocities[tid] = new_vel


@wp.kernel
def apply_ground_collision_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    ground_height: wp.float32,
    restitution: wp.float32,
    num_elements: wp.int32,
):
    """Simple ground plane collision."""
    tid = wp.tid()
    
    if tid >= num_elements:
        return
    
    pos = positions[tid]
    
    if pos[2] < ground_height:
        # Project onto ground
        positions[tid] = wp.vec3f(pos[0], pos[1], ground_height)
        
        # Reflect velocity with restitution
        vel = velocities[tid]
        if vel[2] < 0.0:
            velocities[tid] = wp.vec3f(vel[0], vel[1], -vel[2] * restitution)
```

## Step 4: Create the Main Solver Class

The solver orchestrates the simulation loop.

```python
# my_solver.py
import torch
import warp as wp

from .my_solver_data import MyFullConfig, MySolverData
from .my_solver_kernels import (
    predict_positions_kernel,
    solve_distance_constraint_kernel,
    update_velocities_kernel,
    apply_ground_collision_kernel,
)


class MySolver:
    """Main solver class.
    
    This implements the Position-Based Dynamics algorithm:
    1. Predict positions from velocities
    2. Solve constraints iteratively
    3. Update velocities from position changes
    """
    
    def __init__(self, config: MyFullConfig, num_envs: int = 1):
        """Initialize solver.
        
        Args:
            config: Complete solver configuration.
            num_envs: Number of parallel environments.
        """
        self.config = config
        self.num_envs = num_envs
        
        # Create data structure
        self.data = MySolverData(config, num_envs)
        
        # Allocate temporary arrays
        self._allocate_temp_arrays()
        
        # Precompute constants
        self._setup_constants()
    
    def _allocate_temp_arrays(self):
        """Allocate temporary arrays for solver."""
        n = self.config.geometry.num_elements
        
        # Old positions for velocity update
        self.old_positions = torch.zeros_like(self.data.positions)
        
        # Predicted positions
        self.predicted_positions = torch.zeros_like(self.data.positions)
        
        # Rest lengths between elements
        seg_length = self.config.geometry.total_length / n
        self.rest_lengths = torch.full(
            (self.num_envs, n - 1), 
            seg_length,
            dtype=torch.float32,
            device=self.config.device
        )
        
        # Create Warp arrays
        self.wp_old_positions = wp.from_torch(
            self.old_positions.view(-1, 3), dtype=wp.vec3f
        )
        self.wp_predicted = wp.from_torch(
            self.predicted_positions.view(-1, 3), dtype=wp.vec3f
        )
        self.wp_rest_lengths = wp.from_torch(
            self.rest_lengths.view(-1), dtype=wp.float32
        )
    
    def _setup_constants(self):
        """Precompute solver constants."""
        self.gravity = wp.vec3f(*self.config.solver.gravity)
        self.dt_substep = self.config.solver.dt / self.config.solver.num_substeps
        self.stiffness = self.config.material.stiffness_param1
        self.damping = self.config.material.damping
    
    def step(self, dt: float = None):
        """Advance simulation by one frame.
        
        Args:
            dt: Timestep. If None, uses config.solver.dt.
        """
        if dt is None:
            dt = self.config.solver.dt
        
        dt_substep = dt / self.config.solver.num_substeps
        n = self.config.geometry.num_elements
        
        for _ in range(self.config.solver.num_substeps):
            self._substep(dt_substep)
    
    def _substep(self, dt: float):
        """Perform one substep of the simulation."""
        n = self.config.geometry.num_elements
        
        # 1. Save old positions
        self.old_positions.copy_(self.data.positions)
        
        # 2. Predict positions (apply gravity, integrate velocities)
        wp.launch(
            kernel=predict_positions_kernel,
            dim=n * self.num_envs,
            inputs=[
                self.data.wp_positions,
                self.data.wp_velocities,
                self.wp_predicted,
                wp.from_torch(self.data.inv_masses.view(-1), dtype=wp.float32),
                self.gravity,
                dt,
                n,
            ],
        )
        
        # Copy predicted to positions
        self.data.positions.copy_(self.predicted_positions)
        
        # 3. Solve constraints iteratively
        for _ in range(self.config.solver.num_iterations):
            self._solve_constraints(dt)
        
        # 4. Handle collisions
        self._handle_collisions()
        
        # 5. Update velocities from position changes
        wp.launch(
            kernel=update_velocities_kernel,
            dim=n * self.num_envs,
            inputs=[
                self.data.wp_positions,
                self.wp_old_positions,
                self.data.wp_velocities,
                self.damping,
                dt,
                n,
            ],
        )
    
    def _solve_constraints(self, dt: float):
        """Solve all constraints once."""
        n = self.config.geometry.num_elements
        num_constraints = n - 1
        
        # Distance constraints (keep elements at rest length)
        wp.launch(
            kernel=solve_distance_constraint_kernel,
            dim=num_constraints * self.num_envs,
            inputs=[
                self.data.wp_positions,
                wp.from_torch(self.data.inv_masses.view(-1), dtype=wp.float32),
                self.wp_rest_lengths,
                self.stiffness,
                num_constraints,
            ],
        )
    
    def _handle_collisions(self):
        """Handle collision detection and response."""
        n = self.config.geometry.num_elements
        
        # Ground collision
        wp.launch(
            kernel=apply_ground_collision_kernel,
            dim=n * self.num_envs,
            inputs=[
                self.data.wp_positions,
                self.data.wp_velocities,
                0.0,   # ground height
                0.5,   # restitution
                n,
            ],
        )
    
    def reset(self):
        """Reset simulation to initial state."""
        self.data._initialize_geometry()
        self.data.velocities.zero_()
        self.data.angular_velocities.zero_()
```

## Step 5: Export from __init__.py

```python
# solvers/__init__.py
from .my_solver_data import (
    MyFullConfig,
    MySolverData,
    MySolverMaterialConfig,
    MySolverGeometryConfig,
    MySolverConfig,
)
from .my_solver import MySolver

__all__ = [
    "MySolver",
    "MyFullConfig",
    "MySolverData",
    "MySolverMaterialConfig",
    "MySolverGeometryConfig", 
    "MySolverConfig",
]
```

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

### 1. Configuration Pattern
- Use `@dataclass` for clean, typed configuration
- Separate concerns: material, geometry, solver settings
- Use `field(default_factory=...)` for mutable defaults

### 2. Data Structure Pattern
- Store both PyTorch tensors AND Warp arrays
- PyTorch for easy manipulation, Warp for GPU kernels
- Use `wp.from_torch()` to create Warp views of PyTorch data
- Always call `sync_to_warp()` after PyTorch modifications

### 3. Kernel Pattern
- `@wp.func` for helper functions (can be called from kernels)
- `@wp.kernel` for GPU kernels (launched with `wp.launch`)
- Use `wp.tid()` to get thread ID
- Always bounds-check with element count

### 4. Solver Pattern (Position-Based Dynamics)
```
for each frame:
    for each substep:
        1. save_old_positions()
        2. predict_positions()      # Apply forces, integrate
        3. for iterations:
              solve_constraints()   # Project to satisfy constraints
        4. handle_collisions()      # Collision detection/response
        5. update_velocities()      # v = (x_new - x_old) / dt
```

### 5. Constraint Formulation
- Constraint: `C(x) = 0` when satisfied
- Gradient: `∇C` points in direction of constraint violation
- Correction: `Δx = -λ * ∇C` where `λ = C / |∇C|²`
- Stiffness: Scale `λ` to control constraint strength

## Advanced Topics

### Adding Custom Constraints
1. Define constraint function `C(x)`
2. Compute gradient `∇C`
3. Apply position correction `Δx = -w * λ * ∇C`
4. Handle mass-weighted corrections

### Mesh Collision
1. Build BVH (Bounding Volume Hierarchy)
2. Query nearby triangles per particle
3. Project particles outside mesh
4. Apply friction response

### Parallel Environments
- All arrays have shape `[num_envs, num_elements, ...]`
- Kernels process all environments in parallel
- Use `env_idx = tid // num_elements` to get environment

## Reference: Rod Solver Files

| File | Purpose |
|------|---------|
| `rod_data.py` | Configuration + data structures |
| `rod_kernels.py` | Warp GPU kernels |
| `rod_solver.py` | Main solver class |
| `__init__.py` | Module exports |

## Next Steps

1. Start with a simple particle system
2. Add distance constraints
3. Add your custom physics (bending, twisting, etc.)
4. Add collision detection
5. Integrate with Isaac Sim visualization

