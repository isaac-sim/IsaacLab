# Newton manager configuration and API reference

Read the [design and complete lifecycle examples](newton-manager-design.md) first.
This reference inventories the configuration fields and public call surfaces used
by the new design: the simulation context, manager, functional backend, solver
adapter contract, scheduling objects, construction and scene-data bridges. It also
includes shipped solver configurations and the optional coupling configurations.
Private implementation helpers are outside the extension API.

Navigate: [configuration fields](#configuration-field-reference),
[runtime records](#runtime-records-and-solver-capabilities),
[functions and parameters](#function-reference), [usage recipes](#usage-recipes).

## Imports and reading the examples

- Core Newton types: `from isaaclab_newton.physics import ...`.
- Functional operations: `from isaaclab_newton.physics import newton_backend as nb`.
- Context: `from isaaclab.sim import SimulationCfg, SimulationContext, BackendCfg`.
- Shared interfaces: `from isaaclab.physics import PhysicsCfg, PhysicsEvent`.
- Coupling configs: `isaaclab_contrib.coupling.coupler_cfg`; the specialized
  `CoupledMJWarpVBDSolverCfg` lives in `isaaclab_contrib.custom_coupling.newton_manager_cfg`.

Tables show call shapes, not a sequence to execute from top to bottom. `sim` is an
initialized context; `manager = sim.physics_manager`; `backend = manager.backend`;
`cfg` is a `NewtonCfg`; `adapter = cfg.solver_cfg.class_type`. Objects such as
`view`, `builder`, `controller`, `world_mask`, `batches` and `query` are application
inputs described by their parameter names and the recipes below. Source links give
full type annotations, return contracts and solver-specific constraints.

For application code, start with `SimulationContext`, `NewtonCfg` and the normal
scene APIs. Use manager callbacks for extra work, the functional layer for custom
runners, and adapter hooks for new solver implementations. Registry factories,
clone-record setters and scene-publication hooks are called by integration code;
they are documented for downstream implementers rather than required in every loop.

## Configuration composition

These are construction examples, not recommended tuning values for every task.
The nested Kamino and coupling objects illustrate where each configuration fits;
those solvers may require their optional runtime dependencies and compatible assets.

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg, HydroelasticSDFCfg, KaminoCollisionDetectorCfg,
    KaminoConstraintsCfg, KaminoDVICfg, KaminoDVISolverCfg, KaminoDynamicsCfg,
    KaminoFKCfg, KaminoMaterialsCfg, KaminoPADMMCfg, KaminoPADMMSolverCfg,
    MJWarpSolverCfg, MPMSolverCfg, NewtonBackendCfg, NewtonBuilderCfg, NewtonCfg,
    NewtonCollisionPipelineCfg, NewtonShapeCfg, NewtonSoftContactCfg,
    VBDSolverCfg, XPBDSolverCfg,
)
from isaaclab_contrib.coupling.coupler_cfg import (
    CouplerAdmmCfg, CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg,
)
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg

physics = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(iterations=20, use_mujoco_contacts=False),
    num_substeps=2,
    collision_decimation=1,
    use_cuda_graph=True,
    default_shape_cfg=NewtonShapeCfg(mu=0.8, gap=0.01),
    collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
)
sim_cfg = SimulationCfg(dt=0.01, device="cuda:0", physics=physics)

kamino = KaminoPADMMSolverCfg(
    dynamics_solver_cfg=KaminoPADMMCfg(max_iterations=100),
    dynamics=KaminoDynamicsCfg(),
    constraints=KaminoConstraintsCfg(),
    fk=KaminoFKCfg(),
    collision_detector=KaminoCollisionDetectorCfg(),
    materials=KaminoMaterialsCfg(),
)
kamino_physics = NewtonCfg(solver_cfg=kamino)

# Application selectors must match the actual model; entries own disjoint parts.
entries = [
    CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg(), bodies=["Robot/.*"]),
    CouplerEntryCfg(name="soft", solver_cfg=VBDSolverCfg(), all_particles=True),
]
proxies = [
    CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=["Robot/.*"]),
]
coupled_physics = NewtonCfg(solver_cfg=CouplerProxyCfg(entries=entries, proxies=proxies))
```

For MJWarp, an explicit `collision_cfg` requires `use_mujoco_contacts=False`.
Kamino uses its internal detector when `use_collision_detector=True`, and MPM
handles collision internally; do not assume every solver consumes `collision_cfg`.
Hydroelastic settings additionally require compatible SDF geometry. Determinism and
heterogeneous-world support are checked against the selected adapter's capabilities.
`solver_type` is legacy metadata; dispatch uses `class_type`.

## Configuration field reference

Defaults below are the declared construction expressions. `None` may request automatic
resolution, as described in the meaning column and source link. `{DIR}` factory strings
resolve during configuration initialization. Required fields have no default. Set fields
before registration; use a new descriptor for different registry settings.

Inherited fields are listed under the named base class; subclass rows override them.
`launcher_type` is class metadata, not a per-instance constructor setting. Examples assume
the imports in the configuration recipe above. The shared Kamino base is internal; instantiate
its PADMM or DVI subclass.

### SimulationCfg

[SimulationCfg](../../isaaclab/isaaclab/sim/simulation_cfg.py#L36). Configuration for simulation physics.

```python
sim_cfg = SimulationCfg(dt=0.01, device="cuda:0", physics=NewtonCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `device` | `str` | `'cuda:0'` | The device to run the simulation on. Default is `"cuda:0"`. |
| `dt` | `float` | `1.0 / 60.0` | The physics simulation time-step (in seconds). Default is 0.0167 seconds. |
| `gravity` | `tuple[float, float, float]` | `(0.0, 0.0, -9.81)` | The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81). |
| `physics_prim_path` | `str` | `'/physicsScene'` | The prim path where the USD PhysicsScene is created. Default is "/physicsScene". |
| `physics_material` | `RigidBodyMaterialBaseCfg` | `RigidBodyMaterialBaseCfg()` | Default physics material settings for rigid bodies. Default is RigidBodyMaterialBaseCfg. |
| `use_fabric` | `bool` | `True` | Enable/disable reading of physics buffers directly. Default is True. |
| `render_interval` | `int` | `1` | The number of physics simulation steps per rendering step. Default is 1. |
| `enable_scene_query_support` | `bool` | `False` | Enable/disable scene query support for collision shapes. Default is False. |
| `use_newton_actuators` | `bool` | `True` | Use native actuators for supported explicit actuator configurations. Default is True. |
| `physics` | `PhysicsCfg \| None` | `None` | Physics manager configuration. Default is None (uses PhysxCfg()). |
| `create_stage_in_memory` | `bool` | `False` | If stage is first created in memory. Default is False. |
| `logging_level` | `Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']` | `'WARNING'` | The logging level. Default is "WARNING". |
| `save_logs_to_file` | `bool` | `True` | Save logs to a file. Default is True. |
| `log_dir` | `str \| None` | `None` | The directory to save the logs to. Default is None. |
| `visualizer_cfgs` | `list[VisualizerCfg] \| VisualizerCfg` | `[]` | The visualizer configuration(s). Default is an empty list. |
| `default_visualizer_cfg` | `VisualizerCfg \| None` | `None` | Default visualizer settings applied to any visualizer that is selected at runtime, e.g. with `--visualizer`. |

### PhysicsCfg

[PhysicsCfg](../../isaaclab/isaaclab/physics/physics_manager_cfg.py#L23). Abstract base configuration for physics managers.

```python
# Base class: use NewtonCfg for Newton physics.
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[PhysicsManager] \| Any` | `required` | The physics manager class to use. Must be set by subclasses. |
| `launcher_type` | `ClassVar[str \| None]` | `None` | The launcher that starts the runtime this backend runs in, as `"module:Class"`, or None if none is needed. |
| `deterministic` | `bool` | `False` | Whether to request reproducible physics from the backend. Defaults to False. |

### BackendCfg

[BackendCfg](../../isaaclab/isaaclab/sim/simulation_cfg.py#L25). Declarative settings and value identity for a simulation-owned resource.

```python
# Abstract registry contract: use NewtonBackendCfg below.
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `Callable[[BackendCfg], Any]` | `required` | Constructor called through `instantiate(cfg)`; the returned resource must implement `close()`. |

### NewtonCfg

[NewtonCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L153). Configuration for Newton physics manager. Inherits `PhysicsCfg`.

```python
cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=20), num_substeps=2)
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonManager] \| str` | `'{DIR}.newton_manager:NewtonManager'` | Manager factory called without arguments by SimulationContext, unless an instance is injected. |
| `num_substeps` | `int` | `1` | Number of substeps to use for the solver. |
| `collision_decimation` | `int` | `0` | Re-collide every N solver substeps within a physics tick (`0` = once per tick). |
| `debug_mode` | `bool` | `False` | Whether to enable debug mode for the solver. |
| `use_cuda_graph` | `bool` | `True` | Whether to use CUDA graphing when simulating. |
| `deterministic_mode` | `Literal['not_guaranteed', 'run_to_run', 'gpu_to_gpu']` | `'not_guaranteed'` | Determinism guarantee applied to the Newton solver and collision pipeline. |
| `solver_cfg` | `NewtonSolverCfg \| None` | `None` | Solver configuration. If None (default), MJWarpSolverCfg is used by default. |
| `soft_contact_cfg` | `NewtonSoftContactCfg \| None` | `None` | Global soft-contact parameters applied after model finalization. |
| `collision_cfg` | `NewtonCollisionPipelineCfg \| None` | `None` | Newton collision pipeline configuration. |
| `default_shape_cfg` | `NewtonShapeCfg` | `NewtonShapeCfg()` | Default per-shape collision properties applied to every shape in the scene. |
| `load_visual_shapes` | `bool \| None` | `None` | Whether Newton replication imports visual-only geometry from USD. |
| `bvh_constructor_geometry` | `Literal['lbvh', 'sah', 'cubql']` | `'cubql'` | BVH construction algorithm for mesh geometry colliders. |
| `bvh_constructor_scene` | `Literal['lbvh', 'sah']` | `'sah'` | BVH construction algorithm for the top-level scene (broad-phase) hierarchy. |
| `bvh_constructor_gaussian` | `Literal['lbvh', 'sah', 'cubql']` | `'cubql'` | BVH construction algorithm for Gaussian-splat primitives. |

### NewtonBuilderCfg

[NewtonBuilderCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L35). Share mutable construction data, populated before model allocation and retained across hard resets.

```python
builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics, manager=sim.physics_manager))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `Callable[[NewtonBuilderCfg], ModelBuilder] \| str` | `'{DIR}.newton_manager:create_newton_builder'` | Factory used by the context registry. |
| `physics_cfg` | `PhysicsCfg` | `required (keyword-only)` | Selected physics settings; non-Newton physics requires a render-only Newton representation. |
| `manager` | `PhysicsManager` | `required (keyword-only)` | Owner of this construction model; independently owned models never share registry entries. |

### NewtonBackendCfg

[NewtonBackendCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L47). Allocate model and state from the matching builder; closing them leaves the builder intact. Inherits `BackendCfg`.

```python
backend = sim.get_or_create_backend(NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device, manager=sim.physics_manager))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `Callable[[NewtonBackendCfg], NewtonBackend] \| str` | `'{DIR}.newton_backend:create_newton_backend'` | Factory used by the context registry. |
| `physics_cfg` | `PhysicsCfg` | `required (keyword-only)` | Selected physics settings, also identifying the shared construction builder. |
| `device` | `str` | `required` | Device on which to allocate the model and native buffers. |
| `manager` | `PhysicsManager` | `required (keyword-only)` | Manager owning the model's construction and lifecycle. |

### NewtonSolverCfg

[NewtonSolverCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L61). Configuration for Newton solver-related parameters.

```python
# Base class: subclass for a custom adapter; see the custom-solver example.
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.newton_solver:NewtonSolver'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'None'` | Solver type metadata (deprecated). |

### NewtonShapeCfg

[NewtonShapeCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L108). Default per-shape collision properties applied to all shapes in a Newton scene.

```python
cfg = NewtonCfg(default_shape_cfg=NewtonShapeCfg(mu=0.8, gap=0.01))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `margin` | `float` | `0.0` | Default per-shape collision margin [m]. |
| `gap` | `float` | `0.01` | Default per-shape contact gap [m]. Newton's upstream default is `None`. |
| `ke` | `float` | `2500.0` | Default per-shape normal contact stiffness [N/m]. |
| `kd` | `float` | `100.0` | Default per-shape normal contact damping [N*s/m]. |
| `mu` | `float` | `1.0` | Default per-shape friction coefficient [dimensionless]. |

### NewtonSoftContactCfg

[NewtonSoftContactCfg](../isaaclab_newton/physics/newton_manager_cfg.py#L86). Global soft-contact parameters applied to the finalized Newton model.

```python
cfg = NewtonCfg(soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=1.0e4))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `soft_contact_ke` | `float` | `1000.0` | Body-particle and particle self-contact stiffness [N/m]. |
| `soft_contact_kd` | `float` | `10.0` | Body-particle contact damping [N*s/m]. |
| `soft_contact_mu` | `float` | `0.5` | Body-particle contact friction coefficient [dimensionless]. |

### NewtonCollisionPipelineCfg

[NewtonCollisionPipelineCfg](../isaaclab_newton/physics/newton_collision_cfg.py#L74). Configuration for Newton collision pipeline.

```python
cfg = NewtonCfg(solver_cfg=XPBDSolverCfg(), collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `broad_phase` | `Literal['explicit', 'nxn', 'sap']` | `'explicit'` | Broad phase algorithm for collision detection. |
| `reduce_contacts` | `bool` | `True` | Whether to reduce contacts for mesh-mesh collisions. |
| `rigid_contact_max` | `int \| None` | `None` | Maximum number of rigid contacts to allocate. |
| `max_triangle_pairs` | `int` | `1000000` | Maximum number of triangle pairs allocated by narrow phase for mesh and heightfield collisions. |
| `soft_contact_max` | `int \| None` | `None` | Maximum number of soft contacts to allocate. |
| `soft_contact_margin` | `float` | `0.01` | Margin [m] for soft contact generation. |
| `enable_rigid_soft_full_surface_contact` | `bool` | `False` | Whether to generate soft contacts against full-surface-capable rigid colliders. |
| `requires_grad` | `bool \| None` | `None` | Whether to enable gradient computation for collision. |
| `sdf_hydroelastic_config` | `HydroelasticSDFCfg \| None` | `None` | Configuration for SDF-based hydroelastic collision handling. |

`pipeline_cfg.to_pipeline_args()` converts nested settings into native constructor arguments;
use `newton.CollisionPipeline(model, **pipeline_cfg.to_pipeline_args())` only when constructing
a custom pipeline yourself. Normal initialization handles this automatically.

### HydroelasticSDFCfg

[HydroelasticSDFCfg](../isaaclab_newton/physics/newton_collision_cfg.py#L16). Configuration for SDF-based hydroelastic collision handling.

```python
pipeline_cfg = NewtonCollisionPipelineCfg(sdf_hydroelastic_config=HydroelasticSDFCfg(buffer_fraction=0.5))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `reduce_contacts` | `bool` | `True` | Whether to reduce contacts to a smaller representative set per shape pair. |
| `buffer_fraction` | `float` | `1.0` | Fraction of worst-case hydroelastic buffer allocations. Range: (0, 1]. |
| `normal_matching` | `bool` | `True` | Whether to rotate reduced contact normals to align with aggregate force direction. |
| `anchor_contact` | `bool` | `False` | Whether to add an anchor contact at the center of pressure for each normal bin. |
| `margin_contact_area` | `float` | `0.01` | Contact area [m^2] used for non-penetrating contacts at the margin. |
| `output_contact_surface` | `bool` | `False` | Whether to output hydroelastic contact surface vertices for visualization. |

### MJWarpSolverCfg

[MJWarpSolverCfg](../isaaclab_newton/physics/mjwarp_manager_cfg.py#L22). Configuration for MuJoCo Warp solver-related parameters. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.mjwarp_manager:MJWarpSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'mujoco_warp'` | Solver type. Can be "mujoco_warp". |
| `njmax` | `int` | `300` | Number of constraints per environment (world). |
| `nconmax` | `int \| None` | `None` | Number of contact points per environment (world). |
| `iterations` | `int` | `100` | Number of solver iterations. |
| `ls_iterations` | `int` | `50` | Number of line search iterations for the solver. |
| `solver` | `str` | `'newton'` | Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants. |
| `integrator` | `str` | `'euler'` | Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants. |
| `use_mujoco_cpu` | `bool` | `False` | Whether to use the pure MuJoCo backend instead of `mujoco_warp`. |
| `disable_contacts` | `bool` | `False` | Whether to disable contact computation in MuJoCo. |
| `disable_sensors` | `bool` | `False` | Whether to disable MuJoCo Warp's internal sensor computation. |
| `default_actuator_gear` | `float \| None` | `None` | Default gear ratio for all actuators. |
| `actuator_gears` | `dict[str, float] \| None` | `None` | Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`. |
| `update_data_interval` | `int` | `1` | Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state. |
| `save_to_mjcf` | `str \| None` | `None` | Optional path to save the generated MJCF model file. |
| `impratio` | `float` | `1.0` | Frictional-to-normal constraint impedance ratio. |
| `cone` | `str` | `'pyramidal'` | The type of contact friction cone. Can be "pyramidal" or "elliptic". |
| `ccd_iterations` | `int` | `35` | Maximum iterations for convex collision detection (GJK/EPA). |
| `enable_multiccd` | `bool` | `False` | Whether to enable multiple-contact convex collision detection. Defaults to False. |
| `ls_parallel` | `bool` | `False` | Deprecated parallel line search option. |
| `use_mujoco_contacts` | `bool` | `True` | Whether to use MuJoCo's internal contact solver. |
| `tolerance` | `float` | `1e-06` | Solver convergence tolerance for the constraint residual. |

### FeatherstoneSolverCfg

[FeatherstoneSolverCfg](../isaaclab_newton/physics/featherstone_manager_cfg.py#L21). A semi-implicit integrator using symplectic Euler. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=FeatherstoneSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.featherstone_manager:FeatherstoneSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'featherstone'` | Solver type. Can be "featherstone". |
| `angular_damping` | `float` | `0.05` | Angular damping parameter for rigid contact simulation. |
| `update_mass_matrix_interval` | `int` | `1` | Frequency (in simulation steps) at which to update the mass matrix. |
| `friction_smoothing` | `float` | `1.0` | Friction smoothing parameter. |
| `use_tile_gemm` | `bool` | `False` | Whether to use tile-based GEMM for the mass matrix. |
| `fuse_cholesky` | `bool` | `True` | Whether to fuse the Cholesky decomposition. |

### XPBDSolverCfg

[XPBDSolverCfg](../isaaclab_newton/physics/xpbd_manager_cfg.py#L21). An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=XPBDSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.xpbd_manager:XPBDSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'xpbd'` | Solver type. Can be "xpbd". |
| `iterations` | `int` | `2` | Number of solver iterations. |
| `soft_body_relaxation` | `float` | `0.9` | Relaxation parameter for soft body simulation. |
| `soft_contact_relaxation` | `float` | `0.9` | Relaxation parameter for soft contact simulation. |
| `joint_linear_relaxation` | `float` | `0.7` | Relaxation parameter for joint linear simulation. |
| `joint_angular_relaxation` | `float` | `0.4` | Relaxation parameter for joint angular simulation. |
| `joint_linear_compliance` | `float` | `0.0` | Compliance parameter for joint linear simulation. |
| `joint_angular_compliance` | `float` | `0.0` | Compliance parameter for joint angular simulation. |
| `rigid_contact_relaxation` | `float` | `0.8` | Relaxation parameter for rigid contact simulation. |
| `rigid_contact_con_weighting` | `bool` | `True` | Whether to use contact constraint weighting for rigid contact simulation. |
| `angular_damping` | `float` | `0.0` | Angular damping parameter for rigid contact simulation. |
| `enable_restitution` | `bool` | `False` | Whether to enable restitution for rigid contact simulation. |

### VBDSolverCfg

[VBDSolverCfg](../isaaclab_newton/physics/vbd_manager_cfg.py#L21). Configuration for the Vertex Block Descent solver. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=VBDSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.vbd_manager:VBDSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `iterations` | `int` | `10` | Number of VBD iterations per substep. |
| `integrate_with_external_rigid_solver` | `bool` | `False` | Whether an external solver integrates rigid bodies. |
| `particle_enable_self_contact` | `bool` | `False` | Whether to enable particle self-contact. |
| `particle_self_contact_radius` | `float` | `0.005` | Particle radius used for self-contact detection [m]. |
| `particle_self_contact_margin` | `float` | `0.005` | Self-contact detection margin [m]. |
| `particle_collision_detection_interval` | `int` | `-1` | Self-contact detection: <0 before init, 0 before and after init, k>=1 before every k VBD iterations. |
| `particle_vertex_contact_buffer_size` | `int` | `32` | Preallocation size for each vertex contact buffer. |
| `particle_edge_contact_buffer_size` | `int` | `64` | Preallocation size for each edge contact buffer. |
| `particle_topological_contact_filter_threshold` | `int` | `2` | Topological distance below which self-contacts are discarded. |
| `particle_rest_shape_contact_exclusion_radius` | `float` | `0.0` | Rest-shape separation threshold for filtering contacts [m]. |
| `rigid_compliant_alm` | `bool \| None` | `None` | Whether to use compliant ALM for rigid joints and contacts; `None` preserves Newton's default. |
| `rigid_contact_k_start` | `float` | `100.0` | Initial stiffness seed for rigid-body contacts [N/m]. |
| `rigid_body_contact_buffer_size` | `int` | `64` | Per-body body-body contact capacity when VBD integrates rigid bodies. |
| `rigid_body_particle_contact_buffer_size` | `int` | `256` | Per-body particle, edge, and face soft-contact capacity when VBD integrates rigid bodies. |

### MPMSolverCfg

[MPMSolverCfg](../isaaclab_newton/physics/mpm_manager_cfg.py#L21). Configuration for Newton's implicit Material Point Method (MPM) solver. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=MPMSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.mpm_manager:MPMSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'implicit_mpm'` | Solver type. Can be "implicit_mpm". |
| `max_iterations` | `int` | `250` | Maximum number of iterations for the rheology solver. |
| `tolerance` | `float` | `0.0001` | Tolerance for the rheology solver. |
| `solver` | `str \| tuple[str, ...]` | `'auto'` | Rheology solver, or an ordered warm-start sequence of solvers. |
| `warmstart_mode` | `Literal['none', 'auto', 'particles', 'grid', 'smoothed']` | `'auto'` | Warm-start mode for the rheology solver. |
| `collider_velocity_mode` | `Literal['forward', 'backward', 'instantaneous', 'finite_difference']` | `'forward'` | Collider velocity computation mode. |
| `voxel_size` | `float` | `0.1` | Size of the MPM grid voxels [m]. |
| `grid_type` | `Literal['sparse', 'dense', 'fixed']` | `'sparse'` | Type of grid to use. |
| `grid_padding` | `int` | `0` | Number of empty cells to add around particles when allocating the grid. |
| `max_active_cell_count` | `int` | `-1` | Maximum active grid-cell count shared by all worlds. |
| `max_leaf_node_count` | `int` | `-1` | Maximum sparse-grid leaf-node count shared by all worlds. |
| `max_lower_node_count` | `int` | `-1` | Maximum sparse-grid lower internal-node count shared by all worlds. |
| `max_upper_node_count` | `int` | `-1` | Maximum sparse-grid upper internal-node count shared by all worlds. |
| `separate_worlds` | `bool` | `False` | Whether each Newton world uses an independent local MPM grid environment. |
| `transfer_scheme` | `Literal['apic', 'pic']` | `'apic'` | Particle-grid transfer scheme. |
| `integration_scheme` | `Literal['pic', 'gimp']` | `'pic'` | Integration scheme controlling shape-function support. |
| `critical_fraction` | `float` | `0.0` | Dimensionless fraction under which the yield surface collapses. |
| `air_drag` | `float` | `1.0` | Numerical drag for background air. |
| `collider_normal_from_sdf_gradient` | `bool` | `False` | Whether collider normals are computed from SDF gradients rather than closest points. |
| `collider_basis` | `str` | `'S2'` | Collider basis function, such as `"S2"` or `"Q1"`. |
| `strain_basis` | `str` | `'P0'` | Strain basis function, such as `"P0"`, `"P1d"`, `"Q1"`, or `"Q1d"`. |
| `velocity_basis` | `str` | `'Q1'` | Velocity basis function, such as `"Q1"`, `"B2"`, or `"B3"`. |
| `project_outside_colliders` | `bool` | `False` | Whether to hard-project particles out of collider interiors after each substep. |

### _KaminoSolverCfgBase

[_KaminoSolverCfgBase](../isaaclab_newton/physics/kamino_manager_cfg.py#L226). Common configuration for Kamino solver-related parameters. Inherits `NewtonSolverCfg`.

```python
# Shared fields: KaminoPADMMSolverCfg(integrator="euler")
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.kamino_manager:KaminoSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `solver_type` | `str` | `'kamino'` | Solver type. Can be "kamino". |
| `integrator` | `Literal['euler', 'moreau']` | `'moreau'` | Integrator type. |
| `use_collision_detector` | `bool` | `False` | Whether to use Kamino's internal collision detector instead of Newton's pipeline. |
| `use_fk_solver` | `bool \| None` | `None` | Whether to enable the forward kinematics solver for state resets. |
| `sparse_jacobian` | `bool \| None` | `None` | Whether to use sparse Jacobian computation. `None` lets Newton pick per backend. |
| `sparse_dynamics` | `bool` | `False` | Whether to use sparse dynamics computation. |
| `rotation_correction` | `Literal['twopi', 'continuous', 'none']` | `'twopi'` | Rotation correction mode. |
| `angular_velocity_damping` | `float` | `0.0` | Angular velocity damping factor. Valid range is [0.0, 1.0]. |
| `collect_solver_info` | `bool` | `False` | Whether to collect solver convergence and performance info at each step. |
| `compute_solution_metrics` | `bool` | `False` | Whether to compute solution metrics at each step. |
| `dynamics` | `KaminoDynamicsCfg \| None` | `None` | Constrained dynamics problem parameters. |
| `constraints` | `KaminoConstraintsCfg` | `KaminoConstraintsCfg()` | Constraint stabilization parameters. |
| `fk` | `KaminoFKCfg` | `KaminoFKCfg()` | Forward-kinematics reset solver parameters. |
| `collision_detector` | `KaminoCollisionDetectorCfg` | `KaminoCollisionDetectorCfg()` | Internal collision-detector parameters. |
| `materials` | `KaminoMaterialsCfg` | `KaminoMaterialsCfg()` | Material mixing parameters. |
| `max_contacts_per_world` | `int \| None` | `None` | Cap the per-world contact pre-allocation handed to Kamino. |

### KaminoPADMMSolverCfg

[KaminoPADMMSolverCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L396). Configuration for Kamino with the P-ADMM forward-dynamics solver. Inherits `_KaminoSolverCfgBase`.

```python
cfg = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dynamics_solver_cfg` | `KaminoPADMMCfg` | `KaminoPADMMCfg()` | P-ADMM forward-dynamics solver parameters. |

### KaminoDVISolverCfg

[KaminoDVISolverCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L408). Configuration for Kamino with the DVI forward-dynamics solver. Inherits `_KaminoSolverCfgBase`.

```python
cfg = NewtonCfg(solver_cfg=KaminoDVISolverCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dynamics_solver_cfg` | `KaminoDVICfg` | `KaminoDVICfg()` | DVI forward-dynamics solver parameters. |

### KaminoPADMMCfg

[KaminoPADMMCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L34). P-ADMM forward-dynamics solver parameters for Kamino.

```python
solver_cfg = KaminoPADMMSolverCfg(dynamics_solver_cfg=KaminoPADMMCfg(max_iterations=100))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `max_iterations` | `int` | `100` | Maximum number of P-ADMM solver iterations. |
| `primal_tolerance` | `float` | `0.0001` | Primal residual convergence tolerance. |
| `dual_tolerance` | `float` | `0.0001` | Dual residual convergence tolerance. |
| `compl_tolerance` | `float` | `0.0001` | Complementarity residual convergence tolerance. |
| `restart_tolerance` | `float` | `0.999` | Combined primal-dual residual tolerance for acceleration restarts. |
| `rho_0` | `float` | `0.05` | Initial penalty parameter. |
| `rho_min` | `float` | `1e-05` | Lower bound on the penalty parameter. |
| `a_0` | `float` | `1.0` | Initial acceleration parameter. |
| `alpha` | `float` | `10.0` | Primal-dual residual threshold for penalty updates. |
| `tau` | `float` | `1.5` | Penalty increase/decrease factor. |
| `eta` | `float` | `1e-05` | Proximal regularization parameter. Must be greater than zero. |
| `penalty_update_freq` | `int` | `1` | Frequency of penalty updates. Zero disables updates. |
| `penalty_update_method` | `Literal['fixed', 'balanced']` | `'fixed'` | Penalty update method. |
| `linear_solver_tolerance` | `float` | `0.0` | Absolute tolerance for the iterative linear solver. Zero leaves it unchanged. |
| `linear_solver_tolerance_ratio` | `float` | `0.0` | Ratio adapting the linear solver tolerance from the ADMM primal residual. |
| `use_acceleration` | `bool` | `True` | Whether to use Nesterov-type acceleration (APADMM). |
| `use_graph_conditionals` | `bool` | `False` | Whether to use CUDA graph conditional nodes in the iterative solver. |
| `warmstart_mode` | `Literal['none', 'internal', 'containers']` | `'containers'` | Warmstart mode. |
| `contact_warmstart_method` | `Literal['key_and_position', 'geom_pair_net_force', 'geom_pair_net_wrench', 'key_and_position_with_net_force_backup', 'key_and_position_with_net_wrench_backup']` | `'geom_pair_net_force'` | Contact warm-start method. |

### KaminoDVICfg

[KaminoDVICfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L102). DVI forward-dynamics solver parameters for Kamino.

```python
solver_cfg = KaminoDVISolverCfg(dynamics_solver_cfg=KaminoDVICfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `tolerance` | `float` | `1e-05` | Convergence tolerance on the projected update size. |
| `regularization` | `float` | `1e-06` | Diagonal regularization added to each projected update denominator. |
| `omega` | `float` | `1.0` | Relaxation factor applied to projected Gauss-Seidel updates. |
| `max_alternating_iterations` | `int` | `20` | Maximum outer DVI iterations. |
| `inequality_sweeps_per_iteration` | `int` | `1` | Projected Gauss-Seidel sweeps per DVI iteration. |
| `bilateral_solve_interval` | `int` | `1` | DVI iterations between repeated direct bilateral solves. |
| `bilateral_solver_type` | `Literal['LLTB', 'LLTBRCM']` | `'LLTB'` | Direct linear solver for the bilateral constraint block. |
| `bilateral_solver_kwargs` | `dict[str, Any]` | `dict()` | Additional keyword arguments for the bilateral linear solver. |
| `warmstart_mode` | `Literal['none', 'internal', 'containers']` | `'containers'` | Warmstart mode. |
| `contact_warmstart_method` | `Literal['key_and_position', 'geom_pair_net_force', 'key_and_position_with_net_force_backup']` | `'key_and_position_with_net_force_backup'` | Contact warm-start method when `warmstart_mode` is `containers`. |

### KaminoDynamicsCfg

[KaminoDynamicsCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L141). Constrained forward-dynamics problem parameters for Kamino.

```python
solver_cfg = KaminoPADMMSolverCfg(dynamics=KaminoDynamicsCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `preconditioning` | `bool` | `True` | Whether to precondition the dual problem. Must be `False` when using DVI. |
| `linear_solver_type` | `Literal['LLTB', 'LLTBRCM', 'CR', 'CRF']` | `'LLTB'` | Linear solver for the dynamics problem. |
| `linear_solver_kwargs` | `dict[str, Any]` | `dict()` | Additional keyword arguments for the linear solver. |

### KaminoConstraintsCfg

[KaminoConstraintsCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L155). Global constraint stabilization parameters for Kamino.

```python
solver_cfg = KaminoPADMMSolverCfg(constraints=KaminoConstraintsCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.1` | Baumgarte stabilization for bilateral joint constraints. Valid range is [0, 1]. |
| `beta` | `float` | `0.01` | Baumgarte stabilization for unilateral joint-limit constraints. Valid range is [0, 1]. |
| `gamma` | `float` | `0.01` | Baumgarte stabilization for unilateral contact constraints. Valid range is [0, 1]. |
| `delta` | `float` | `1e-06` | Contact penetration margin [m]. |

### KaminoFKCfg

[KaminoFKCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L172). Forward-kinematics reset solver parameters for Kamino.

```python
solver_cfg = KaminoPADMMSolverCfg(fk=KaminoFKCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `use_regularization` | `bool` | `True` | Whether to regularize the FK reset solve (Tikhonov term on body poses). |
| `regularization_weight` | `float` | `1e-05` | Weight of the FK reset regularizer when `use_regularization` is `True`. |
| `tolerance` | `float` | `1e-05` | Convergence tolerance of the FK reset solve. |

### KaminoCollisionDetectorCfg

[KaminoCollisionDetectorCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L186). Internal Kamino collision-detector parameters.

```python
solver_cfg = KaminoPADMMSolverCfg(collision_detector=KaminoCollisionDetectorCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pipeline` | `Literal['primitive', 'unified'] \| None` | `None` | Collision-detection pipeline. `None` uses Newton's default (`unified`). |
| `broadphase` | `Literal['nxn', 'sap', 'explicit'] \| None` | `None` | Broad-phase algorithm. `None` uses Newton's default. |
| `bvtype` | `Literal['aabb', 'bs'] \| None` | `None` | Bounding-volume type. `None` uses Newton's default. |
| `max_contacts` | `int \| None` | `None` | Model-wide contact buffer capacity cap. |
| `max_contacts_per_world` | `int \| None` | `None` | Per-world contact buffer capacity override. |
| `max_contacts_per_pair` | `int \| None` | `None` | Maximum contacts generated per candidate geometry pair. |
| `max_triangle_pairs` | `int \| None` | `None` | Maximum triangle-primitive shape pairs in narrow phase. |
| `default_gap` | `float \| None` | `None` | Default detection gap [m] applied as a floor to per-geometry gaps. |

### KaminoMaterialsCfg

[KaminoMaterialsCfg](../isaaclab_newton/physics/kamino_manager_cfg.py#L215). Material mixing parameters for Kamino contacts.

```python
solver_cfg = KaminoPADMMSolverCfg(materials=KaminoMaterialsCfg())
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `friction_mix_mode` | `Literal['average', 'multiply', 'max', 'min']` | `'average'` | How friction coefficients are mixed for a contact pair. |
| `restitution_mix_mode` | `Literal['average', 'multiply', 'max', 'min']` | `'min'` | How restitution coefficients are mixed for a contact pair. |

### CouplerEntryCfg

[CouplerEntryCfg](../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py#L30). Configuration for one named sub-solver and its model ownership.

```python
entry = CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg(), bodies=["Robot/.*"])
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | `required` | Unique name used by coupling mappings to reference this entry. |
| `solver_cfg` | `NewtonSolverCfg` | `required` | Configuration used to construct this entry's Newton solver. |
| `bodies` | `list[str]` | `list()` | Bodies owned by this entry. |
| `particles` | `list[int]` | `list()` | Parent-model particle indices owned by this entry. |
| `all_particles` | `bool` | `False` | Whether this entry owns every particle in the parent model. |
| `include_child_joints` | `bool` | `True` | Whether fully selected child joints are owned by this entry. |
| `include_body_shapes` | `bool` | `True` | Whether shapes attached to selected bodies are owned by this entry. |
| `include_static_shapes` | `bool` | `False` | Whether this entry owns all shapes whose body index is `-1`. |
| `shape_label_patterns` | `list[str]` | `list()` | Regexes matched against full Newton shape labels for additional ownership. |
| `substeps` | `int` | `1` | Number of equal substeps this entry runs inside one coupled step. |
| `in_place` | `bool` | `False` | Whether this entry steps in-place instead of using a second state buffer. |

### CouplerProxyMappingCfg

[CouplerProxyMappingCfg](../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py#L86). Configuration for one directed virtual-proxy mapping.

```python
proxy = CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=["Robot/.*"])
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `source` | `str` | `required` | Name of the entry that owns the source bodies. |
| `destination` | `str` | `required` | Name of the entry that receives the proxy bodies. |
| `bodies` | `list[str \| int]` | `list()` | Source bodies exposed as proxies in the destination entry. |
| `particles` | `list[int]` | `list()` | Source particle indices exposed as proxies in the destination entry. |
| `mode` | `Literal['lagged', 'staggered']` | `'lagged'` | Proxy transfer mode passed to Newton's coupled-proxy solver. |
| `mass_scale` | `float` | `1.0` | Scale applied to proxy body mass/inertia and particle mass in the destination view. |
| `collide_interval` | `int \| None` | `None` | Proxy-local collision refresh interval. |
| `collision_pipeline` | `NewtonCollisionPipelineCfg \| Callable[[ModelView], CollisionPipeline \| None] \| None` | `NewtonCollisionPipelineCfg()` | Configuration or factory for the proxy destination collision pipeline. |

### CouplerCfg

[CouplerCfg](../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py#L131). Base configuration for a Newton experimental coupled solver. Inherits `NewtonSolverCfg`.

```python
# Base class: select CouplerProxyCfg or CouplerAdmmCfg.
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.coupler:CouplerSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `entries` | `list[CouplerEntryCfg]` | `list()` | Ordered named sub-solver entries and their ownership selectors. |

### CouplerProxyCfg

[CouplerProxyCfg](../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py#L147). Configuration for Newton's lagged-impulse virtual-proxy coupling. Inherits `CouplerCfg`.

```python
solver_cfg = CouplerProxyCfg(entries=entries, proxies=proxies)
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `proxies` | `list[CouplerProxyMappingCfg]` | `list()` | Directed proxy mappings between named solver entries. |
| `iterations` | `int` | `1` | Number of proxy relaxation passes per coupled step. |

### CouplerAdmmCfg

[CouplerAdmmCfg](../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py#L161). Configuration for Newton's linearized ADMM coupling. Inherits `CouplerCfg`.

```python
solver_cfg = CouplerAdmmCfg(entries=entries, iterations=5)
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `contact_max_triangle_pairs` | `int \| None` | `None` | Internal ADMM triangle-pair capacity across all environments in one process. |
| `contact_reduction_hashtable_size_factor` | `float \| None` | `None` | Contact-reduction hash table size relative to the internal triangle-pair capacity. |
| `contact_pairs` | `list[tuple[str, str]] \| None` | `None` | Symmetric contact interfaces as `(entry_name, entry_name)` pairs. |
| `iterations` | `int` | `5` | Number of ADMM dual iterations per coupled step. |
| `rho` | `float` | `1.0` | ADMM penalty parameter [dimensionless]. |
| `gamma` | `float` | `0.0` | Proximal mass scaling parameter [dimensionless]. |
| `baumgarte` | `float` | `0.0` | Position-error correction fraction [dimensionless]. |
| `joint_stiffness` | `float` | `10000.0` | Translational cross-solver joint stiffness [N/m]. |
| `joint_damping` | `float` | `0.0` | Translational cross-solver joint damping [N*s/m]. |
| `joint_angular_stiffness` | `float` | `10000.0` | Angular cross-solver joint stiffness [N*m/rad]. |
| `joint_angular_damping` | `float` | `0.0` | Angular cross-solver joint damping [N*m*s/rad]. |
| `joint_proximal_bodies` | `bool` | `True` | Whether cross-solver joint neighbors remain visible as inertial proxies. |
| `joint_proximal_destination_entries` | `list[str] \| None` | `None` | Optional entries that receive cross-solver joint proximal bodies. |
| `joint_proximal_mass_scale` | `float` | `1.0` | Mass scale applied to cross-solver joint proximal bodies. |
| `rigid_contact_matching` | `Literal['disabled', 'latest', 'sticky']` | `'disabled'` | Frame-to-frame matching mode for collision-detected rigid contacts. |
| `contact_matching_pos_threshold` | `float \| None` | `None` | Maximum midpoint distance for matching rigid contacts [m]. |
| `contact_matching_normal_dot_threshold` | `float \| None` | `None` | Minimum normal dot product for matching rigid contacts. |
| `contact_matching_force_scale` | `float` | `0.9` | Scale applied to the previous ADMM dual when a rigid contact matches. |

### CoupledMJWarpVBDSolverCfg

[CoupledMJWarpVBDSolverCfg](../../isaaclab_contrib/isaaclab_contrib/custom_coupling/newton_manager_cfg.py#L21). Configuration for the custom MJWarp and VBD coupling manager. Inherits `NewtonSolverCfg`.

```python
cfg = NewtonCfg(solver_cfg=CoupledMJWarpVBDSolverCfg(coupling_mode="two_way"))
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `class_type` | `type[NewtonSolver] \| str` | `'{DIR}.coupled_mjwarp_vbd_manager:CoupledMJWarpVBDSolverAdapter'` | Stateless solver-adapter class or import path; independent of NewtonCfg.class_type. |
| `rigid_solver_cfg` | `MJWarpSolverCfg` | `MJWarpSolverCfg()` | MJWarp rigid-body solver configuration. |
| `soft_solver_cfg` | `VBDSolverCfg` | `VBDSolverCfg(integrate_with_external_rigid_solver=True)` | VBD deformable solver configuration. |
| `coupling_mode` | `Literal['one_way', 'two_way']` | `'two_way'` | Coupling direction between the rigid and deformable solvers. |

## Runtime records and solver capabilities

These are runtime objects or adapter metadata, not SimulationCfg fields.

### StepCallback fields

[StepCallback](../isaaclab_newton/physics/newton_backend.py#L161). A function run at a `StepPhase` of every step, until it is unregistered or the backend closes.

| Field | Default | Meaning |
| --- | --- | --- |
| `fn` | `required` | Function to run. `StepPhase.STATE_FORCE` callbacks receive the input state; others take no arguments. |
| `phase` | `required` | Where the callback runs. |
| `graphable` | `True` | Whether the function can be recorded into a CUDA graph: fixed buffers and no host branching on device data. Other callbacks run eagerly at the same position. |
| `name` | `''` | Label used in errors and profiles. |

```python
callback = nb.register_step_callback(backend, controller, StepPhase.CONTROL)
```

### StepOp fields

[StepOp](../isaaclab_newton/physics/newton_backend.py#L179). One operation of a `StepGraph`, with every buffer bound.

| Field | Default | Meaning |
| --- | --- | --- |
| `fn` | `required` | Bound operation taking no arguments. |
| `graphable` | `required` | Whether this operation can join CUDA capture. |
| `name` | `required` | Operation label for errors and profiles. |

```python
op = nb.StepOp(fn=controller, graphable=True, name="controller")
schedule = nb.StepGraph([op], steps=1)  # Custom schedule; does not add physics automatically.
```

### NewtonCloneRecord fields

[NewtonCloneRecord](../isaaclab_newton/physics/newton_backend.py#L257). Native replication outputs, kept across hard resets and consumed when the model is finalized.

| Field | Default | Meaning |
| --- | --- | --- |
| `world_xforms` | `required` | Root transform of each cloned world. |
| `source_builders` | `required` | Per-source builders retained so single-model consumers can finalize one environment. |
| `particle_ranges` | `required` | Native `(start, count)` particle range of each imported particle prim. |
| `cable_bindings` | `required` | Native capsule shape indices of each open cable. |
| `geometry_batches` | `required` | Deformable and particle geometry published through scene data. |

```python
record = NewtonCloneRecord(
    world_xforms=world_xforms, source_builders=source_builders,
    particle_ranges=particle_ranges, cable_bindings=cable_bindings,
    geometry_batches=geometry_batches,
)
manager.record_clone(record, site_index_map)  # Cloner integration, before finalization.
```

### NewtonSolver fields

[NewtonSolver](../isaaclab_newton/physics/newton_solver.py#L27). Construction, stepping, reset and capability hooks over an explicit backend, with no runtime state.

| Field | Default | Meaning |
| --- | --- | --- |
| `solver_class` | `None` | Newton solver the default `create_solver` constructs from the matching configuration fields. |
| `builder_attribute_solvers` | `()` | Solvers whose custom builder attributes are registered before import. |
| `single_state` | `False` | Whether the solver steps in place on one `newton.State`. |
| `supports_deterministic` | `False` | Whether the solver can honor a Warp determinism guarantee. |
| `supports_contact_sensors` | `True` | Whether Newton contact sensors can read the solver's contacts. |
| `prepares_step` | `False` | Whether `prepare_step` does work, so the step graph runs it. |
| `supports_heterogeneous_worlds` | `False` | Whether one solver can step worlds with different contents, such as different robots per world. |
| `ignored_model_changes` | `{}` | Model changes the solver does not apply after construction, mapped to the warning logged once. |

```python
adapter = backend.solver_adapter
can_capture = adapter.supports_graph_capture(backend)
uses_one_state = adapter.single_state
```

`StepGraph` retains `ops`, `steps`, grouped `segments`, `graphable` and captured `graphs`.
Its `captured` property indicates whether capture has occurred. Normally obtain it
from `nb.prepare(backend)` rather than building the operation list yourself.
`CapturedGraph` retains the Torch graph, Warp graph, device and allocator; keep the
wrapper alive for all replays. Use `nb.capture_graph` to construct it.

### Shipped adapters

| Adapter | Selected by | Role |
| --- | --- | --- |
| `MJWarpSolverAdapter` | `MJWarpSolverCfg` | MuJoCo Warp or optional CPU MuJoCo; CPU mode cannot join CUDA capture. |
| `FeatherstoneSolverAdapter` | `FeatherstoneSolverCfg` | Articulated rigid-body dynamics. |
| `XPBDSolverAdapter` | `XPBDSolverCfg` | XPBD rigid and deformable simulation. |
| `VBDSolverAdapter` | `VBDSolverCfg` | VBD deformable/rigid simulation. |
| `MPMSolverAdapter` | `MPMSolverCfg` | Implicit MPM with solver-owned grid/history state. |
| `KaminoSolverAdapter` | `KaminoPADMMSolverCfg or KaminoDVISolverCfg` | Kamino with the chosen dynamics solver. |
| `CouplerSolverAdapter` | `CouplerProxyCfg or CouplerAdmmCfg` | Entries sharing one model, with proxy or ADMM coupling. |
| `CoupledMJWarpVBDSolverAdapter` | `CoupledMJWarpVBDSolverCfg` | Specialized rigid/deformable coupling. |

Usage: `cfg = NewtonCfg(solver_cfg=XPBDSolverCfg())`; the backend resolves
`cfg.solver_cfg.class_type` and invokes the common hooks below. Adapter overrides
retain those signatures; their source and the extension guide describe solver-specific
behavior. MPM additionally exposes `MPMSolverAdapter.reset_solver_state(backend,
state=None, world_mask=None, flags=None)`: it clears solver history after task state
writes. Use `MPMSolverAdapter.reset_solver_state(backend)` for both state buffers.
An explicit mask uses the native `(world_count + 1,)` convention, with the final
entry selecting global entities, unlike the local-world reset mask in `invalidate_worlds`.

## Function reference

Signatures list every parameter and default. A `*` marks keyword-only parameters.
Properties appear without parentheses. The usage column supplies a call shape;
use the lifecycle and recipes for ordering and input preparation. Class-level
context operations such as `clear_instance()` may also be called on
`SimulationContext` directly. `CapturedGraph` is constructed by `capture_graph()`;
applications call its `launch()` method rather than constructing allocator internals.

### Important parameter contracts

- `env_mask` in `invalidate_fk`/`invalidate_body_state` selects **view rows**.
  Pass `view.articulation_ids` for FK or the cached `view_row_worlds(...)` mapping
  for body-state writes. `invalidate_fk` without an articulation mapping flags all
  articulations; a mask alone does not narrow it.
- `worlds` in `invalidate_worlds` and `env_mask` in `mark_model_changed` select
  **backend worlds**, with shape `(model.world_count,)` and Warp boolean dtype.
  Integer `env_ids` select view rows. Use stable device masks for replayable resets.
- `view_row_worlds` reads device arrays at bind time. Cache its result outside the
  captured/hot path. `-1` denotes global articulations rather than a simulated world.
- State writers invalidate FK or maximal-coordinate state; property writers queue
  `newton.ModelFlags`. `nb.forward()` commits state invalidation;
  `nb.notify_model_changes()` commits properties. `manager.forward()` does both.
- A step callback takes no arguments except `STATE_FORCE`, which receives the
  current substep input state. `graphable=True` promises replay-safe device work;
  fixed input buffers may change contents between launches, but Python branches,
  indices, shapes and buffer identities must not change implicitly.
- Lifecycle callbacks receive an event payload. `order` is ascending;
  `wrap_weak_ref=True` avoids retaining bound-method owners. Keep the subscriber
  alive while registered. These callbacks author/bind resources; they are distinct
  from callbacks recorded into every simulation step.
- Contact selectors are full-label regular expressions; body and shape selectors
  identify sensing objects, and partner selectors identify their counterparts.
  IMU `sites` are finalized site indices, one for each sensor instance. Request
  `body_qdd` and sites before finalization; bind sensors after finalization.
- `prepare` is required before outer `record_step`. `use_cuda_graph` configures
  ordinary stepping; it does not waive solver or callback capture restrictions.
  `relaxed=True` on capture selects the Kit-compatible capture mode/stream.
- `run_query` takes a monotonic scene-publication timestamp, a bound query callable
  and that consumer's previous graph cache. Store its returned cache. A hard reset
  requires rebinding the query to the new backend and discarding the old cache.
- Hooks accepting a `ModelBuilder` run before finalization; hooks accepting
  `state_in`, `state_out`, `contacts`, and `dt` operate on bound native buffers,
  with `dt` equal to one solver substep. Do not store an active backend on the
  adapter class.

### SimulationContext

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `instance() -> SimulationContext \| None` | Get the singleton instance, or None if not created. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L80) | `SimulationContext.instance()` |
| `add_reset_callback(name, fn) -> None` | Register a callback to fire after every `reset` of any simulation context. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L85) | `sim.add_reset_callback(name, fn)` |
| `remove_reset_callback(name) -> None` | Unregister a previously registered reset callback. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L98) | `sim.remove_reset_callback(name)` |
| `__init__(cfg=None, *, physics_manager=None)` | Initialize the simulation context. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L106) | `SimulationContext(sim_cfg, physics_manager=my_manager)` |
| `physics_sim_view` | Returns the physics simulation view. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L303) | `sim.physics_sim_view` |
| `device` | Returns the device on which the simulation is running. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L308) | `sim.device` |
| `backend` | Returns the tensor backend being used ("numpy" or "torch"). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L313) | `sim.backend` |
| `has_gui` | Returns whether GUI is enabled (cached at init). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L318) | `sim.has_gui` |
| `has_offscreen_render` | Returns whether offscreen rendering is enabled (cached at init). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L323) | `sim.has_offscreen_render` |
| `has_active_visualizers() -> bool` | Return whether any visualizer path is active for rendering/camera control. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L327) | `sim.has_active_visualizers()` |
| `is_running() -> bool` | Return whether the simulation should keep running. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L331) | `sim.is_running()` |
| `require_visual_shapes() -> None` | Record that something in this simulation draws the physics model's visual-only shapes. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L339) | `sim.require_visual_shapes()` |
| `visual_shapes_required` | Whether `require_visual_shapes` was called for this simulation. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L349) | `sim.visual_shapes_required` |
| `can_render_rgb_array() -> bool` | Return whether rgb-array rendering is currently available, including from a headless visualizer. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L353) | `sim.can_render_rgb_array()` |
| `is_rendering` | Returns whether *continuous* rendering is active (GUI, RTX sensors, visualizers, or XR). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L360) | `sim.is_rendering` |
| `get_physics_dt() -> float` | Returns the physics time step [s]. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L375) | `sim.get_physics_dt()` |
| `get_physics_step_count() -> int` | Return the monotonic physics step counter (incremented each `step`). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L379) | `sim.get_physics_step_count()` |
| `render_context` | Shared rendering state for camera backends and visual materials. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L384) | `sim.render_context` |
| `render_generation` | Returns a monotonic counter for render() executions. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L389) | `sim.render_generation` |
| `resolve_visualizer_types() -> list[str]` | Return the types of the visualizers in `SimulationCfg.visualizer_cfgs`. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L415) | `sim.resolve_visualizer_types()` |
| `initialize_visualizers() -> None` | Initialize the constructed visualizers after their shared scene has been cloned. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L461) | `sim.initialize_visualizers()` |
| `get_scene_data_provider() -> SceneDataProvider` | Return the scene data provider shared by visualizers and renderers. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L475) | `sim.get_scene_data_provider()` |
| `register_interactive_scene(scene) -> None` | Register the active scene so scene data providers can expose scene-owned sensors. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L479) | `sim.register_interactive_scene(scene)` |
| `get_clone_plan() -> ClonePlan \| None` | Return the clone plan published by the scene. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L485) | `sim.get_clone_plan()` |
| `set_clone_plan(plan) -> None` | Set the cloner's active clone plan. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L493) | `sim.set_clone_plan(plan)` |
| `visualizers` | Returns the list of active visualizers. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L498) | `sim.visualizers` |
| `get_rendering_dt() -> float` | Return rendering dt, allowing visualizer-specific override. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L502) | `sim.get_rendering_dt()` |
| `set_camera_view(eye, target) -> None` | Set camera view on all visualizers that support it. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L510) | `sim.set_camera_view(eye, target)` |
| `add_render_callback(name, fn, order=0) -> None` | Register a callback to fire after every render step. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L516) | `sim.add_render_callback(name, fn)` |
| `remove_render_callback(name) -> None` | Unregister a previously registered render callback. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L526) | `sim.remove_render_callback(name)` |
| `forward() -> None` | Update kinematics without stepping physics. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L534) | `sim.forward()` |
| `reset(soft=False) -> None` | Reset the simulation. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L538) | `sim.reset()` |
| `step(render=True) -> None` | Step physics and optionally render. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L556) | `sim.step()` |
| `render(mode=None, skip_app_pumping=False) -> None` | Update visualizers and render the scene. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L573) | `sim.render()` |
| `update_visualizers(dt, skip_app_pumping=False) -> None` | Update visualizers without triggering renderer/GUI. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L603) | `sim.update_visualizers(backend.dt)` |
| `play() -> None` | Start or resume the simulation. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L668) | `sim.play()` |
| `pause() -> None` | Pause the simulation (can be resumed with play). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L676) | `sim.pause()` |
| `stop() -> None` | Stop the simulation completely. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L683) | `sim.stop()` |
| `request_reset() -> None` | Request an episode reset from a UI control (e.g. the Kit window button). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L691) | `sim.request_reset()` |
| `consume_reset_request() -> bool` | Return `True` if any visualizer or UI control requested an episode reset and clear the flag. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L698) | `sim.consume_reset_request()` |
| `is_playing() -> bool` | Returns True if simulation is playing (not paused or stopped). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L714) | `sim.is_playing()` |
| `is_stopped() -> bool` | Returns True if simulation is stopped (not just paused). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L718) | `sim.is_stopped()` |
| `set_setting(name, value) -> None` | Set a setting value. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L722) | `sim.set_setting(name, value)` |
| `get_setting(name) -> Any` | Get a setting value. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L726) | `sim.get_setting(name)` |
| `get_or_create_backend(cfg) -> Any` | Return the simulation-owned object for a construction configuration. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L730) | `sim.get_or_create_backend(builder_cfg)` |
| `close_backend(backend) -> None` | Release one registered object by identity. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L754) | `sim.close_backend(backend)` |
| `clear_instance() -> None` | Stop the simulation, clean up resources, and clear the singleton instance. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L777) | `sim.clear_instance()` |
| `clear_stage() -> None` | Clear the current USD stage (preserving /World and PhysicsScene). [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L841) | `sim.clear_stage()` |

### Context construction helper

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `build_simulation_context(create_new_stage=True, gravity_enabled=True, device=None, dt=0.01, sim_cfg=None, add_ground_plane=False, add_lighting=False, auto_add_lighting=False) -> Iterator[SimulationContext]` | Context manager to build a simulation context with the provided settings. [Source](../../isaaclab/isaaclab/sim/simulation_context.py#L857) | `build_simulation_context()` |

### NewtonManager

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `__init__(builder=None, cfg=None, *, dt=1.0 / 60.0, device='cuda:0')` | Create an independent manager, optionally from a populated Newton builder. [Source](../isaaclab_newton/physics/newton_manager.py#L91) | `NewtonManager(builder, cfg, dt=0.01, device="cuda:0")` |
| `initialize(sim_context) -> None` | Initialize the manager with simulation context. [Source](../isaaclab_newton/physics/newton_manager.py#L125) | `manager.initialize(sim)` |
| `reset(soft=False) -> None` | Reset physics simulation. [Source](../isaaclab_newton/physics/newton_manager.py#L138) | `manager.reset()` |
| `forward() -> None` | Reset solver internals and recompute body state for worlds whose state was authored. [Source](../isaaclab_newton/physics/newton_manager.py#L178) | `manager.forward()` |
| `step() -> None` | Advance physics by one environment step, or one physics step when the environment runs decimation. [Source](../isaaclab_newton/physics/newton_manager.py#L185) | `manager.step()` |
| `record_step() -> nb.StepGraph` | Record physics into the caller's capture, without host time or rendering bookkeeping. [Source](../isaaclab_newton/physics/newton_manager.py#L209) | `manager.record_step()` |
| `prepare() -> None` | Build the step graph ahead of a caller's capture of `step`, without advancing physics. [Source](../isaaclab_newton/physics/newton_manager.py#L213) | `manager.prepare()` |
| `close() -> None` | Clean up Newton physics resources. [Source](../isaaclab_newton/physics/newton_manager.py#L220) | `manager.close()` |
| `clear() -> None` | Release the backend and all session state. Lifecycle helper; use close() for normal teardown. [Source](../isaaclab_newton/physics/newton_manager.py#L233) | `manager.clear()` |
| `bind_control(actions, scene) -> bool` | Schedule each controller and asset command writer, preserving their execution order. [Source](../isaaclab_newton/physics/newton_manager.py#L262) | `manager.bind_control(actions, scene)` |
| `set_decimation(decimation, *, apply_every_physics_step=None) -> None` | Set the physics steps of one environment step. [Source](../isaaclab_newton/physics/newton_manager.py#L291) | `manager.set_decimation(decimation)` |
| `handles_decimation() -> bool` | Whether one `step` advances the whole decimation loop. [Source](../isaaclab_newton/physics/newton_manager.py#L305) | `manager.handles_decimation()` |
| `require_env_decimation() -> None` | Keep decimation in the environment for consumers requiring scene publication between physics steps. [Source](../isaaclab_newton/physics/newton_manager.py#L320) | `manager.require_env_decimation()` |
| `register_step_callback(fn, phase, *, graphable=True, name='') -> StepCallback` | Run `fn` at `phase` of every step until it is unregistered or the model is rebuilt. [Source](../isaaclab_newton/physics/newton_manager.py#L333) | `manager.register_step_callback(controller, StepPhase.CONTROL, graphable=True)` |
| `unregister_step_callback(callback) -> None` | Stop running a callback; a no-op after the model is rebuilt. [Source](../isaaclab_newton/physics/newton_manager.py#L351) | `manager.unregister_step_callback(callback)` |
| `activate_actuators() -> NewtonActuatorAdapter \| None` | Run the model's Newton actuators inside the step. Idempotent. Call after authored actuator data is available and before schedule preparation. [Source](../isaaclab_newton/physics/newton_manager.py#L360) | `manager.activate_actuators()` |
| `invalidate_fk(env_mask=None, env_ids=None, articulation_ids=None) -> None` | Mark articulations as needing FK and their worlds as needing a solver reset. [Source](../isaaclab_newton/physics/newton_manager.py#L372) | `manager.invalidate_fk(env_mask=row_mask, articulation_ids=view.articulation_ids)` |
| `invalidate_body_state(env_ids=None, env_mask=None, row_worlds=None) -> None` | Mark worlds whose maximal-coordinate body state was written, without requesting FK. [Source](../isaaclab_newton/physics/newton_manager.py#L393) | `manager.invalidate_body_state(env_mask=row_mask, row_worlds=row_worlds)` |
| `view_row_worlds(articulation_ids) -> wp.array` | Return the world of each row of an articulation view; see `newton_backend.view_row_worlds`. [Source](../isaaclab_newton/physics/newton_manager.py#L410) | `manager.view_row_worlds(view.articulation_ids)` |
| `add_model_change(change, env_mask=None) -> None` | Queue a property edit for the next state-read or step boundary. [Source](../isaaclab_newton/physics/newton_manager.py#L418) | `manager.add_model_change(flags, env_mask=world_mask)` |
| `transforms_may_change_on_graph_replay() -> bool` | Whether state was written during an outer capture, so graph replays may change it without notice. [Source](../isaaclab_newton/physics/newton_manager.py#L428) | `manager.transforms_may_change_on_graph_replay()` |
| `mark_particles_dirty() -> None` | Invalidate scene-data geometry after native particle writes. [Source](../isaaclab_newton/physics/newton_manager.py#L433) | `manager.mark_particles_dirty()` |
| `add_contact_sensor(body_names_expr=None, shape_names_expr=None, contact_partners_body_expr=None, contact_partners_shape_expr=None, verbose=False) -> SensorContact` | Add a contact sensor for reporting contacts between bodies or shapes; see `add_contact_sensor`. [Source](../isaaclab_newton/physics/newton_manager.py#L453) | `manager.add_contact_sensor(body_names_expr="Robot/.*")` |
| `add_imu_sensor(sites) -> SensorIMU` | Bind native IMU updates at the supplied site indices; request body_qdd before model finalization. [Source](../isaaclab_newton/physics/newton_manager.py#L472) | `manager.add_imu_sensor(sites)` |
| `prepare_builder(builder, solver_cfg) -> None` | Apply unresolved site requests and solver-specific normalization to the builder about to be finalized. [Source](../isaaclab_newton/physics/newton_manager.py#L478) | `manager.prepare_builder(builder, cfg.solver_cfg)` |
| `register_site(body_pattern, xform, *, per_world=False) -> str` | Request a site for injection into prototypes before replication. [Source](../isaaclab_newton/physics/newton_manager.py#L496) | `manager.register_site("Robot/base", wp.transform_identity())` |
| `inject_sites(main_builder, source_builders) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]` | Add unresolved sites to source builders, or to the main builder for global and shared-asset sites. [Source](../isaaclab_newton/physics/newton_manager.py#L520) | `manager.inject_sites(main_builder, source_builders)` |
| `get_world_builder_hooks() -> list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]]` | Return the hooks extending every world built by Newton replication. [Source](../isaaclab_newton/physics/newton_manager.py#L562) | `manager.get_world_builder_hooks()` |
| `record_clone(record, site_index_map) -> None` | Record native replication outputs for model finalization and consumers. [Source](../isaaclab_newton/physics/newton_manager.py#L566) | `manager.record_clone(record, site_index_map)` |
| `get_clone_record() -> NewtonCloneRecord \| None` | Return the outputs of the last native replication, or `None` without replication. [Source](../isaaclab_newton/physics/newton_manager.py#L576) | `manager.get_clone_record()` |
| `get_site_index_map() -> dict[str, SiteEntry]` | Return resolved sites by label. [Source](../isaaclab_newton/physics/newton_manager.py#L580) | `manager.get_site_index_map()` |
| `get_world_xforms() -> list[wp.transform] \| None` | Return the root transform of each cloned world, or `None` without replication. [Source](../isaaclab_newton/physics/newton_manager.py#L584) | `manager.get_world_xforms()` |
| `get_clone_source_builders() -> dict[str, ModelBuilder]` | Return per-source builders retained from replication, keyed by clone-plan source path. [Source](../isaaclab_newton/physics/newton_manager.py#L589) | `manager.get_clone_source_builders()` |
| `request_extended_state_attribute(attr) -> None` | Request an extended state attribute (e.g. `"body_qdd"`) from the builder before finalization. [Source](../isaaclab_newton/physics/newton_manager.py#L594) | `manager.request_extended_state_attribute(attr)` |
| `request_extended_contact_attribute(attr) -> None` | Request an extended contact attribute (e.g. `"force"`) from the builder before finalization. [Source](../isaaclab_newton/physics/newton_manager.py#L602) | `manager.request_extended_contact_attribute(attr)` |
| `finalize_backend() -> NewtonBackend` | Finalize this manager's builder into independently owned simulation buffers. [Source](../isaaclab_newton/physics/newton_manager.py#L616) | `manager.finalize_backend()` |
| `get_physics_dt() -> float` | Return the physics timestep, including standalone simulations. [Source](../isaaclab_newton/physics/newton_manager.py#L633) | `manager.get_physics_dt()` |
| `get_solver() -> SolverBase \| None` | Return the active Newton solver, or `None` before it is constructed. [Source](../isaaclab_newton/physics/newton_manager.py#L639) | `manager.get_solver()` |
| `get_model() -> Model \| None` | Return the active physics model. [Source](../isaaclab_newton/physics/newton_manager.py#L644) | `manager.get_model()` |
| `get_state_0() -> State \| None` | Return the current state. [Source](../isaaclab_newton/physics/newton_manager.py#L649) | `manager.get_state_0()` |
| `get_state_1() -> State \| None` | Return the spare state of double-buffered solvers. [Source](../isaaclab_newton/physics/newton_manager.py#L654) | `manager.get_state_1()` |
| `get_control() -> Control \| None` | Return the control inputs. [Source](../isaaclab_newton/physics/newton_manager.py#L659) | `manager.get_control()` |
| `get_contacts() -> Contacts \| None` | Return the current Newton contacts, if the active solver exposes them. [Source](../isaaclab_newton/physics/newton_manager.py#L664) | `manager.get_contacts()` |
| `get_scene_data_backend() -> SceneDataBackend \| None` | Return the SceneDataBackend for the SceneDataProvider. [Source](../isaaclab_newton/physics/newton_manager.py#L669) | `manager.get_scene_data_backend()` |
| `get_scene_data_provider() -> SceneDataProvider` | Return the active scene data provider. [Source](../isaaclab_newton/physics/newton_manager.py#L673) | `manager.get_scene_data_provider()` |
| `get_physics_sim_view() -> list` | Return the registered articulation views. [Source](../isaaclab_newton/physics/newton_manager.py#L677) | `manager.get_physics_sim_view()` |
| `create_visual_material_writer(batches) -> VisualMaterialWriter` | Compile material-to-shape addresses for the active Newton model. [Source](../isaaclab_newton/physics/newton_manager.py#L681) | `manager.create_visual_material_writer(batches)` |
| `create_visual_shape_color_writer(asset, body_names) -> VisualShapeColorWriter` | Compile selected articulation-body shape addresses for the active Newton model. [Source](../isaaclab_newton/physics/newton_manager.py#L685) | `manager.create_visual_shape_color_writer(asset, body_names)` |
| `video_capture_backend() -> str` | Newton GL headless perspective video capture. [Source](../isaaclab_newton/physics/newton_manager.py#L702) | `manager.video_capture_backend()` |
| `setup_deformable_body(prim, deformable_type, sim_mesh_prim, vis_mesh_prim) -> None` | Apply Newton's token deformable anchor schemas and sync the visual mesh geometry. [Source](../isaaclab_newton/physics/newton_manager.py#L707) | `manager.setup_deformable_body(prim, deformable_type, sim_mesh_prim, vis_mesh_prim)` |

### Inherited PhysicsManager methods

These are inherited by NewtonManager; methods overridden above are omitted here.

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `fix_articulation_root(articulation_prim, stage=None) -> Any` | Ensure that an articulation root has one enabled world fixed joint. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L108) | `manager.fix_articulation_root(articulation_prim)` |
| `register_callback(callback, event, order=0, name=None, wrap_weak_ref=True) -> CallbackHandle` | Register a callback for a physics event. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L236) | `manager.register_callback(callback, event)` |
| `deregister_callback(callback_id) -> None` | Remove a registered callback. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L276) | `manager.deregister_callback(callback_id)` |
| `dispatch_event(event, payload=None) -> None` | Dispatch an event to all registered callbacks. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L289) | `manager.dispatch_event(event)` |
| `clear_callbacks() -> None` | Remove all registered callbacks. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L305) | `manager.clear_callbacks()` |
| `pre_render() -> None` | Sync deferred physics state to the rendering backend. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L431) | `manager.pre_render()` |
| `after_visualizers_render() -> None` | Hook after visualizers have stepped during `isaaclab.sim.SimulationContext.render`. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L441) | `manager.after_visualizers_render()` |
| `get_device() -> str` | Get the physics simulation device. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L517) | `manager.get_device()` |
| `get_simulation_time() -> float` | Get the current simulation time in seconds. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L521) | `manager.get_simulation_time()` |
| `play() -> None` | Start or resume physics simulation. Default is no-op. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L529) | `manager.play()` |
| `pause() -> None` | Pause physics simulation. Default is no-op. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L533) | `manager.pause()` |
| `stop() -> None` | Stop physics simulation. Default is no-op. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L537) | `manager.stop()` |
| `wait_for_playing() -> None` | Block until the timeline is playing. Default is no-op. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L541) | `manager.wait_for_playing()` |
| `get_backend() -> str` | Get the tensor backend being used ("numpy" or "torch"). [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L580) | `manager.get_backend()` |
| `safe_callback_invoke(fn, *args, physics_manager=None) -> None` | Invoke a callback, catching exceptions that would be swallowed by external event buses. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L585) | `manager.safe_callback_invoke(fn)` |

### Lifecycle callback handles

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `__init__(callback_id, manager)` | Construct the object from the supplied arguments. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L55) | `manager.register_callback(on_ready, PhysicsEvent.PHYSICS_READY)` |
| `id` | Owner-local callback identifier; pass the handle or ID to deregister_callback. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L60) | `handle.id` |
| `deregister() -> None` | Remove this callback from the manager. [Source](../../isaaclab/isaaclab/physics/physics_manager.py#L63) | `handle.deregister()` |

### Functional backend API

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `create_newton_backend(cfg) -> NewtonBackend` | Finalize the shared builder of the simulation and bind a backend to the model. [Source](../isaaclab_newton/physics/newton_backend.py#L408) | `nb.create_newton_backend(backend_cfg)` |
| `init_solver(backend, *, relaxed_capture=False) -> None` | Construct the solver and contacts, compute body state from the initial joint state, and resolve capture. [Source](../isaaclab_newton/physics/newton_backend.py#L441) | `nb.init_solver(backend)` |
| `resolve_deterministic_mode(cfg) -> wp.DeterministicMode` | Translate the determinism request of a Newton configuration into a Warp mode. [Source](../isaaclab_newton/physics/newton_backend.py#L498) | `nb.resolve_deterministic_mode(cfg)` |
| `invalidate_fk(backend, env_mask=None, env_ids=None, articulation_ids=None) -> None` | Flag articulations for forward kinematics and their worlds for a solver reset, without host synchronization. [Source](../isaaclab_newton/physics/newton_backend.py#L523) | `nb.invalidate_fk(backend, env_mask=row_mask, articulation_ids=view.articulation_ids)` |
| `invalidate_body_state(backend, env_ids=None, env_mask=None, row_worlds=None) -> None` | Flag worlds whose maximal-coordinate body state was written, without requesting FK. [Source](../isaaclab_newton/physics/newton_backend.py#L555) | `nb.invalidate_body_state(backend, env_mask=row_mask, row_worlds=row_worlds)` |
| `invalidate_worlds(backend, worlds) -> None` | Flag whole worlds for a solver reset and forward kinematics, without host synchronization. [Source](../isaaclab_newton/physics/newton_backend.py#L579) | `nb.invalidate_worlds(backend, worlds)` |
| `view_row_worlds(backend, articulation_ids) -> wp.array` | Return the world of each row of an articulation view. Call once at bind time; it reads device arrays. [Source](../isaaclab_newton/physics/newton_backend.py#L600) | `nb.view_row_worlds(backend, view.articulation_ids)` |
| `forward(backend, *, force=False) -> None` | Reset solver internals and recompute body state for the flagged worlds, then clear the flags. Call notify_model_changes separately for queued property edits; manager.forward does both. [Source](../isaaclab_newton/physics/newton_backend.py#L614) | `nb.forward(backend)` |
| `mark_model_changed(backend, change, env_mask=None) -> None` | Queue a model-property transaction, selecting affected worlds entirely on the device. [Source](../isaaclab_newton/physics/newton_backend.py#L638) | `nb.mark_model_changed(backend, flags, env_mask=world_mask)` |
| `notify_model_changes(backend) -> None` | Commit queued property edits once, skipping solver work when every edit mask was empty. [Source](../isaaclab_newton/physics/newton_backend.py#L655) | `nb.notify_model_changes(backend)` |
| `register_step_callback(backend, fn, phase, *, graphable=True, name='') -> StepCallback` | Run `fn` at `phase` of every subsequent step. [Source](../isaaclab_newton/physics/newton_backend.py#L703) | `nb.register_step_callback(backend, controller, StepPhase.CONTROL, graphable=True)` |
| `unregister_step_callback(backend, callback) -> None` | Stop running a callback; unregistering an absent callback is a no-op. [Source](../isaaclab_newton/physics/newton_backend.py#L728) | `nb.unregister_step_callback(backend, callback)` |
| `activate_actuators(backend) -> NewtonActuatorAdapter \| None` | Run the model's Newton actuators inside the step. Idempotent. Call after authored actuator data is available and before schedule preparation. [Source](../isaaclab_newton/physics/newton_backend.py#L740) | `nb.activate_actuators(backend)` |
| `add_contact_sensor(backend, body_names_expr=None, shape_names_expr=None, contact_partners_body_expr=None, contact_partners_shape_expr=None, verbose=False) -> SensorContact` | Add a contact sensor between bodies or shapes; identical requests share one sensor. [Source](../isaaclab_newton/physics/newton_backend.py#L768) | `nb.add_contact_sensor(backend, body_names_expr="Robot/.*")` |
| `add_imu_sensor(backend, sites) -> SensorIMU` | Bind native IMU updates at the supplied site indices; request body_qdd before model finalization. [Source](../isaaclab_newton/physics/newton_backend.py#L820) | `nb.add_imu_sensor(backend, sites)` |
| `compile_label_pattern(expr) -> re.Pattern[str] \| None` | Compile Isaac Lab selector expressions for Newton's full label matching. [Source](../isaaclab_newton/physics/newton_backend.py#L836) | `nb.compile_label_pattern(expr)` |
| `build_step_graph(backend, steps) -> StepGraph` | Unroll `steps` physics steps into straight-line operations with every buffer bound. [Source](../isaaclab_newton/physics/newton_backend.py#L846) | `nb.build_step_graph(backend, steps)` |
| `prepare(backend) -> StepGraph` | Build the step graph for `NewtonBackend.steps_per_call` unless the current one matches. [Source](../isaaclab_newton/physics/newton_backend.py#L959) | `nb.prepare(backend)` |
| `step(backend) -> StepGraph` | Advance the prepared schedule with live reads throughout its controller phases. [Source](../isaaclab_newton/physics/newton_backend.py#L977) | `nb.step(backend)` |
| `record_step(backend) -> StepGraph` | Record `NewtonBackend.steps_per_call` physics steps into the caller's active CUDA graph capture. [Source](../isaaclab_newton/physics/newton_backend.py#L1013) | `nb.record_step(backend)` |
| `execution_stream(device)` | Order mixed eager Torch/Warp work with the caller, using the simulation's Warp stream. [Source](../isaaclab_newton/physics/newton_backend.py#L1099) | `nb.execution_stream(device)` |
| `capture_graph(device, fn, *, relaxed=False) -> CapturedGraph` | Record Torch and Warp work on one stream without advancing the simulation. [Source](../isaaclab_newton/physics/newton_backend.py#L1116) | `nb.capture_graph(device, fn)` |
| `run_query(backend, timestamp, query, graph, *, use_cuda_graph=True) -> tuple[tuple[int, ...], CapturedGraph] \| None` | Refresh shared BVHs once per publication and run a consumer-owned query graph. [Source](../isaaclab_newton/physics/newton_backend.py#L1169) | `nb.run_query(backend, timestamp, query, graph)` |

### NewtonBackend methods

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `__init__(model, physics_cfg=None, *, dt=None, deformable_ranges=None)` | Bind native buffers to a finalized model. [Source](../isaaclab_newton/physics/newton_backend.py#L291) | `NewtonBackend(model, cfg, dt=0.01)` |
| `create_visual_material_writer(batches) -> VisualMaterialWriter` | Bind material writes to this resource's native shape-color buffer. [Source](../isaaclab_newton/physics/newton_backend.py#L390) | `backend.create_visual_material_writer(batches)` |
| `close() -> None` | Drop native handles after consumers release their bindings. [Source](../isaaclab_newton/physics/newton_backend.py#L394) | `backend.close()` |

### StepGraph methods

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `__init__(ops, steps)` | Initialize the graph. [Source](../isaaclab_newton/physics/newton_backend.py#L195) | `StepGraph(ops, steps)` |
| `captured` | Whether graphable segments replay from captured graphs. [Source](../isaaclab_newton/physics/newton_backend.py#L211) | `schedule.captured` |
| `capture(capture) -> None` | Record every graphable segment without executing it. [Source](../isaaclab_newton/physics/newton_backend.py#L215) | `schedule.capture(capture)` |
| `launch() -> None` | Advance `steps` physics steps, replaying captured segments. [Source](../isaaclab_newton/physics/newton_backend.py#L223) | `schedule.launch()` |
| `record() -> None` | Launch every operation so that the caller's active capture records them. [Source](../isaaclab_newton/physics/newton_backend.py#L234) | `schedule.record()` |

### CapturedGraph methods

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `launch() -> None` | Replay on the simulation stream, ordered with the caller's Torch work. [Source](../isaaclab_newton/physics/newton_backend.py#L1092) | `graph.launch()` |

### NewtonSolver adapter hooks

These are engine/adapter calls, not extra calls to add around `nb.step`. Override hooks
when implementing a solver; keep runtime state in the supplied backend.

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `create_builder(up_axis=None, *, physics_cfg=None, **kwargs) -> ModelBuilder` | Create a `ModelBuilder` configured with default settings. [Source](../isaaclab_newton/physics/newton_solver.py#L31) | `adapter.create_builder()` |
| `get_usd_import_schema_resolvers(solver_cfg) -> list[SchemaResolver]` | Return ordered schema resolvers for physics-model USD imports. [Source](../isaaclab_newton/physics/newton_solver.py#L63) | `adapter.get_usd_import_schema_resolvers(cfg.solver_cfg)` |
| `create_solver(model, solver_cfg, deterministic_mode=wp.DeterministicMode.NOT_GUARANTEED) -> SolverBase` | Construct the configured solver. Coupled solvers call this to build their entries. [Source](../isaaclab_newton/physics/newton_solver.py#L106) | `adapter.create_solver(backend.model, cfg.solver_cfg)` |
| `solver_kwargs(solver_cls, solver_cfg, deterministic_mode) -> dict` | Return the configuration fields that match the solver constructor. [Source](../isaaclab_newton/physics/newton_solver.py#L129) | `adapter.solver_kwargs(solver_cls, cfg.solver_cfg, deterministic_mode)` |
| `validate_cfg(backend) -> None` | Reject configurations the solver cannot run, before it is constructed. [Source](../isaaclab_newton/physics/newton_solver.py#L147) | `adapter.validate_cfg(backend)` |
| `register_builder_attributes(builder, solver_cfg) -> None` | Register custom attributes the solver reads from the model. [Source](../isaaclab_newton/physics/newton_solver.py#L161) | `adapter.register_builder_attributes(builder, cfg.solver_cfg)` |
| `registers_builder_attributes_from(solver_cls, solver_cfg) -> bool` | Whether this adapter registers the supplied native solver class's custom builder attributes. [Source](../isaaclab_newton/physics/newton_solver.py#L172) | `adapter.registers_builder_attributes_from(solver_cls, cfg.solver_cfg)` |
| `prepare_solver_builder(builder, solver_cfg) -> None` | Normalize a complete builder for the solver before finalization. The default is a no-op. [Source](../isaaclab_newton/physics/newton_solver.py#L184) | `adapter.prepare_solver_builder(builder, cfg.solver_cfg)` |
| `uses_collision_pipeline(backend) -> bool` | Whether contacts come from Newton's `newton.CollisionPipeline` instead of the solver. [Source](../isaaclab_newton/physics/newton_solver.py#L193) | `adapter.uses_collision_pipeline(backend)` |
| `supports_body_forces(backend) -> bool` | Whether the solver consumes applied rigid-body forces from `newton.State.body_f`. [Source](../isaaclab_newton/physics/newton_solver.py#L198) | `adapter.supports_body_forces(backend)` |
| `supports_graph_capture(backend) -> bool` | Whether the configured solver can be recorded into a CUDA graph. [Source](../isaaclab_newton/physics/newton_solver.py#L203) | `adapter.supports_graph_capture(backend)` |
| `create_contacts(backend) -> Contacts \| None` | Allocate contacts the solver reports from internal collision detection. [Source](../isaaclab_newton/physics/newton_solver.py#L208) | `adapter.create_contacts(backend)` |
| `prepare_contacts(backend) -> None` | Bind solver-owned buffers to newly allocated contacts. The default is a no-op. [Source](../isaaclab_newton/physics/newton_solver.py#L216) | `adapter.prepare_contacts(backend)` |
| `initialize_output_state(backend, state) -> None` | Initialize solver-owned buffers of the output state of double-buffered solvers. The default is a no-op. [Source](../isaaclab_newton/physics/newton_solver.py#L220) | `adapter.initialize_output_state(backend, backend.state_0)` |
| `prepare_step(backend, state) -> None` | Refresh solver acceleration structures once per physics step, before collision; set `prepares_step`. [Source](../isaaclab_newton/physics/newton_solver.py#L224) | `adapter.prepare_step(backend, backend.state_0)` |
| `step_solver(backend, state_in, state_out, contacts, dt) -> None` | Advance one solver substep. [Source](../isaaclab_newton/physics/newton_solver.py#L231) | `adapter.step_solver(backend, state_in, state_out, backend.contacts, backend.dt)` |
| `reset_solver(backend, state, world_mask) -> None` | Clear solver-owned history for masked worlds while keeping authored joint state. [Source](../isaaclab_newton/physics/newton_solver.py#L246) | `adapter.reset_solver(backend, backend.state_0, world_mask)` |
| `eval_fk(backend, state, world_mask, fk_mask) -> None` | Update body state from joint coordinates for masked articulations. [Source](../isaaclab_newton/physics/newton_solver.py#L257) | `adapter.eval_fk(backend, backend.state_0, world_mask, fk_mask)` |
| `check_status(backend, captured) -> None` | Raise asynchronous solver failures after a step. The default is a no-op. [Source](../isaaclab_newton/physics/newton_solver.py#L271) | `adapter.check_status(backend, captured)` |
| `log_debug(backend) -> None` | Log solver diagnostics after a step when debug mode is enabled. The default is a no-op. [Source](../isaaclab_newton/physics/newton_solver.py#L280) | `adapter.log_debug(backend)` |
| `create_fixed_tendon_control(articulation, model) -> Any` | Build the solver's fixed-tendon command adapter for `articulation`. [Source](../isaaclab_newton/physics/newton_solver.py#L284) | `adapter.create_fixed_tendon_control(articulation, backend.model)` |

### Builder factory

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `create_newton_builder(cfg) -> ModelBuilder` | Return the owner's shared builder; registry factory, not a second independent builder. [Source](../isaaclab_newton/physics/newton_manager.py#L56) | `create_newton_builder(builder_cfg)` |

### Terrain construction helper

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `inject_terrain_heightfields(stage, builder, *, root_paths, device='cpu') -> list[str]` | Replace height-field-tagged terrain colliders with Newton heightfields. [Source](../isaaclab_newton/physics/newton_builder.py#L22) | `newton_builder.inject_terrain_heightfields(stage, builder, root_paths=root_paths)` |

### Scene-data bridge

| Signature | Purpose | Usage shape |
| --- | --- | --- |
| `__init__(backend)` | Initialize the scene data backend. [Source](../isaaclab_newton/physics/newton_scene_data.py#L29) | `NewtonSceneDataBackend(lambda: manager.backend)` |
| `initialize_geometry(geometry_batches, cable_bindings) -> None` | Bind imported geometry paths to native particle ranges and capsule endpoints. [Source](../isaaclab_newton/physics/newton_scene_data.py#L41) | `scene_data.initialize_geometry(geometry_batches, cable_bindings)` |
| `native_geometry_formats` | Native geometry formats supported by this scene-data bridge. [Source](../isaaclab_newton/physics/newton_scene_data.py#L72) | `scene_data.native_geometry_formats` |
| `get_geometry_batches(output_format=SceneDataFormat.Points)` | Publish native arrays; SDP derives cable endpoints and applies destination layouts. [Source](../isaaclab_newton/physics/newton_scene_data.py#L75) | `scene_data.get_geometry_batches()` |
| `transforms` | Publish the authoritative native pointer, including solver state-buffer swaps. [Source](../isaaclab_newton/physics/newton_scene_data.py#L87) | `scene_data.transforms` |
| `transform_count` | Return the number of rigid body transforms in the Newton sim. [Source](../isaaclab_newton/physics/newton_scene_data.py#L96) | `scene_data.transform_count` |
| `transform_paths` | Return the prim paths for each rigid body transform. [Source](../isaaclab_newton/physics/newton_scene_data.py#L101) | `scene_data.transform_paths` |
| `model` | Current finalized model, or None before model construction. [Source](../isaaclab_newton/physics/newton_scene_data.py#L108) | `scene_data.model` |
| `state` | Return native physics state, consistent with authored state, without entering the rendering path. [Source](../isaaclab_newton/physics/newton_scene_data.py#L113) | `scene_data.state` |

## Usage recipes

### Bind a controller on every hard reset

This integrates with the scene setup from the design guide. The controller writes
raw generalized efforts, suitable for a model using direct efforts; native actuator
models may instead require their target fields. The fixed input buffer can be
updated between steps without recapturing.

```python
import warp as wp
from isaaclab.physics import PhysicsEvent
from isaaclab_newton.physics import StepPhase


def install_effort_controller(manager):
    buffers = {}

    def on_ready(payload):
        backend = manager.backend
        effort = wp.zeros_like(backend.control.joint_f)
        buffers["effort"] = effort

        def controller():
            wp.copy(backend.control.joint_f, effort)

        manager.register_step_callback(
            controller, StepPhase.CONTROL, graphable=True, name="direct_effort"
        )

    handle = manager.register_callback(on_ready, PhysicsEvent.PHYSICS_READY)
    return buffers, handle


# Before sim.reset(), after constructing the context/scene:
# buffers, handle = install_effort_controller(sim.physics_manager)
# sim.reset()
# buffers["effort"].fill_(0.1)
# sim.step(render=False)
# handle.deregister()  # Prevent future rebinding; existing step callback remains.
```

To stop an active step callback, keep the returned `StepCallback` and call
`manager.unregister_step_callback(callback)`; the next step rebuilds its schedule.
`handle.deregister()` above removes only the lifecycle subscription. Hard reset
discards the old backend's step callbacks automatically.

| Phase | Callback shape | When it runs |
| --- | --- | --- |
| `StepPhase.CONTROL` | `controller()` | After collision, before Newton actuators, each physics tick. |
| `StepPhase.POST_ACTUATOR` | `publish_effort()` | After the final physics tick's actuators, before its solver substeps. |
| `StepPhase.STATE_FORCE` | `apply_force(state_in)` | Before every solver substep; add to the supplied state's forces. |
| `StepPhase.POST_STEP` | `after_tick()` | After every physics tick, before native sensor refresh. |

A `POST_STEP` callback sees the newly advanced state, but native sensor values
refresh immediately afterward. Controllers on the next tick see the refreshed data.

### Record masked state and property resets

The writer functions below are explicit application stubs: implement masked device
writes using the model's joint/body indexing, and initialize their scratch buffers
before capture. State and property masks are device arrays whose values can change
on every replay. The recipe assumes an already initialized backend, such as the
`make_backend()` example in the design guide.

```python
import warp as wp
from newton import ModelFlags
from isaaclab_newton.physics import newton_backend as nb


def capture_reset_and_step(backend, write_state, write_properties):
    """Writers are graph-safe callables: writer(backend, world_mask)."""
    world_mask = wp.zeros(backend.model.world_count, dtype=wp.bool, device=backend.device)
    nb.prepare(backend)

    def advance():
        write_state(backend, world_mask)       # Application: restore selected states.
        nb.invalidate_worlds(backend, world_mask)
        write_properties(backend, world_mask)  # Application: edit mass/inertia consistently.
        nb.mark_model_changed(backend, ModelFlags.BODY_INERTIAL_PROPERTIES, world_mask)
        nb.record_step(backend)                # Commits pending changes, then advances.

    graph = nb.capture_graph(backend.device, advance)
    return world_mask, graph


# mask, graph = capture_reset_and_step(backend, my_state_writer, my_property_writer)
# mask.fill_(True)
# graph.launch()    # Reset selected worlds and step.
# mask.zero_()
# graph.launch()    # Step without a reset/property refresh.
```

Choose property categories supported by the solver. For example, Featherstone
reports runtime inertia edits as unsupported; a queued notification cannot add
support the native solver lacks. For state-only resets, omit the property writer
and `mark_model_changed`. For reset-only execution, commit properties with
`notify_model_changes`, then use `forward(backend, force=True)` instead of
`record_step`. Use the provided capture helper for property-changing resets so
conditional refresh nodes retain their scratch allocations.

### Request sites and bind sensors

This example shows where native sensor setup belongs. Replace the body label with
one in the imported prototype. Finalized site mappings distinguish a global site
from per-world/per-body sites; a wildcard may create several sites in each world.

```python
import warp as wp
from isaaclab.physics import PhysicsEvent


def install_sensors(manager):
    manager.request_extended_state_attribute("body_qdd")
    site_label = manager.register_site("Robot/base", wp.transform_identity())
    sensors = {}

    def on_ready(payload):
        global_index, world_sites = manager.get_site_index_map()[site_label]
        sites = [global_index] if global_index is not None else [
            index for indices in world_sites for index in indices
        ]
        sensors["imu"] = manager.add_imu_sensor(sites)
        sensors["contacts"] = manager.add_contact_sensor(
            body_names_expr="Robot/.*", contact_partners_body_expr="Ground.*"
        )

    handle = manager.register_callback(on_ready, PhysicsEvent.PHYSICS_READY)
    return sensors, handle


# Call install_sensors(sim.physics_manager) before scene replication/finalization.
# sim.reset() binds sensors; each subsequent physics tick updates them.
```

Contact reporting depends on solver capabilities and allocated contact attributes.
`add_contact_sensor` requests force reporting and refreshes contact allocation if
needed. Sensor setup changes the schedule, so complete setup before outer capture.
Scene-level sensor wrappers normally perform these lifecycle calls for users.

### Define a downstream solver adapter

This complete example wraps Newton's existing XPBD solver to show the minimal
adapter/configuration contract. A new native solver supplies its own constructor,
capabilities and any additional hooks from the reference tables.

```python
from newton.solvers import SolverXPBD
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, NewtonSolver, XPBDSolverCfg


class MyXPBDAdapter(NewtonSolver):
    solver_class = SolverXPBD
    supports_deterministic = True
    supports_heterogeneous_worlds = True


@configclass
class MyXPBDCfg(XPBDSolverCfg):
    class_type: type[NewtonSolver] = MyXPBDAdapter


physics_cfg = NewtonCfg(solver_cfg=MyXPBDCfg(iterations=8))
# SimulationCfg(physics=physics_cfg) uses the standard NewtonManager.
# Or use NewtonCfg(class_type=MyNewtonManager, solver_cfg=MyXPBDCfg()).
```

Adapter defaults provide double-buffered stepping, Newton collision handling and
FK. Set `single_state=True` only for solvers that accept identical input/output
state. Register custom builder attributes before asset import; initialize special
output state in `initialize_output_state`; keep optional pre-step device work in
`prepare_step` with `prepares_step=True`. Native solver creation and GPU execution
remain separate from the stateless adapter class.

### Registry and query integration

```python
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonBackendCfg
from isaaclab_newton.physics import newton_backend as nb


def get_construction_builder(sim):
    return sim.get_or_create_backend(
        NewtonBuilderCfg(physics_cfg=sim.cfg.physics, manager=sim.physics_manager)
    )


def get_finalized_backend(sim):
    # Integration hook: use only after model authoring is complete.
    # Normal applications let sim.reset() request this resource.
    return sim.get_or_create_backend(
        NewtonBackendCfg(
            physics_cfg=sim.cfg.physics, device=sim.device, manager=sim.physics_manager
        )
    )


class QueryConsumer:
    def __init__(self, backend, query):
        self.backend = backend
        self.query = query       # Application stub: bound camera/raycast device work.
        self.graph = None

    def update(self, publication_timestamp):
        self.graph = nb.run_query(
            self.backend, publication_timestamp, self.query, self.graph
        )
```

`publication_timestamp` is the sum of the scene-data backend's monotonic transform
and geometry timestamps. The backend refits shared acceleration structures once
per publication; each consumer owns its query graph. Reconstruct consumers after
a hard reset. `NewtonSceneDataBackend` reads `manager.backend` through a callback,
so scene publication follows the replacement backend rather than stale buffers.
