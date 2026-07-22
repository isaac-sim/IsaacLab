# OVPhysX Deformable Support Design

## Summary

Add volume and surface deformable-object support to the OVPhysX backend while
preserving Isaac Lab's backend-neutral deformable API and matching the PhysX
implementation wherever the OVPhysX runtime exposes equivalent behavior.

The implementation is based on current `origin/develop` and uses Marco
Malesiani's prototype branch as a behavioral reference. It is not a direct
port: OVPhysX 0.5 now exposes tensor dtype metadata, surface deformable
tensors, and additional material properties, while Isaac Lab has introduced a
shared `OvPhysxView` binding abstraction since Marco's branch diverged.

## Goals

- Register an OVPhysX implementation for the existing
  `isaaclab.assets.DeformableObject` factory.
- Support both volume and surface deformable bodies.
- Match the PhysX deformable asset, data, and view-style APIs where OVPhysX
  exposes equivalent tensors.
- Preserve OVPhysX-specific device residency, full-buffer write semantics,
  dtype metadata, and stage-loading constraints.
- Support indexed and masked writes for nodal state and material properties.
- Expose volume and surface connectivity with the wheel-reported `int32`
  dtype.
- Expose every deformable material property available in OVPhysX 0.5:
  dynamic friction, Young's modulus, Poisson's ratio, elasticity damping,
  bending stiffness, thickness, and bending damping.
- Test volume and surface assets through both isolated adapters and the real
  CUDA backend.

## Non-goals

- Changing the OVPhysX dependency or Warp pins. PR #6660 owns those changes.
- Adding Kit, `omni.physics.tensors`, or other Isaac Sim runtime dependencies
  to `isaaclab_ovphysx`.
- Implementing procedural mesh-to-volume tetrahedralization in the kitless
  runtime. Tests and users must provide already-authored deformable schemas and
  topology.
- Emulating unsupported surface kinematic targets.
- Adding implicit CPU/GPU data staging to `OvPhysxView`.
- Refactoring unrelated OVPhysX assets or the backend-neutral deformable API.

## Prerequisites and compatibility

PR #6660 updates Isaac Lab to OVPhysX 0.5 and the stable Warp 1.15 release. It
also migrates stage loading from an exported file to an in-memory OVStage. This
feature treats that PR as an external prerequisite and does not edit
`pyproject.toml`, `uv.lock`, or OV dependency-resolution workflows.

Development and validation use the explicitly installed local wheel:

`/home/antoiner/ovphysx-0.5.2+head.f62c22207c-py3-none-manylinux_2_35_x86_64.whl`

The deformable adapter requires OVPhysX bindings that expose `shape`, `dtype`,
and `spec` metadata and the deformable tensor types used below. If dtype
metadata is missing, the adapter raises a focused compatibility error rather
than reinterpreting the buffer as `float32`.

## Architecture

### Generic OVPhysX view

`OvPhysxView` remains the common binding-management layer. It will stop
assuming that every binding is `float32` and instead derive the scalar Warp
dtype from `binding.dtype`.

The conversion is generic and keyed by the DLPack type code, bit width, and
lane count. It is not a per-`TensorType` dtype table. The initial supported
scalar mappings cover the wheel's current types:

- DLPack float, 32 bits, one lane -> `wp.float32`
- DLPack signed integer, 32 bits, one lane -> `wp.int32`
- DLPack unsigned integer, 8 bits, one lane -> `wp.uint8`

Existing semantic structured mappings, such as seven `float32` components to
`wp.transformf`, remain Isaac Lab-side because DLPack describes scalar layout,
not physical meaning. Existing rigid-body and articulation callers continue to
receive the same structured buffers.

Buffer allocation and validation compare against the binding-reported scalar
dtype and shape. A structured Warp array may be reinterpreted only when its
scalar dtype matches the binding dtype and its byte size exactly matches the
binding shape. No numeric conversion or device transfer occurs implicitly.

### Deformable body adapter

A thin `OvPhysxDeformableBodyView` composes `OvPhysxView` and exposes the
PhysX-style method and property names used by `DeformableObject` and advanced
callers. It selects one immutable tensor mapping at construction based on the
detected deformable type.

Volume mapping:

- simulation nodal positions
- simulation nodal velocities
- simulation nodal kinematic targets
- rest nodal positions
- simulation element indices
- collision element indices

Surface mapping:

- simulation nodal positions
- simulation nodal velocities
- rest nodal positions
- simulation element indices

The adapter exposes count, prim paths, maximum node counts, maximum element
counts, typed getters, indexed setters, and cleanup. Volume-only operations
remain explicit; surface kinematic-target access raises the same `ValueError`
as the PhysX asset.

The adapter retains OVPhysX's full-buffer partial-write contract. A setter's
data buffer always has the binding's complete first dimension. `indices` or
`mask` selects the rows that OVPhysX applies.

### Deformable material adapter

`OvPhysxDeformableMaterialView` also composes `OvPhysxView`. It exposes
PhysX-style getters and setters for all material properties available in the
wheel. Material bindings are CPU-resident even when simulation state is on a
CUDA device.

Setters accept a full material buffer plus optional indices or mask, following
OVPhysX binding semantics. The adapter performs no hidden CUDA-to-CPU copy.
The higher-level asset-facing paths may accept Torch or Warp inputs, but must
make any required staging explicit.

### Asset and data classes

`isaaclab_ovphysx.assets.deformable_object.DeformableObject` subclasses
`BaseDeformableObject`. It follows the PhysX class's public method names,
argument names, shape validation, deprecated wrappers inherited from the base
class, debug visualization behavior, and surface/volume error behavior.

`DeformableObjectData` subclasses `BaseDeformableObjectData`. It owns stable
Warp buffers for:

- nodal positions in world frame `[m]`
- nodal velocities in world frame `[m/s]`
- combined nodal state `[m, m/s]`
- derived mean root position `[m]`
- derived mean root velocity `[m/s]`
- default nodal state
- volume-only kinematic targets `[m, flag]`

Properties refresh lazily using the inherited simulation timestamp. Reads fill
stable caller-owned buffers through `OvPhysxView.read_into`, preserving binding
read caches and stable `ProxyArray` wrappers.

### Factory and public exports

The package exports `DeformableObject` and `DeformableObjectData` through
runtime modules and `.pyi` stubs consistent with other OVPhysX assets. The
backend factory resolves the OVPhysX implementation without changing the
backend-neutral public class names.

No public symbol is removed or renamed.

## Initialization and discovery

The asset resolves the configured template prim and finds exactly one authored
`OmniPhysicsDeformableBodyAPI` root below it. Kitless USD can omit registered
schema objects from `GetAppliedSchemas()`, so discovery also checks authored
`apiSchemas` metadata as in Marco's prototype.

The deformable type is determined from authored material/body schemas and mesh
topology:

- volume material or a `TetMesh` descendant -> volume
- surface material or a `Mesh` descendant without a volume indication ->
  surface

Ambiguous or unsupported combinations fail during initialization with the
asset path and detected schemas in the error message.

The material binding is resolved through the direct physics-purpose material
relationship. If a material exists, the material adapter is created using the
equivalent wildcard path. If no material is bound, initialization succeeds,
OVPhysX uses its default material behavior, a warning explains that runtime
material edits are unavailable, and `material_physx_view` is `None`.

After views are created, the asset allocates data buffers and captures the
spawn-time nodal state as the default state. Volume kinematic targets are read,
their flags are initialized to Isaac Lab's free-node convention (`1.0`), and
the initialized buffer is written back to OVPhysX before partial target writes
are allowed. Surface assets leave `nodal_kinematic_target` as `None`.

## Runtime data flow

### Reads

The body adapter reads directly into stable device-resident Warp buffers. Nodal
state is recomputed only after position or velocity timestamps become stale.
Root position and velocity are the per-body means of simulation nodes, matching
PhysX behavior.

Rest positions and connectivity are available through the body view. The
simulation connectivity dtype comes from the wheel and is exposed as
`wp.int32`; it is never cast through `float32`.

### Writes

The asset accepts `torch.Tensor`, `wp.array`, and `ProxyArray` inputs where the
backend-neutral interface permits them. Inputs are validated against the
number of selected environments, maximum node count, component width, and
expected scalar dtype.

For indexed writes:

1. Resolve environment indices to device-resident `wp.int32`.
2. Copy selected input rows into the asset's complete internal buffer.
3. Pass the complete buffer and indices to the body adapter.
4. Invalidate or update the corresponding lazy buffer timestamp.

Mask writes use the inherited base behavior to resolve selected rows, while
the material adapter may forward a wheel-compatible mask directly where that
preserves the established OVPhysX pattern.

Volume kinematic-target writes update the complete target buffer before the
binding write. Surface target writes fail before buffer access.

## Stage loading and replication

OVPhysX runtime cloning does not replicate authored deformable body and
material schema prims. Creating a deformable therefore marks the manager as
requiring the complete authored environment set for the next stage load.

For current `develop`, that means exporting the full composed USD rather than
the env-0-only stage. After PR #6660, the same neutral manager flag selects
full-stage in-memory serialization rather than env-0-only serialization. The
feature will avoid coupling the public marker name to the transport mechanism
(`export` versus OVStage).

When full-stage loading is active, queued runtime clone operations are skipped
and cleared because the authored environment copies already exist. Scenes
without deformables retain the existing env-0 plus runtime-clone fast path.
Mixed rigid/deformable scenes load all authored objects from the same full
stage and do not duplicate rigid actors through runtime cloning.

## Error handling

- Missing or multiple matching deformable body roots: `RuntimeError` with the
  configured path and match count.
- Unsupported or ambiguous deformable type: `RuntimeError` with discovered
  schema/topology evidence.
- Missing required tensor binding: `RuntimeError` chaining the OVPhysX binding
  creation failure.
- Missing dtype metadata or unsupported DLPack scalar: focused compatibility
  or dtype error with the reported type fields.
- Wrong shape, dtype, or device: the existing nested `OvPhysxView` error types.
- Surface kinematic-target access: the PhysX-compatible `ValueError`.
- Missing material binding: warning plus `material_physx_view = None`; the
  deformable itself remains supported.
- CPU deformable simulation: clear runtime failure because current OVPhysX
  deformable tensors require DirectGPU mode.

Cleanup destroys body and material bindings and clears references during asset
invalidation. The manager resets the full-stage requirement between simulation
contexts.

## Testing strategy

Development follows red-green-refactor. Regression tests are run before
implementation and must fail for the missing behavior, then pass after the
minimal implementation.

### Generic view tests

Extend `source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py` with fake bindings
that report DLPack float32, int32, and uint8 specs. Tests cover allocation,
structured reinterpretation, explicit output buffers, wrong-dtype rejection,
read-only rejection, and preservation of current float behavior.

### Adapter tests

Add focused tests using fake OVPhysX bindings for:

- volume and surface tensor mappings
- count and maximum-dimension properties
- nodal getter/setter shapes
- int32 volume and surface connectivity
- volume-only kinematic targets
- indexed and masked full-buffer writes
- all seven material getters and setters
- CPU material residency
- binding cleanup

### Real backend tests

Add CUDA tests under `source/isaaclab_ovphysx/test/assets/` using explicitly
authored deformable schemas and topology:

- Volume initialization, nodal state read/write, int32 tetrahedral
  connectivity, kinematic targets, material properties, stepping, and derived
  root state.
- Surface initialization, nodal state read/write, int32 triangular
  connectivity, all applicable material properties, stepping, absence of a
  target buffer, and rejected kinematic-target writes.
- Multiple cloned environments through the full-stage fallback.
- A mixed rigid/deformable scene proving runtime clones are not duplicated.
- A soft-lift task smoke derived from Marco's prototype when the current task
  remains compatible with authored deformable input.

Tests requiring OVPhysX deformables are CUDA-gated and use the existing
OVPhysX/IsaacLab test markers.

### Verification

The fresh development environment is `.venv` in the isolated worktree and has
the local OVPhysX wheel installed explicitly. Verification includes:

- Focused generic view and adapter unit tests.
- OVPhysX manager replication/full-stage tests.
- Real volume and surface backend tests on CUDA.
- Relevant task smoke tests.
- Python compilation for touched modules.
- `./isaaclab.sh -f` for repository-wide pre-commit hooks.
- A second `./isaaclab.sh -f` run if the first run modifies files.

## Documentation and changelog

Add one `isaaclab_ovphysx` minor changelog fragment describing volume and
surface deformable support. Update generated public API documentation only if
the repository's existing OVPhysX asset documentation requires explicit new
module entries; if new public exports change generated docs, run
`./isaaclab.sh -d` as required by the repository guidelines.
