# Newton Fused Inertial-Property Writers

## Goal

Keep Newton's primary and inverse body inertial properties consistent while
reducing each mass or inertia update from two Warp kernel launches to one.
The change remains private to the Newton backend and does not alter the public
asset APIs.

## Design

Add Newton-specific fused mass and inertia writers to
`isaaclab_newton.assets.kernels`, with index and mask variants for each
property.

The mass writers will:

1. Resolve the selected environment and public body.
2. Resolve the backend body through the optional articulation ordering map.
3. Write the new mass to the public and backend buffers as required.
4. Update inverse mass and inverse inertia for that backend body. A nonpositive
   mass makes both inverse values zero; a positive mass restores them from the
   new mass and current inertia.

The inertia writers will perform the same selection and ordering steps, copy
all nine inertia components, and update inverse inertia from the new matrix
when the current mass is positive. They will leave inverse mass unchanged.

Each kernel will launch over `(environment, body)`. In particular, inertia
copying will happen in a loop inside one thread so only one thread updates the
corresponding inverse matrix.

Indexed kernels will accept both 32-bit and 64-bit Torch or Warp selectors
through `Any` selector types and `IndexKernelDispatcher`. Mask kernels will
retain boolean selectors. The same kernels will serve Newton articulations,
rigid objects, and rigid-object collections; assets without body ordering will
pass aliased public/backend buffers and disable the ordering scatter.

The shared `isaaclab.assets.articulation.ordering_kernels` utilities will not
change. PhysX and OvPhysX do not expose Newton's inverse arrays and should not
acquire Newton-specific arguments or branches.

## Integration

Replace the paired primary-write and inverse-update launches in Newton's
`set_masses_index`, `set_masses_mask`, `set_inertias_index`, and
`set_inertias_mask` implementations with the corresponding fused launch.
Continue notifying `SimulationManager` once after the launch.

No public configuration, method signature, property, or changelog category
changes are required beyond the existing PR scope.

## Testing

Keep coverage compact by extending the existing inverse-property regression
tests rather than adding parallel suites:

- Exercise indexed mass and inertia writes with both `int32` and `int64`
  selectors.
- Retain the reordered-articulation assertions to verify public-to-backend
  mapping.
- Retain positive-to-zero and zero-to-positive mass transitions.
- Retain mask coverage for each asset family.
- Confirm the focused regression tests fail when the inverse update is removed
  from the fused writer and pass with it present.

Pre-commit and the focused Newton asset tests must pass before committing and
pushing implementation changes.

## Out of Scope

This design does not expose inverse mass or inverse inertia publicly, add
inverse-property handling to PhysX or OvPhysX, or change actuator target-mode
inference. The separately identified articulation-root and free/fixed-joint
target-mode corrections remain independent follow-up changes in the same PR.
