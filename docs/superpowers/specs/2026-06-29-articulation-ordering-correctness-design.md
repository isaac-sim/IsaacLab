# Articulation Ordering Correctness Follow-up

## Status

Approved.

This specification tightens the articulation ordering implementation after the
first cross-backend integration review. It does not change the default:
`joint_ordering=None` and `body_ordering=None` continue to expose the active
backend order through the existing direct path.

## Goals

- Make every public joint/body quantity obey the configured public order.
- Reorder data exactly once at the public/backend boundary.
- Keep the `None` ordering path allocation-free and free of reorder launches.
- Preserve existing `BaseArticulation` subclasses through a deprecation period.
- Define a clear fixed-base root-body contract that preserves existing Jacobian
  indexing.
- Add fast value-level regression coverage for each corrected path.

## Non-goals

- Supporting a fixed root at an arbitrary public body index.
- Changing the ordering resolver conventions or making `robot_schema` a
  default.
- Reordering fixed or spatial tendon axes.
- Replacing contiguous public buffers with `wp.indexedarray` views.
- Adding a second ordering abstraction for actuator code.

## Public Contract

### Fixed-base root body

For a fixed-base articulation, `body_ordering` must keep the root body at public
index zero. The remaining bodies may appear in any order.

This restriction preserves the established fixed-base Jacobian contract:

```text
jacobian_body_idx = body_idx - 1
```

The fixed root has no Jacobian row. Keeping it at public index zero means the
compact Jacobian body axis is exactly `body_names[1:]`, so existing task-space
controllers and kernels remain correct without another runtime map.

The ordering is validated after concrete body names have been resolved and
before the map is installed. An invalid explicit sequence or symbolic
convention raises `ValueError` during articulation initialization.

For the supported backends, the fixed root is backend body zero. Validation
therefore locates `backend_body_names[0]` in the resolved public sequence and
requires its public index to be zero.

The message must identify the articulation, root body, requested index, and
remediation. For example:

```text
Invalid body_ordering for fixed-base articulation '/World/Robot': root body
'panda_link0' must remain at public index 0, but was requested at index 8. Put
'panda_link0' first; all remaining bodies may be reordered freely.
```

The message must not describe the request as an incomplete permutation: the
name list is valid, but unsupported because it relocates the fixed root.

Floating-base articulations retain complete arbitrary body permutations.

### Backend introspection compatibility

`backend_joint_names`, `backend_body_names`, `joint_ordering`, and
`body_ordering` remain part of the `BaseArticulation` public contract, but they
will not become abstract requirements in this release.

The base class provides compatibility implementations:

- `backend_joint_names` returns `joint_names`.
- `backend_body_names` returns `body_names`.
- `joint_ordering` returns `data.joint_ordering` when present, otherwise `None`.
- `body_ordering` returns `data.body_ordering` when present, otherwise `None`.

Using a backend-name fallback emits a `DeprecationWarning` directing backend
authors to override the property before it becomes abstract in a future
release. Built-in PhysX, OVPhysX, and Newton articulations continue to override
all four properties and do not warn.

The major changelog fragment must describe this migration path. Existing
third-party subclasses must remain instantiable.

## Data-path Design

### External wrenches

`WrenchComposer` inputs and outputs are public body order because body IDs and
the body state used for frame conversion are public-order APIs. Each backend
must convert the body axis before writing to its backend-order simulation
buffer.

The conversion stays fused with work that already occurs in `write_data_to_sim`:

- PhysX uses one fused body-wrench reorder kernel that writes backend-order
  force and torque staging buffers before the existing Tensor API call.
- Newton maps the public body index to the backend body index inside the
  existing wrench update launch.
- OVPhysX maps the output body index inside `_body_wrench_to_world`, producing
  `LINK_WRENCH` directly in backend order.

When body ordering is disabled, each kernel writes the same index it reads. No
additional launch is introduced on the `None` path. Backend staging buffers are
allocated only when a non-identity body ordering needs them.

### Jacobians and dynamics

Joint axes continue to follow the public joint order after the floating-base
prefix. Floating-base Jacobian body axes use the full public body permutation.

For fixed-base articulations, validation guarantees that the root is public
body zero. The compact Jacobian map therefore contains the requested order of
`body_names[1:]`, and COM-to-link shifting may continue to use
`full_body_idx = jacobian_body_idx + 1`.

Mass matrices and generalized vectors retain their existing base prefix and
reorder only actuated-joint rows/columns.

### OVPhysX joint acceleration

Finite differencing must operate entirely in public joint order. With a
non-identity joint ordering, OVPhysX reads `DOF_VELOCITY` into the backend
shadow, reorders it into the public velocity buffer, and then differences the
public velocity against the public `_previous_joint_vel` buffer.

The acceleration getter must not temporarily overwrite a public buffer with
backend-order values.

### Newton actuator defaults

Newton's global actuator adapter indexes controller gains in backend-local DOF
order. Per-articulation default stiffness, damping, and managed-joint indices
must be converted to public joint order once during articulation
initialization.

The PhysX Newton-actuator adapter already indexes its wrapper in public order
and does not apply this conversion. Runtime domain-randomization writes keep
accepting public joint IDs and values; no reorder is added to that hot path.

### OVPhysX cache invalidation

Every timestamped backend shadow used to refresh a public quantity must be
invalidated when a same-step write changes its source:

- Pose resets invalidate both public and backend link-pose buffers.
- Velocity resets invalidate the public COM/link velocity buffers and both
  backend velocity shadows used by body and root accessors.
- COM-offset writes retain `develop`'s full
  `_reset_body_com_pose_b_dependents()` behavior, including root/body velocity,
  body-frame velocity, and concatenated-state caches.

The ordering changes must extend these invalidation sets rather than replacing
newer `develop` behavior with narrower manual timestamp resets.

## Error Handling

- Invalid fixed-base root placement raises before any public buffers are pinned
  to the requested order.
- Existing duplicate, missing, and extra-name diagnostics remain unchanged.
- A convention resolver that places a fixed root after index zero receives the
  same actionable root-placement error as an explicit sequence.
- No ordering failure silently falls back to backend order.

## Performance Requirements

### `None` ordering

- No map construction.
- No user-order shadow allocation caused by ordering.
- No reorder kernel launch.
- Existing backend buffers remain the public buffers where they were before
  this feature.

### Non-identity ordering

- Writes fuse index conversion with an existing write/transform operation where
  possible.
- Static initialization work, including Newton gain conversion, may allocate or
  gather once.
- Lazy read caches launch at most once per stale property per simulation step.
- No host synchronization is added to simulation-step hot paths.

## Testing Strategy

Fast mocked tests are the primary guardrail. Each regression test must be run
against the unfixed implementation and observed to fail for the expected value
mismatch before production code is changed.

### Shared mocked coverage

- Apply distinct per-body forces and torques under a fully reversed
  floating-base ordering and a root-preserving fixed-base permutation; assert
  each backend receives backend-order wrench values.
- Exercise floating-base Jacobians with fully reversed body ordering.
- Exercise fixed-base Jacobians with the root first and all remaining bodies
  reversed; compare body/link Jacobians and mass matrices against an identity
  articulation permuted by name.
- Request a fixed-base ordering with the root last and assert the complete
  actionable `ValueError` message.
- Verify a floating-base articulation accepts the same complete reversed body
  permutation.
- Feed distinct OVPhysX backend joint velocities and assert both `joint_vel` and
  `joint_acc` are public ordered.
- Prime OVPhysX root/body velocity shadows, perform a same-step write, and
  assert the next read observes the new value.
- Prime OVPhysX COM-dependent derived buffers, write a new COM offset, and
  assert every dependent timestamp is invalidated.
- Snapshot heterogeneous Newton controller gains and assert defaults and
  managed IDs are associated with the correct public joint names.
- Instantiate an old-style `BaseArticulation` test subclass without the four
  new overrides and assert compatibility fallbacks work and emit the documented
  deprecation warning.

### Resolver-specific coverage

Resolver tests must use an articulation whose branching topology produces
different PhysX and MJWarp joint/body orders. A small deterministic branching
USD fixture is the primary unit/integration guardrail; its expected convention
name sequences are asserted explicitly so the test cannot pass after the two
orders accidentally converge.

ANYmal-D is the representative live robot because it exercises the locomotion
sim-to-sim use case and has convention-dependent traversal order. Before any
value assertions, the test must assert that the requested preset produced a
non-identity joint map, body map, or both. If both maps are identity, the test
fails with a setup message explaining that the asset no longer covers
cross-backend ordering.

Panda must not be used as evidence that the `physx` or `mjwarp` convention
resolvers work: those backends can agree on its body order. Panda may still be
used with an explicit manual permutation to test backend read/write machinery.

### Live integration coverage

Keep two distinct live checks:

- Update the existing PhysX Panda smoke to keep the fixed root first while
  manually reversing all remaining bodies. This covers the fixed-base data path
  only and is named/documented accordingly.
- Run ANYmal-D with the opposite backend convention (`mjwarp` on PhysX or
  `physx` on Newton), assert that the resolved map is non-identity, and compare
  named state/command values at the backend boundary.

Add one external-wrench assertion to either live smoke if the backend view
exposes stable force readback. Real sim-to-sim tests remain smoke tests; they do
not replace the deterministic mocked value matrix.

### Verification commands

At minimum:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py::test_live_manual_root_preserving_ordering_reorders_backend_reads_and_writes -q
./isaaclab.sh -f
```

Backend-specific tests for Newton and OVPhysX are added to the command set when
the corresponding regression lives outside the shared interface suite.

## Integration Sequence

1. Preserve the current review-fix work in a focused commit, excluding unrelated
   debug output.
2. Rebase onto the latest `origin/develop` and resolve OVPhysX conflicts by
   retaining the newer view and cache-invalidation behavior.
3. Add one failing regression at a time.
4. Implement the smallest backend/shared fix that makes that regression pass.
5. Run the shared matrix after each backend fix.
6. Run formatting, targeted live smoke tests, and a final diff review before
   pushing.

## Decisions

- Fixed-base root relocation is rejected rather than silently normalized.
- The root-first restriction is initialization-only and has no step-time cost.
- External wrench conversion is a backend-boundary responsibility.
- Newton actuator defaults are normalized once, not during randomization.
- Base-class compatibility is preserved through deprecation rather than an
  immediate abstract-method break.
- Fast deterministic parity tests remain the main coverage; live simulation is
  an integration smoke.
