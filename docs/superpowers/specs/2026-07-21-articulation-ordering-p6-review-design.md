# Articulation Ordering P6 Review Design

## Context

P6 adds articulation ordering support to the OVPhysX backend. Review of P2
through P6 identified one correctness defect in the indexed joint-state writer
and several cleanup opportunities that are still applicable after P5 merged.
P7 through P9 are stacked on P6 and must be restacked after P6 changes.

## Goals

- Preserve the public, user-facing joint and body order while OVPhysX uses its
  backend order internally.
- Fix full and partial indexed joint-state writes when user and backend joint
  orders differ.
- Make write bindings and ordering-buffer invariants explicit without storing
  unused pointer aliases.
- Remove duplicated joint-state staging code and the duplicated deprecated
  joint-state writer implementation.
- Keep P6 focused on OVPhysX behavior and package-local tests.
- Move cross-package ordering-fixture deduplication to P7 and keep it explicit
  in the stacked-series plan.

## Non-goals

- Redesign the ordering maps established in P2 through P5.
- Change the identity-ordering fast path.
- Alter public API names or remove deprecated APIs.
- Fold unrelated Newton, PhysX, adapter, Jacobian, or material changes into P6.

## P6 Design

### Indexed joint-state writes

`write_joint_state_to_sim_index` will translate user-ordered positions and
velocities into backend-ordered staging buffers before calling the OVPhysX
view. The implementation will reuse the same ordering kernel and buffer
selection rules as the mask-based writer so full and partial environment/joint
selections behave consistently.

Regression tests will use non-identity joint ordering and cover both full and
partial indexed writes. Following the repository guidelines, each new
regression must be observed failing against the pre-fix implementation and
passing after the fix.

### Write binding capabilities

The three pointer-aliased tensors used only as `None` sentinels will be replaced
with explicit boolean capability flags for effort, position-target, and
velocity-target bindings. Ordered target writes will pass `None` for the
disabled joint-acceleration output rather than aliasing the applied-effort
buffer into two output slots.

### One-time ordering setup

Ordering buffers are configured once after the articulation view resolves.
The setup API will therefore drop the unused previous-ordering lifecycle
arguments and the unreachable release/reseed branches. Hot-path guards for
ordering staging buffers will be removed where setup guarantees their
existence; genuinely optional backend bindings and lazily allocated pinned
transfer buffers will retain their guards.

Pure capability checks will use `has_joint_ordering` or `has_body_ordering`.
Direct map checks remain appropriate where the map is immediately
dereferenced.

### Joint-state helper consolidation

The position and velocity refresh helpers will become thin wrappers over one
parameterized joint-state refresh helper. The corresponding write-buffer
selection helpers will follow the same pattern. Tensor-kind parameters will
select the correct error context while buffers remain specifically typed as
`torch.Tensor`.

The deprecated `write_joint_state_to_sim` method will delegate to the public
position and velocity writers, preserving its API while eliminating a second
ordering implementation.

### Documentation and tests

The articulation-data documentation will state that non-identity backend data
is published to user-facing buffers once per simulation step. Root COM pose
composition will use the backend COM-pose buffer directly because its property
already handles identity ordering.

OVPhysX test docstrings will describe the scenario directly and will not refer
to private finding numbers or an unavailable internal design document.

## P7 Design

P7 will extract the repeated ANYmal and Panda ordering fixtures into the
existing shared articulation-ordering test utility. Newton, PhysX, OVPhysX,
and core test consumers will import the shared constants and helpers while
retaining backend-specific assertions. This is intentionally separate from P6
because it crosses package boundaries without changing OVPhysX production
behavior.

## Validation

P6 validation will include:

1. New indexed-write regressions failing before the production fix.
2. The same regressions passing after the fix.
3. Existing focused OVPhysX articulation-ordering tests.
4. Relevant core and Newton ordering tests used to guard stack compatibility.
5. `./isaaclab.sh -f`, reviewed after any automatic formatting, followed by a
   second clean run before committing and pushing.

P7 will additionally run all consumers of the shared fixture utility. P7
through P9 will then be rebased onto their rewritten predecessor, checked with
`git range-diff`, and pushed only to the PR author's fork with force-with-lease.

## Rollout and Recovery

P6 will receive focused review-fix commits rather than amended historical
commits. P7 will receive the fixture-deduplication commit. P8 and P9 should
contain no semantic changes beyond conflict resolution required by the
restack. Existing backup branches remain available until the rewritten stack
has been verified on GitHub.
