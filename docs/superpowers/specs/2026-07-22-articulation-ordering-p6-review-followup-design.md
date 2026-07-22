# Articulation Ordering P6 Review Follow-up Design

## Context

P6 adds articulation ordering to OVPhysX. Its first review-cleanup pass fixed
ordered indexed joint-state writes, removed dead staging lifecycle branches,
replaced write-view sentinels with capability flags, consolidated position and
velocity state helpers, clarified tests, and simplified ordering gates.

A fresh audit of every inline review thread from `hujc7` and `ooctipus` found
four remaining implementation or documentation gaps:

1. P6 lacks a live OVPhysX regression for the ordered joint-target path through
   `write_data_to_sim`.
2. The reordered joint-property read helper still duplicates the three-component
   friction path.
3. Joint-property writers repeat backend reordering, CPU staging, and binding
   submission.
4. The OVPhysX changelog omits two user-visible fixes already shipped by P6.

Octi's shared ordering-fixture request is already implemented on P7. It remains
on P7, following the prior stack-layering decision, because the extraction
crosses core, Newton, PhysX, and OVPhysX test packages without changing P6's
OVPhysX production behavior.

## Goals

- Exercise ordered position and velocity targets through the live OVPhysX
  `write_data_to_sim` path on CPU and CUDA.
- Consolidate reordered scalar and multi-component joint-property reads without
  changing timestamps, binding availability checks, or ordering semantics.
- Consolidate the repeated property-write transport path without changing
  validation, mutation kernels, pinned-buffer ownership, or selector behavior.
- Document both remaining user-visible P6 fixes.
- Reply to and resolve every inline P6 thread from `hujc7` and `ooctipus` after
  the verified stack is pushed.
- Preserve the original P7, P8, and P9 patches when restacking the series.

## Non-goals

- Move the shared ordering fixtures from P7 down to P6.
- Change public APIs, ordering maps, property values, or device support.
- Replace the existing index and mask property-writer entry points.
- Change OVPhysX allocation policy or introduce asynchronous copy batching.
- Refactor unrelated body-property writers, tendon writers, or actuator logic.

## Live Ordered-Target Regression

Add a live OVPhysX test beside the existing articulation-ordering tests. The
test creates an articulation whose public joint order is a non-self-inverse
permutation of the backend order. A cyclic permutation is preferred over a
simple reversal because its `user_to_backend` and `backend_to_user` maps differ,
so using the wrong map cannot accidentally pass.

The test writes distinct per-joint position and velocity targets through the
public target setters, calls `write_data_to_sim`, reads `DOF_POSITION_TARGET`
and `DOF_VELOCITY_TARGET` directly from the OVPhysX view, and verifies that the
backend columns contain the expected public values. It runs once on CPU and
once on CUDA in separate pytest processes because the OVPhysX device lock is
process-global.

This is coverage for an existing production path rather than a new production
bug fix. To prove the test has teeth, temporarily replace the fused gather's
`backend_to_user` map with `user_to_backend` (or bypass the gather), run the new
test, and record the expected failure. Restore production byte-for-byte before
the passing run and commit.

## Joint-Property Read Consolidation

Replace the scalar-only private read helper with a general joint-property read
helper:

```python
def _read_joint_property_binding(
    self,
    tensor_type: int,
    user_buffer: TimestampedBuffer,
    backend_buffer: TimestampedBuffer | None,
    component_count: int | None = None,
) -> None:
```

Identity ordering reads directly into `user_buffer`. Nonidentity ordering reads
the backend buffer once when stale and gathers into public order. A `None`
component count selects the existing two-dimensional reorder; a positive count
selects the three-dimensional reorder with shape
`(num_instances, num_joints, component_count)`.

Existing scalar property call sites use the helper with the default component
count. `_read_joint_friction_binding` stays as a thin named wrapper and calls it
with `component_count=3`, preserving readable friction-writer call sites and
the combined friction buffer's timestamp behavior.

## Joint-Property Write Consolidation

Add one private transport helper on the OVPhysX articulation. Callers continue
to validate inputs, resolve joint/environment selections, update the public
property buffer, and perform property-specific operations such as clamping
default positions. The helper owns only the repeated transport tail:

1. Select or populate backend-order storage through
   `_get_backend_ordered_joint_buffer`.
2. Reinterpret structured storage as flat `float32` only when the destination
   binding requires it.
3. Copy into the caller-provided pinned CPU buffer, or reuse the existing
   tensor-type write staging cache for combined friction properties.
4. Call `set_attribute` with exactly one resolved CPU `indices` or `mask`
   selector.

The helper accepts explicit user and backend buffers, the tensor type, an
optional preallocated CPU buffer, an optional component count, and mutually
exclusive `indices`/`mask` keyword arguments. Stiffness, damping, limits, and
armature retain their dedicated preallocated CPU buffers. Friction retains its
existing lazy `(tensor_type, "write")` staging cache. The helper does not
validate public inputs, mutate timestamps, or choose which environments are
selected. Keeping those responsibilities at the call sites avoids hiding
differences between index/mask APIs and scalar, position-limit, and friction
mutation kernels.

All stiffness, damping, position-limit, velocity-limit, effort-limit, armature,
and friction writers use the helper for their final transport step. Existing
preallocated and lazy pinned buffers remain in use.

## Changelog

Extend `source/isaaclab_ovphysx/changelog.d/articulation-ordering.major.rst`
with two `Fixed` entries:

- Root-link velocity refresh no longer overwrites or falsely marks the
  body-link velocity cache fresh.
- Partial joint position and velocity writes preserve newer unselected backend
  rows instead of rewriting stale cached values.

These entries describe user-visible behavior in past tense and do not expose
internal staging details.

## Validation

P6 validation includes:

1. Mutation-failure and restored-pass runs of the new live target-write test.
2. Separate CPU and CUDA OVPhysX processes for device-specific coverage.
3. Existing ordered indexed-write and stale-row regressions.
4. Existing stiffness, damping, limit, armature, and friction writer tests.
5. Focused property-read tests, including three-component friction under
   nonidentity ordering.
6. OVPhysX view and articulation-ordering selections.
7. `./isaaclab.sh -f`, repeated after any formatter modifications.

After P6 review approval, rebase P7 onto P6, P8 onto P7, and P9 onto P8. Run
exact range-diffs against the currently pushed branches and verify that every
pre-existing P7-P9 patch remains equivalent. Final-stack validation includes
the cross-backend ordering interface with all backends discovered, shared
ordering utilities, Newton ordering, live OVPhysX CPU ordering, and
pre-commit.

## Review-Thread Closure

Push P6-P9 atomically to the `antoine` fork with an explicit force-with-lease
for every branch. Never push to `origin`.

After GitHub reports the expected heads:

- Reply in each of the seven `hujc7` and four `ooctipus` inline threads with
  the implementing commit or the deliberate P7 fixture ownership.
- Resolve each replied inline thread.
- Add one top-level P6 comment mapping the six items in Octi's review body to
  their final commits.
- Re-query all thread states, PR heads, mergeability, and CI startup.

Thread replies are factual and concise. Already-addressed feedback cites its
existing P6 commit; new feedback cites the new P6 commits; the fixture thread
links P7 and its shared-fixture commits.

## Rollback and Safety

Create fresh dated backup refs for the current pushed P6-P9 heads before any
rebase. Do not modify existing backup refs. Keep every review fix as a new,
focused commit. Fetch the fork immediately before pushing and abort if any
remote head differs from its expected lease.
