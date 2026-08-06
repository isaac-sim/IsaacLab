# Newton Wrench Integration Test Design

## Context

The Newton wrench-frame fix rotates WrenchComposer body-frame center-of-mass
forces and torques into the world frame while packing Newton's `body_f` array.
The focused kernel tests and the rigid-object integration suites pass in a fresh
environment created from the repository lockfile.

`test_external_force_on_multiple_bodies_at_position` applies each wrench twice
(`set` followed by `add`), producing 200 N and 200 N·m per selected shank for
100 simulation steps. With correct frame conversion, this trajectory overflows
Newton's constraint capacity and produces NaNs. Its pre-fix trajectory passes
because the unrotated wrenches act in different world directions.

## Selected Approach

Keep the production rotation unchanged. Revise only the affected integration
test so that it verifies two distinct responsibilities:

1. After `write_data_to_sim`, directly compare the selected Newton `body_f`
   entries with independently calculated world-frame force and torque values.
   Cover both global inputs and body-frame inputs, including articulation body
   ordering.
2. Apply a moderate wrench for a bounded number of steps and verify that the
   resulting articulation state remains finite and has a nontrivial angular
   response.

Use 20 N input loads, which become 40 N after the existing `set` plus `add`, and
50 simulation steps. This operating point passed all CPU/CUDA and one/two-world
parameterizations while preserving the existing angular-velocity threshold.

## Alternatives Considered

- Keep the original 200 N/100-step stress case. This is rejected because it
  deterministically produces Newton constraint overflow with the now-correct
  world wrench and therefore tests solver divergence rather than wrench frames.
- Only reduce the load and duration. This avoids overflow but leaves the test's
  frame claim indirect. Direct `body_f` assertions are required so the test
  fails specifically when rotation or body ordering is wrong.
- Remove or skip the test. This is rejected because articulation-level coverage
  is valuable and the corrected behavior should remain exercised.

## Verification

- Demonstrate the direct world-wrench assertion fails when the packing kernels
  are replaced with pre-fix behavior that omits rotation.
- Demonstrate it passes with production packing on CPU and CUDA.
- Run all Newton external-force tests separately for rigid objects,
  rigid-object collections, and articulations.
- Run the focused WrenchComposer and Newton packing-kernel suites.
- Run all pre-commit checks and confirm a clean worktree.

## Branch Integration

Rebase the feature branch onto current `origin/develop` before final
verification. Push the rewritten branch using `--force-with-lease` only after
all checks pass.
