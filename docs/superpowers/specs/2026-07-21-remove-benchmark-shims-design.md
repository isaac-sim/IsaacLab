# Remove Benchmark Compatibility Shims

## Goal

Make PR #6564 reviewable as a direct promotion of the benchmark implementation from
`isaaclab.test.benchmark` and standalone scripts to the public `isaaclab.benchmark`
package and unified `isaaclab benchmark` CLI.

## Scope

- Delete the complete `isaaclab.test.benchmark` compatibility namespace, including
  its lazy module proxies and forwarding type stubs.
- Delete the legacy runtime, startup, training, play, early-stop, and RL framework
  adapter scripts under `scripts/benchmarks`.
- Keep standalone benchmark scripts that are not compatibility wrappers, such as
  camera, Hydra resolution, and robot-loading benchmarks.
- Replace internal use of removed paths with `isaaclab.benchmark` imports and the
  unified benchmark CLI.

## Compatibility Decision

This intentionally removes previously public benchmark imports and script paths
without retaining deprecation shims. The exception to the normal deprecation policy
will be stated explicitly in the changelog and PR description. Migration guidance
will point users directly to `isaaclab.benchmark` and `isaaclab benchmark`.

## Tests

- Remove tests whose only purpose is validating compatibility shims.
- Update existing runtime, startup, training, and play smoke tests to invoke the
  unified CLI or package entrypoints.
- Keep behavioral coverage for the promoted implementation and all RL backends.
- Do not add tests that merely assert deleted files or imports remain absent.

## Documentation

- Replace deprecation language with removal and direct migration instructions.
- Remove examples that invoke deleted standalone scripts.
- Update the PR description to explain the intentional compatibility break and the
  resulting reduction in review noise.

## Validation

- Run focused benchmark API, CLI, and smoke tests.
- Run the complete benchmark unit-test directory.
- Run changelog validation, documentation build, and all pre-commit hooks.
