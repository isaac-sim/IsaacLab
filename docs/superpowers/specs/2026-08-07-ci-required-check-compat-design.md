# CI Required-Check Compatibility Design

## Goal

Ensure docs-only pull requests satisfy the repository's existing required demo
and kit-less rendering checks while continuing to skip the expensive test jobs.

## Approach

Add three lightweight compatibility jobs to `.github/workflows/build.yaml` with
the exact status names currently required by branch protection:

- `standalone demos (headless, Kit)`
- `standalone demos (headless, non-Kit)`
- `rendering-correctness-kitless (legacy)`

Each compatibility job will use `if: always()` and depend on the shared change
detector plus the corresponding current matrix job. It will exit successfully
when `changes.outputs.should_run` is false, allowing intentional docs-only
skips to satisfy the exact required context. When tests are required, it will
mirror the aggregate result of the corresponding matrix job and fail if that
job fails or is cancelled.

No branch-protection settings will be changed, and no duplicate test execution
will be introduced. The existing test jobs remain gated by the base-image build
and therefore remain skipped for docs-only changes.

## Verification

Validate the workflow YAML and add a focused static test that checks the three
compatibility job names, their `always()` conditions, and their dependencies.
Run the focused test, pre-commit, and the workflow validation available in the
repository before opening the pull request.
