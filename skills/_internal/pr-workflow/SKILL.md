---
name: isaaclab-preparing-pr-workflow
description: Prepares Isaac Lab changes for review using the repository PR checklist, validation commands, commit rules, and changelog policy. Use when opening a PR, finishing a branch, preparing a commit, or checking contribution readiness.
license: BSD-3-Clause
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
---

# Preparing PR Workflow

## When To Use

Use this skill when preparing Isaac Lab changes for review, checking a branch before a PR, or helping a contributor understand final readiness steps.

Do not use this skill to bypass repository checks or to push to `origin`.

## Workflow

1. Inspect the changed files and identify touched packages.
2. Confirm the branch is focused on one logical change.
3. Run targeted tests for the touched behavior. For added, changed, or removed tests, apply
   [the test-audit skill](../test-audit/SKILL.md) and record the focused evidence it requires.
4. For skill changes, inspect the changed skill's adjacent `evaluations.md` when present, plus directly linked `examples.md` or `reference.md`, and confirm the representative scenarios still match the skill guidance.
5. Follow the contribution guide's documentation validation scope: skip Sphinx when rendered docs are unaffected, use incremental previews during editing, and require one clean, warning-free build for the final documentation-affecting changes.
6. Run formatting and lint checks with `uv run isaaclab -f`.
7. Add package changelog fragments when `source/<package>/` code changes.
8. Check whether `CONTRIBUTORS.md` needs an update for a new contributor.
9. Draft a commit message in imperative mood with no AI attribution.
10. Use the PR checklist in `.github/PULL_REQUEST_TEMPLATE.md`.

## Validation

Run the feedback loop until checks pass:

```bash
uv run isaaclab -f
```

For targeted tests, use:

```bash
uv run python -m pytest PATH_TO_TEST
```

Use the [contribution guide's documentation validation guidance](../../../docs/source/refs/contributing.rst#contributing-documentation) to decide whether Sphinx is needed and which build to run. Reuse a successful clean build of the final changes; do not repeat it for unrelated edits.

If skills changed, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `.github/PULL_REQUEST_TEMPLATE.md`, and `docs/source/refs/contributing.rst`. If a PR workflow rule changes, update the authoritative file first and keep this skill as a short routing checklist.

## References

- [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Changelog skill](../changelog-fragments/SKILL.md)
- [Examples](examples.md)
