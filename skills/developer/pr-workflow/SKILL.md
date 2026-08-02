---
name: isaaclab-preparing-pr-workflow
description: Prepares Isaac Lab changes for review using the repository PR checklist, validation commands, commit rules, and changelog policy. Use when opening a PR, finishing a branch, preparing a commit, or checking contribution readiness.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Preparing PR Workflow

## When To Use

Use this skill when preparing Isaac Lab changes for review, checking a branch before a PR, or helping a contributor understand final readiness steps.

Do not use this skill to bypass repository checks or to push to `origin`.

## Workflow

1. Inspect the changed files and identify touched packages.
2. Confirm the branch is focused on one logical change.
3. Run targeted tests for the touched behavior.
4. For skill changes, inspect the changed skill's adjacent `evaluations.md` when present, plus directly linked `examples.md` or `reference.md`, and confirm the representative scenarios still match the skill guidance.
5. If the PR changes documentation, run `uv run isaaclab -d` and require a successful build with no warnings or errors. The docs target treats Sphinx warnings as errors.
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

If documentation changed, require a warning-free build:

```bash
uv run isaaclab -d
```

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
