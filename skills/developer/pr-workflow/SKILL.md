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
5. Run formatting and lint checks with `./isaaclab.sh -f`.
6. Add package changelog fragments when `source/<package>/` code changes.
7. Check whether `CONTRIBUTORS.md` needs an update for a new contributor.
8. Draft a commit message in imperative mood with no AI attribution.
9. Use the PR checklist in `.github/PULL_REQUEST_TEMPLATE.md`.

## Validation

Run the feedback loop until checks pass:

```bash
./isaaclab.sh -f
```

For targeted tests, use:

```bash
./isaaclab.sh -p -m pytest PATH_TO_TEST
```

If skills changed, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `.github/PULL_REQUEST_TEMPLATE.md`, and `docs/source/refs/contributing.rst`. If a PR workflow rule changes, update the authoritative file first and keep this skill as a short routing checklist.

## References

- [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Changelog skill](../changelog-fragments/SKILL.md)
- [Examples](examples.md)
