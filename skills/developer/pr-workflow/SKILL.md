---
name: isaaclab-preparing-pr-workflow
description: Checks Isaac Lab contribution readiness. Use when preparing a branch for a commit or PR.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Preparing PR Workflow

## When To Use

Use when preparing the final change for a commit or PR. Follow the user's existing authorization for commits, pushes, and PR creation.

## Workflow

1. Inspect the final diff, identify touched packages, and confirm the branch has one coherent scope.
2. Apply the [contribution guide](../../../docs/source/refs/contributing.rst) sections relevant to the diff and the [PR checklist](../../../.github/PULL_REQUEST_TEMPLATE.md). Reuse current guidance and successful checks as described in Agent Development.
3. For changed tests, apply [test audit](../test-audit/SKILL.md). For source package changes, use [changelog fragments](../changelog-fragments/SKILL.md).
4. For skill changes, inspect the affected examples and evaluation scenarios. Confirm they still match the guidance; read supporting references only when needed.
5. Inspect staged changes and prepare the authorized commit or PR with the validation results and remaining limitations.

## Validation

Use the guide's Unit Testing, Tools, and Contributing Documentation sections for final checks. For skill changes, run the [skill validator](../../../tools/skills/cli.py):

```bash
uv run --no-project python tools/skills/cli.py check
```

Report checks actually completed separately from CI that is still running.

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `.github/PULL_REQUEST_TEMPLATE.md`, and `docs/source/refs/contributing.rst`. If a PR workflow rule changes, update the authoritative file first and keep this skill as a short routing checklist.

## References

- [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Changelog skill](../changelog-fragments/SKILL.md)
- [Examples](examples.md)
