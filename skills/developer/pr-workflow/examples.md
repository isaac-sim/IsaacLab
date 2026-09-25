# PR Workflow Examples

## Contents

- Source package change
- Docs-only change
- Skill-only change

## Source Package Change

Input: a PR modifies `source/isaaclab/isaaclab/assets/`.

Expected workflow:

1. Run targeted tests for the changed asset behavior.
2. Add a fragment under `source/isaaclab/changelog.d/`.
3. Run `uv run isaaclab -f`.
4. Fill the PR checklist with test results.

## Docs-Only Change

Input: a PR modifies `docs/source/overview/`.

Expected workflow:

1. Run `uv run --isolated --extra test -- make -C docs current-docs` and require the build to complete without warnings or errors.
2. Run `uv run isaaclab -f`.
3. Do not add a package changelog fragment unless `source/<package>/` changed.

## Skill-Only Change

Input: a PR modifies `skills/user/domain-randomization-events/SKILL.md`.

Expected workflow:

1. Run `uv run --no-project python tools/skills/cli.py check`.
2. Inspect `skills/user/domain-randomization-events/evaluations.md` and directly linked `examples.md` or `reference.md` to confirm scenarios, examples, and source references still match the changed guidance.
3. Let the path-scoped skills CI gate validate the change on the PR.
