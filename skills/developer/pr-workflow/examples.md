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
3. Format selected files during editing, then run `uv run isaaclab -f` on the final changes. Reuse successful focused tests while their inputs remain current.
4. Fill the PR checklist with test results.

## Docs-Only Change

Input: a PR modifies `docs/source/overview/`.

Expected workflow:

1. Use the contribution guide's incremental preview during editing, then run one clean, warning-free build of the final documentation changes.
2. Run `uv run isaaclab -f` on the final changes. Inspect any automatic edits before rerunning failed checks.
3. Do not add a package changelog fragment unless `source/<package>/` changed.

## Skill-Only Change

Input: a PR modifies `skills/user/domain-randomization-events/SKILL.md`.

Expected workflow:

1. Run `uv run --no-project python tools/skills/cli.py check`.
2. Inspect `skills/user/domain-randomization-events/evaluations.md` when present, and directly linked `examples.md` or `reference.md` to confirm scenarios, examples, and source references still match the changed guidance.
3. Let the path-scoped skills CI gate validate the change on the PR.
4. Skip Sphinx because standalone skill Markdown is not rendered in the documentation site.
