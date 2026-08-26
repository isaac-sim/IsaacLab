---
name: isaaclab-auditing-an-issue
description: Audits one open GitHub issue against current code to decide close or keep-open, and drafts the closing comment. Use when deciding whether a specific issue is still valid, whether a reported bug was fixed, or what to reply before closing.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Auditing an Issue

## When To Use

Use this skill for a single issue: deciding whether it is still valid against current code, and drafting the reply if it is being closed.

For auditing many issues at once, use `isaaclab-triaging-issue-backlog`, which applies this skill per issue and adds the batch controls.

Do not close an issue purely because it is old. Age is not evidence.

## Core Principle

A comment that reads well is not evidence that its claims are true. Verify every factual claim against current code before posting, and ask what is lost by closing, not only whether closing is defensible.

## Workflow

1. Read the whole thread, not just the first post: `gh issue view NUMBER --repo isaac-sim/IsaacLab --comments`.
2. Identify each checkable claim you intend to make in the reply.
3. Verify each one using the table below. Prefer running code over reading it.
4. Check the Do Not Close list before deciding.
5. Draft the comment in the style below, then confirm every link resolves.

## Mandatory Checks

| Claim shape | Required check |
|---|---|
| "X is fixed / landed in #N" | `gh pr view N` returns MERGED, and its diff touches the reported behavior. A generic "Merges changes from main" PR is never an acceptable citation. |
| "Isaac Lab doesn't have X" | Grep `source/isaaclab_tasks/isaaclab_tasks/contrib/`, `source/isaaclab_contrib/` and the `*_experimental` packages. Features move to contrib and absence claims go stale. |
| "`SYMBOL` no longer exists" | `grep -rn "SYMBOL" source/` — any live hit refutes it. |
| "`BACKEND` supports X" | Check that backend's own package. Backends differ; several classes raise `NotImplementedError` on one and not another. |
| Behavioral claim about an error or a limit | Write a temporary script and run it with `uv run python`. Reading the raise site is not enough — the guard may be narrower than it looks. |
| A usage answer | Confirm it is still the best current advice, not merely that some answer was given. Check the relevant capability on the Isaac Sim side too. |
| Any URL | `curl -s -o /dev/null -w "%{http_code}" -L "URL"` returns 200. |

## URL Traps

- `github.com/isaac-sim/IsaacLab/blob/main/...` **404s** for anything under `source/isaaclab_newton/`, `source/isaaclab_ov/`, `source/isaaclab_ovphysx/` or `isaaclab_contrib` — those packages are not on `main`. Use `blob/develop/...`, or cite the path in backticks with no link.
- `isaac-sim.github.io/IsaacLab/main/...` is the stale 2.x docs build. Use `release/3.0.0/...`, and confirm the anchor still exists — several 2.x anchors were removed in the 3.x reorganisation.

## Do Not Close

Hold the issue when any of these apply, even if the stated close reason is technically defensible:

- A maintainer asked the reporter a question and the thread is still recent.
- It is pre-merge feedback on an open PR.
- It is a duplicate of an issue that was itself closed without a fix.
- The titled ask is unmet, even if a related bug in the thread was fixed.
- It is blocked on upstream work that is planned but not landed. Confirm the upstream issue exists; "file it upstream" is not a resolution if nobody has.

When holding, record the concrete next step and the evidence, so the next audit does not redo the work.

## Comment Style

- Thank the reporter by handle in the first sentence, specifically and briefly.
- State what actually resolves it, with file paths in backticks and a verified PR link.
- Invite a reopen if it still reproduces on a current release.
- Do not assign fault. Avoid "we never heard back" (blames the reporter) and "nothing on our side" (blames a sibling team). Describe where the behavior lives, and route rather than deflect.
- Do not name a colleague as the source of a negative finding. Crediting a helpful answer is fine.
- If the reply was AI-assisted, say so, and do not claim human verification that has not happened.

## Common Mistakes

- Trusting a polished draft. Fluency correlates with neither accuracy nor completeness.
- Answering "was a reply given?" instead of "is that answer still correct and complete?"
- Citing a PR whose title matches the topic without reading its diff.
- Closing an issue whose stated reason is sound but which is the only remaining record of a live problem.

## Validation

Verify citations and links before posting:

```bash
gh pr view PR_NUMBER --repo isaac-sim/IsaacLab --json state,mergedAt,title
curl -s -o /dev/null -w "%{http_code}" -L "URL"
```

For a behavioral claim, write a temporary script and run it:

```bash
uv run python PATH_TO_SCRIPT
```

If skills changed, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep the URL traps section synchronized with the current package layout. When a package moves between `main` and `develop`, or the docs build version changes, update it before it produces a dead link in a public comment.

## References

- [Backlog triage skill](../issue-backlog-triage/SKILL.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
