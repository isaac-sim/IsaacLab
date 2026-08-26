---
name: isaaclab-triaging-issue-backlog
description: Triages large batches of open GitHub issues into close/keep-open with drafted closing comments. Use when auditing the issue backlog, deciding which issues can be closed, or drafting closing responses at scale.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Triaging the Issue Backlog

## When To Use

Use this skill when auditing many open issues at once and deciding which can be closed, and when drafting the closing comments that go with them.

Do not use it for a single issue you already understand. Do not use it to close issues purely because they are old — age is not evidence.

## Core Principle

Agents triaging issues are fluent and confidently wrong. A drafted comment that reads well is not evidence that its claims are true. Every factual claim must be checked against current code before it is posted, and a human must decide what is lost by closing.

## Workflow

1. Dump the open issues with metadata (`gh issue list --json number,title,author,createdAt,updatedAt,labels,comments,assignees`) and split into batches.
2. For each issue, read the full thread (`gh issue view <n> --comments`), then verify it against current code: grep the referenced symbol, `git log -S` for a landed fix, `gh pr view` to confirm a cited PR actually merged.
3. Run an adversarial pass over every close recommendation. Instruct the reviewer to refute, and to default to REFUTED or UNCERTAIN when it cannot confirm the evidence. Expect this to reject a meaningful share of recommendations.
4. Fact-check the drafted comments against the checks below.
5. Review every comment with a maintainer before posting anything.

## Mandatory Checks Before Posting

| Claim shape | Required check |
|---|---|
| "X is fixed / landed in #N" | `gh pr view N` returns MERGED, and its diff touches the relevant behavior. A generic "Merges changes from main" PR is never an acceptable citation. |
| "Isaac Lab doesn't have X" | Grep `source/isaaclab_tasks/isaaclab_tasks/contrib/`, `source/isaaclab_contrib/` and the `*_experimental` packages. Features move to contrib and absence claims go stale. |
| "`<symbol>` no longer exists" | `grep -rn "<symbol>" source/` — any live hit refutes it. |
| "`<backend>` supports X" | Check that backend's own package. Backends differ; several classes raise `NotImplementedError` on one and not another. |
| Behavioral claim about an error or limit | Write a temporary script and run it with `uv run python`. Reading the raise site is not enough — the guard may be narrower than it looks. |
| Any URL | `curl -s -o /dev/null -w "%{http_code}" -L "<url>"` returns 200. |

## URL Traps

- `github.com/isaac-sim/IsaacLab/blob/main/...` **404s** for anything under `source/isaaclab_newton/`, `source/isaaclab_ov/`, `source/isaaclab_ovphysx/` or `isaaclab_contrib` — those packages are not on `main`. Use `blob/develop/...` or cite the path in backticks without a link.
- `isaac-sim.github.io/IsaacLab/main/...` is the stale 2.x docs build. Use `release/3.0.0/...`, and check the anchor still exists — several 2.x anchors were removed in the 3.x reorganisation.

## Do Not Close

Closing is not free. Hold an issue when any of these apply, even if the stated close reason is technically defensible:

- A maintainer asked the reporter a question and the thread is still recent.
- It is pre-merge feedback on an open PR.
- It is a duplicate of an issue that was itself closed without a fix.
- The titled ask is unmet, even if a related bug in the thread was fixed.
- It is blocked on upstream work that is planned but not landed. Confirm the upstream issue exists first; "file it upstream" is not a resolution if nobody has.

## Comment Style

- Thank the reporter by handle in the first sentence, specifically and briefly.
- State what actually resolves it, with file paths in backticks and a verified PR link.
- Invite a reopen if it still reproduces.
- Do not assign fault. Avoid "we never heard back" (blames the reporter) and "nothing on our side" (blames a sibling team). Describe where the behaviour lives, not whose fault it is, and route rather than deflect.
- Do not name a colleague as the source of a negative finding. Crediting a helpful answer is fine.
- Mark AI-assisted triage explicitly, and do not claim human verification that has not happened.

## Common Mistakes

- Trusting a polished draft. Fluency correlates with neither accuracy nor completeness.
- Answering "was a reply given?" instead of "is that answer still correct and complete?"
- Citing a PR whose title matches the topic without reading its diff.
- Closing an issue whose reason is sound but which is the only remaining record of a live problem.

## Validation

Verify every cited PR is merged and every URL resolves before posting:

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

Keep the URL traps table synchronized with the current package layout. When a package moves between `main` and `develop`, or the docs build version changes, update the table before it produces dead links in a public comment.

## References

- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [PR workflow skill](../pr-workflow/SKILL.md)
