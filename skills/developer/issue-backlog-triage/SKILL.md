---
name: isaaclab-triaging-issue-backlog
description: Triages many open GitHub issues at once into close/keep-open with drafted closing comments. Use when auditing the whole issue backlog, running a bulk close pass, or reporting how many issues can be retired.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Triaging the Issue Backlog

## When To Use

Use this skill when auditing many issues at once: a backlog sweep, a bulk close pass, or a report on how much of the backlog can be retired.

For a single issue, use `isaaclab-auditing-an-issue` directly.

**REQUIRED SUB-SKILL:** Use `isaaclab-auditing-an-issue` for every individual issue. This skill only adds batch orchestration; it does not restate the per-issue checks.

## Core Principle

Scale multiplies confident errors. A batch of drafted comments will read uniformly well whether or not the claims are true, so verification must be adversarial and the human review gate is not optional.

## Workflow

1. Dump the backlog with metadata and split it into batches:

   ```bash
   gh issue list --repo isaac-sim/IsaacLab --state open --limit 500 \
     --json number,title,author,createdAt,updatedAt,labels,comments,assignees,url
   ```

2. Audit each issue per `isaaclab-auditing-an-issue`, recording a verdict, the evidence, and a confidence level.
3. Run an adversarial pass over every close recommendation. Instruct the reviewer to **refute**, and to return REFUTED or UNCERTAIN whenever it cannot independently confirm the evidence. Verify low-confidence recommendations as strictly as the rest — they refute at a far higher rate.
4. Sweep the drafted comments for claims that a per-issue audit misses across a batch: capability assertions, citations whose PR does not match, and dead links.
5. Review every comment with a maintainer, in batches, before anything is posted.
6. Post only what a maintainer explicitly approved.

## Batch Controls

- **Never infer approval.** Approve on an explicit statement, not on a reply that merely discusses a batch. Record approvals so a resumed session cannot post something unreviewed.
- **Keep one plan file** listing every issue to be closed with its final comment, plus appendices for held issues and for issues pulled out during review with their next steps.
- **Re-show anything edited after approval.** Later passes rewrite earlier comments; approval does not survive a rewrite.
- **Report what was excluded.** State how many issues were refuted, held, or pulled out, not only how many are closeable.

## Expect These Failure Modes

Measured over one full pass of the backlog; treat them as the default, not the exception.

| Failure mode | What it looks like |
|---|---|
| Stale absence claims | "Isaac Lab doesn't support X" when X moved to `contrib/`. |
| Mismatched citations | A merged PR whose title fits the topic but whose diff never touches the behavior. |
| Capability drift | Asserting a backend or release supports something it does not. |
| Answered-but-wrong | A usage answer that was correct when written and is now outdated. |
| Blame framing | "We never heard back" or "nothing on our side", which read badly in public. |
| Defensible but lossy | A close whose reason is sound but which discards the only record of a live problem. |

## Common Mistakes

- Treating a high-confidence label as a substitute for verification.
- Batching approval across issues that were edited at different times.
- Reporting a reduction figure without the refuted and held counts.
- Posting before a maintainer has read the specific comment being posted.

## Validation

Before posting, confirm the batch is internally consistent: every comment ends with the intended marker, every cited PR is merged, and every URL resolves.

```bash
gh pr view PR_NUMBER --repo isaac-sim/IsaacLab --json state,mergedAt,title
curl -s -o /dev/null -w "%{http_code}" -L "URL"
```

If skills changed, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep the per-issue checks in `isaaclab-auditing-an-issue` and do not duplicate them here. Update the failure-mode table when a new class of error survives verification into a posted comment.

## References

- [Issue audit skill](../issue-audit/SKILL.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
