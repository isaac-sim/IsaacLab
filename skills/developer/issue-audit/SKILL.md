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

- `github.com/isaac-sim/IsaacLab/blob/main/...` **404s** for anything under `source/isaaclab_newton/`, `source/isaaclab_ov/` or `source/isaaclab_contrib/` — those packages are not on `main`. Use `blob/develop/...`, or cite the path in backticks with no link. Package layout shifts over time (e.g. `isaaclab_ovphysx` was merged into `isaaclab_ov`), so verify the path exists on the ref you link.
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
- If the reply was AI-assisted, end with the disclaimer footer below, and do not claim a level of human review that did not happen.

### Disclaimer Footer

End an AI-assisted reply with a horizontal rule and a one-line footer:

```markdown
---
*🤖 This issue was reviewed as part of an AI-assisted triage of the Isaac Lab backlog, with automated verification against the current codebase. If we got this wrong, please reopen — we'd rather hear about it!*
```

Match the claim to what actually happened. "with automated verification against the current codebase" is accurate for a machine-checked reply. Only say "verified by a maintainer" if a maintainer read that specific comment.

### Worked Example

A close for a bug that was genuinely fixed:

```markdown
Thanks @Mitchell-Torok for catching that `reset_joints_by_offset` ignored the `joint_names` you passed in `SceneEntityCfg`.

It's fixed: both `reset_joints_by_offset` and `reset_joints_by_scale` in `isaaclab/envs/mdp/events.py` now index the default joint state and the soft-limit clamp with `asset_cfg.joint_ids` (broadcasting `env_ids[:, None]` for subsets), so only the named joints are touched. Shipped in https://github.com/isaac-sim/IsaacLab/pull/2899. Sorry for the delay — please reopen if extra joints still move on a current release.

---
*🤖 This issue was reviewed as part of an AI-assisted triage of the Isaac Lab backlog, with automated verification against the current codebase. If we got this wrong, please reopen — we'd rather hear about it!*
```

Each part is doing work: the handle and the specific symptom in sentence one, the mechanism plus file path so the comment stands alone in search results, a PR link confirmed MERGED, a three-word apology rather than a paragraph, and a reopen invitation tied to a current release.

### Fault Assignment

The rewrites below are the difference between a reply that reads as closure and one that reads as a brush-off.

| Instead of | Write |
|---|---|
| "We asked twice and never heard back." | "We weren't able to get to the bottom of this at the time, and a lot has changed since." |
| "There's nothing on our side to fix." | "That code path lives in `<component>`, so `<tracker>` is the better home for it." |
| "This is an Isaac Sim bug that their team owns." | "The `sim.reset()` requirement lives in that layer and is tracked there." |
| "As @SomeDev confirmed, it's broken." | State the finding directly, without attributing it to a colleague. |

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
