# IsaacLab Agent Skills Overview

This directory contains automated workflow skills for the `isaac-sim/IsaacLab` repository.
Each skill is a self-contained `SKILL.md` file that an agent reads and executes.

## Skill Index

| Skill | Trigger phrase | SKILL.md |
|-------|---------------|----------|
| `isaaclab-issue-triage` | `"Run the issue triage workflow for issue #N"` | `skills/isaaclab-issue-triage/SKILL.md` |
| `isaaclab-bug-reproduce` | Called internally by triage | `skills/isaaclab-bug-reproduce/SKILL.md` |
| `isaaclab-bug-fix` | Called internally by reproduce, or `"Fix issue #N and open a PR"` | `skills/isaaclab-bug-fix/SKILL.md` |
| `isaaclab-pr-respond` | `"Address review comments on PR #N"` | `skills/isaaclab-pr-respond/SKILL.md` |
| `isaaclab-pr-resolve-conflicts` | `"Resolve merge conflicts on PR #N"` | `skills/isaaclab-pr-resolve-conflicts/SKILL.md` |

---

## When to use which skill

```
A GitHub issue is filed
└─► isaaclab-issue-triage
    ├─ Not a bug / missing info → comment & stop
    └─ Valid bug report
       └─► isaaclab-bug-reproduce
           ├─ Cannot reproduce → comment & stop
           ├─ Fixed on latest develop → comment, close & stop
           └─ Still broken on latest
              └─► isaaclab-bug-fix
                  ├─ Search open/merged PRs for existing fix
                  │  ├─ Open PR found   → comment pointing to it & stop
                  │  ├─ Merged PR found → comment "already fixed on develop" & stop
                  │  └─ No prior PR     → Branch → fix → test → changelog → pre-commit → PR → comment on issue


A PR is open and has reviewer comments
└─► isaaclab-pr-respond
    ├─ Question
    │  └─ Post a direct reply
    ├─ Improvement request
    │  └─ Implement → pre-commit → commit → push → reply with hash
    └─ New issue reported by reviewer
       ├─ Recent PR already fixes it
       │  └─ Comment pointing to that PR & stop
       └─ No prior fix found
          ├─ Related to current PR & small scope
          │  └─ Fix in current PR branch → commit → push → reply
          └─ Unrelated or would make PR too large
             └─► isaaclab-bug-fix (new branch + separate PR)
                 └─ Reply with link to new PR


A PR has merge conflicts with its target branch
└─► isaaclab-pr-resolve-conflicts
    ├─ Already mergeable → report & stop
    └─ Has conflicts
       ├─ Rebase onto target branch (default)
       │  └─ For each conflict: read both sides → resolve → stage → continue
       └─ Rebase irrecoverable → fall back to merge commit
       └─ pre-commit → tests → force-push → comment on PR
```

---

## Skill Details

### `isaaclab-issue-triage`
**When:** A GitHub issue is opened or needs processing.
**What it does:**
1. Fetches open bug issues from `isaac-sim/IsaacLab`.
2. Validates required fields (steps to reproduce, commit hash).
3. Routes: comments requesting missing info, or hands off to `isaaclab-bug-reproduce`.

### `isaaclab-bug-reproduce`
**When:** Called by `isaaclab-issue-triage` once a valid bug report is confirmed.
**What it does:**
1. Checks out the commit reported in the issue (falls back to latest `develop`).
2. Runs the reproduction steps.
3. If reproducible, re-tests on the latest `develop` commit.
4. Routes: comments "cannot reproduce", closes as "fixed on latest", or hands off to `isaaclab-bug-fix`.

### `isaaclab-bug-fix`
**When:** A bug is confirmed to still reproduce on `develop`; or when directly asked to fix an issue and open a PR.
**What it does:**
1. Searches open and recently merged PRs for an existing fix (by issue number and keywords); comments and stops if one is found.
2. Creates a branch `isaaclab-bot/fix-issue-<N>` from `develop`.
3. Implements the fix following `AGENTS.md` coding standards.
4. Writes a regression test (verified to fail without the fix).
5. Updates the changelog and bumps the version in `extension.toml`.
6. Runs pre-commit (`./isaaclab.sh -f`) until clean.
7. Commits, pushes, opens a PR, and comments on the original issue.

### `isaaclab-pr-respond`
**When:** A PR has reviewer comments that need to be addressed.
**What it does:**
Categorizes each comment and acts accordingly:
- **Question** → posts a direct reply.
- **Improvement** → implements the change, commits on the PR branch, replies with the commit hash.
- **New issue** → searches recent merged/open PRs for an existing fix first:
  - If found: comments pointing to that PR.
  - If not found and in-scope: fixes it in the current PR branch.
  - If not found and out-of-scope / too large: opens a dedicated new PR via `isaaclab-bug-fix` logic.

### `isaaclab-pr-resolve-conflicts`
**When:** A PR has merge conflicts with its target branch (e.g., `develop`) and cannot be merged.
**What it does:**
1. Checks out the PR branch and fetches the latest target branch.
2. Rebases the PR branch onto the target (preferred) or falls back to a merge commit.
3. For each conflict: reads both sides, understands intent from git log, resolves manually.
4. Runs pre-commit and the affected tests.
5. Force-pushes with `--force-with-lease`.
6. Posts a comment on the PR listing each resolved conflict.
