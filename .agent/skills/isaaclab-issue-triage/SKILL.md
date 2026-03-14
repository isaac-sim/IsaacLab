---
name: isaaclab-issue-triage
description: Triage GitHub issues for the IsaacLab repo. Validates bug reports against the bug template, checks for required fields (commit, steps to reproduce), comments requesting missing info, and routes reproducible bugs to the reproduce and fix workflows. Use when asked to triage issues, check bugs, or process GitHub issues.
---

# IsaacLab Issue Triage

Automated workflow to triage bug issues on `isaac-sim/IsaacLab`.

## Prerequisites

- `gh` CLI authenticated (`gh auth status`)
- Git repo cloned with remote `origin` pointing to `isaac-sim/IsaacLab`

## Workflow

### Step 1: Fetch open bug issues

```bash
gh issue list --repo isaac-sim/IsaacLab --label bug --state open --json number,title,body,labels,comments --limit 50
```

If no `--label bug` filter works, fetch all open issues and filter for `[Bug Report]` in the title or `bug` label.

### Step 2: For each bug issue, validate the report

Parse the issue body and check for required fields from the bug template:

**Required fields:**
- **Steps to reproduce**: Look for `### Steps to reproduce` section with actual content (not just the template placeholder)
- **Commit hash**: Look for `### System Info` section containing `- Commit:` with an actual hash (not `[e.g. 8f3b9ca]`)

**Decision tree:**

```
Issue body parsed
├─ Missing steps to reproduce AND no other useful details?
│  └─ Comment requesting steps (see "Comment: Missing Steps" below)
│     Then STOP for this issue.
├─ Has steps to reproduce but missing commit?
│  └─ Use the latest commit on `develop` branch.
│     Proceed to reproduce.
├─ Has both commit and steps?
│  └─ Proceed to reproduce.
└─ Not a bug report (feature request, question, etc.)?
   └─ Skip.
```

### Step 3: Route to reproduction

Follow the **isaaclab-bug-reproduce** skill with:
- The issue number
- The commit hash (from issue or latest `develop`)
- The reproduction steps extracted from the issue body

### Comment Templates

**Comment: Missing Steps**

```markdown
Thank you for filing this issue. To help us investigate, could you please provide:

1. **Minimal steps to reproduce** the bug (a small script or sequence of commands)
2. **The full error message / stack trace** (if applicable)

These details are required per our [bug report template](https://github.com/isaac-sim/IsaacLab/blob/main/.github/ISSUE_TEMPLATE/bug.md) and are essential for us to diagnose the problem. We'll revisit this issue once the reproduction steps are available.
```

Post via:
```bash
gh issue comment <NUMBER> --repo isaac-sim/IsaacLab --body "<BODY>"
```

## Batch Processing

When triaging multiple issues, process them sequentially. For each issue, complete the full triage cycle (validate → reproduce → fix or comment) before moving to the next.

## References

- Bug report template: read `.github/ISSUE_TEMPLATE/bug.md` for the expected issue fields
- Contributing guidelines: read `docs/source/refs/contributing.rst` for coding style and PR requirements
- Project rules: read `AGENTS.md` for naming, changelog, commit, and testing conventions
