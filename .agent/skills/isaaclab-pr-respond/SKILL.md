---
name: isaaclab-pr-respond
description: Respond to PR review comments on an IsaacLab pull request. Handles three comment types: (1) questions — posts a direct answer, (2) improvement requests — implements the improvement as a new commit, (3) new issue reports — searches recent PRs for a prior fix and either points to it or implements the fix in the current PR. Use when asked to address PR review feedback, respond to reviewer comments, or iterate on an open pull request.
---

# IsaacLab PR Review Responder

Address reviewer feedback on an open IsaacLab pull request by analyzing comments and taking the appropriate action for each one.

## Inputs

- **PR number**: GitHub PR `#N`
- **Comment ID(s)** *(optional)*: Specific review comment IDs to address. If omitted, all unresolved comments are processed.

## Workflow

### Step 1: Fetch the PR and its review comments

```bash
# Get PR metadata (branch, base, title, body)
gh pr view <PR_NUMBER> --repo isaac-sim/IsaacLab --json number,title,body,headRefName,baseRefName,state

# Get all review comments (inline + review-level)
gh pr view <PR_NUMBER> --repo isaac-sim/IsaacLab --json reviews,comments

# Get individual review comments with full context
gh api repos/isaac-sim/IsaacLab/pulls/<PR_NUMBER>/comments
gh api repos/isaac-sim/IsaacLab/pulls/<PR_NUMBER>/reviews
```

Also fetch the current PR diff for context:
```bash
gh pr diff <PR_NUMBER> --repo isaac-sim/IsaacLab
```

### Step 2: Checkout the PR branch

```bash
git fetch origin
git checkout <HEAD_REF_NAME>
```

If the branch is from a fork, fetch it explicitly:
```bash
gh pr checkout <PR_NUMBER> --repo isaac-sim/IsaacLab
```

### Step 3: Categorize each comment

For each unresolved review comment, classify it into one of three types:

| Type | Description | Examples |
|------|-------------|---------|
| **Question** | Reviewer asks for clarification about the approach, code intent, or design decision | "Why did you choose X?", "What happens when Y is None?" |
| **Improvement** | Reviewer requests a concrete code change to the existing fix | "Rename this variable", "Add a docstring", "Handle edge case Z", "Extract this into a helper" |
| **New issue** | Reviewer identifies an additional bug, regression, or missing functionality not covered by the current PR | "This doesn't handle the case when…", "This will break if…", "The same bug exists in file X" |

A single comment may contain multiple items of different types — handle each sub-item separately.

### Step 4: Handle each comment by type

---

#### Type A — Question: Post a direct answer

Read the relevant code, understand the design, and post a concise factual reply.

```bash
gh api repos/isaac-sim/IsaacLab/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="<ANSWER>"
```

If the comment is a top-level review (not inline), use:
```bash
gh pr comment <PR_NUMBER> --repo isaac-sim/IsaacLab --body "<ANSWER>"
```

Keep answers short and precise. If the question reveals a misunderstanding that warrants a code clarification (e.g., a comment/docstring), address that under Type B.

---

#### Type B — Improvement: Implement the change as a new commit

1. Make the requested change in the codebase.
2. Run pre-commit:
   ```bash
   ./isaaclab.sh -f
   ```
   If pre-commit modifies files, stage them and re-run until clean.
3. Commit the change:
   ```bash
   git add -A
   git reset HEAD -- .agent/
   git commit -m "$(cat <<'EOF'
   <Short imperative description of the improvement>

   Address reviewer feedback: <one-line summary of what was requested>.
   EOF
   )"
   ```
4. If the improvement is substantial enough to mention in the changelog, update `source/<package>/docs/CHANGELOG.rst` and `source/<package>/config/extension.toml` before committing (follow the same bump/format rules as `isaaclab-bug-fix`).

Reply to the reviewer's comment after pushing (Step 6):
```
Done in commit <SHORT_HASH> — <one-sentence summary of what changed>.
```

---

#### Type C — New issue: Search for a prior fix, then act

**Step C-1: Search recent PRs for an existing fix**

```bash
# Search merged PRs for keywords from the issue description
gh pr list --repo isaac-sim/IsaacLab --state merged --limit 50 \
  --json number,title,body,mergedAt \
  | jq '.[] | select(.title | test("<KEYWORD>"; "i"))'

# Also search by label and body keywords
gh search prs --repo isaac-sim/IsaacLab --state merged \
  "<keyword1> <keyword2>" --limit 20 \
  --json number,title,body,mergedAt
```

Additionally search open PRs — a fix may already be in review:
```bash
gh pr list --repo isaac-sim/IsaacLab --state open --limit 50 \
  --json number,title,body,headRefName \
  | jq '.[] | select(.title | test("<KEYWORD>"; "i"))'
```

Use multiple keyword searches to cover different phrasings of the issue.

**Step C-2: Evaluate search results**

For each candidate PR, read its body and diff to confirm it actually addresses the same problem:
```bash
gh pr diff <CANDIDATE_PR_NUMBER> --repo isaac-sim/IsaacLab
```

```
Candidate found?
├─ YES — confirmed overlap → Go to Step C-3 (comment pointing to other PR)
└─ NO  — no relevant prior PR → Go to Step C-4 (implement fix in current PR)
```

**Step C-3: Comment pointing to the existing PR**

Post a comment on the current PR referencing the prior fix:

```bash
gh pr comment <PR_NUMBER> --repo isaac-sim/IsaacLab --body "$(cat <<'EOF'
The issue you raised (re: <brief description>) appears to have been addressed in PR #<OTHER_PR_NUMBER> (_<other PR title>_).

<If merged>:
That fix was merged on <MERGED_DATE>. Once the current PR is rebased onto the latest `<BASE_BRANCH>`, the fix will be included automatically.

<If open>:
That fix is currently open in PR #<OTHER_PR_NUMBER> and not yet merged. We'll track resolution there and avoid duplicating the fix here to keep the diffs clean.

Let me know if the scope there differs from what you had in mind.
EOF
)"
```

**Step C-4: Decide where to fix — current PR or a new one**

Before implementing, assess scope:

```
Is the new issue related to the current PR's topic AND small enough to keep the PR focused?
├─ YES → Fix in the current PR branch (Step C-4a)
└─ NO  → Open a separate PR for the new issue (Step C-4b)
```

Criteria for opening a **separate PR**:
- The issue is in a completely different subsystem or file area from the current PR's changes.
- Fixing it would require touching many additional files, making the diff substantially larger.
- The fix deserves its own changelog entry, review audience, or backport target.

**Step C-4a: Fix in the current PR branch**

Follow the same process as the **isaaclab-bug-fix** skill (Steps 2–5: implement, test, changelog, pre-commit), committing directly to the current PR branch.

```bash
# Implement the fix, then:
./isaaclab.sh -f              # pre-commit
git add -A
git reset HEAD -- .agent/
git commit -m "$(cat <<'EOF'
Fix <short description of new issue>

Address additional issue raised in PR review: <what was wrong and why>.
EOF
)"
```

Update the changelog if the fix touches a `source/<package>/` directory.

After pushing, reply to the reviewer:
```
Fixed in commit <SHORT_HASH>. <One sentence explaining the root cause and what changed.>
```

**Step C-4b: Open a separate PR for the new issue**

Create a new branch from the same base branch as the current PR:

```bash
BASE_BRANCH=$(gh pr view <PR_NUMBER> --repo isaac-sim/IsaacLab --json baseRefName -q .baseRefName)
git checkout origin/$BASE_BRANCH
git checkout -b isaaclab-bot/fix-<short-slug>
```

Implement the fix (same process as **isaaclab-bug-fix**: code, test, changelog, pre-commit, commit), then push and open a PR:

```bash
git push -u origin HEAD
gh pr create --repo isaac-sim/IsaacLab --base $BASE_BRANCH \
  --title "Fix: <short description>" \
  --body "$(cat <<'EOF'
# Description

<Summary of the bug and fix.>

Addresses issue raised in review of PR #<ORIGINAL_PR_NUMBER>.

## Type of change

- Bug fix (non-breaking change which fixes an issue)

## Checklist

- [x] I have read and understood the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)
- [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with `./isaaclab.sh --format`
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] I have updated the changelog and the corresponding version in the extension's `config/extension.toml` file
- [ ] I have added my name to the `CONTRIBUTORS.md` or my name already exists there
EOF
)"
```

Then return to the original PR branch:
```bash
git checkout <ORIGINAL_HEAD_REF>
```

Reply to the reviewer on the original PR:
```
The issue you raised is outside the scope of this PR. I've opened a dedicated fix in PR #<NEW_PR_NUMBER> to keep both diffs focused.
```

### Step 5: Run all changed tests

After any code change (Type B or C-4), run the tests most relevant to the modified code:

```bash
./isaaclab.sh -p -m pytest <RELEVANT_TEST_PATH>
```

If a regression test does not exist for the new fix, add one (following the `isaaclab-bug-fix` guideline: verify it fails without the fix, passes with it).

### Step 6: Push and reply

```bash
git push origin <HEAD_REF_NAME>
```

Then post replies to each addressed comment (as described in the type-specific steps above). Collect all short commit hashes after pushing:
```bash
git log --oneline -5
```

Use the real hash in every reply so the reviewer can navigate directly to the change.

### Step 7: Request re-review (optional)

If all open comments have been addressed, optionally request a re-review:
```bash
gh pr edit <PR_NUMBER> --repo isaac-sim/IsaacLab --add-reviewer <REVIEWER_HANDLE>
# or simply leave a top-level comment summarizing all changes made
gh pr comment <PR_NUMBER> --repo isaac-sim/IsaacLab --body "All review comments addressed — please take another look when you have a chance."
```

## Decision Summary

```
For each review comment
├─ Question
│  └─ Post a direct answer as a reply
├─ Improvement request
│  └─ Implement change → pre-commit → commit → push → reply with hash
└─ New issue
   ├─ Search recent merged/open PRs for existing fix
   │  ├─ Found → comment pointing to other PR → STOP for this item
   │  └─ Not found
   │     ├─ Related & small → fix in current PR → commit → push → reply
   │     └─ Unrelated or large → open separate PR → reply with new PR link
```

## Important Notes

- **Never amend existing commits** when iterating on PR feedback. Always create new commits so the reviewer can easily verify each request was addressed.
- **No AI attribution lines** in commit messages.
- **Stage only project files** — always run `git reset HEAD -- .agent/` before committing.
- **Follow `AGENTS.md` naming and changelog rules** for any new code or changelog entries.
- **Prefer fixing in the current PR** when the issue is closely related and small. Open a separate PR when the fix is unrelated to the current topic or would make the diff too large to review coherently.
- **Isaac Sim source access.** When a reviewer's comment involves behavior in `isaacsim.*` or `omni.*` modules, use the `_isaac_sim` symlink at the repo root to trace through Isaac Sim internals.

## References

- Coding standards and changelog rules: read `AGENTS.md`
- Contributing guidelines: read `docs/source/refs/contributing.rst`
- PR template: read `.github/PULL_REQUEST_TEMPLATE.md`
