---
name: isaaclab-pr-resolve-conflicts
description: Resolve merge conflicts between an IsaacLab pull request branch and its target branch (e.g., develop or main). Checks out the PR branch, rebases or merges the target branch, resolves each conflict by understanding both sides, runs pre-commit and tests, then force-pushes the resolved branch. Use when asked to resolve merge conflicts, rebase a PR, or update a PR branch against its base.
---

# IsaacLab PR Merge Conflict Resolver

Bring a PR branch up to date with its target branch by resolving all merge conflicts.

## Inputs

- **PR number**: GitHub PR `#N`
- **Strategy** *(optional)*: `rebase` (default) or `merge`. Prefer rebase to keep a linear history; use merge only when the branch has already been shared with other contributors who have based work on it.

## Workflow

### Step 1: Fetch the PR metadata

```bash
gh pr view <PR_NUMBER> --repo isaac-sim/IsaacLab \
  --json number,title,headRefName,baseRefName,state,mergeable,mergeStateStatus
```

Check `mergeable` and `mergeStateStatus`. If the PR is already mergeable (`mergeable: MERGEABLE`), report that to the user and stop — no action needed.

### Step 2: Checkout the PR branch

```bash
git fetch origin
git checkout <HEAD_REF_NAME>
```

If the branch is from a fork, use:
```bash
gh pr checkout <PR_NUMBER> --repo isaac-sim/IsaacLab
```

Record the current tip commit before making any changes:
```bash
git rev-parse HEAD   # save as ORIGINAL_TIP
```

### Step 3: Fetch the latest target branch

```bash
git fetch origin <BASE_REF_NAME>
```

Identify exactly which commits are in conflict:
```bash
git log --oneline origin/<BASE_REF_NAME>..HEAD          # commits unique to PR branch
git diff --name-only HEAD...origin/<BASE_REF_NAME>      # files changed on both sides
```

### Step 4: Resolve conflicts (rebase strategy — default)

```bash
git rebase origin/<BASE_REF_NAME>
```

When the rebase stops at a conflict:

```bash
git status          # lists files with conflict markers
git diff            # shows the full conflict diff
```

For each conflicted file:

1. **Read the file** to understand both sides of the conflict:
   - `<<<<<<< HEAD` (or `<<<<<<< <branch>`) — changes from the PR branch
   - `=======` — divider
   - `>>>>>>> origin/<BASE_REF_NAME>` — changes from the target branch

2. **Understand the intent** of each side before resolving:
   - Read the relevant commit messages: `git log --oneline origin/<BASE_REF_NAME> -- <FILE>` and `git log --oneline HEAD -- <FILE>`
   - If the conflict is in code you don't understand, read surrounding context in the file.

3. **Choose the correct resolution**:
   - If both sides add independent changes → keep both, in logical order.
   - If the target branch refactored code the PR also modified → apply the PR's logical change on top of the refactored version.
   - If the changes are semantically identical → keep one copy.
   - If unsure → prefer the PR branch's intent (it's the change being reviewed), but ensure the target branch's invariants are preserved.

4. **Edit the file** to remove all conflict markers and produce a correct result.

5. Stage the resolved file:
   ```bash
   git add <FILE>
   ```

6. Continue the rebase:
   ```bash
   git rebase --continue
   ```
   Use a commit message that accurately describes what the commit does (do not change existing messages unless they are now inaccurate due to the resolution).

Repeat for each conflict stop until the rebase completes:
```bash
# rebase finished when:
git status   # shows "nothing to commit, working tree clean" on the PR branch
```

**If rebase produces an irrecoverable state**, abort and fall back to the merge strategy:
```bash
git rebase --abort
# then proceed with Step 4b
```

#### Step 4b: Resolve conflicts (merge strategy — fallback or explicit)

```bash
git merge origin/<BASE_REF_NAME>
```

Resolve conflicts using the same read-understand-edit-stage process described above, then:

```bash
git add -A
git reset HEAD -- .agent/
git commit -m "$(cat <<'EOF'
Merge origin/<BASE_REF_NAME> into <HEAD_REF_NAME>

Resolve merge conflicts introduced by <brief description of what changed
on the target branch that conflicted>.
EOF
)"
```

### Step 5: Run pre-commit on the resolved files

```bash
./isaaclab.sh -f
```

If pre-commit modifies files, stage them and re-run:
```bash
git add -A
./isaaclab.sh -f
```

If this is a rebase, amend the last commit rather than creating a new one for pre-commit fixes:
```bash
git add -A
git commit --amend --no-edit
```

Repeat until all checks pass.

### Step 6: Run the affected tests

Run tests for every file that had a conflict or was touched during resolution:

```bash
./isaaclab.sh -p -m pytest <RELEVANT_TEST_PATH>
```

If tests fail, diagnose the failure — it may indicate the conflict was resolved incorrectly. Fix the code, re-run pre-commit, and amend or add a commit as appropriate.

### Step 7: Force-push the updated branch

Because rebase rewrites history, a force-push is required:

```bash
git push --force-with-lease origin <HEAD_REF_NAME>
```

Use `--force-with-lease` (not `--force`) — it refuses to push if someone else has pushed to the branch since you fetched, preventing accidental overwrites.

### Step 8: Verify and comment

Confirm the PR is now conflict-free:
```bash
gh pr view <PR_NUMBER> --repo isaac-sim/IsaacLab --json mergeable,mergeStateStatus
```

Post a summary comment on the PR:
```bash
gh pr comment <PR_NUMBER> --repo isaac-sim/IsaacLab --body "$(cat <<'EOF'
Rebased onto `<BASE_REF_NAME>` (now at `<NEW_TIP_SHORT_HASH>`). Resolved the following conflicts:

- `<file1>`: <one sentence describing what each side changed and how it was resolved>
- `<file2>`: <same>

All pre-commit checks pass. Please re-review the updated diff.
EOF
)"
```

## Decision Summary

```
PR has merge conflicts?
├─ NO  → Report already mergeable, stop
└─ YES
   ├─ Fetch PR branch and target branch
   ├─ Attempt rebase onto target
   │  ├─ Conflict → read both sides → resolve → stage → continue
   │  └─ Irrecoverable → abort → fall back to merge
   ├─ Run pre-commit → fix if needed
   ├─ Run affected tests → fix if failing
   ├─ Force-push (--force-with-lease)
   └─ Comment on PR with conflict summary
```

## Important Notes

- **Never force-push to `main` or `develop`** — only force-push the PR's own feature branch.
- **Prefer `--force-with-lease` over `--force`** to avoid overwriting concurrent pushes.
- **Do not change commit messages** during rebase unless the message is now factually wrong after resolution.
- **Rebase is preferred** over merge to keep history linear and make the diff easier to review. Use merge only when the branch is shared with other contributors.
- **Stage only project files** — always run `git reset HEAD -- .agent/` before any merge commit.
- **No AI attribution lines** in commit messages.
- **When in doubt about intent**, read the relevant git log and the surrounding code before resolving — a wrong resolution is worse than stopping and asking.

## References

- Coding standards: read `AGENTS.md`
- Contributing guidelines: read `docs/source/refs/contributing.rst`
